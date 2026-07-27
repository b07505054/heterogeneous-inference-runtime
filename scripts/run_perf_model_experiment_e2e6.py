#!/usr/bin/env python3
"""E2E-6: N simultaneously-launched, non-staggered, identical requests
decoding together at a fixed batch size, with a common validated steady-
state measurement window (tokens 9..40 by default). max_num_seqs is fixed
at 8 for every group (>= the largest tested batch size) to remove admission/
capacity effects entirely, per this slice's explicit isolation goal.

Reuses build_command / gpu_memory_used_mib / attention_backend_from_log /
oom_occurred / run_reference_completion (E2E-2), OpenAICompatibleBackend
(warmup only), and perf_model.token_timeline (E2E-3/4/5, unmodified).

Optional torch-profiler groups (--enable-profiler) use vLLM's own
/start_profile /stop_profile REST endpoints (--profiler-config.profiler=torch
CLI flag) -- no vLLM source patch. Profiled groups are run SEPARATELY from
the quantitative sweep and their timing is excluded from model fitting
(profiler overhead can perturb latency), used only for kernel attribution.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.backends.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from deployment.vllm_adapter import metrics_client, server_info_client
from perf_model.token_timeline import stream_completion_with_timeline
from scripts.run_perf_model_experiment import (
    build_command, gpu_memory_used_mib, attention_backend_from_log, oom_occurred, run_reference_completion,
)


def sample_loop(port: int, samples: list[dict], stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            text = metrics_client.fetch_metrics_text(port)
            parsed = metrics_client.parse_prometheus_text(text)
            samples.append({
                "time": time.perf_counter(),
                "num_requests_running": metrics_client.gauge_value(parsed, "vllm:num_requests_running"),
                "num_requests_waiting": metrics_client.gauge_value(parsed, "vllm:num_requests_waiting"),
                "kv_cache_usage_perc": metrics_client.gauge_value(parsed, "vllm:kv_cache_usage_perc"),
            })
        except metrics_client.MetricsUnavailable:
            pass
        time.sleep(0.1)


def trigger_profile(port: int, action: str) -> None:
    url = f"http://127.0.0.1:{port}/{action}_profile"
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        assert response.status == 200, f"{action}_profile returned {response.status}"


def gpu_utilization_sample() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        vals = [x.strip() for x in out.split(",")]
        return {"gpu_util_percent": float(vals[0]), "memory_used_mib": float(vals[1]),
                "power_draw_w": float(vals[2]) if len(vals) > 2 else None}
    except Exception:
        return {}


def cpu_sample(pid: int) -> dict:
    try:
        out = subprocess.check_output(["ps", "-o", "%cpu,nlwp", "-p", str(pid)], text=True).strip().splitlines()
        if len(out) >= 2:
            cpu, nlwp = out[1].split()
            return {"cpu_percent": float(cpu), "thread_count": int(nlwp)}
    except Exception:
        pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--enable-profiler", action="store_true")
    parser.add_argument("--profiler-dir", type=Path, default=None)
    parser.add_argument("--profiler-record-shapes", action="store_true")
    parser.add_argument("--profiler-with-flops", action="store_true")
    parser.add_argument("--profiler-pre-wait-s", type=float, default=3.0)
    parser.add_argument("--profiler-window-s", type=float, default=3.0)
    parser.add_argument("--reference-baseline", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=240)
    # E2E-8: tiny-M GEMV dispatch (OOT LogitsProcessor override, zero vLLM source
    # patch). Defaults preserve exact E2E-6 behavior (wrapper=False -> unchanged
    # `-m vllm.entrypoints.openai.api_server` launch; tiny_m_enable unset -> the
    # wrapper, if used, is behaviorally identical to stock vLLM).
    parser.add_argument("--use-tiny-m-launcher", action="store_true",
                         help="Launch via scripts/launch_vllm_with_tiny_m.py instead of -m vllm..."
                              " Safe for baseline runs too: registration is a no-op unless --tiny-m-enable.")
    parser.add_argument("--tiny-m-enable", action="store_true")
    parser.add_argument("--tiny-m-ops", default="lm_head")
    parser.add_argument("--tiny-m-threshold", type=int, default=8)
    # E2E-9: opt-in unified-selector LM-head path (see
    # perf_model/tiny_m_oot_logits_processor.py's VLLM_TINY_M_UNIFIED_SELECTOR
    # branch). Mutually exclusive with --tiny-m-enable in practice (both flip
    # the same call site); default off preserves exact E2E-6/E2E-8 behavior.
    parser.add_argument("--use-unified-selector", action="store_true")
    args = parser.parse_args()

    fixed = json.loads(args.fixed.read_text())
    manifest = json.loads(args.workload_manifest.read_text())
    requests = manifest["pools"][str(args.prompt_length)][: args.batch_size]
    assert len(requests) == args.batch_size

    python = ".venv/bin/python"
    command = build_command(fixed, args.max_num_seqs, args.port, python)
    if args.use_tiny_m_launcher:
        launcher = str(Path(__file__).resolve().parent / "launch_vllm_with_tiny_m.py")
        # build_command's first 3 elements are [python, "-m", "vllm.entrypoints.openai.api_server"];
        # replace only that prefix, keep every resolved CLI arg identical.
        assert command[:3] == [python, "-m", "vllm.entrypoints.openai.api_server"]
        command = [python, launcher] + command[3:]
    if args.enable_profiler:
        assert args.profiler_dir is not None
        command += ["--profiler-config.profiler=torch", f"--profiler-config.torch_profiler_dir={args.profiler_dir}"]
        if args.profiler_record_shapes:
            command += ["--profiler-config.torch_profiler_record_shapes=true"]
        if args.profiler_with_flops:
            command += ["--profiler-config.torch_profiler_with_flops=true"]
        args.profiler_dir.mkdir(parents=True, exist_ok=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.reference_baseline.parent.mkdir(parents=True, exist_ok=True)

    idle_mib = gpu_memory_used_mib()
    log = args.log.open("w")
    env = os.environ.copy()
    env["VLLM_HOST_IP"] = "127.0.0.1"
    env["VLLM_SERVER_DEV_MODE"] = "1"
    if args.tiny_m_enable:
        env["VLLM_TINY_M_GEMV_ENABLE"] = "1"
        env["VLLM_TINY_M_GEMV_OPS"] = args.tiny_m_ops
        env["VLLM_TINY_M_GEMV_THRESHOLD"] = str(args.tiny_m_threshold)
    if args.use_unified_selector:
        env["VLLM_TINY_M_UNIFIED_SELECTOR"] = "1"
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)

    classification = "VALID"
    startup_seconds = None
    server_info_raw = None
    server_info_error = None
    pre_metrics_text = None
    final_metrics_text = None
    reference_result = None
    timelines: list[dict] = []
    samples: list[dict] = []
    gpu_samples: list[dict] = []
    profiler_started = False
    profiler_stopped = False
    peak_mib = idle_mib

    try:
        deadline = time.time() + args.startup_timeout
        launch_start = time.perf_counter()
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError("SERVER_START_FAILED")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as response:
                    if response.status == 200:
                        startup_seconds = time.perf_counter() - launch_start
                        break
            except Exception:
                time.sleep(1)
        if startup_seconds is None:
            raise RuntimeError("READINESS_TIMEOUT")

        try:
            server_info_raw = server_info_client.fetch_server_info(args.port)
        except server_info_client.ServerInfoUnavailable as exc:
            server_info_error = str(exc)

        peak_mib = max(peak_mib, gpu_memory_used_mib())

        warmup_backend = OpenAICompatibleBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{args.port}", model=fixed["served_model_name"], concurrency=1,
            timeout_s=120, endpoint="/v1/completions",
        ))
        for _ in range(2):
            warmup_backend.execute(requests[0], True)
        peak_mib = max(peak_mib, gpu_memory_used_mib())

        pre_metrics_text = metrics_client.fetch_metrics_text(args.port)

        stop_sampling = threading.Event()
        sampler_thread = threading.Thread(target=sample_loop, args=(args.port, samples, stop_sampling), daemon=True)
        sampler_thread.start()

        base_url = f"http://127.0.0.1:{args.port}"
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as pool:
            futures = [
                pool.submit(stream_completion_with_timeline, base_url, fixed["served_model_name"],
                            req["metadata"]["request_id"], req["prompt"], max_tokens=req["max_tokens"],
                            temperature=req["temperature"], seed=req["seed"], ignore_eos=req.get("ignore_eos", True))
                for req in requests
            ]
            if args.enable_profiler:
                time.sleep(args.profiler_pre_wait_s)  # let requests reach steady state before profiling
                try:
                    trigger_profile(args.port, "start")
                    profiler_started = True
                except Exception:
                    pass
                gpu_samples.append(gpu_utilization_sample())
                time.sleep(args.profiler_window_s)  # short, bounded profiling window
                gpu_samples.append(gpu_utilization_sample())
                if profiler_started:
                    try:
                        trigger_profile(args.port, "stop")
                        profiler_stopped = True
                    except Exception:
                        pass
            else:
                for _ in range(8):
                    gpu_samples.append(gpu_utilization_sample())
                    time.sleep(0.5)
            timelines = [f.result().to_dict() for f in futures]

        stop_sampling.set()
        sampler_thread.join(2)

        final_metrics_text = metrics_client.fetch_metrics_text(args.port)
        reference_result = run_reference_completion(args.port)

    except RuntimeError as exc:
        classification = str(exc)
    finally:
        cpu_info = cpu_sample(process.pid) if process.pid else {}
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        cleanup_status = "graceful_sigterm"
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            cleanup_status = "forced_sigkill"
        log.close()
        time.sleep(2)
        after_mib = gpu_memory_used_mib()

    attn_backend = attention_backend_from_log(args.log)
    oom = oom_occurred(args.log)

    reference_baseline = None
    reference_match = None
    if args.reference_baseline.exists():
        reference_baseline = json.loads(args.reference_baseline.read_text())
        if reference_result and reference_result.get("http_status") == 200:
            reference_match = reference_result.get("text") == reference_baseline.get("text")
    elif reference_result and reference_result.get("http_status") == 200:
        args.reference_baseline.write_text(json.dumps(reference_result, indent=2))

    good = [t for t in timelines if t.get("ok")]
    if len(good) != len(timelines):
        classification = "REQUEST_FAILURE"

    result = {
        "raw_result_schema_version": "perf_model.e2e6.raw_result.v1",
        "batch_size": args.batch_size, "prompt_length": args.prompt_length, "candidate_id": args.candidate_id,
        "max_num_seqs_requested": args.max_num_seqs, "enable_profiler": args.enable_profiler,
        "use_tiny_m_launcher": args.use_tiny_m_launcher, "tiny_m_enable": args.tiny_m_enable,
        "tiny_m_ops": args.tiny_m_ops if args.tiny_m_enable else None,
        "tiny_m_threshold": args.tiny_m_threshold if args.tiny_m_enable else None,
        "use_unified_selector": args.use_unified_selector,
        "profiler_started": profiler_started, "profiler_stopped": profiler_stopped,
        "profiler_dir": str(args.profiler_dir) if args.profiler_dir else None,
        "classification": classification,
        "command": command, "fixed_configuration": fixed,
        "startup_seconds": startup_seconds, "server_pid": process.pid,
        "server_info_raw": server_info_raw, "server_info_error": server_info_error,
        "attention_backend_from_log": attn_backend,
        "pre_metrics_text": pre_metrics_text, "final_metrics_text": final_metrics_text,
        "reference_completion": reference_result, "reference_baseline": reference_baseline or reference_result,
        "reference_match": reference_match, "oom_detected_in_log": oom,
        "idle_gpu_memory_mib": idle_mib, "peak_gpu_memory_mib": peak_mib, "after_shutdown_gpu_memory_mib": after_mib,
        "process_cleanup_status": cleanup_status, "cpu_info": cpu_info,
        "success_count": len(good), "failure_count": len(timelines) - len(good),
        "timelines": timelines, "running_waiting_samples": samples, "gpu_samples": gpu_samples,
        "output_tokens": manifest["output_tokens"],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} classification={classification} success={len(good)}/{len(timelines)}")


if __name__ == "__main__":
    main()
