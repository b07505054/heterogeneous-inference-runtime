#!/usr/bin/env python3
"""Phase 1 of the perf-model slice: launch real vLLM, run one (workload,
candidate) pair, and capture raw evidence only (no predictions here --
predictions/calibration happen offline in analyze_perf_model_results.py so
the two calibrated constants can be derived once and reused everywhere).

Extends the existing real vLLM diagnostic path (scripts/run_vllm_max_num_seqs_
diagnostic.py) by reusing its server_command/readiness/warmup/measurement
pattern, and adds:
  - VLLM_SERVER_DEV_MODE=1 + a GET /server_info capture right after readiness
  - a GET /metrics capture after warmup and again after the measured rounds
    (delta of the two isolates measured-window phase histograms from warmup)
  - a fixed-reference correctness request, compared against a stored baseline
  - multiple measured repetition rounds (not one pass) for distribution stats
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.backends.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from deployment.vllm_adapter import metrics_client, server_info_client


def gpu_memory_used_mib() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return sum(int(x.strip()) for x in out.splitlines() if x.strip())
    except Exception:
        return -1


def build_command(fixed: dict, max_num_seqs: int | None, port: int, python: str) -> list[str]:
    cmd = [
        python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", fixed["model"], "--tokenizer", fixed["tokenizer"], "--dtype", fixed["dtype"],
        "--max-model-len", str(fixed["max_model_len"]),
        "--gpu-memory-utilization", str(fixed["gpu_memory_utilization"]),
        "--block-size", str(fixed["block_size"]),
        "--max-num-batched-tokens", str(fixed["max_num_batched_tokens"]),
        "--tensor-parallel-size", str(fixed["tensor_parallel_size"]),
        "--pipeline-parallel-size", str(fixed["pipeline_parallel_size"]),
        "--served-model-name", fixed["served_model_name"],
        "--host", "127.0.0.1", "--port", str(port),
    ]
    cmd.append("--enable-chunked-prefill" if fixed["enable_chunked_prefill"] else "--no-enable-chunked-prefill")
    cmd.append("--enable-prefix-caching" if fixed["enable_prefix_caching"] else "--no-enable-prefix-caching")
    if max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(max_num_seqs)])
    return cmd


REFERENCE_PROMPT = "The capital of France is"


def run_reference_completion(port: int) -> dict:
    import requests
    payload = {"model": "qwen2.5-0.5b", "prompt": REFERENCE_PROMPT, "max_tokens": 16,
               "temperature": 0.0, "seed": 1234, "logprobs": 0}
    try:
        resp = requests.post(f"http://127.0.0.1:{port}/v1/completions", json=payload, timeout=60)
        body = resp.json() if resp.content else None
        text = (body.get("choices") or [{}])[0].get("text") if body else None
        finish_reason = (body.get("choices") or [{}])[0].get("finish_reason") if body else None
        return {"http_status": resp.status_code, "text": text, "finish_reason": finish_reason,
                "usage": body.get("usage") if body else None}
    except Exception as exc:
        return {"http_status": -1, "error": str(exc)}


def attention_backend_from_log(log_path: Path) -> str | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    match = re.search(r"Using (\S+) attention backend", text)
    return match.group(1) if match else None


def oom_occurred(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="replace")
    return bool(re.search(r"out of memory|CUDA OOM|OutOfMemoryError", text, re.I))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--reference-baseline", type=Path, required=True,
                         help="Path storing the first-seen reference completion text for this workload; "
                              "later candidates are compared against it.")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=240)
    args = parser.parse_args()

    fixed = json.loads(args.fixed.read_text())
    manifest = json.loads(args.workload_manifest.read_text())
    workload = next(w for w in manifest["workloads"] if w["workload_id"] == args.workload)

    python = ".venv/bin/python"  # relative to repo root cwd, matches policy_executor.py convention
    command = build_command(fixed, args.max_num_seqs, args.port, python)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.reference_baseline.parent.mkdir(parents=True, exist_ok=True)

    idle_mib = gpu_memory_used_mib()
    log = args.log.open("w")
    env = os.environ.copy()
    env["VLLM_HOST_IP"] = "127.0.0.1"
    env["VLLM_SERVER_DEV_MODE"] = "1"
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)

    classification = "VALID"
    startup_seconds = None
    server_info_raw = None
    server_info_error = None
    post_warmup_metrics_text = None
    final_metrics_text = None
    reference_result = None
    pooled_rows: list[dict] = []
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

        # endpoint MUST be /v1/completions: our prompts are exact token-ID arrays
        # (see generate_perf_model_workloads.py), and OpenAICompatibleBackend's
        # default /v1/chat/completions endpoint stringifies non-string prompts
        # into chat message text, silently corrupting the exact token count this
        # whole workload design depends on.
        backend = OpenAICompatibleBackend(OpenAICompatibleConfig(
            base_url=f"http://127.0.0.1:{args.port}", model=fixed["served_model_name"],
            concurrency=workload["concurrency"], timeout_s=120, record_timeline=True,
            endpoint="/v1/completions",
        ))
        for request in workload["requests"][: workload["warmup_requests"]]:
            backend.execute(request, True)
        peak_mib = max(peak_mib, gpu_memory_used_mib())

        post_warmup_metrics_text = metrics_client.fetch_metrics_text(args.port)

        measured = workload["requests"][workload["warmup_requests"]:]
        for _round in range(args.repetitions):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workload["concurrency"]) as executor:
                futures = []
                for idx, request in enumerate(measured):
                    submitted = time.perf_counter()
                    futures.append(executor.submit(
                        lambda r=request, s=submitted, i=idx: {
                            **backend.execute(r), "submit_time": s, "request_index": i,
                        }
                    ))
                pooled_rows.extend(future.result() for future in futures)
            peak_mib = max(peak_mib, gpu_memory_used_mib())

        final_metrics_text = metrics_client.fetch_metrics_text(args.port)
        reference_result = run_reference_completion(args.port)

    except RuntimeError as exc:
        classification = str(exc)
    finally:
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

    good_rows = [r for r in pooled_rows if r.get("ok")]
    result = {
        "raw_result_schema_version": "perf_model.raw_result.v1",
        "workload_id": args.workload,
        "candidate_id": args.candidate_id,
        "max_num_seqs_requested": args.max_num_seqs,
        "repetitions": args.repetitions,
        "classification": classification if len(good_rows) == len(pooled_rows) else "REQUEST_FAILURE",
        "command": command,
        "fixed_configuration": fixed,
        "startup_seconds": startup_seconds,
        "server_pid": process.pid,
        "server_info_raw": server_info_raw,
        "server_info_error": server_info_error,
        "attention_backend_from_log": attn_backend,
        "post_warmup_metrics_text": post_warmup_metrics_text,
        "final_metrics_text": final_metrics_text,
        "reference_completion": reference_result,
        "reference_baseline": reference_baseline or reference_result,
        "reference_match": reference_match,
        "oom_detected_in_log": oom,
        "idle_gpu_memory_mib": idle_mib,
        "peak_gpu_memory_mib": peak_mib,
        "after_shutdown_gpu_memory_mib": after_mib,
        "process_cleanup_status": cleanup_status,
        "request_count": len(pooled_rows),
        "success_count": len(good_rows),
        "failure_count": len(pooled_rows) - len(good_rows),
        "pooled_request_rows": pooled_rows,
        "workload_definition": {k: v for k, v in workload.items() if k != "requests"},
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} classification={result['classification']} success={len(good_rows)}/{len(pooled_rows)}")


if __name__ == "__main__":
    main()
