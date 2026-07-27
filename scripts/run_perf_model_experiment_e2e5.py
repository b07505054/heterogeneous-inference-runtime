#!/usr/bin/env python3
"""E2E-5: max_num_seqs x active-anchor-count x admitted-request-count
capacity-deficit experiment. Generalizes E2E-4's single-anchor staggered
admission to a multi-anchor barrier: all active anchors must independently
reach `release_after_tokens` real streamed tokens before admissions release.

Reuses build_command / gpu_memory_used_mib / attention_backend_from_log /
oom_occurred / run_reference_completion (E2E-2), OpenAICompatibleBackend
(unmodified), and perf_model.token_timeline (E2E-3/E2E-4, unmodified).
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
from perf_model.token_timeline import stream_completion_with_timeline, TokenTimeline
from scripts.run_perf_model_experiment import (
    build_command, gpu_memory_used_mib, attention_backend_from_log, oom_occurred, run_reference_completion,
)


def sample_running_waiting(port: int) -> dict | None:
    try:
        text = metrics_client.fetch_metrics_text(port)
        parsed = metrics_client.parse_prometheus_text(text)
        return {
            "time": time.perf_counter(),
            "num_requests_running": metrics_client.gauge_value(parsed, "vllm:num_requests_running"),
            "num_requests_waiting": metrics_client.gauge_value(parsed, "vllm:num_requests_waiting"),
            "kv_cache_usage_perc": metrics_client.gauge_value(parsed, "vllm:kv_cache_usage_perc"),
        }
    except metrics_client.MetricsUnavailable:
        return None


def run_round(base_url: str, port: int, model: str, anchors: list[dict], admissions: list[dict], release_after_tokens: int, round_index: int):
    barrier_events = [threading.Event() for _ in anchors]
    samples: list[dict] = []
    stop_sampling = threading.Event()

    def sampler():
        while not stop_sampling.is_set():
            s = sample_running_waiting(port)
            if s:
                samples.append(s)
            time.sleep(0.12)

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    def submit_admissions():
        backend = OpenAICompatibleBackend(OpenAICompatibleConfig(
            base_url=base_url, model=model, concurrency=max(1, len(admissions)), timeout_s=120,
            record_timeline=True, endpoint="/v1/completions",
        ))
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(admissions))) as executor:
            futures = []
            for idx, request in enumerate(admissions):
                submitted = time.perf_counter()
                futures.append(executor.submit(
                    lambda r=request, s=submitted, i=idx: {**backend.execute(r), "submit_time": s,
                                                            "request_index": i, "round": round_index}
                ))
            rows = [f.result() for f in futures]
        return rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(anchors) + 1) as pool:
        anchor_futures = [
            pool.submit(stream_completion_with_timeline, base_url, model, a["metadata"]["request_id"], a["prompt"],
                        max_tokens=a["max_tokens"], temperature=a["temperature"], seed=a["seed"],
                        ignore_eos=a.get("ignore_eos", True), release_after_tokens=release_after_tokens,
                        on_release=barrier_events[i].set)
            for i, a in enumerate(anchors)
        ]
        all_released = all(e.wait(timeout=90) for e in barrier_events)
        admission_rows = submit_admissions() if all_released and admissions else []
        anchor_timelines = [f.result() for f in anchor_futures]

    stop_sampling.set()
    sampler_thread.join(2)

    return [t.to_dict() for t in anchor_timelines], admission_rows, all_released, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--active-anchor-count", type=int, required=True)
    parser.add_argument("--admitted-request-count", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--release-after-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--reference-baseline", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=240)
    args = parser.parse_args()

    fixed = json.loads(args.fixed.read_text())
    manifest = json.loads(args.workload_manifest.read_text())
    anchors = manifest["anchors"][: args.active_anchor_count]
    admissions = manifest["admissions"][: args.admitted_request_count]

    python = ".venv/bin/python"
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
    pre_round_metrics_text = None
    final_metrics_text = None
    reference_result = None
    all_anchor_timelines: list[list[dict]] = []
    admission_pooled_rows: list[dict] = []
    all_running_waiting_samples: list[dict] = []
    all_released_overall = True
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
            warmup_backend.execute(manifest["anchors"][0], True)
        peak_mib = max(peak_mib, gpu_memory_used_mib())

        pre_round_metrics_text = metrics_client.fetch_metrics_text(args.port)

        base_url = f"http://127.0.0.1:{args.port}"
        for _round in range(args.repetitions):
            timelines, admission_rows, released, samples = run_round(
                base_url, args.port, fixed["served_model_name"], anchors, admissions,
                args.release_after_tokens, _round,
            )
            all_anchor_timelines.append(timelines)
            admission_pooled_rows.extend(admission_rows)
            all_running_waiting_samples.extend(samples)
            all_released_overall = all_released_overall and released
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

    flat_timelines = [t for round_timelines in all_anchor_timelines for t in round_timelines]
    good_anchors = [t for t in flat_timelines if t.get("ok")]
    good_admissions = [r for r in admission_pooled_rows if r.get("ok")]
    total_requests = len(flat_timelines) + len(admission_pooled_rows)
    total_success = len(good_anchors) + len(good_admissions)

    if not all_released_overall:
        classification = "RELEASE_BARRIER_TIMEOUT"

    result = {
        "raw_result_schema_version": "perf_model.e2e5.raw_result.v1",
        "active_anchor_count": args.active_anchor_count, "admitted_request_count": args.admitted_request_count,
        "max_num_seqs_requested": args.max_num_seqs, "candidate_id": args.candidate_id,
        "release_after_tokens": args.release_after_tokens, "repetitions": args.repetitions,
        "classification": classification if total_success == total_requests and all_released_overall else "REQUEST_FAILURE",
        "command": command, "fixed_configuration": fixed,
        "startup_seconds": startup_seconds, "server_pid": process.pid,
        "server_info_raw": server_info_raw, "server_info_error": server_info_error,
        "attention_backend_from_log": attn_backend,
        "pre_round_metrics_text": pre_round_metrics_text, "final_metrics_text": final_metrics_text,
        "reference_completion": reference_result, "reference_baseline": reference_baseline or reference_result,
        "reference_match": reference_match, "oom_detected_in_log": oom,
        "idle_gpu_memory_mib": idle_mib, "peak_gpu_memory_mib": peak_mib, "after_shutdown_gpu_memory_mib": after_mib,
        "process_cleanup_status": cleanup_status,
        "request_count": total_requests, "success_count": total_success, "failure_count": total_requests - total_success,
        "anchor_timelines_by_round": all_anchor_timelines, "admission_pooled_rows": admission_pooled_rows,
        "running_waiting_samples": all_running_waiting_samples,
        "prompt_tokens": manifest["prompt_tokens"], "anchor_output_tokens": manifest["anchor_output_tokens"],
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} classification={result['classification']} success={total_success}/{total_requests}")


if __name__ == "__main__":
    main()
