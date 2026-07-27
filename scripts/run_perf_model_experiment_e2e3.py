#!/usr/bin/env python3
"""E2E-3: controlled chunked-prefill x max_num_seqs experiment with per-token
timelines and (optionally) staggered arrival.

Reuses the E2E-2 pipeline directly (imported, not duplicated):
  - build_command / gpu_memory_used_mib / attention_backend_from_log /
    oom_occurred / run_reference_completion from run_perf_model_experiment.py
  - OpenAICompatibleBackend for the "rest" of the concurrent requests
    (aggregate ttft/tpot/e2e, exactly as in E2E-2)
  - deployment.vllm_adapter.server_info_client / metrics_client, unchanged

New, narrow addition: perf_model.token_timeline captures a full per-token
arrival timeline for exactly one anchor request per round ("request 0"),
which is the request whose decode progress we are checking for interruption
by newly-admitted prefill work. This keeps the change minimal: every other
request in the workload still goes through the unmodified E2E-2 backend.
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
from perf_model.token_timeline import stream_completion_with_timeline, inter_token_stats
from scripts.run_perf_model_experiment import (
    build_command, gpu_memory_used_mib, attention_backend_from_log, oom_occurred,
    run_reference_completion,
)


def run_round(
    base_url: str, model: str, measured_requests: list[dict], concurrency: int, arrival_mode: str,
) -> tuple[dict, list[dict]]:
    """Returns (request0_timeline_dict, rest_pooled_rows)."""
    req0 = measured_requests[0]
    rest = measured_requests[1:]

    first_token_event = threading.Event()

    def submit_rest():
        rest_backend = OpenAICompatibleBackend(OpenAICompatibleConfig(
            base_url=base_url, model=model, concurrency=max(1, concurrency - 1), timeout_s=120,
            record_timeline=True, endpoint="/v1/completions",
        ))
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency - 1)) as executor:
            futures = []
            for idx, request in enumerate(rest):
                submitted = time.perf_counter()
                futures.append(executor.submit(
                    lambda r=request, s=submitted, i=idx: {**rest_backend.execute(r), "submit_time": s, "request_index": i + 1}
                ))
            rows = [f.result() for f in futures]
        return rows

    if arrival_mode == "burst" or not rest:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            req0_future = executor.submit(
                stream_completion_with_timeline, base_url, model, req0["metadata"]["request_id"],
                req0["prompt"], max_tokens=req0["max_tokens"], temperature=req0["temperature"],
                seed=req0["seed"], ignore_eos=req0.get("ignore_eos", True),
            )
            rest_future = executor.submit(submit_rest) if rest else None
            timeline = req0_future.result()
            rest_rows = rest_future.result() if rest_future else []
    elif arrival_mode == "staggered":
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            req0_future = executor.submit(
                stream_completion_with_timeline, base_url, model, req0["metadata"]["request_id"],
                req0["prompt"], max_tokens=req0["max_tokens"], temperature=req0["temperature"],
                seed=req0["seed"], ignore_eos=req0.get("ignore_eos", True),
                on_first_token=first_token_event.set,
            )
            got_first_token = first_token_event.wait(timeout=60)
            rest_rows = submit_rest() if got_first_token else []
            timeline = req0_future.result()
    else:
        raise ValueError(f"unknown arrival_mode {arrival_mode!r}")

    return timeline.to_dict(), rest_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--enable-chunked-prefill", choices=("true", "false"), required=True)
    parser.add_argument("--arrival-mode", choices=("burst", "staggered"), required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--reference-baseline", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=240)
    args = parser.parse_args()

    fixed = json.loads(args.fixed.read_text())
    chunked_prefill_requested = args.enable_chunked_prefill == "true"
    fixed = {**fixed, "enable_chunked_prefill": chunked_prefill_requested}

    manifest = json.loads(args.workload_manifest.read_text())
    workload = next(w for w in manifest["workloads"] if w["workload_id"] == args.workload)

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
    post_warmup_metrics_text = None
    final_metrics_text = None
    reference_result = None
    request0_timelines: list[dict] = []
    rest_pooled_rows: list[dict] = []
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
            base_url=f"http://127.0.0.1:{args.port}", model=fixed["served_model_name"],
            concurrency=workload["concurrency"], timeout_s=120, endpoint="/v1/completions",
        ))
        for request in workload["requests"][: workload["warmup_requests"]]:
            warmup_backend.execute(request, True)
        peak_mib = max(peak_mib, gpu_memory_used_mib())

        post_warmup_metrics_text = metrics_client.fetch_metrics_text(args.port)

        measured = workload["requests"][workload["warmup_requests"]:]
        base_url = f"http://127.0.0.1:{args.port}"
        for _round in range(args.repetitions):
            timeline_dict, rest_rows = run_round(
                base_url, fixed["served_model_name"], measured, workload["concurrency"], args.arrival_mode,
            )
            request0_timelines.append(timeline_dict)
            rest_pooled_rows.extend(rest_rows)
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

    good_rest = [r for r in rest_pooled_rows if r.get("ok")]
    good_req0 = [t for t in request0_timelines if t.get("ok")]
    total_requests = len(rest_pooled_rows) + len(request0_timelines)
    total_success = len(good_rest) + len(good_req0)

    req0_stats = []
    for t in request0_timelines:
        from perf_model.token_timeline import TokenTimeline
        tl = TokenTimeline(**t)
        req0_stats.append(inter_token_stats(tl))

    result = {
        "raw_result_schema_version": "perf_model.e2e3.raw_result.v1",
        "workload_id": args.workload, "candidate_id": args.candidate_id,
        "max_num_seqs_requested": args.max_num_seqs,
        "enable_chunked_prefill_requested": chunked_prefill_requested,
        "arrival_mode": args.arrival_mode,
        "repetitions": args.repetitions,
        "classification": classification if total_success == total_requests else "REQUEST_FAILURE",
        "command": command, "fixed_configuration": fixed,
        "startup_seconds": startup_seconds, "server_pid": process.pid,
        "server_info_raw": server_info_raw, "server_info_error": server_info_error,
        "attention_backend_from_log": attn_backend,
        "post_warmup_metrics_text": post_warmup_metrics_text, "final_metrics_text": final_metrics_text,
        "reference_completion": reference_result, "reference_baseline": reference_baseline or reference_result,
        "reference_match": reference_match, "oom_detected_in_log": oom,
        "idle_gpu_memory_mib": idle_mib, "peak_gpu_memory_mib": peak_mib, "after_shutdown_gpu_memory_mib": after_mib,
        "process_cleanup_status": cleanup_status,
        "request_count": total_requests, "success_count": total_success, "failure_count": total_requests - total_success,
        "request0_timelines": request0_timelines, "request0_inter_token_stats": req0_stats,
        "rest_pooled_request_rows": rest_pooled_rows,
        "workload_definition": {k: v for k, v in workload.items() if k != "requests"},
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} classification={result['classification']} success={total_success}/{total_requests}")


if __name__ == "__main__":
    main()
