#!/usr/bin/env python3
"""E2E-4: prompt-length x admitted-multiplicity x max_num_seqs controlled
experiment, using the E2E-3 dynamically-triggered staggered admission
generalized to release after `release_after_tokens` (default 8, giving a
real pre-admission baseline window) instead of the first token.

Reuses build_command / gpu_memory_used_mib / attention_backend_from_log /
oom_occurred / run_reference_completion from E2E-2's orchestrator unchanged,
and OpenAICompatibleBackend for the admitted requests (their own aggregate
ttft/tpot/e2e, exactly as in E2E-2/E2E-3).
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
    build_command, gpu_memory_used_mib, attention_backend_from_log, oom_occurred, run_reference_completion,
)


def run_round(base_url: str, model: str, anchor: dict, admission_requests: list[dict], release_after_tokens: int,
              round_index: int = 0):
    release_event = threading.Event()

    def submit_admissions():
        backend = OpenAICompatibleBackend(OpenAICompatibleConfig(
            base_url=base_url, model=model, concurrency=max(1, len(admission_requests)), timeout_s=120,
            record_timeline=True, endpoint="/v1/completions",
        ))
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(admission_requests))) as executor:
            futures = []
            for idx, request in enumerate(admission_requests):
                submitted = time.perf_counter()
                futures.append(executor.submit(
                    lambda r=request, s=submitted, i=idx: {**backend.execute(r), "submit_time": s, "request_index": i,
                                                            "round": round_index}
                ))
            rows = [f.result() for f in futures]
        return rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as anchor_executor:
        anchor_future = anchor_executor.submit(
            stream_completion_with_timeline, base_url, model, anchor["metadata"]["request_id"], anchor["prompt"],
            max_tokens=anchor["max_tokens"], temperature=anchor["temperature"], seed=anchor["seed"],
            ignore_eos=anchor.get("ignore_eos", True), release_after_tokens=release_after_tokens,
            on_release=release_event.set,
        )
        released = release_event.wait(timeout=60)
        admission_rows = submit_admissions() if released and admission_requests else []
        timeline = anchor_future.result()

    return timeline.to_dict(), admission_rows, released


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--multiplicity", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--release-after-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--reference-baseline", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=240)
    args = parser.parse_args()

    fixed = json.loads(args.fixed.read_text())
    manifest = json.loads(args.workload_manifest.read_text())
    anchor = manifest["anchor"]
    admission_requests = manifest["admission_pools"][str(args.prompt_length)][: args.multiplicity]

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
    anchor_timelines: list[dict] = []
    admission_pooled_rows: list[dict] = []
    all_released = True
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
            warmup_backend.execute(anchor, True)
        peak_mib = max(peak_mib, gpu_memory_used_mib())

        post_warmup_metrics_text = metrics_client.fetch_metrics_text(args.port)

        base_url = f"http://127.0.0.1:{args.port}"
        for _round in range(args.repetitions):
            timeline_dict, admission_rows, released = run_round(
                base_url, fixed["served_model_name"], anchor, admission_requests, args.release_after_tokens,
                round_index=_round,
            )
            anchor_timelines.append(timeline_dict)
            admission_pooled_rows.extend(admission_rows)
            all_released = all_released and released
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

    good_admissions = [r for r in admission_pooled_rows if r.get("ok")]
    good_anchors = [t for t in anchor_timelines if t.get("ok")]
    total_requests = len(admission_pooled_rows) + len(anchor_timelines)
    total_success = len(good_admissions) + len(good_anchors)

    from perf_model.token_timeline import TokenTimeline
    anchor_stats = [inter_token_stats(TokenTimeline(**t)) for t in anchor_timelines]

    if not all_released:
        classification = "RELEASE_EVENT_TIMEOUT"

    result = {
        "raw_result_schema_version": "perf_model.e2e4.raw_result.v1",
        "prompt_length": args.prompt_length, "multiplicity": args.multiplicity,
        "max_num_seqs_requested": args.max_num_seqs, "candidate_id": args.candidate_id,
        "release_after_tokens": args.release_after_tokens, "repetitions": args.repetitions,
        "classification": classification if total_success == total_requests and all_released else "REQUEST_FAILURE",
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
        "anchor_timelines": anchor_timelines, "anchor_inter_token_stats": anchor_stats,
        "admission_pooled_rows": admission_pooled_rows,
        "anchor_definition": {"prompt_tokens": len(anchor["prompt"]), "output_tokens": anchor["max_tokens"]},
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} classification={result['classification']} success={total_success}/{total_requests}")


if __name__ == "__main__":
    main()
