#!/usr/bin/env python3
"""Aggregate raw real-vLLM sessions into small canonical evidence artifacts."""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.metrics import latency_summary_ms


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--order-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sessions = [json.loads(path.read_text()) for path in sorted(args.raw_dir.glob("*.json"))]
    if len(sessions) != 45:
        raise ValueError(f"expected 45 sessions, found {len(sessions)}")
    if sum(x["request_count"] for x in sessions) != 2250:
        raise ValueError("expected 2250 measured requests")
    groups = {}
    for session in sessions:
        groups.setdefault((session["workload_id"], session["candidate_id"]), []).append(session)
    rows = []
    for (workload, candidate), group in sorted(groups.items()):
        requests = [request for session in group for request in session["request_results"]]
        good = [request for request in requests if request.get("ok")]
        tokens = sum(request.get("output_tokens", 0) for request in good)
        wall = sum(session["wall_seconds"] for session in group)
        row = {
            "target_identity": group[0]["target_identity"],
            "model_identity": group[0]["model_identity"],
            "workload_id": workload,
            "candidate_id": candidate,
            "max_num_seqs": group[0]["max_num_seqs"],
            "effective_max_num_seqs": group[0]["effective_default_max_num_seqs"] if group[0]["value_source"] == "default" else group[0]["max_num_seqs"],
            "value_source": group[0]["value_source"],
            "session_count": len(group),
            "request_count": len(requests),
            "success_count": len(good),
            "failure_count": len(requests) - len(good),
            "classification": "VALID" if all(x["classification"] == "VALID" for x in group) else "MIXED_FAILURE",
            "ttft_ms": latency_summary_ms(x["ttft_ms"] for x in good if x.get("ttft_ms") is not None),
            "tpot_ms": latency_summary_ms(x["tpot_ms"] for x in good if x.get("tpot_ms") is not None),
            "e2e_ms": latency_summary_ms(x["e2e_latency_ms"] for x in good),
            "output_token_throughput": round(tokens / wall, 6),
            "request_throughput": round(len(good) / wall, 6),
            "queue_wait_ms": {"p50": "not_available", "p95": "not_available", "p99": "not_available"},
            "prefill_time": "not_available",
            "decode_time": "not_available",
            "kv_cache_usage": "not_available",
            "idle_gpu_memory_mib": round(statistics.mean(x["idle_gpu_memory_mib"] for x in group), 6),
            "model_loaded_gpu_memory_mib": max(x["model_loaded_gpu_memory_mib"] for x in group),
            "peak_gpu_memory_mib": max(x["peak_gpu_memory_mib"] for x in group),
            "after_shutdown_gpu_memory_mib": max(x["after_shutdown_gpu_memory_mib"] for x in group),
            "oom_count": sum(x["oom_count"] for x in group),
            "timeout_count": sum(x["timeout_count"] for x in group),
            "http_server_error_count": sum(x["http_server_error_count"] for x in group),
        }
        rows.append(row)
    first_plan = json.loads(next(args.plans.glob("S1-*.json")).read_text())
    write(args.out / "candidate_matrix.json", first_plan["candidate_matrix"])
    write(args.out / "latency_summary.json", rows)
    write(args.out / "throughput_summary.json", [{k: row[k] for k in ("workload_id", "candidate_id", "output_token_throughput", "request_throughput")} for row in rows])
    write(args.out / "memory_summary.json", [{k: row[k] for k in ("workload_id", "candidate_id", "idle_gpu_memory_mib", "model_loaded_gpu_memory_mib", "peak_gpu_memory_mib", "after_shutdown_gpu_memory_mib")} for row in rows])
    write(args.out / "correctness_and_failures.json", [{k: row[k] for k in ("workload_id", "candidate_id", "classification", "request_count", "success_count", "failure_count", "oom_count", "timeout_count", "http_server_error_count")} for row in rows])
    raw_manifest = []
    for path in sorted(args.raw_dir.glob("*.json")):
        item = json.loads(path.read_text())
        raw_manifest.append({"workload_id": item["workload_id"], "candidate_id": item["candidate_id"], "session": item["session"], "classification": item["classification"], "session_sha256": sha(path), "raw_log_sha256": item["raw_log_sha256"], "plan_sha256": item["plan_sha256"], "server_pid": item["server_pid"], "runtime_launched_max_num_seqs": item["runtime_launched_max_num_seqs"], "runtime_policy_reselection_count": item["runtime_policy_reselection_count"]})
    write(args.out / "session_manifest.json", {"candidate_order_seed": 20260715, "order": json.loads(args.order_log.read_text()), "sessions": raw_manifest})
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"], text=True).strip()
    target = {"hostname": platform.node(), "kernel": platform.platform(), "gpu_csv": gpu, "target_identity": sessions[0]["target_identity"]}
    write(args.out / "target_identity.json", target)
    versions = subprocess.check_output([".venv/bin/python", "-c", "import torch,vllm;print(vllm.__version__);print(torch.__version__);print(torch.version.cuda)"], text=True).splitlines()
    write(args.out / "environment.json", {**target, "python": platform.python_version(), "vllm_version": versions[0], "pytorch_version": versions[1], "pytorch_cuda_runtime": versions[2], "execution_environment": "native_runtime_repository_virtual_environment", "metric_availability": {"TTFT": "measured_directly", "TPOT": "derived_from_stream_timestamps", "E2E": "measured_directly", "throughput": "derived_from_tokens_and_concurrent_wall_time", "queue_wait": "not_available", "prefill_time": "not_available", "decode_time": "not_available", "KV_cache_usage": "not_available", "GPU_memory": "measured_directly_with_nvidia_smi"}})
    source_files = [
        Path(__file__),
        Path(__file__).with_name("generate_vllm_max_num_seqs_workloads.py"),
        Path(__file__).with_name("run_vllm_max_num_seqs_session.py"),
        Path(__file__).with_name("run_vllm_max_num_seqs_matrix.py"),
        Path(__file__).with_name("run_vllm_max_num_seqs_proof_matrix.py"),
        Path(__file__).with_name("finalize_vllm_max_num_seqs_evaluation.py"),
        Path(__file__).resolve().parents[1] / "deployment/vllm_adapter/policy_executor.py",
        args.out / "fixed_configuration.json",
        args.out / "objective_definitions.json",
        args.out / "workload_manifest.json",
    ]
    repo_root = Path(__file__).resolve().parents[1]
    hashes = {str(path.resolve().relative_to(repo_root)): sha(path) for path in source_files}
    compiler_selector = repo_root.parent / "ml-graph-compiler-runtime/tools/select_vllm_max_num_seqs.py"
    hashes["../ml-graph-compiler-runtime/tools/select_vllm_max_num_seqs.py"] = sha(compiler_selector)
    write(args.out / "artifact_provenance.json", {"raw_sessions_location": "outside_source_control:/tmp/vllm-max-num-seqs-raw", "raw_logs_location": "outside_source_control:/tmp/vllm-max-num-seqs-logs", "raw_session_count": len(sessions), "measured_request_count": sum(x["request_count"] for x in sessions), "all_sessions_valid": all(x["classification"] == "VALID" for x in sessions), "all_after_shutdown_gpu_memory_mib": sorted(set(x["after_shutdown_gpu_memory_mib"] for x in sessions)), "source_sha256": hashes, "truth_boundary": "real_target_model_workload_specific_measured_vllm_serving"})


if __name__ == "__main__":
    main()
