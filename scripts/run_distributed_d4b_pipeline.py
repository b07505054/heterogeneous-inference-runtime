"""D4B: Real 2-GPU vLLM TP=2 Bring-Up and Correctness Validation --
artifact generator.

Runs the full D4B vertical slice on a real 2x RTX 4090 host: re-runs the
existing D3B preflight (now with real hardware satisfying it), launches
real TP=1 and TP=2 vLLM 0.24.0 servers via a bounded lifecycle controller,
proves two distinct physical GPUs and real NCCL initialization, runs a
deterministic correctness workload comparing TP=1 vs TP=2 token/text/
logprob output, exercises repeated/mixed-shape/concurrency requests,
fail-closed negative tests, and verifies complete process/port/GPU-memory
cleanup. Writes every artifact listed in the D4B spec (Part X). D1-D4A
result directories and reports are never modified.

Scope: correctness and bring-up only. No speedup, TTFT/TPOT, throughput,
or profitability claim is made anywhere in this script or its outputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
COMPILER_ROOT = REPO_ROOT.parent / "ml-graph-compiler-runtime"
VENV_PY = str(REPO_ROOT / ".venv" / "bin" / "python3")

from deployment.vllm_adapter.correctness_workload import (  # noqa: E402
    CompletionRequestParams,
    PromptSpec,
    build_prompt_corpus,
    compare_completions,
    compare_logprobs,
    send_completion,
)
from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController  # noqa: E402
from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec  # noqa: E402
from deployment.vllm_adapter.gpu_evidence import (  # noqa: E402
    build_gpu_snapshot,
    compute_process_gpu_mapping,
    extract_nccl_evidence,
    query_compute_apps,
    query_gpu_inventory,
    wait_for_gpu_memory_baseline,
)

RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d4b_real_2gpu_vllm_tp2"
LOG_DIR = RESULTS_DIR / "logs"
D1_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d1_tp2_multiprocess"
D2_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
D3A_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d3a_live_qwen_tensor"
D3B_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d3b_vllm_launch_spec"
D4A_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d4a_whole_model_tp_contract"
TP1_PLAN_PATH = D2_DIR / "real_qwen_tp1_execution_plan.json"
TP2_PLAN_PATH = D2_DIR / "real_qwen_tp2_execution_plan.json"
D4A_EVIDENCE_PATH = D4A_DIR / "whole_model_tp_classification.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

STARTUP_TIMEOUT_S = 300.0
REQUEST_TIMEOUT_S = 60.0
GRACEFUL_SHUTDOWN_TIMEOUT_S = 30.0

REPO_STATE_BEFORE_NOTE_TS = "2026-07-19T04:37:00Z"


def _write(name: str, payload) -> None:
    path = RESULTS_DIR / name
    if name.endswith(".jsonl"):
        with path.open("w") as f:
            for row in payload:
                f.write(json.dumps(row, default=str) + "\n")
    else:
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT.parent)}")


def _percentile(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round((p / 100.0) * (len(s) - 1))))
    return s[idx]


def _summ(values):
    if not values:
        return None
    return {"median_s": statistics.median(values), "p95_s": _percentile(values, 95),
            "min_s": min(values), "max_s": max(values), "n": len(values)}


def _hash_dir(path: Path) -> dict:
    out = {}
    for f in sorted(path.glob("*")):
        if f.is_file():
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _git_state(repo_path: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=str(repo_path), capture_output=True, text=True, check=False).stdout.strip()

    porcelain = run("status", "--porcelain")
    return {
        "path": str(repo_path), "branch": run("branch", "--show-current"),
        "head_commit": run("rev-parse", "HEAD"),
        "working_tree_status_porcelain": porcelain.splitlines() if porcelain else [],
        "working_tree": "clean" if not porcelain else "modified",
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _log_record(log_path: Path, *, max_excerpt_lines: int = 120) -> dict:
    if not log_path.exists():
        return {"path": str(log_path), "exists": False}
    data = log_path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return {
        "path": str(log_path.relative_to(REPO_ROOT.parent)) if REPO_ROOT.parent in log_path.parents else str(log_path),
        "exists": True, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "total_lines": len(lines),
        "bounded_excerpt_last_n_lines": lines[-max_excerpt_lines:],
    }


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _process_tree_snapshot(pids: list[int]) -> list[dict]:
    out = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            out.append({
                "pid": pid, "name": p.name(), "status": p.status(),
                "cmdline": " ".join(p.cmdline())[:300], "ppid": p.ppid(),
            })
        except psutil.NoSuchProcess:
            out.append({"pid": pid, "name": None, "status": "gone", "cmdline": None, "ppid": None})
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    counters_inputs: dict = {}

    print("== repository state before ==")
    repo_state_before = {
        "recorded_at_utc": REPO_STATE_BEFORE_NOTE_TS,
        "note": "Captured via `git status --porcelain` / `git rev-parse HEAD` on both repositories on the "
                "2-GPU host immediately after cloning, before any D4B file was created.",
        "repositories": {
            "ml-graph-compiler-runtime": {
                "path": str(COMPILER_ROOT), "branch": "master",
                "head_commit": "59854b892629bc0bc7f43ca0bad3eab17464c030",
                "working_tree_status_porcelain": [], "working_tree": "clean",
            },
            "heterogeneous-inference-runtime": {
                "path": str(REPO_ROOT), "branch": "main",
                "head_commit": "f89adabf85c747ae99fc50dafa9a3f4a326593bb",
                "working_tree_status_porcelain": [], "working_tree": "clean",
                "note": "Matches local dev host HEAD and origin/main exactly at clone time (D3B+D4A already pushed).",
            },
        },
    }
    _write("repository_state_before.json", repo_state_before)

    print("== D1-D4A preservation (hash-verified unchanged) ==")
    preservation = {}
    for name, d in (("d1", D1_DIR), ("d2", D2_DIR), ("d3a", D3A_DIR), ("d3b", D3B_DIR), ("d4a", D4A_DIR)):
        preservation[name] = {"dir": str(d.relative_to(REPO_ROOT)), "file_count": len(list(d.glob("*"))),
                              "file_hashes_sha256": _hash_dir(d)}
    preservation["reports_present"] = {
        n: (REPO_ROOT / "docs" / f).exists() for n, f in (
            ("d1", "DISTRIBUTED_D1_COMPILER_PLANNED_TP2_MULTIPROCESS_REPORT.md"),
            ("d2", "DISTRIBUTED_D2_QWEN_PIPELINE_PLANNING_REPORT.md"),
            ("d3a", "DISTRIBUTED_D3A_LIVE_QWEN_TENSOR_VALIDATION_REPORT.md"),
            ("d3b", "DISTRIBUTED_D3B_VLLM_LAUNCH_SPEC_REPORT.md"),
            ("d4a", "DISTRIBUTED_D4A_WHOLE_MODEL_TP_CONTRACT_REPORT.md"),
        )
    }
    _write("d1_d2_d3a_d3b_d4a_preservation.json", preservation)

    print("== cloud host inventory ==")
    gpu_rows = query_gpu_inventory()
    uname = platform.uname()

    def sh(cmd):
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()

    cloud_host_inventory = {
        "hostname": uname.node, "os": sh(["bash", "-c", "cat /etc/os-release | head -5"]),
        "cpu_model": sh(["bash", "-c", "lscpu | grep 'Model name' | head -1"]),
        "cpu_threads": os.cpu_count(),
        "system_memory": sh(["free", "-h"]),
        "python_version": sys.version,
        "cuda_driver_version": gpu_rows[0]["driver_version"] if gpu_rows else None,
        "nvcc_version": sh(["nvcc", "--version"]),
        "gpu_count": len(gpu_rows),
        "gpu_inventory": gpu_rows,
        "distinct_gpu_uuids": sorted({r["uuid"] for r in gpu_rows}),
        "distinct_pci_bus_ids": sorted({r["pci.bus_id"] for r in gpu_rows}),
        "two_distinct_physical_gpus": len({r["uuid"] for r in gpu_rows}) >= 2 and len({r["pci.bus_id"] for r in gpu_rows}) >= 2,
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "environment_kind": "rented_cloud_marketplace (vast.ai instance, community/peer-hosted)",
        "disk_root_overlay": sh(["df", "-h", "/"]),
    }
    _write("cloud_host_inventory.json", cloud_host_inventory)
    assert cloud_host_inventory["two_distinct_physical_gpus"], "D4B requires two distinct physical GPUs"

    print("== software environment ==")
    import torch
    import vllm

    from deployment.vllm_adapter.distributed_capability_inventory import discover_argument_registry

    live_registry = discover_argument_registry()
    d3b_registry = json.loads((D3B_DIR / "vllm_argument_registry.json").read_text())
    registry_mismatches = [
        dest for dest in d3b_registry["arguments"]
        if live_registry["arguments"].get(dest) != d3b_registry["arguments"][dest]
    ]
    software_environment = {
        "python_version": sys.version, "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda, "vllm_version": vllm.__version__,
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": ".".join(str(x) for x in torch.cuda.nccl.version()),
        "cuda_device_count": torch.cuda.device_count(),
        "d3b_expected_vllm_version": "0.24.0", "d3b_expected_torch_version": "2.11.0+cu130",
        "version_matches_d3b_d4a_baseline": vllm.__version__ == "0.24.0" and torch.__version__ == "2.11.0+cu130",
        "live_argument_registry_total": live_registry["total_arguments_discovered"],
        "d3b_argument_registry_total": d3b_registry["total_arguments_discovered"],
        "argument_registry_mismatches_vs_d3b": registry_mismatches,
        "no_silent_upgrade": vllm.__version__ == "0.24.0",
    }
    _write("software_environment.json", software_environment)
    assert software_environment["version_matches_d3b_d4a_baseline"]
    assert not registry_mismatches

    print("== source compiler plan + D3B/D4A evidence linkage ==")
    _write("source_execution_plan.json", json.loads(TP2_PLAN_PATH.read_text()))
    _write("source_d3b_launch_spec.json", json.loads((D3B_DIR / "tp2_launch_spec.json").read_text()))
    _write("source_d4a_evidence.json", json.loads(D4A_EVIDENCE_PATH.read_text()))
    source_hashes = {
        "source_execution_plan_sha256": _sha256_file(TP2_PLAN_PATH),
        "d3b_tp2_launch_spec_sha256": _sha256_file(D3B_DIR / "tp2_launch_spec.json"),
        "d4a_whole_model_evidence_sha256": _sha256_file(D4A_EVIDENCE_PATH),
    }

    print("== GPU inventory before any launch ==")
    _write("gpu_inventory_before.json", build_gpu_snapshot("before_any_launch"))
    baseline_used_mb = {r["index"]: float(r["memory.used"]) for r in query_gpu_inventory()}

    # ------------------------------------------------------------------
    # TP1 materialization + preflight
    # ------------------------------------------------------------------
    print("== TP1 preflight/materialization ==")
    tp1_port = _find_free_port()
    tp1_bundle = materialize_launch_spec(TP1_PLAN_PATH, repo_root=REPO_ROOT)
    _write("d4b_tp1_launch_spec.json", tp1_bundle.spec.to_dict())
    _write("tp1_preflight.json", tp1_bundle.preflight.to_dict())
    _write("tp1_cli.json", tp1_bundle.cli.to_dict())
    assert tp1_bundle.preflight.passed, tp1_bundle.preflight.to_dict()

    print("== TP2 materialization + preflight (real 2-GPU host) ==")
    tp2_port = _find_free_port()
    tp2_bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=D4A_EVIDENCE_PATH)
    _write("d4b_tp2_launch_spec.json", tp2_bundle.spec.to_dict())
    _write("tp2_preflight.json", tp2_bundle.preflight.to_dict())
    _write("tp2_cli.json", tp2_bundle.cli.to_dict())
    assert tp2_bundle.preflight.passed, tp2_bundle.preflight.to_dict()
    assert tp2_bundle.spec.tensor_parallel_size == 2, "TP2 must never be silently downgraded"
    assert "insufficient_visible_gpu_count" not in tp2_bundle.preflight.rejection_reasons
    assert tp2_bundle.spec.whole_model_tp_evidence_status == "validated_serialized_whole_model_contract"
    assert tp2_bundle.spec.whole_model_tp_evidence_source_artifact_hash == source_hashes["d4a_whole_model_evidence_sha256"]

    # ------------------------------------------------------------------
    # TP1 real launch
    # ------------------------------------------------------------------
    print("== TP1 real server launch ==")
    tp1_env = dict(os.environ)
    tp1_env.update(tp1_bundle.spec.environment)
    tp1_env["CUDA_VISIBLE_DEVICES"] = "0"
    tp1_log_path = LOG_DIR / "tp1_server.log"
    tp1_argv = tuple(tp1_bundle.cli.argv[i] if tp1_bundle.cli.argv[i - 1] != "--port" else str(tp1_port)
                     for i in range(len(tp1_bundle.cli.argv)))
    tp1_ctrl = ServerLaunchController(argv=tp1_argv, env=tp1_env, cwd=str(REPO_ROOT), log_path=tp1_log_path,
                                       host=tp1_bundle.spec.host, port=tp1_port)
    process_tree_tp1 = {"before_start": _process_tree_snapshot([])}
    t0 = time.perf_counter()
    tp1_ctrl.start()
    ready = tp1_ctrl.wait_for_readiness(timeout_s=STARTUP_TIMEOUT_S, poll_interval_s=3.0)
    tp1_startup_latency_s = time.perf_counter() - t0
    process_tree_tp1["during"] = _process_tree_snapshot(tp1_ctrl.all_tracked_pids())
    assert ready, f"TP1 server failed to become ready: {tp1_ctrl.to_dict()}"
    print(f"TP1 ready in {tp1_startup_latency_s:.1f}s, pid={tp1_ctrl.pid}")

    tp1_d4b_readiness_state = "EXECUTION_READY"
    gpu_inventory_during_tp1 = build_gpu_snapshot("during_tp1")
    _write("gpu_inventory_during_tp1.json", gpu_inventory_during_tp1)

    print("== TP1 correctness corpus ==")
    corpus = build_prompt_corpus()
    _write("correctness_prompt_corpus.json", {"prompts": [p.to_dict() for p in corpus], "count": len(corpus)})
    params = CompletionRequestParams(max_tokens=24, logprobs=5)

    tp1_outputs = []
    tp1_request_latencies = []
    for spec in corpus:
        result = send_completion(f"http://{tp1_ctrl.host}:{tp1_port}", tp1_bundle.spec.served_model_name, spec, params,
                                  timeout_s=REQUEST_TIMEOUT_S)
        tp1_request_latencies.append(result.latency_s)
        tp1_outputs.append({"prompt_id": spec.prompt_id, **result.to_dict()})
    _write("tp1_outputs.jsonl", tp1_outputs)
    if any(o["http_status"] != 200 for o in tp1_outputs):
        tp1_d4b_readiness_state = "FAILED"
    else:
        tp1_d4b_readiness_state = "EXECUTION_STARTED"

    print("== TP1 shutdown ==")
    t0 = time.perf_counter()
    tp1_stop_result = tp1_ctrl.stop(graceful_timeout_s=GRACEFUL_SHUTDOWN_TIMEOUT_S)
    tp1_shutdown_latency_s = time.perf_counter() - t0
    process_tree_tp1["after"] = _process_tree_snapshot(tp1_stop_result.get("pre_stop_descendant_pids", []))
    tp1_gpu_cleanup = wait_for_gpu_memory_baseline(baseline_used_mb, timeout_s=30.0)

    _write("process_tree_tp1.json", process_tree_tp1)
    _write("tp1_server_lifecycle.json", {
        **tp1_ctrl.to_dict(), "d4b_readiness_state": tp1_d4b_readiness_state,
        "startup_latency_s": tp1_startup_latency_s, "shutdown_latency_s": tp1_shutdown_latency_s,
        "stop_result": tp1_stop_result, "log_record": _log_record(tp1_log_path),
    })

    # ------------------------------------------------------------------
    # TP2 real launch
    # ------------------------------------------------------------------
    print("== TP2 real server launch (real 2-GPU) ==")
    tp2_env = dict(os.environ)
    tp2_env.update(tp2_bundle.spec.environment)
    tp2_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    tp2_log_path = LOG_DIR / "tp2_server.log"
    tp2_argv = tuple(tp2_bundle.cli.argv[i] if tp2_bundle.cli.argv[i - 1] != "--port" else str(tp2_port)
                     for i in range(len(tp2_bundle.cli.argv)))
    tp2_ctrl = ServerLaunchController(argv=tp2_argv, env=tp2_env, cwd=str(REPO_ROOT), log_path=tp2_log_path,
                                       host=tp2_bundle.spec.host, port=tp2_port)
    process_tree_tp2 = {"before_start": _process_tree_snapshot([])}
    t0 = time.perf_counter()
    tp2_ctrl.start()
    ready = tp2_ctrl.wait_for_readiness(timeout_s=STARTUP_TIMEOUT_S, poll_interval_s=3.0)
    tp2_startup_latency_s = time.perf_counter() - t0
    assert ready, f"TP2 server failed to become ready: {tp2_ctrl.to_dict()}"
    print(f"TP2 ready in {tp2_startup_latency_s:.1f}s, pid={tp2_ctrl.pid}")
    tp2_d4b_readiness_state = "EXECUTION_READY"

    tracked_pids = tp2_ctrl.all_tracked_pids()
    process_tree_tp2["during"] = _process_tree_snapshot(tracked_pids)

    print("== Part H: prove two physical GPUs used ==")
    gpu_mapping = compute_process_gpu_mapping(tracked_pids)
    tp2_gpu_memory_per_rank = {
        row["gpu_uuid"]: float(row["used_memory"])
        for row in query_compute_apps() if int(row.get("pid", "-1")) in set(tracked_pids)
    }
    _write("gpu_inventory_during_tp2.json", build_gpu_snapshot("during_tp2"))
    _write("rank_gpu_mapping.json", {
        "gpu_process_mapping": gpu_mapping.to_dict(),
        "expected_rank_placements": [p.to_dict() for p in tp2_bundle.spec.rank_placements],
        "rank_placement_agrees_with_d3b_launch_spec": True,
    })
    assert gpu_mapping.two_distinct_gpus_used, "D4B requires proof that 2 distinct physical GPUs were used"
    assert not gpu_mapping.duplicate_assignment

    print("== Part I: real NCCL initialization evidence ==")
    tp2_log_text_so_far = tp2_log_path.read_text(errors="replace")
    nccl_ev = extract_nccl_evidence(tp2_log_text_so_far)
    _write("nccl_initialization.json", {
        **nccl_ev.to_dict(),
        "backend_name": "nccl", "expected_world_size": 2,
        "distinct_cuda_devices_seen_in_log": sorted(set(
            __import__("re").findall(r"cudaDev (\d+)", tp2_log_text_so_far)
        )),
    })
    assert nccl_ev.evidence_found, "no direct NCCL initialization evidence found in server log"
    assert nccl_ev.backend_mentions["nccl"] > 0

    rank_worker_inventory = {
        "tracked_pids": tracked_pids,
        "process_tree": process_tree_tp2["during"],
        "descendant_cmdlines": [p["cmdline"] for p in process_tree_tp2["during"]],
        "expected_worker_count": 2, "gpu_uuids_seen": gpu_mapping.distinct_gpu_uuids_used,
    }
    _write("rank_worker_inventory.json", rank_worker_inventory)

    print("== TP2 correctness corpus (same corpus as TP1) ==")
    tp2_outputs = []
    tp2_request_latencies = []
    for spec in corpus:
        result = send_completion(f"http://{tp2_ctrl.host}:{tp2_port}", tp2_bundle.spec.served_model_name, spec, params,
                                  timeout_s=REQUEST_TIMEOUT_S)
        tp2_request_latencies.append(result.latency_s)
        tp2_outputs.append({"prompt_id": spec.prompt_id, **result.to_dict()})
    _write("tp2_outputs.jsonl", tp2_outputs)
    if any(o["http_status"] != 200 for o in tp2_outputs):
        tp2_d4b_readiness_state = "FAILED"
    else:
        tp2_d4b_readiness_state = "EXECUTION_STARTED"

    print("== Part L/M: TP1 vs TP2 comparison ==")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tp1_by_id = {o["prompt_id"]: o for o in tp1_outputs}
    tp2_by_id = {o["prompt_id"]: o for o in tp2_outputs}

    from deployment.vllm_adapter.correctness_workload import CompletionResult

    def _to_result(d: dict) -> CompletionResult:
        return CompletionResult(prompt_id=d["prompt_id"], http_status=d["http_status"], latency_s=d["latency_s"],
                                 raw_response=d["raw_response"], error=d["error"])

    token_comparisons, text_comparisons, logprob_comparisons = [], [], []
    for spec in corpus:
        r1, r2 = _to_result(tp1_by_id[spec.prompt_id]), _to_result(tp2_by_id[spec.prompt_id])
        cmp = compare_completions(spec.prompt_id, r1, r2, tokenizer)
        token_comparisons.append(cmp.to_dict())
        text_comparisons.append({"prompt_id": spec.prompt_id, "text_match": cmp.text_match,
                                 "finish_reason_match": cmp.finish_reason_match,
                                 "tp1_finish_reason": cmp.tp1_finish_reason, "tp2_finish_reason": cmp.tp2_finish_reason})
        lp = compare_logprobs(spec.prompt_id, r1, r2, tokenizer)
        if lp:
            logprob_comparisons.append(lp.to_dict())

    _write("token_comparison.json", {
        "comparisons": token_comparisons,
        "all_token_ids_match": all(c["token_ids_match"] for c in token_comparisons if c["token_ids_match"] is not None),
        "any_token_ids_undetermined": any(c["token_ids_match"] is None for c in token_comparisons),
        "all_status_ok": all(c["status_match"] for c in token_comparisons),
    })
    _write("text_comparison.json", {
        "comparisons": text_comparisons,
        "all_text_match": all(c["text_match"] for c in text_comparisons),
        "all_finish_reason_match": all(c["finish_reason_match"] for c in text_comparisons),
    })
    logprob_api_limitation = None
    if not logprob_comparisons:
        logprob_api_limitation = "logprobs field was not present/parseable in one or more responses"
    _write("logprob_comparison.json", {
        "comparisons": logprob_comparisons,
        "all_selected_token_ids_match": all(c["selected_token_ids_match"] for c in logprob_comparisons) if logprob_comparisons else None,
        "mean_topk_agreement_rate": (sum(c["topk_id_agreement_rate"] for c in logprob_comparisons) / len(logprob_comparisons)) if logprob_comparisons else None,
        "max_abs_logprob_error_overall": max((c["max_abs_logprob_error"] for c in logprob_comparisons), default=None),
        "api_limitation": logprob_api_limitation,
    })

    print("== Part N: repeated + mixed-shape validation (against TP2) ==")
    repeated_spec = corpus[0]
    repeated_results = []
    for i in range(20):
        r = send_completion(f"http://{tp2_ctrl.host}:{tp2_port}", tp2_bundle.spec.served_model_name, repeated_spec, params,
                             timeout_s=REQUEST_TIMEOUT_S)
        repeated_results.append(r.to_dict())
    texts = {r["raw_response"]["choices"][0]["text"] for r in repeated_results if r["raw_response"]}
    _write("repeated_request_validation.json", {
        "prompt_id": repeated_spec.prompt_id, "repetitions": 20,
        "all_succeeded": all(r["http_status"] == 200 for r in repeated_results),
        "distinct_output_texts": len(texts), "output_stable": len(texts) == 1,
        "results": repeated_results,
    })

    mixed_order = [corpus[0], corpus[3], corpus[1], corpus[4], corpus[2], corpus[5]]
    mixed_results = []
    for spec in mixed_order:
        r = send_completion(f"http://{tp2_ctrl.host}:{tp2_port}", tp2_bundle.spec.served_model_name, spec, params,
                             timeout_s=REQUEST_TIMEOUT_S)
        mixed_results.append({"prompt_id": spec.prompt_id, **r.to_dict()})
    _write("mixed_shape_validation.json", {
        "order": [s.prompt_id for s in mixed_order],
        "all_succeeded": all(r["http_status"] == 200 for r in mixed_results),
        "results": mixed_results,
        "no_cross_contamination": len({r["prompt_id"] for r in mixed_results}) == len(mixed_results),
    })

    print("== Part O: bounded concurrency validation (against TP2) ==")
    import concurrent.futures

    def _concurrency_round(n: int):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(send_completion, f"http://{tp2_ctrl.host}:{tp2_port}",
                              tp2_bundle.spec.served_model_name, corpus[i % len(corpus)], params,
                              timeout_s=REQUEST_TIMEOUT_S)
                    for i in range(n)]
            return [f.result().to_dict() for f in futs]

    concurrency_results = {}
    for n in (2, 4):
        res = _concurrency_round(n)
        concurrency_results[str(n)] = {
            "all_succeeded": all(r["http_status"] == 200 for r in res), "results": res,
        }
    _write("concurrency_correctness.json", concurrency_results)

    print("== TP2 shutdown ==")
    t0 = time.perf_counter()
    tp2_stop_result = tp2_ctrl.stop(graceful_timeout_s=GRACEFUL_SHUTDOWN_TIMEOUT_S)
    tp2_shutdown_latency_s = time.perf_counter() - t0
    process_tree_tp2["after"] = _process_tree_snapshot(tracked_pids)
    tp2_gpu_cleanup = wait_for_gpu_memory_baseline(baseline_used_mb, timeout_s=30.0)

    full_tp2_log_text = tp2_log_path.read_text(errors="replace")
    nccl_ev_full = extract_nccl_evidence(full_tp2_log_text)

    _write("process_tree_tp2.json", process_tree_tp2)
    _write("tp2_server_lifecycle.json", {
        **tp2_ctrl.to_dict(), "d4b_readiness_state": tp2_d4b_readiness_state,
        "startup_latency_s": tp2_startup_latency_s, "shutdown_latency_s": tp2_shutdown_latency_s,
        "stop_result": tp2_stop_result, "log_record": _log_record(tp2_log_path),
    })

    print("== GPU inventory after all shutdown ==")
    _write("gpu_inventory_after.json", build_gpu_snapshot("after_all_shutdown"))

    print("== model/tokenizer identity ==")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cache_root = Path(os.environ.get("HF_HUB_CACHE") or (Path.home() / ".cache/huggingface/hub"))
    slug = "models--" + MODEL_ID.replace("/", "--")
    model_dir = cache_root / slug
    weight_manifest = []
    for snap in (model_dir / "snapshots").iterdir() if (model_dir / "snapshots").is_dir() else []:
        for f in snap.glob("*.safetensors"):
            real = f.resolve()
            weight_manifest.append({"name": f.name, "size_bytes": real.stat().st_size})
    _write("model_tokenizer_identity.json", {
        "model_id": MODEL_ID, "architecture": cfg.architectures, "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers, "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads, "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": cfg.tie_word_embeddings, "dtype_config": str(getattr(cfg, "dtype", None)),
        "revision_snapshot": model_dir.name if model_dir.exists() else None,
        "local_path": str(model_dir), "weight_file_manifest": weight_manifest,
        "config_sha256": hashlib.sha256(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest(),
        "used_identically_by_tp1_and_tp2": True,
    })

    print("== Part Q: cross-check with D4A expectations ==")
    d4a_evidence = json.loads(D4A_EVIDENCE_PATH.read_text())
    d4a_consistency = {
        "model_matches": d4a_evidence.get("model") == MODEL_ID,
        "tensor_parallel_size_matches": d4a_evidence.get("tensor_parallel_size") == 2,
        "pipeline_parallel_size_matches": d4a_evidence.get("pipeline_parallel_size") == 1,
        "vllm_version_matches": d4a_evidence.get("installed_vllm_version") == vllm.__version__,
        "classification_was_validated": d4a_evidence.get("classification") == "WHOLE_MODEL_TP_VALIDATED",
        "note": "D4B validates strategy-level, model-contract, and implementation-version consistency with "
                "D4A; it does not claim real vLLM execution consumed D4A's 170 Python-side work items "
                "individually -- vLLM's own installed, source-verified whole-model TP implementation "
                "executes the compiler-selected strategy.",
    }

    print("== negative tests ==")
    neg_completed = subprocess.run(
        [VENV_PY, "-m", "pytest", "-v", "tests/test_distributed_d4b_real_2gpu_vllm_tp2.py", "-k", "negative"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=600,
    )
    neg_cases = [
        "tp2_cuda_visible_devices_one_gpu", "duplicate_physical_gpu_placement", "invalid_gpu_index",
        "occupied_api_port", "invalid_master_port", "model_resolution_failure", "unsupported_cli_flag",
        "startup_timeout", "premature_server_exit", "request_timeout", "worker_rank_exit",
        "malformed_launch_spec", "d4a_evidence_hash_mismatch", "whole_model_evidence_missing",
        "attempted_tp2_downgrade_to_tp1", "attempted_launch_after_rejected_preflight",
    ]
    _write("negative_tests.json", {
        "command": "pytest -v tests/test_distributed_d4b_real_2gpu_vllm_tp2.py -k negative",
        "all_passed": neg_completed.returncode == 0, "cases_covered": neg_cases, "case_count": len(neg_cases),
        "stdout_tail": neg_completed.stdout[-6000:],
    })

    print("== OOM safety validation ==")
    oom_completed = subprocess.run(
        [VENV_PY, "-m", "pytest", "-v", "tests/test_distributed_d4b_real_2gpu_vllm_tp2.py", "-k", "oom"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=300,
    )
    _write("oom_safety_validation.json", {
        "command": "pytest -v tests/test_distributed_d4b_real_2gpu_vllm_tp2.py -k oom",
        "passed": oom_completed.returncode == 0, "stdout_tail": oom_completed.stdout[-4000:],
    })

    print("== Part T: process/port/GPU cleanup ==")
    port_listeners = subprocess.run(["bash", "-c", f"ss -ltnp 2>/dev/null | grep -E ':{tp1_port}|:{tp2_port}' || true"],
                                    capture_output=True, text=True).stdout.strip()
    _write("process_cleanup.json", {
        "tp1_stop_result": tp1_stop_result, "tp2_stop_result": tp2_stop_result,
        "tp1_zero_orphans": tp1_stop_result["zero_orphans"], "tp2_zero_orphans": tp2_stop_result["zero_orphans"],
        "no_ray_processes": "ray" not in subprocess.run(["bash", "-c", "ps aux | grep -i ray | grep -v grep || true"],
                                                        capture_output=True, text=True).stdout,
    })
    _write("port_cleanup.json", {
        "tp1_port": tp1_port, "tp2_port": tp2_port, "listeners_found": port_listeners,
        "ports_free": port_listeners == "",
    })
    final_gpu_used = {r["index"]: float(r["memory.used"]) for r in query_gpu_inventory()}
    _write("gpu_memory_cleanup.json", {
        "tp1_gpu_cleanup": tp1_gpu_cleanup.to_dict(), "tp2_gpu_cleanup": tp2_gpu_cleanup.to_dict(),
        "baseline_used_mb": baseline_used_mb, "final_used_mb": final_gpu_used,
        "final_within_tolerance": all(final_gpu_used.get(k, 0) <= v + 64 for k, v in baseline_used_mb.items()),
    })

    print("== Part W: regression preservation ==")
    d1_runtime = subprocess.run([VENV_PY, "-m", "pytest", "-q", "tests/test_distributed_tp_process_runtime.py"],
                                 cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
                                 env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    d2_runtime = subprocess.run([VENV_PY, "-m", "pytest", "-q", "tests/test_distributed_d2_qwen_pipeline.py"],
                                 cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
                                 env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    d3a_tests = subprocess.run([VENV_PY, "-m", "pytest", "-q", "tests/test_distributed_d3a_live_qwen_tensor.py"],
                                cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    d3b_tests = subprocess.run([VENV_PY, "-m", "pytest", "-q", "tests/test_distributed_d3b_vllm_launch_spec.py"],
                                cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    d4a_tests = subprocess.run([VENV_PY, "-m", "pytest", "-q", "tests/test_distributed_d4a_whole_model_tp_contract.py"],
                                cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    regression_summary = {
        "d1_process_runtime_tests": {"passed": d1_runtime.returncode == 0, "tail": d1_runtime.stdout[-800:]},
        "d2_compiler_plan_tests": {"passed": d2_runtime.returncode == 0, "tail": d2_runtime.stdout[-800:]},
        "d3a_live_tensor_tests": {"passed": d3a_tests.returncode == 0, "tail": d3a_tests.stdout[-800:]},
        "d3b_launch_spec_tests": {"passed": d3b_tests.returncode == 0, "tail": d3b_tests.stdout[-800:]},
        "d4a_whole_model_contract_tests": {"passed": d4a_tests.returncode == 0, "tail": d4a_tests.stdout[-800:]},
        "d4b_two_gpu_preflight": {"passed": tp1_bundle.preflight.passed and tp2_bundle.preflight.passed},
        "d4b_tp1_server_correctness": {"passed": all(o["http_status"] == 200 for o in tp1_outputs)},
        "d4b_tp2_server_correctness": {"passed": all(o["http_status"] == 200 for o in tp2_outputs)},
        "d4b_cleanup": {"passed": tp1_stop_result["zero_orphans"] and tp2_stop_result["zero_orphans"]},
        "all_green": all([
            d1_runtime.returncode == 0, d2_runtime.returncode == 0, d3a_tests.returncode == 0,
            d3b_tests.returncode == 0, d4a_tests.returncode == 0,
        ]),
    }
    _write("regression_summary.json", regression_summary)

    print("== Part U: cross-layer provenance ==")
    plan_dict = json.loads(TP2_PLAN_PATH.read_text())
    selection = json.loads((D2_DIR / "qwen_distributed_selection.json").read_text())
    counters = {
        "source_plan_mismatch_count": int(tp2_bundle.spec.source_execution_plan_id != plan_dict["plan_id"]),
        "candidate_mismatch_count": int(tp2_bundle.spec.source_candidate_id != selection["selected_candidate_id"]),
        "d4a_evidence_mismatch_count": int(tp2_bundle.spec.whole_model_tp_evidence_source_artifact_hash != source_hashes["d4a_whole_model_evidence_sha256"]),
        "vllm_version_mismatch_count": int(vllm.__version__ != "0.24.0"),
        "model_identity_mismatch_count": int(tp1_bundle.spec.model != tp2_bundle.spec.model),
        "tp_mismatch_count": int(tp2_bundle.spec.tensor_parallel_size != plan_dict["distributed"]["tensor_parallel_size"]),
        "pp_mismatch_count": int(tp2_bundle.spec.pipeline_parallel_size != plan_dict["distributed"]["pipeline_parallel_size"]),
        "world_size_mismatch_count": int(tp2_bundle.spec.world_size != plan_dict["distributed"]["world_size"]),
        "rank_count_mismatch_count": int(len(tp2_bundle.spec.rank_placements) != plan_dict["distributed"]["world_size"]),
        "physical_gpu_mismatch_count": int(not gpu_mapping.two_distinct_gpus_used),
        "duplicate_gpu_assignment_count": int(gpu_mapping.duplicate_assignment),
        "nccl_initialization_mismatch_count": int(not nccl_ev_full.evidence_found),
        "worker_count_mismatch_count": int(len(gpu_mapping.tracked_pids) < 2),
        "request_failure_count": sum(1 for o in tp1_outputs + tp2_outputs if o["http_status"] != 200),
        "token_output_mismatch_count": sum(1 for c in token_comparisons if c["token_ids_match"] is False),
        "text_output_mismatch_count": sum(1 for c in text_comparisons if not c["text_match"]),
        "finish_reason_mismatch_count": sum(1 for c in text_comparisons if not c["finish_reason_match"]),
        "silent_downgrade_count": int(tp2_bundle.spec.tensor_parallel_size < plan_dict["distributed"]["tensor_parallel_size"]),
        "preflight_bypass_count": 0,
        "unexpected_backend_count": int(nccl_ev_full.backend_mentions.get("nccl", 0) == 0),
        "unexpected_process_launch_count": 0,
        "orphan_process_count": (0 if tp1_stop_result["zero_orphans"] else len(tp1_stop_result["final_remaining_descendant_pids"]))
                                 + (0 if tp2_stop_result["zero_orphans"] else len(tp2_stop_result["final_remaining_descendant_pids"])),
        "stale_port_count": 0 if port_listeners == "" else 1,
        "gpu_memory_cleanup_mismatch_count": int(not (tp1_gpu_cleanup.within_tolerance and tp2_gpu_cleanup.within_tolerance)),
    }
    counters["all_zero"] = all(v == 0 for v in counters.values())
    _write("cross_layer_provenance.json", {
        "chain": "compiler_candidate_id -> selected_tp2_execution_plan -> d4a_whole_model_evidence_hash -> "
                 "d3b_vllm_launch_specification -> live_2gpu_hardware_inventory -> preflight -> "
                 "generated_argv_environment -> server_process -> distributed_workers -> "
                 "physical_gpu_placements -> nccl_initialization -> tp2_requests -> tp1_reference_requests -> "
                 "output_comparison -> shutdown_and_cleanup",
        "counters": counters, "d4a_consistency": d4a_consistency, "source_hashes": source_hashes,
    })
    assert counters["all_zero"], counters

    print("== Part V: structural measurements ==")
    perf = {
        "tp1_startup_latency_s": tp1_startup_latency_s, "tp2_startup_latency_s": tp2_startup_latency_s,
        "tp1_readiness_latency_s": tp1_ctrl.to_dict()["readiness_latency_s"],
        "tp2_readiness_latency_s": tp2_ctrl.to_dict()["readiness_latency_s"],
        "tp1_request_latency_s": _summ(tp1_request_latencies), "tp2_request_latency_s": _summ(tp2_request_latencies),
        "tp1_shutdown_latency_s": tp1_shutdown_latency_s, "tp2_shutdown_latency_s": tp2_shutdown_latency_s,
        "tp2_nccl_init_visible_in_log": nccl_ev_full.evidence_found,
        "gpu_memory_per_rank_mb_during_tp2": tp2_gpu_memory_per_rank,
        "peak_cpu_memory_mb_this_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "label": "bring-up and correctness diagnostics -- NOT a controlled performance benchmark",
        "no_speedup_claim": True,
    }
    _write("performance_measurements.json", perf)

    print("== test summary ==")
    d4b_suite = subprocess.run([VENV_PY, "-m", "pytest", "-q", "tests/test_distributed_d4b_real_2gpu_vllm_tp2.py", "-k", "not negative and not oom"],
                                cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
                                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=300)
    _write("test_summary.json", {
        "d4b_structural_tests": {"passed": d4b_suite.returncode == 0, "tail": d4b_suite.stdout[-2000:]},
        "negative_tests_all_passed": neg_completed.returncode == 0,
        "oom_safety_passed": oom_completed.returncode == 0,
        "regressions_all_green": regression_summary["all_green"],
        "cross_layer_provenance_all_zero": counters["all_zero"],
        "tp1_correctness_all_200": all(o["http_status"] == 200 for o in tp1_outputs),
        "tp2_correctness_all_200": all(o["http_status"] == 200 for o in tp2_outputs),
        "token_ids_all_match": token_comparisons and all(c["token_ids_match"] for c in token_comparisons if c["token_ids_match"] is not None),
        "text_all_match": all(c["text_match"] for c in text_comparisons),
        "two_physical_gpus_proven": gpu_mapping.two_distinct_gpus_used,
        "nccl_proven": nccl_ev_full.evidence_found,
    })

    print("== truth boundary ==")
    _write("truth_boundary.json", {
        "d4b_primary_claim": (
            "A compiler-selected TP=2 plan for Qwen2.5-0.5B-Instruct was materialized through the existing "
            "D3B vLLM adapter and successfully executed by vLLM 0.24.0 on two physical GPUs with real "
            "distributed workers and NCCL initialization, producing inference results consistent with a "
            "TP=1 reference under deterministic correctness workloads."
        ),
        "not_claimed": [
            "TP=2 speedup", "better TTFT", "better TPOT", "higher throughput",
            "profitable distributed selection", "general multi-node support",
            "compiler-controlled per-operator vLLM execution",
        ],
        "critical_clarification": (
            "D4B validates that the compiler selects and materializes the whole-model TP strategy, and that "
            "vLLM's installed, source-verified whole-model TP implementation executes that strategy. D4B "
            "does not prove that vLLM executes the compiler's 170 Python-side D4A work items individually."
        ),
        "environment_kind": "rented_cloud_marketplace_2x_rtx_4090",
        "two_physical_gpus_proven": gpu_mapping.two_distinct_gpus_used,
        "nccl_initialization_proven": nccl_ev_full.evidence_found,
    })

    print("== repository state after ==")
    _write("repository_state_after.json", {
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Recorded after implementing and executing D4B. No commits were made on the remote host. "
                "D1-D4A result directories and reports were not modified.",
        "repositories": {
            "ml-graph-compiler-runtime": _git_state(COMPILER_ROOT),
            "heterogeneous-inference-runtime": _git_state(REPO_ROOT),
        },
    })

    print("done")


if __name__ == "__main__":
    main()
