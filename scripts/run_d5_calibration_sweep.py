"""D5: real TP=1 vs TP=2 workload-matrix benchmark sweep.

Launches the exact same D2/D3B/D4A-derived TP1 and TP2 launch specs used in
D4B (unmodified execution chain, unmodified materializer, unmodified model
-- Qwen2.5-0.5B-Instruct) on the real 2x RTX 4090 host, and against each
already-running server, measures every workload cell in the D5 workload
matrix (`tp_workload_matrix.build_full_matrix()`), using real streaming
HTTP requests (`tp_benchmark_harness`), with warmup requests discarded
before measured repetitions.

This script measures both the calibration-split and held-out-split
workload cells in the same real sweep, because the calibration/held-out
partition (`calibration_holdout_split.json`) was declared, written to
disk, and hash-recorded *before* this script ran -- so collecting all
numbers in one pass cannot bias which workloads happen to be labeled
"calibration". The separation that matters -- that the cost model is only
*fit* on calibration numbers, and held-out numbers are only *consulted*
after the model is frozen -- is enforced later, in the cost-model-fitting
and held-out-validation scripts, not here.

No claim of any kind (speedup, profitability, which policy is better) is
made anywhere in this script. It only measures.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController  # noqa: E402
from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec  # noqa: E402
from deployment.vllm_adapter.gpu_evidence import (  # noqa: E402
    build_gpu_snapshot,
    query_gpu_inventory,
    wait_for_gpu_memory_baseline,
)
from deployment.vllm_adapter.tp_benchmark_harness import run_workload_benchmark  # noqa: E402
from deployment.vllm_adapter.tp_workload_matrix import build_full_matrix, is_held_out  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d5_compiler_tp_policy"
LOG_DIR = RESULTS_DIR / "logs"
D2_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
D4A_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d4a_whole_model_tp_contract"
TP1_PLAN_PATH = D2_DIR / "real_qwen_tp1_execution_plan.json"
TP2_PLAN_PATH = D2_DIR / "real_qwen_tp2_execution_plan.json"
D4A_EVIDENCE_PATH = D4A_DIR / "whole_model_tp_classification.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

STARTUP_TIMEOUT_S = 300.0


def _write(name: str, payload) -> None:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT.parent)}", flush=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_tp_sweep(tp_degree: int, tokenizer, matrix) -> list[dict]:
    plan_path = TP1_PLAN_PATH if tp_degree == 1 else TP2_PLAN_PATH
    if tp_degree == 1:
        bundle = materialize_launch_spec(plan_path, repo_root=REPO_ROOT)
    else:
        bundle = materialize_launch_spec(plan_path, repo_root=REPO_ROOT, d4a_evidence_path=D4A_EVIDENCE_PATH)
    assert bundle.preflight.passed, bundle.preflight.to_dict()
    assert bundle.spec.tensor_parallel_size == tp_degree

    port = _find_free_port()
    env = dict(os.environ)
    env.update(bundle.spec.environment)
    env["CUDA_VISIBLE_DEVICES"] = "0" if tp_degree == 1 else "0,1"
    log_path = LOG_DIR / f"d5_tp{tp_degree}_server.log"
    argv = tuple(bundle.cli.argv[i] if bundle.cli.argv[i - 1] != "--port" else str(port)
                 for i in range(len(bundle.cli.argv)))
    ctrl = ServerLaunchController(argv=argv, env=env, cwd=str(REPO_ROOT), log_path=log_path,
                                   host=bundle.spec.host, port=port)

    baseline_used_mb = {r["index"]: float(r["memory.used"]) for r in query_gpu_inventory()}
    t0 = time.perf_counter()
    ctrl.start()
    ready = ctrl.wait_for_readiness(timeout_s=STARTUP_TIMEOUT_S, poll_interval_s=3.0)
    startup_latency_s = time.perf_counter() - t0
    assert ready, f"TP{tp_degree} server failed to become ready: {ctrl.to_dict()}"
    print(f"TP{tp_degree} ready in {startup_latency_s:.1f}s, pid={ctrl.pid}", flush=True)

    base_url = f"http://{bundle.spec.host}:{port}"
    served_model_name = bundle.spec.served_model_name
    results = []
    for i, wl in enumerate(matrix):
        print(f"  [{i + 1}/{len(matrix)}] TP{tp_degree} {wl.workload_id} "
              f"({'held_out' if is_held_out(wl) else 'calibration'})", flush=True)
        bench = run_workload_benchmark(base_url, served_model_name, tokenizer, wl, tp_degree)
        row = bench.to_dict()
        row["split"] = "held_out" if is_held_out(wl) else "calibration"
        results.append(row)
        _write(f"_partial_tp{tp_degree}_sweep.json", results)  # incremental save for resumability

    stop_result = ctrl.stop(graceful_timeout_s=30.0)
    gpu_cleanup = wait_for_gpu_memory_baseline(baseline_used_mb, timeout_s=30.0)
    return {
        "tp_degree": tp_degree, "startup_latency_s": startup_latency_s,
        "launch_spec": bundle.spec.to_dict(), "preflight": bundle.preflight.to_dict(),
        "workload_results": results, "stop_result": stop_result, "gpu_cleanup": gpu_cleanup,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    matrix = build_full_matrix()
    _write("gpu_inventory_before_sweep.json", build_gpu_snapshot("before_d5_sweep"))

    source_hashes = {
        "tp1_plan_sha256": _sha256_file(TP1_PLAN_PATH), "tp2_plan_sha256": _sha256_file(TP2_PLAN_PATH),
        "d4a_evidence_sha256": _sha256_file(D4A_EVIDENCE_PATH),
    }
    _write("source_artifact_hashes.json", source_hashes)

    print("== TP1 sweep (all 36 workload cells) ==", flush=True)
    tp1_bundle_result = _run_tp_sweep(1, tokenizer, matrix)
    _write("tp1_sweep_full.json", tp1_bundle_result)

    print("== TP2 sweep (all 36 workload cells) ==", flush=True)
    tp2_bundle_result = _run_tp_sweep(2, tokenizer, matrix)
    _write("tp2_sweep_full.json", tp2_bundle_result)

    _write("gpu_inventory_after_sweep.json", build_gpu_snapshot("after_d5_sweep"))

    manifest = {
        "model_id": MODEL_ID, "workload_matrix_size": len(matrix),
        "source_artifact_hashes": source_hashes,
        "tp1_gpu_cleanup": tp1_bundle_result["gpu_cleanup"],
        "tp2_gpu_cleanup": tp2_bundle_result["gpu_cleanup"],
        "tp1_startup_latency_s": tp1_bundle_result["startup_latency_s"],
        "tp2_startup_latency_s": tp2_bundle_result["startup_latency_s"],
        "note": "Both calibration-split and held-out-split workload cells were measured in this same "
                "sweep. This is safe because the calibration/held-out partition was declared and "
                "hash-recorded in calibration_holdout_split.json before this sweep ran. The cost "
                "model (built in a later stage) must only be FIT on rows where split == 'calibration'; "
                "'held_out' rows must only be consulted after the cost model is frozen.",
    }
    _write("d5_calibration_sweep_manifest.json", manifest)
    print("== D5 calibration sweep complete ==")


if __name__ == "__main__":
    main()
