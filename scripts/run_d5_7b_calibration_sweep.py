"""D5 Part B (7B): representative TP1 vs TP2 calibration sweep for
Qwen2.5-7B-Instruct.

Smaller grid than the 0.5B sweep (12 cells vs 36 -- see
tp_workload_matrix.build_representative_matrix_7b) so that 10 measured
repetitions per cell are affordable within the same real-hardware time
budget. Uses the same default D3B serving config (max_model_len=2048,
max_num_seqs=4, max_num_batched_tokens=2048, gpu_memory_utilization=0.9)
as the 0.5B sweep, for an apples-to-apples comparison isolating the
model-size axis. A second, memory-pressure config (if the legal-range
probe found one where TP1 is illegal but TP2 is legal) is swept
separately by run_d5_7b_memory_pressure_sweep.py, not here.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.vllm_adapter.correctness_workload import (  # noqa: E402
    CompletionRequestParams,
    build_prompt_corpus,
    compare_completions,
    compare_logprobs,
    send_completion,
)
from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController  # noqa: E402
from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec  # noqa: E402
from deployment.vllm_adapter.gpu_evidence import build_gpu_snapshot, query_gpu_inventory, wait_for_gpu_memory_baseline  # noqa: E402
from deployment.vllm_adapter.tp_benchmark_harness import run_workload_benchmark  # noqa: E402
from deployment.vllm_adapter.tp_workload_matrix import build_representative_matrix_7b, is_held_out  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d5_compiler_tp_policy" / "7b"
LOG_DIR = RESULTS_DIR / "logs"
D2_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
TP1_PLAN_PATH = D2_DIR / "real_qwen7b_tp1_execution_plan.json"
TP2_PLAN_PATH = D2_DIR / "real_qwen7b_tp2_execution_plan.json"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

STARTUP_TIMEOUT_S = 600.0  # 7B weight loading is real and slower than 0.5B's
WARMUP_REQUESTS = 2
MEASURED_REPETITIONS = 10


def _write(name: str, payload) -> None:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT.parent)}", flush=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_tp_sweep(tp_degree: int, tokenizer, matrix) -> dict:
    plan_path = TP1_PLAN_PATH if tp_degree == 1 else TP2_PLAN_PATH
    bundle = materialize_launch_spec(plan_path, repo_root=REPO_ROOT)
    assert bundle.preflight.passed, bundle.preflight.to_dict()
    assert bundle.spec.tensor_parallel_size == tp_degree

    port = _find_free_port()
    env = dict(os.environ)
    env.update(bundle.spec.environment)
    env["CUDA_VISIBLE_DEVICES"] = "0" if tp_degree == 1 else "0,1"
    log_path = LOG_DIR / f"d5_7b_tp{tp_degree}_server.log"
    argv = tuple(bundle.cli.argv[i] if bundle.cli.argv[i - 1] != "--port" else str(port)
                 for i in range(len(bundle.cli.argv)))
    ctrl = ServerLaunchController(argv=argv, env=env, cwd=str(REPO_ROOT), log_path=log_path,
                                   host=bundle.spec.host, port=port)

    baseline_used_mb = {r["index"]: float(r["memory.used"]) for r in query_gpu_inventory()}
    t0 = time.perf_counter()
    ctrl.start()
    ready = ctrl.wait_for_readiness(timeout_s=STARTUP_TIMEOUT_S, poll_interval_s=5.0)
    startup_latency_s = time.perf_counter() - t0
    assert ready, f"TP{tp_degree} server failed to become ready: {ctrl.to_dict()}"
    print(f"TP{tp_degree} ready in {startup_latency_s:.1f}s, pid={ctrl.pid}", flush=True)

    base_url = f"http://{bundle.spec.host}:{port}"
    served_model_name = bundle.spec.served_model_name
    results = []
    for i, wl in enumerate(matrix):
        print(f"  [{i + 1}/{len(matrix)}] TP{tp_degree} {wl.workload_id} "
              f"({'held_out' if is_held_out(wl) else 'calibration'})", flush=True)
        bench = run_workload_benchmark(base_url, served_model_name, tokenizer, wl, tp_degree,
                                        warmup_requests=WARMUP_REQUESTS, measured_repetitions=MEASURED_REPETITIONS,
                                        timeout_s=180.0)
        row = bench.to_dict()
        row["split"] = "held_out" if is_held_out(wl) else "calibration"
        results.append(row)
        _write(f"_partial_tp{tp_degree}_sweep_7b.json", results)

    print(f"  correctness corpus against TP{tp_degree} (preserving D4B-style guarantees) ==", flush=True)
    corpus = build_prompt_corpus()
    params = CompletionRequestParams(max_tokens=24, logprobs=5)
    correctness_outputs = []
    for spec in corpus:
        r = send_completion(base_url, served_model_name, spec, params, timeout_s=120.0)
        correctness_outputs.append(r.to_dict())
    _write(f"tp{tp_degree}_correctness_outputs_7b.json", correctness_outputs)

    stop_result = ctrl.stop(graceful_timeout_s=45.0)
    gpu_cleanup = wait_for_gpu_memory_baseline(baseline_used_mb, timeout_s=60.0)
    return {
        "tp_degree": tp_degree, "startup_latency_s": startup_latency_s,
        "launch_spec": bundle.spec.to_dict(), "preflight": bundle.preflight.to_dict(),
        "workload_results": results, "correctness_outputs": correctness_outputs,
        "stop_result": stop_result, "gpu_cleanup": str(gpu_cleanup),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    matrix = build_representative_matrix_7b()
    _write("gpu_inventory_before_sweep_7b.json", build_gpu_snapshot("before_d5_7b_sweep"))

    print("== TP1 sweep (7B, 12 representative cells, 10 reps each) ==", flush=True)
    tp1_bundle_result = _run_tp_sweep(1, tokenizer, matrix)
    _write("tp1_sweep_full_7b.json", tp1_bundle_result)

    print("== TP2 sweep (7B, 12 representative cells, 10 reps each) ==", flush=True)
    tp2_bundle_result = _run_tp_sweep(2, tokenizer, matrix)
    _write("tp2_sweep_full_7b.json", tp2_bundle_result)

    _write("gpu_inventory_after_sweep_7b.json", build_gpu_snapshot("after_d5_7b_sweep"))

    print("== correctness comparison: TP1 vs TP2 (7B) ==", flush=True)
    corpus = build_prompt_corpus()
    tp1_by_id = {r["prompt_id"]: r for r in tp1_bundle_result["correctness_outputs"]}
    tp2_by_id = {r["prompt_id"]: r for r in tp2_bundle_result["correctness_outputs"]}

    def _to_result(d):
        from deployment.vllm_adapter.correctness_workload import CompletionResult
        return CompletionResult(prompt_id=d["prompt_id"], http_status=d["http_status"], latency_s=d["latency_s"],
                                 raw_response=d["raw_response"], error=d["error"])

    text_comparisons = []
    logprob_comparisons = []
    for spec in corpus:
        if spec.prompt_id not in tp1_by_id or spec.prompt_id not in tp2_by_id:
            continue
        r1, r2 = _to_result(tp1_by_id[spec.prompt_id]), _to_result(tp2_by_id[spec.prompt_id])
        text_comparisons.append(compare_completions(spec.prompt_id, r1, r2, tokenizer).to_dict())
        lp = compare_logprobs(spec.prompt_id, r1, r2, tokenizer)
        if lp is not None:
            logprob_comparisons.append(lp.to_dict())
    _write("correctness_comparison_7b.json", {"text_comparisons": text_comparisons, "logprob_comparisons": logprob_comparisons})
    n_text_match = sum(1 for c in text_comparisons if c["text_match"])
    print(f"text match: {n_text_match}/{len(text_comparisons)}")

    manifest = {
        "model_id": MODEL_ID, "workload_matrix_size": len(matrix),
        "measured_repetitions_per_cell": MEASURED_REPETITIONS, "warmup_requests_per_cell": WARMUP_REQUESTS,
        "tp1_gpu_cleanup": tp1_bundle_result["gpu_cleanup"], "tp2_gpu_cleanup": tp2_bundle_result["gpu_cleanup"],
        "tp1_startup_latency_s": tp1_bundle_result["startup_latency_s"],
        "tp2_startup_latency_s": tp2_bundle_result["startup_latency_s"],
        "correctness_text_match_count": n_text_match, "correctness_text_total": len(text_comparisons),
    }
    _write("d5_7b_calibration_sweep_manifest.json", manifest)
    print("== D5 7B calibration sweep complete ==")


if __name__ == "__main__":
    main()
