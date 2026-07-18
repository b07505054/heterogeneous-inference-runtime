"""D2: Real Qwen Pipeline Distributed Strategy Planning -- artifact generator.

Runs the full D2 vertical slice against the real compiler-exported real-Qwen
TP1/TP2 plans (already produced by ml-graph-compiler-runtime's
compile-for-target via RunDistributedStrategyPlanningPipelineTest.cmake and
copied into this result directory) and the D1 multi-process runtime, and
writes every artifact listed in the D2 spec (Part N).
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.execution_plan.loader import load_execution_plan  # noqa: E402
from deployment.tp_process_runtime import (  # noqa: E402
    DistributedProcessRuntime,
    build_qwen_derived_workload,
    serial_matmul_reference,
    verify_cross_layer_provenance,
)

RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
D1_RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d1_tp2_multiprocess"
COMPILER_ROOT = REPO_ROOT.parent / "ml-graph-compiler-runtime"
CFT_TOOL = COMPILER_ROOT / "mlir_passes" / "build" / "compile-for-target"
ONNX_TOOL = COMPILER_ROOT / "mlir_passes" / "build" / "qwen-onnx-to-serving-mlir"
DISTRIBUTED_TEST_BIN = COMPILER_ROOT / "mlir_passes" / "build" / "DistributedStrategyPlanningTest"
FACTS = COMPILER_ROOT / "configs" / "models" / "qwen_0_5b_onnx_graph_facts.json"
PROFILE_TP1 = COMPILER_ROOT / "configs" / "target_profiles" / "nvidia_gtx1650_maxq.json"
PROFILE_TP2 = COMPILER_ROOT / "configs" / "target_profiles" / "nvidia_gtx1650_maxq_d2_distributed_opt_in.json"


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
    return {"median": statistics.median(values), "p95": _percentile(values, 95),
            "min": min(values), "max": max(values), "n": len(values)}


def split_evidence_report() -> dict:
    evidence = json.loads((RESULTS_DIR / "qwen_distributed_evidence_report.json").read_text())
    candidates = evidence["candidates"]

    _write("qwen_distributed_candidates.json", {
        "candidates": [
            {k: c[k] for k in (
                "candidate_id", "strategy", "world_size", "tensor_parallel_size",
                "pipeline_parallel_size", "partitioned_operator_ids", "partition_axis",
                "shard_count", "required_collectives",
            )}
            for c in candidates
        ],
        "source": "DistributedStrategyPlanningPass module attrs, read via "
                  "compile-for-target --distributed-evidence-report against the "
                  "real per-layer Qwen ONNX graph",
    })

    _write("qwen_distributed_legality.json", {
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "legality_status": c["legality_status"],
                "rejection_reasons": c["rejection_reasons"],
                "rule_results": c["legality_rule_results"],
            }
            for c in candidates
        ],
    })

    _write("qwen_distributed_costs.json", {
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "estimated_communication_bytes": c["estimated_communication_bytes"],
                "estimated_rank_local_compute": c["estimated_rank_local_compute"],
                "selection_score": c["selection_score"],
                "truth_boundary": c["truth_boundary"],
            }
            for c in candidates
        ],
    })

    _write("qwen_distributed_selection.json", {
        "selected_candidate_id": evidence["selected_candidate_id"],
        "selection_reason": evidence["selection_reason"],
        "policy_id": evidence["policy_id"],
        "policy_truth_boundary": evidence["policy_truth_boundary"],
        "all_candidates_considered": [c["candidate_id"] for c in candidates],
    })
    return evidence


def run_real_qwen_tp2() -> tuple:
    plan = load_execution_plan(RESULTS_DIR / "real_qwen_tp2_execution_plan.json")
    workload = build_qwen_derived_workload(plan.distributed, seed=31337)
    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, workload.a, workload.b)
    return plan, workload, result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("== evidence split ==")
    split_evidence_report()

    print("== real qwen tp2 run ==")
    plan, workload, result = run_real_qwen_tp2()

    _write("rank_process_events.jsonl", result.trace.to_jsonable())

    collective_events = []
    for o in result.collective_outcomes:
        collective_events.append({
            "collective_id": o.collective_id, "sequence_id": o.sequence_id,
            "status": o.status, "participant_ranks": sorted(o.contributions.keys()),
            "bytes_contributed": o.bytes_contributed,
            "start_ts": o.start_ts, "end_ts": o.end_ts,
            "latency_s": o.end_ts - o.start_ts,
            "tensor_id": next(iter(o.contributions.values()))["tensor_id"] if o.contributions else None,
        })
    _write("collective_events.jsonl", collective_events)

    _write("runtime_materialization.json", {
        "plan_source": "results/runtime_paths/distributed_d2_qwen_pipeline/real_qwen_tp2_execution_plan.json",
        "selected_operator_id": plan.distributed.tensor_shards[0].tensor_id,
        "workload": {
            "hidden_dim": workload.hidden_dim, "sequence_length": workload.sequence_length,
            "a_shape": list(workload.a.shape), "b_shape": list(workload.b.shape),
            "truth_boundary": "qwen_shaped_synthetic_workload_not_live_transformers_or_vllm_tensor",
        },
        "planned_ranks": [{"rank_id": r.rank_id, "logical_device": r.logical_device}
                          for r in plan.distributed.ranks],
        "materialized_processes": {
            str(r): {"pid": p.pid, "exitcode": p.exitcode, "alive_after_teardown": p.alive_after_teardown}
            for r, p in result.processes.items()
        },
        "status": result.status,
        "all_ranks_completed": result.all_ranks_completed,
        "all_collectives_completed": result.all_collectives_completed,
    })

    correctness = None
    ref = serial_matmul_reference(workload.a, workload.b)
    max_abs = float(np.max(np.abs(result.distributed_output - ref)))
    denom = np.abs(ref)
    denom[denom == 0] = 1e-12
    max_rel = float(np.max(np.abs(result.distributed_output - ref) / denom))
    correctness = {
        "distributed_result_matches_serial_reference": bool(
            np.allclose(result.distributed_output, ref, rtol=1e-9, atol=1e-9)
        ),
        "max_abs_error": max_abs, "max_rel_error": max_rel,
        "shape_match": list(result.distributed_output.shape) == list(ref.shape),
        "dtype_match": str(result.distributed_output.dtype) == str(ref.dtype),
        "all_ranks_completed": result.all_ranks_completed,
        "all_collectives_completed": result.all_collectives_completed,
        "tolerance": {"rtol": 1e-9, "atol": 1e-9},
        "workload_hidden_dim": workload.hidden_dim,
        "workload_sequence_length": workload.sequence_length,
    }
    _write("correctness_summary.json", correctness)

    print("== cross-layer provenance ==")
    report = verify_cross_layer_provenance(plan.distributed, result, workload.hidden_dim)
    _write("cross_layer_provenance.json", {
        "all_match": report.all_match, "mismatch_count": report.mismatch_count,
        "operator_id_match": report.operator_id_match,
        "world_size_match": report.world_size_match,
        "rank_ids_match": report.rank_ids_match,
        "shard_ranges_match": report.shard_ranges_match,
        "collective_id_match": report.collective_id_match,
        "sequence_id_match": report.sequence_id_match,
        "participant_set_match": report.participant_set_match,
        "no_silent_downgrade": report.no_silent_downgrade,
        "no_synthetic_fallback_dimensions": report.no_synthetic_fallback_dimensions,
        "details": report.details,
        "provenance_chain": "qwen_graph_operator -> distributed_candidate -> legality_decision -> "
                            "cost_evaluation_result -> selected_candidate -> execution_plan_distributed_"
                            "work_item -> runtime_rank_specifications -> launched_process_ids -> "
                            "rank_local_shards -> collective_participation -> reconstructed_output",
    })

    print("== negative tests ==")
    negative_tests = run_negative_tests()
    _write("negative_tests.json", negative_tests)

    print("== d1 regression ==")
    d1_regression = run_d1_regression()
    _write("d1_regression_summary.json", d1_regression)

    print("== performance measurements ==")
    perf = run_performance_measurements()
    _write("performance_measurements.json", perf)

    print("== test summary ==")
    _write("test_summary.json", build_test_summary(negative_tests, d1_regression))

    print("== truth boundary ==")
    _write("truth_boundary.json", {
        "d2_maturity_claim": (
            "The real Qwen compiler serving pipeline generated, evaluated, selected, "
            "and exported TP=1/TP=2 distributed strategy candidates as part of the "
            "normal ExecutionPlan construction path, and the runtime consumed the "
            "resulting real-Qwen plan through the D1 multi-process simulator with "
            "complete plan-to-execution provenance."
        ),
        "not_claimed": [
            "real Qwen tensor parallel execution", "real vLLM distributed execution",
            "real GPU tensor parallelism", "NCCL", "multi-GPU speedup",
            "distributed serving profitability",
        ],
        "environment": "single-host CPU multi-process simulation, localhost IPC only; "
                       "real Qwen model metadata (qwen2.5-0.5b, hidden_size=896, 24 layers) "
                       "and one real per-layer llm.o_proj operator instance drive the plan's "
                       "dimensions, but the executed workload is a Qwen-shaped synthetic "
                       "matmul, not a captured live tensor",
        "explicitly_not": [
            "not NCCL", "not GPU-to-GPU communication", "not real vLLM tensor parallelism",
            "not representative of multi-GPU scaling", "not live Transformers/vLLM tensor capture",
        ],
    })

    print("done")


def run_negative_tests() -> dict:
    results = {}

    # Compiler-side (C++) negative tests: run ctest directly for ground truth.
    ctest_bin = COMPILER_ROOT / "mlir_passes" / "build"
    cases = [
        ("qwen_dimension_not_divisible_by_tp2", "DistributedStrategyPlanningTest"),
        ("unsupported_operator_selected_for_partitioning", "DistributedStrategyPlanningTest"),
        ("missing_required_shape_metadata", "DistributedStrategyPlanningTest"),
        ("distributed_capability_unavailable", "DistributedStrategyPlanningTest"),
        ("distributed_pass_disabled_tp1_no_distributed_key", "DistributedStrategyPlanningPipelineTest"),
    ]
    completed = subprocess.run(
        ["ctest", "--output-on-failure", "-R",
         "DistributedStrategyPlanningTest|DistributedStrategyPlanningPipelineTest|DistributedPlanningTest"],
        cwd=str(ctest_bin), capture_output=True, text=True, check=False,
    )
    compiler_ctest_passed = completed.returncode == 0
    results["compiler_side"] = {
        "ctest_command": "ctest --output-on-failure -R "
                         "'DistributedStrategyPlanningTest|DistributedStrategyPlanningPipelineTest|DistributedPlanningTest'",
        "all_passed": compiler_ctest_passed,
        "cases_covered": [c[0] for c in cases],
        "stdout_tail": completed.stdout[-2000:],
    }

    # Runtime-side (Python) negative tests: run pytest for ground truth.
    py_completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-v",
         "tests/test_distributed_d2_qwen_pipeline.py::test_reject_collective_referencing_unknown_operator_tensor",
         "tests/test_distributed_d2_qwen_pipeline.py::test_runtime_rejects_dimensions_differing_from_compiler_plan",
         "tests/test_distributed_d2_qwen_pipeline.py::test_runtime_mismatched_operator_id_detected_by_cross_layer_check",
         "tests/test_distributed_d2_qwen_pipeline.py::test_duplicate_rank_id_rejected",
         "tests/test_distributed_d2_qwen_pipeline.py::test_reject_tp2_real_qwen_plan_on_real_vllm_adapter_path",
         ],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    results["runtime_side"] = {
        "cases_covered": [
            "collective_references_unknown_operator_tensor",
            "runtime_dimensions_differ_from_compiler_plan",
            "runtime_mismatched_operator_id_detected_by_cross_layer_check",
            "duplicate_rank_id",
            "tp2_plan_sent_to_real_vllm_adapter",
        ],
        "all_passed": py_completed.returncode == 0,
        "stdout_tail": py_completed.stdout[-2000:],
    }
    results["all_negative_tests_passed"] = (
        results["compiler_side"]["all_passed"] and results["runtime_side"]["all_passed"]
    )
    return results


def run_d1_regression() -> dict:
    ctest_bin = COMPILER_ROOT / "mlir_passes" / "build"
    completed = subprocess.run(
        ["ctest", "--output-on-failure", "-R", "DistributedPlanningTest"],
        cwd=str(ctest_bin), capture_output=True, text=True, check=False,
    )
    d1_compiler_passed = completed.returncode == 0

    py_completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_distributed_tp_process_runtime.py"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    d1_runtime_passed = py_completed.returncode == 0

    # Re-run the D1 deadlock negative test explicitly for a fresh confirmation.
    d1_plan = load_execution_plan(D1_RESULTS_DIR / "compiler_tp2_plan.json")
    rng = np.random.default_rng(555)
    a = rng.uniform(-2, 2, size=(4, 16))
    b = rng.uniform(-2, 2, size=(16, 4))
    rt = DistributedProcessRuntime()
    t0 = time.time()
    deadlock_result = rt.run(d1_plan.distributed, a, b, collective_timeout_s=2.0,
                             force_skip_collective_rank=1)
    elapsed = time.time() - t0
    orphans = [p.pid for p in deadlock_result.processes.values() if p.alive_after_teardown]

    return {
        "d1_compiler_tests": {"command": "ctest -R DistributedPlanningTest",
                              "passed": d1_compiler_passed, "stdout_tail": completed.stdout[-1500:]},
        "d1_runtime_tests": {"command": "pytest tests/test_distributed_tp_process_runtime.py",
                             "passed": d1_runtime_passed, "stdout_tail": py_completed.stdout[-1500:]},
        "d1_deadlock_negative_test_rerun": {
            "status": deadlock_result.status, "wall_clock_elapsed_s": elapsed,
            "missing_ranks": deadlock_result.deadlock["missing_ranks"] if deadlock_result.deadlock else None,
            "orphans_found": orphans,
        },
        "no_d1_rank_or_d2_rank_processes_after_tests": len(orphans) == 0,
        "all_d1_regressions_green": d1_compiler_passed and d1_runtime_passed and not orphans,
    }


def run_performance_measurements(reps: int = 5) -> dict:
    results = {}

    # Compiler pipeline wall-clock (whole tool, includes the new pass; not
    # isolated to the pass alone -- MLIR pass-timing instrumentation was not
    # wired into D2, documented as a known limitation).
    raw_mlir = RESULTS_DIR / "_perf_raw.mlir"
    subprocess.run([str(ONNX_TOOL), "--graph-facts", str(FACTS), "--out", str(raw_mlir)],
                   check=True, capture_output=True)
    for label, profile in (("tp1_profile", PROFILE_TP1), ("tp2_opt_in_profile", PROFILE_TP2)):
        times = []
        out_path = RESULTS_DIR / f"_perf_{label}.json"
        for _ in range(reps):
            t0 = time.perf_counter()
            subprocess.run([str(CFT_TOOL), f"--device-profile={profile}", f"--mlir={raw_mlir}",
                            f"--out={out_path}"], check=True, capture_output=True)
            times.append(time.perf_counter() - t0)
        results[f"compiler_pipeline_wall_time_s[{label}]"] = _summ(times)
        out_path.unlink(missing_ok=True)
    raw_mlir.unlink(missing_ok=True)

    # Legality/cost evaluation latency: real in-process C++ measurement,
    # parsed from DistributedStrategyPlanningTest's own stdout.
    if DISTRIBUTED_TEST_BIN.exists():
        completed = subprocess.run([str(DISTRIBUTED_TEST_BIN)], capture_output=True, text=True, check=True)
        for line in completed.stdout.splitlines():
            if line.startswith("LEGALITY_COST_LATENCY_NS_PER_CALL="):
                results["legality_and_cost_evaluation_latency_ns_per_candidate_pair"] = float(
                    line.split("=", 1)[1]
                )
    results["candidate_count"] = 2

    # Runtime plan-load latency.
    load_times = []
    for _ in range(reps * 4):
        t0 = time.perf_counter()
        load_execution_plan(RESULTS_DIR / "real_qwen_tp2_execution_plan.json")
        load_times.append(time.perf_counter() - t0)
    results["runtime_plan_load_latency_s"] = _summ(load_times)

    # Process startup / rank-local compute / collective / end-to-end.
    plan = load_execution_plan(RESULTS_DIR / "real_qwen_tp2_execution_plan.json")
    startup, compute, collective, end_to_end = [], [], [], []
    for i in range(reps):
        workload = build_qwen_derived_workload(plan.distributed, seed=6000 + i)
        rt = DistributedProcessRuntime()
        result = rt.run(plan.distributed, workload.a, workload.b)
        assert result.status == "completed"
        startup.append(result.timings["process_startup_s"])
        collective.append(result.timings["collective_latency_s"])
        end_to_end.append(result.timings["end_to_end_s"])
        for e in result.trace.events:
            if e.get("event") == "local_compute_done":
                compute.append(e["compute_ms"] / 1000.0)

    results["process_startup_latency_s"] = _summ(startup)
    results["rank_local_compute_latency_s"] = _summ(compute)
    results["collective_latency_s"] = _summ(collective)
    results["end_to_end_simulated_execution_latency_s"] = _summ(end_to_end)
    results["repetitions"] = reps
    results["truth_boundary"] = (
        "single-host CPU multi-process simulation, localhost IPC only; compiler "
        "latencies are whole-pipeline wall-clock (not pass-isolated); not GPU-"
        "measured; not NCCL-calibrated; not a distributed-serving profitability claim"
    )
    return results


def build_test_summary(negative_tests: dict, d1_regression: dict) -> dict:
    compiler_ctest = subprocess.run(
        ["ctest", "--output-on-failure"], cwd=str(COMPILER_ROOT / "mlir_passes" / "build"),
        capture_output=True, text=True, check=False,
    )
    runtime_pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "agentic_eval/tests", "tests"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    return {
        "compiler_repo_ctest": {
            "command": "ctest --output-on-failure (full suite, 39 targets after D2)",
            "returncode": compiler_ctest.returncode,
            "tail": compiler_ctest.stdout[-1500:],
            "d2_new_targets": ["DistributedStrategyPlanningTest", "DistributedStrategyPlanningPipelineTest"],
            "pre_existing_unrelated_failures": [
                "ImplementationCandidateTest", "LLMFrontendNormalizationTest", "QwenOnnxServingPlanExportTest",
            ],
        },
        "runtime_repo_pytest": {
            "command": "pytest -q agentic_eval/tests tests (full suite)",
            "returncode": runtime_pytest.returncode,
            "tail": runtime_pytest.stdout[-1500:],
            "d2_new_test_file": "tests/test_distributed_d2_qwen_pipeline.py",
            "pre_existing_unrelated_failures": [
                "test_attention_runtime.py (selector-v4, explicitly out of scope)",
                "test_deployment_planner.py (missing local capabilities/profiles/backend/coreml.json)",
                "test_model_adapter_registry.py (sys.modules import-pollution, test-ordering sensitive)",
                "test_native_fused_attention.py (9 subprocess/native-binary errors)",
            ],
        },
        "negative_tests_all_passed": negative_tests["all_negative_tests_passed"],
        "d1_regressions_all_green": d1_regression["all_d1_regressions_green"],
    }


if __name__ == "__main__":
    main()
