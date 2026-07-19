"""D5: tests for the workload matrix, benchmark harness math, and cost
model -- all pure-logic, no GPU/vLLM required. Real hardware execution is
covered by the D5 result artifacts (results/runtime_paths/
distributed_d5_compiler_tp_policy/), not by this suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment.vllm_adapter.tp_benchmark_harness import (
    StreamedRequestResult,
    WorkloadBenchmarkResult,
    _percentile,
)
from deployment.vllm_adapter.tp_cost_model import (
    MODEL_IDENTITY_FEATURES,
    TPCostModel,
    build_feature_vector,
    fit_linear_regression,
    is_feasible,
    kv_cache_bytes_per_token_per_gpu,
    per_gpu_weight_mb,
)
from deployment.vllm_adapter.tp_workload_matrix import (
    build_full_matrix,
    build_representative_matrix_7b,
    is_held_out,
    matrix_manifest,
    split_matrix,
    workload_weight,
    WorkloadSpec,
)


# --------------------------------------------------------------------------
# tp_workload_matrix
# --------------------------------------------------------------------------

def test_full_matrix_size_matches_declared_grid():
    matrix = build_full_matrix()
    assert len(matrix) == 3 * 3 * 4  # INPUT_LENGTHS x OUTPUT_LENGTHS x CONCURRENCY_LEVELS


def test_representative_7b_matrix_size():
    matrix = build_representative_matrix_7b()
    assert len(matrix) == 2 * 2 * 3


def test_split_is_deterministic_and_workload_identity_only():
    matrix = build_full_matrix()
    first = {w.workload_id: is_held_out(w) for w in matrix}
    second = {w.workload_id: is_held_out(w) for w in build_full_matrix()}
    assert first == second


def test_split_covers_every_workload_exactly_once():
    matrix = build_full_matrix()
    calibration, held_out = split_matrix(matrix)
    assert len(calibration) + len(held_out) == len(matrix)
    assert set(w.workload_id for w in calibration).isdisjoint(w.workload_id for w in held_out)


def test_split_is_not_degenerate():
    """Neither split may be empty or capture nearly everything -- a
    degenerate split would make calibration/held-out evaluation
    meaningless."""
    matrix = build_full_matrix()
    calibration, held_out = split_matrix(matrix)
    assert len(calibration) >= len(matrix) * 0.25
    assert len(held_out) >= len(matrix) * 0.25


def test_weighting_scheme_is_uniform():
    matrix = build_full_matrix()
    weights = {workload_weight(w) for w in matrix}
    assert weights == {1.0}


def test_matrix_manifest_is_internally_consistent():
    manifest = matrix_manifest()
    assert manifest["calibration_count"] + manifest["held_out_count"] == manifest["total_workloads"]
    assert manifest["total_workloads"] == len(build_full_matrix())


# --------------------------------------------------------------------------
# tp_benchmark_harness
# --------------------------------------------------------------------------

def test_streamed_request_result_ttft_tpot_e2e_math():
    r = StreamedRequestResult(
        request_index=0, ok=True, error=None, request_start_ts=10.0,
        first_token_ts=10.1, last_token_ts=10.5, output_token_count=5, prompt_token_count=32,
    )
    assert r.ttft_s == pytest.approx(0.1)
    assert r.e2e_latency_s == pytest.approx(0.5)
    assert r.tpot_s == pytest.approx((10.5 - 10.1) / 4)


def test_streamed_request_result_failed_request_has_no_derived_metrics():
    r = StreamedRequestResult(
        request_index=1, ok=False, error="timeout", request_start_ts=0.0,
        first_token_ts=None, last_token_ts=None, output_token_count=0, prompt_token_count=None,
    )
    assert r.ttft_s is None and r.e2e_latency_s is None and r.tpot_s is None


def test_single_token_output_has_no_tpot():
    r = StreamedRequestResult(
        request_index=0, ok=True, error=None, request_start_ts=0.0,
        first_token_ts=0.05, last_token_ts=0.05, output_token_count=1, prompt_token_count=10,
    )
    assert r.tpot_s is None  # tpot is undefined with fewer than 2 tokens


def test_workload_benchmark_summary_aggregates_only_ok_requests():
    wl = WorkloadSpec(32, 32, 2)
    ok = StreamedRequestResult(0, True, None, 0.0, 0.1, 0.5, 5, 32)
    failed = StreamedRequestResult(1, False, "err", 0.0, None, None, 0, None)
    bench = WorkloadBenchmarkResult(workload=wl, tp_degree=1, warmup_count=1,
                                     measured_results=[ok, failed], wall_clock_batch_s=0.5)
    summary = bench.summary()
    assert summary["requests_ok"] == 1
    assert summary["requests_total"] == 2
    assert summary["mean_ttft_s"] == pytest.approx(0.1)
    assert summary["aggregate_throughput_tokens_per_s"] == pytest.approx(10.0)


def test_percentile_bounds():
    assert _percentile([], 0.5) is None
    assert _percentile([1, 2, 3, 4, 5], 0.0) == 1
    assert _percentile([1, 2, 3, 4, 5], 1.0) == 5


# --------------------------------------------------------------------------
# tp_cost_model
# --------------------------------------------------------------------------

def test_larger_model_has_larger_per_gpu_weight_and_kv_cost():
    small = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]
    large = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-7B-Instruct"]
    assert per_gpu_weight_mb(large, 1) > per_gpu_weight_mb(small, 1)
    assert kv_cache_bytes_per_token_per_gpu(large, 1) > kv_cache_bytes_per_token_per_gpu(small, 1)


def test_tp2_halves_per_gpu_weight_and_kv_cost_vs_tp1():
    model = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-7B-Instruct"]
    assert per_gpu_weight_mb(model, 2) == pytest.approx(per_gpu_weight_mb(model, 1) / 2)
    assert kv_cache_bytes_per_token_per_gpu(model, 2) == pytest.approx(kv_cache_bytes_per_token_per_gpu(model, 1) / 2)


def test_feasibility_check_rejects_when_budget_too_small():
    model = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-7B-Instruct"]
    feasible, detail = is_feasible(model, 1, gpu_total_mb=1000.0, gpu_memory_utilization=0.9,
                                    max_model_len=2048, max_num_seqs=4)
    assert feasible is False
    assert detail["required_mb"] > detail["budget_mb"]


def test_feasibility_check_accepts_ample_budget():
    model = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]
    feasible, _ = is_feasible(model, 1, gpu_total_mb=24564.0, gpu_memory_utilization=0.9,
                               max_model_len=2048, max_num_seqs=4)
    assert feasible is True


def test_fit_linear_regression_recovers_exact_linear_relationship():
    # y = 2*x0 + 3*x1 + 1, noise-free -- regression must recover it exactly.
    X = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2]]
    y = [2 * x0 + 3 * x1 + 1 for x0, x1 in X]
    reg = fit_linear_regression(X, y)
    assert reg.r_squared == pytest.approx(1.0, abs=1e-6)
    intercept, c0, c1 = reg.coefficients
    assert intercept == pytest.approx(1.0, abs=1e-6)
    assert c0 == pytest.approx(2.0, abs=1e-6)
    assert c1 == pytest.approx(3.0, abs=1e-6)


def test_cost_model_decision_uses_performance_regression_when_both_feasible():
    model = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]
    calibration_rows = []
    for tp_degree, base_throughput in ((1, 1000.0), (2, 500.0)):
        for concurrency in (1, 2, 4):
            fv = build_feature_vector(model, tp_degree, input_length=32, output_length=32, concurrency=concurrency)
            calibration_rows.append({"tp_degree": tp_degree, "feature_vector": fv,
                                      "aggregate_throughput_tokens_per_s": base_throughput * concurrency})
    model_fit = TPCostModel()
    model_fit.fit(calibration_rows)
    decision = model_fit.decide(
        model_features=model, input_length=32, output_length=32, concurrency=2,
        gpu_total_mb=24564.0, gpu_memory_utilization=0.9, max_model_len=2048, max_num_seqs=4,
    )
    assert decision["decision"] == "tp1"  # TP1's calibration throughput is always 2x TP2's here
    assert decision["reason"] == "performance_regression"


def test_cost_model_forces_tp2_when_tp1_infeasible():
    model = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-7B-Instruct"]
    model_fit = TPCostModel()
    model_fit.fit([
        {"tp_degree": 1, "feature_vector": build_feature_vector(model, 1, input_length=32, output_length=32, concurrency=1),
         "aggregate_throughput_tokens_per_s": 100.0},
        {"tp_degree": 2, "feature_vector": build_feature_vector(model, 2, input_length=32, output_length=32, concurrency=1),
         "aggregate_throughput_tokens_per_s": 100.0},
    ])
    # A tiny GPU budget makes TP1 (full weight shard) illegal but TP2 (half shard) legal.
    decision = model_fit.decide(
        model_features=model, input_length=32, output_length=32, concurrency=1,
        gpu_total_mb=10000.0, gpu_memory_utilization=0.9, max_model_len=2048, max_num_seqs=4,
    )
    assert decision["decision"] == "tp2"
    assert decision["reason"] == "capacity_forced"


def test_cost_model_refit_after_freeze_raises():
    model_fit = TPCostModel()
    model = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]
    rows = [{"tp_degree": tp, "feature_vector": build_feature_vector(model, tp, input_length=32, output_length=32, concurrency=1),
             "aggregate_throughput_tokens_per_s": 100.0} for tp in (1, 2)]
    model_fit.fit(rows)
    try:
        model_fit.fit(rows)
        assert False, "refit after freeze must raise"
    except RuntimeError:
        pass
