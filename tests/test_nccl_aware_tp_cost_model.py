import json
from pathlib import Path

import pytest

from deployment.vllm_adapter.tp_cost_model import (
    CommunicationCalibrationError,
    FittedRegression,
    MODEL_IDENTITY_FEATURES,
    TPCostModel,
    D9_COMPUTE_REFERENCE_WEIGHT_MB,
    D9_COMPUTE_SAVINGS_US_PER_WEIGHT_MB_ABOVE_REFERENCE,
    adjust_throughput_for_communication,
    estimate_collective_call_count,
    estimate_collective_demand,
    estimate_communication_penalty_us,
    estimated_communication_bytes,
    load_communication_predictor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_real_predictor():
    profile = json.loads(
        (REPO_ROOT / "results/runtime_paths/nccl_calibration/communication_cost_profile.json").read_text()
    )
    fit = json.loads((REPO_ROOT / "results/runtime_paths/nccl_calibration/fit_report.json").read_text())
    return load_communication_predictor(profile, fit)


def _toy_model(tp1: float = 1000.0, tp2: float = 1001.0) -> TPCostModel:
    model = TPCostModel()
    # Only intercept differs; all other features are neutral.
    model.throughput_models[1] = FittedRegression(1, [tp1, 0, 0, 0, 0, 0, 0], 1, 1.0)
    model.throughput_models[2] = FittedRegression(2, [tp2, 0, 0, 0, 0, 0, 0], 1, 1.0)
    model.frozen = True
    return model


def test_exact_lookup_point_uses_raw_measured_time():
    predictor = _load_real_predictor()
    point = next(p for p in predictor.points if p.bytes == 8192)
    assert predictor.predict_time_us(8192) == pytest.approx(point.time_us)


def test_interpolation_between_measured_sizes():
    predictor = _load_real_predictor()
    predicted = predictor.predict_time_us(12 * 1024)
    left = next(p for p in predictor.points if p.bytes == 8192)
    right = next(p for p in predictor.points if p.bytes == 16384)
    assert min(left.time_us, right.time_us) <= predicted <= max(left.time_us, right.time_us)
    assert predicted != pytest.approx(left.time_us)
    assert predicted != pytest.approx(right.time_us)


def test_out_of_range_fails_closed():
    predictor = _load_real_predictor()
    with pytest.raises(CommunicationCalibrationError, match="outside calibrated range"):
        predictor.predict_time_us(predictor.points[-1].bytes * 2)


def test_topology_mismatch_fails_closed():
    profile = json.loads(
        (REPO_ROOT / "results/runtime_paths/nccl_calibration/communication_cost_profile.json").read_text()
    )
    fit = json.loads((REPO_ROOT / "results/runtime_paths/nccl_calibration/fit_report.json").read_text())
    profile["machine_calibration_boundary"]["topology_class"] = "NVLINK"
    with pytest.raises(CommunicationCalibrationError, match="topology mismatch"):
        load_communication_predictor(profile, fit)


def test_legacy_profile_behavior_is_explicit():
    decision = _toy_model().decide(
        model_features=MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"],
        input_length=32,
        output_length=32,
        concurrency=1,
        gpu_total_mb=24564.0,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_num_seqs=4,
    )
    assert decision["legacy_behavior"] is True
    assert decision["legacy_reason"] == "legacy_no_communication_calibration"
    assert decision["communication"]["communication_profile_id"] is None


def test_communication_cost_can_flip_tp_decision():
    predictor = _load_real_predictor()
    decision = _toy_model(tp1=1000.0, tp2=1400.0).decide(
        model_features=MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"],
        input_length=32,
        output_length=256,
        concurrency=8,
        gpu_total_mb=24564.0,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_num_seqs=4,
        communication_calibration=predictor,
    )
    assert decision["pre_communication_decision"] == "tp2"
    assert decision["decision"] == "tp1"
    assert decision["communication_changed_decision"] is True
    assert decision["communication"]["estimated_nccl_comm_time_us_tp2"] > 0


def test_communication_adjustment_formula_penalizes_positive_comm_time():
    adjusted = adjust_throughput_for_communication(
        1000.0,
        estimated_nccl_comm_time_us=10.0,
        output_length=100,
        concurrency=1,
    )
    assert adjusted < 1000.0


def test_communication_adjustment_preserves_non_positive_regression_outputs():
    assert adjust_throughput_for_communication(
        -10.0,
        estimated_nccl_comm_time_us=10.0,
        output_length=100,
        concurrency=1,
    ) == -10.0


def test_estimated_communication_bytes_matches_d6_contract():
    mf = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]
    assert estimated_communication_bytes(mf, 1) == 0
    assert estimated_communication_bytes(mf, 2) == 7168


def test_d9_large_structural_compute_saving_remains_tp2():
    predictor = _load_real_predictor()
    decision = _toy_model(tp1=1000.0, tp2=1000.0).decide(
        model_features=MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-7B-Instruct"],
        input_length=32,
        output_length=32,
        concurrency=1,
        gpu_total_mb=24564.0,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_num_seqs=4,
        communication_calibration=predictor,
    )
    assert decision["decision"] == "tp2"
    assert decision["break_even"]["structural_compute_savings_adjustment_us"] > 6000.0
    assert decision["break_even"]["estimated_net_tp2_benefit_us"] > decision["break_even"]["decision_margin_us"]


def test_increasing_collective_count_increases_communication_penalty():
    predictor = _load_real_predictor()
    mf = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]
    c1, d1 = estimate_communication_penalty_us(mf, concurrency=1, tp_degree=2, communication_calibration=predictor)
    c4, d4 = estimate_communication_penalty_us(mf, concurrency=4, tp_degree=2, communication_calibration=predictor)
    assert d4["estimated_collective_call_count"] == 4 * d1["estimated_collective_call_count"]
    assert c4 == pytest.approx(4 * c1)


def test_same_total_bytes_different_call_counts_have_different_costs():
    predictor = _load_real_predictor()
    one_large = predictor.predict_time_us(8192)
    two_small = 2 * predictor.predict_time_us(4096)
    assert 8192 == 2 * 4096
    assert two_small != pytest.approx(one_large)
    assert two_small > one_large


def test_zero_overlap_assumption_is_explicit():
    predictor = _load_real_predictor()
    decision = _toy_model(tp1=1000.0, tp2=1000.0).decide(
        model_features=MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"],
        input_length=32,
        output_length=32,
        concurrency=1,
        gpu_total_mb=24564.0,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        max_num_seqs=4,
        communication_calibration=predictor,
    )
    assert decision["break_even"]["overlap_assumption"] == "zero"


def test_python_cpp_d9_formula_constants_match_contract_fixture():
    mf = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-7B-Instruct"]
    expected = max(0.0, mf["weight_footprint_mb"] - D9_COMPUTE_REFERENCE_WEIGHT_MB) * D9_COMPUTE_SAVINGS_US_PER_WEIGHT_MB_ABOVE_REFERENCE
    assert expected == pytest.approx(6797.175143241882)
    demand = estimate_collective_demand(mf, concurrency=4, tp_degree=2)
    assert demand["bytes_per_collective_call"] == 28672
    assert demand["estimated_collective_call_count"] == 112
