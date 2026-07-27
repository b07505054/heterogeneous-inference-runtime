import json

import pytest

from perf_model.cost_model_registry import AnalyticalFallbackCostModel, CostModelRegistry
from perf_model.implementation_candidate import generate_linear_candidates, generate_rmsnorm_candidates
from perf_model.operation_descriptor import (
    LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)
from perf_model.rmsnorm_cost_model_adapter import DEFAULT_EVIDENCE_PATH, RMSNormCostModel
from perf_model.tiny_m_linear_cost_model import E2E7_LM_HEAD_ISOLATED_MS, TinyMLinearCostModel


def _rmsnorm_op(tokens, hidden):
    env = OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                             dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(tokens, hidden))
    payload = RMSNormDescriptor(token_count=tokens, hidden_size=hidden, epsilon=1e-6, has_weight=True,
                                 input_contiguous=True, output_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def _linear_op(m):
    env = OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                             dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(m, 151936, 896))
    payload = LinearDescriptor(M=m, N=151936, K=896, has_bias=False, decode_or_prefill="decode",
                                graph_captured=False, eager_execution=True, tensor_parallel_size=1,
                                weight_layout="row_major", input_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def test_rmsnorm_cost_model_evidence_file_exists_and_has_48_rows():
    assert DEFAULT_EVIDENCE_PATH.exists()
    raw = json.loads(DEFAULT_EVIDENCE_PATH.read_text())
    assert len(raw["exact_candidates"]) == 48
    assert all(row["correct"] for row in raw["exact_candidates"])


def test_rmsnorm_cost_model_exact_shape_lookup_matches_json():
    model = RMSNormCostModel()
    op = _rmsnorm_op(1, 768)
    cands = generate_rmsnorm_candidates(op)
    cand64 = next(c for c in cands if c.parameters.get("block_size") == 64)
    est = model.predict(op, cand64)
    assert est.source == "measured_lookup"
    assert not est.is_extrapolation
    assert abs(est.predicted_latency_ms - 0.03488) < 1e-6


def test_rmsnorm_cost_model_extrapolates_for_unmeasured_shape():
    model = RMSNormCostModel()
    op = _rmsnorm_op(3, 2000)  # not in the measured sweep
    cands = generate_rmsnorm_candidates(op)
    cand = next(c for c in cands if c.parameters.get("block_size") == 256)
    est = model.predict(op, cand)
    assert est.is_extrapolation
    assert est.source == "measured_interpolation"
    assert est.confidence < 0.9


def test_rmsnorm_cost_model_measured_winner_matches_min_of_four():
    model = RMSNormCostModel()
    winner_bs, winner_ms = model.measured_winner(128, 4096)
    assert winner_bs == 512  # DIRECTLY_MEASURED from Phase 0 sweep printout


def test_tiny_m_linear_cost_model_exact_lookup():
    model = TinyMLinearCostModel()
    op = _linear_op(2)
    cands = generate_linear_candidates(op)
    gemv = next(c for c in cands if "gemv" in c.candidate_id)
    est = model.predict(op, gemv)
    assert est.source == "measured_lookup"
    assert abs(est.predicted_latency_ms - E2E7_LM_HEAD_ISOLATED_MS[2]["gemv"]) < 1e-6


def test_tiny_m_linear_cost_model_interpolates_between_measured_m():
    model = TinyMLinearCostModel()
    op = _linear_op(5)  # between measured M=4 and M=6
    cands = generate_linear_candidates(op)
    gemv = next(c for c in cands if "gemv" in c.candidate_id)
    est = model.predict(op, gemv)
    assert est.is_extrapolation
    lo, hi = E2E7_LM_HEAD_ISOLATED_MS[4]["gemv"], E2E7_LM_HEAD_ISOLATED_MS[6]["gemv"]
    assert min(lo, hi) <= est.predicted_latency_ms <= max(lo, hi)


def test_analytical_fallback_model_produces_low_confidence_estimate():
    model = AnalyticalFallbackCostModel()
    op = _rmsnorm_op(16, 4096)
    cands = generate_rmsnorm_candidates(op)
    est = model.predict(op, cands[1])
    assert est.confidence < 0.5
    assert est.is_extrapolation
    assert est.predicted_latency_ms > 0


def test_registry_returns_family_specific_models_not_one_universal_model():
    registry = CostModelRegistry()
    registry.register(OperationFamily.RMS_NORM, RMSNormCostModel())
    registry.register(OperationFamily.LINEAR, TinyMLinearCostModel())
    rms_model = registry.get_model(OperationFamily.RMS_NORM)
    lin_model = registry.get_model(OperationFamily.LINEAR)
    assert rms_model is not lin_model
    assert type(rms_model) is not type(lin_model)
    assert rms_model.model_id != lin_model.model_id


def test_registry_falls_back_to_analytical_model_for_unregistered_family():
    registry = CostModelRegistry()  # nothing registered
    model = registry.get_model(OperationFamily.RMS_NORM)
    assert isinstance(model, AnalyticalFallbackCostModel)
