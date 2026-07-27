import json

import pytest

from perf_model.cost_model_registry import CostModelRegistry
from perf_model.matmul_bias_relu_candidates import (
    FP32_TILED_E2E10_ID, INT8_PACKED_B_A76DOTPROD_E2E10_ID, INT8_PACKED_B_GENERIC_E2E10_ID, INT8_SCALAR_E2E10_ID,
    generate_matmul_bias_relu_candidates,
)
import perf_model.matmul_bias_relu_legality  # noqa: F401
from perf_model.matmul_bias_relu_cost_model import DEFAULT_EVIDENCE_PATH, MatMulBiasReLUCostModel
from perf_model.matmul_bias_relu_descriptor import MatMulBiasReLUDescriptor
from perf_model.operation_descriptor import OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype


def _op(m, n, k, quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=True):
    env = OperationEnvelope(operation_family=OperationFamily.MATMUL_BIAS_RELU, operation_subtype=OperationSubtype.FUSED_MATMUL_BIAS_RELU,
                             dtype="int8" if quantized else "fp32", device_type="cpu", target_arch="aarch64", phase="n/a",
                             logical_shape=(m, n, k))
    payload = MatMulBiasReLUDescriptor(
        M=m, N=n, K=k, input_dtype="int8" if quantized else "fp32", weight_dtype="int8" if quantized else "fp32",
        accumulator_dtype="int32" if quantized else "fp32", output_dtype="fp32", has_bias=True, activation="relu",
        input_layout="row_major", weight_layout=weight_layout, output_layout="row_major",
        input_contiguous=True, weight_contiguous=True, output_contiguous=True, quantized=quantized,
        target_arch="aarch64", target_cpu="cortex-a76", thread_count=1,
        input_scale=0.01 if quantized else None, weight_scale=0.01 if quantized else None,
        input_zero_point=0 if quantized else None, weight_zero_point=0 if quantized else None,
        per_tensor_or_per_channel="per_tensor" if quantized else None, packed_b_available=packed_b_available,
    )
    return OperationDescriptor(common=env, payload=payload)


def test_evidence_file_exists_and_is_real_pi_data():
    assert DEFAULT_EVIDENCE_PATH.exists()
    raw = json.loads(DEFAULT_EVIDENCE_PATH.read_text())
    assert raw["hardware_identity"]["machine"] == "aarch64"
    assert len(raw["results"]) == 9


def test_exact_measured_lookup_matches_json():
    model = MatMulBiasReLUCostModel()
    op = _op(8, 8, 8)
    cand = next(c for c in generate_matmul_bias_relu_candidates(op) if c.candidate_id == INT8_PACKED_B_A76DOTPROD_E2E10_ID)
    est = model.predict(op, cand)
    assert est.source == "measured_lookup"
    assert not est.is_extrapolation
    assert est.predicted_latency_ms > 0


def test_extrapolation_for_unmeasured_shape():
    model = MatMulBiasReLUCostModel()
    op = _op(50, 60, 70)  # not in the measured set
    cand = next(c for c in generate_matmul_bias_relu_candidates(op) if c.candidate_id == FP32_TILED_E2E10_ID)
    est = model.predict(op, cand)
    assert est.is_extrapolation
    assert est.source == "measured_interpolation"
    assert est.confidence < 0.9


def test_measured_winner_is_real_and_shape_dependent():
    model = MatMulBiasReLUCostModel()
    # 384x384x384: packed-B a76-dotprod has the lowest raw measured median on
    # the real Pi, and is also what deployment.execution_plan.
    # slice3c_target_selection.select_candidate() picked (both correctness
    # and stability gates passed at this shape).
    winner_384, _ = model.measured_winner(384, 384, 384)
    assert "cortex_a76_dotprod" in winner_384


def test_measured_winner_raw_latency_can_diverge_from_gated_selection_at_tiny_shapes():
    """DIRECTLY_MEASURED, real finding from Phase 11: at shape 2x2x2, all 3
    INT8 candidates have lower raw median latency than FP32, but
    select_candidate()'s real stability gate (max_p95_over_median<=1.25)
    rejects all 3 (p95/median ~1.32-1.35 at microsecond scale, vs FP32's
    ~1.12) -- so the real system falls back to FP32 despite it not having
    the lowest raw median. MatMulBiasReLUCostModel.measured_winner() reports
    the raw-latency winner (a different, simpler question than the gated
    selection) -- this test documents that the two questions are NOT always
    the same answer, honestly, rather than assuming they must agree."""
    model = MatMulBiasReLUCostModel()
    raw_winner, raw_ms = model.measured_winner(2, 2, 2)
    assert "int8" in raw_winner  # raw-latency winner: an INT8 candidate
    raw = json.loads(DEFAULT_EVIDENCE_PATH.read_text())
    row = next(r for r in raw["results"] if (r["shape"]["M"], r["shape"]["N"], r["shape"]["K"]) == (2, 2, 2))
    assert row["selection"]["selected_candidate_id"].endswith(":fp32:row_major_kx_n:generic_aarch64")  # gated selection: FP32


def test_registry_holds_matmul_model_separately_from_other_families():
    from perf_model.rmsnorm_cost_model_adapter import RMSNormCostModel
    from perf_model.tiny_m_linear_cost_model import TinyMLinearCostModel

    registry = CostModelRegistry()
    registry.register(OperationFamily.RMS_NORM, RMSNormCostModel())
    registry.register(OperationFamily.LINEAR, TinyMLinearCostModel())
    registry.register(OperationFamily.MATMUL_BIAS_RELU, MatMulBiasReLUCostModel())
    models = {registry.get_model(f).model_id for f in
              (OperationFamily.RMS_NORM, OperationFamily.LINEAR, OperationFamily.MATMUL_BIAS_RELU)}
    assert len(models) == 3  # three distinct model_ids -- no shared/universal model


def test_accuracy_gate_helper_uses_real_measured_metrics():
    model = MatMulBiasReLUCostModel()
    passed = model.accuracy_gate_passed(384, 384, 384, "slice3c:portable_cpu:int8_static_symmetric:packed_b_transposed_nxk:cortex_a76_dotprod")
    assert passed is True
