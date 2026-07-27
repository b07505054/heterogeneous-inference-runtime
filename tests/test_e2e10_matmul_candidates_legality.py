import pytest

from perf_model.implementation_candidate import generate_candidates
from perf_model.legality import ReasonCode, filter_legal_candidates
from perf_model.matmul_bias_relu_candidates import (
    FP32_TILED_E2E10_ID, INT8_PACKED_B_A76DOTPROD_E2E10_ID, INT8_PACKED_B_GENERIC_E2E10_ID, INT8_SCALAR_E2E10_ID,
    MatMulBiasReLUImplementationKind, generate_matmul_bias_relu_candidates,
)
import perf_model.matmul_bias_relu_legality  # noqa: F401 -- registers the legality checker
from perf_model.matmul_bias_relu_descriptor import MatMulBiasReLUDescriptor
from perf_model.operation_descriptor import OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype


def _op(quantized, weight_layout, packed_b_available=False, thread_count=1, target_cpu="cortex-a76", target_arch="aarch64", m=8, n=8, k=8):
    env = OperationEnvelope(operation_family=OperationFamily.MATMUL_BIAS_RELU, operation_subtype=OperationSubtype.FUSED_MATMUL_BIAS_RELU,
                             dtype="int8" if quantized else "fp32", device_type="cpu", target_arch=target_arch, phase="n/a",
                             logical_shape=(m, n, k))
    payload = MatMulBiasReLUDescriptor(
        M=m, N=n, K=k, input_dtype="int8" if quantized else "fp32", weight_dtype="int8" if quantized else "fp32",
        accumulator_dtype="int32" if quantized else "fp32", output_dtype="fp32", has_bias=True, activation="relu",
        input_layout="row_major", weight_layout=weight_layout, output_layout="row_major",
        input_contiguous=True, weight_contiguous=True, output_contiguous=True, quantized=quantized,
        target_arch=target_arch, target_cpu=target_cpu, thread_count=thread_count,
        input_scale=0.01 if quantized else None, weight_scale=0.01 if quantized else None,
        input_zero_point=0 if quantized else None, weight_zero_point=0 if quantized else None,
        per_tensor_or_per_channel="per_tensor" if quantized else None, packed_b_available=packed_b_available,
    )
    return OperationDescriptor(common=env, payload=payload)


def test_candidate_generation_produces_all_4_real_candidates():
    op = _op(quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=True)
    cands = generate_matmul_bias_relu_candidates(op)
    ids = {c.candidate_id for c in cands}
    assert ids == {FP32_TILED_E2E10_ID, INT8_SCALAR_E2E10_ID, INT8_PACKED_B_GENERIC_E2E10_ID, INT8_PACKED_B_A76DOTPROD_E2E10_ID}


def test_fp32_fallback_always_emitted():
    for quantized, layout, packed in [(False, "row_major_kx_n", False), (True, "packed_b_transposed_nxk", True)]:
        op = _op(quantized=quantized, weight_layout=layout, packed_b_available=packed)
        cands = generate_matmul_bias_relu_candidates(op)
        assert any(c.candidate_id == FP32_TILED_E2E10_ID for c in cands)


def test_stable_candidate_ids_deterministic_ordering():
    op = _op(quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=True)
    ids_1 = [c.candidate_id for c in generate_matmul_bias_relu_candidates(op)]
    ids_2 = [c.candidate_id for c in generate_matmul_bias_relu_candidates(op)]
    assert ids_1 == ids_2
    assert all(cid.startswith("matmul_bias_relu_") and cid.endswith("_v1") for cid in ids_1)


def test_generate_candidates_dispatches_through_generic_registry():
    op = _op(quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=True)
    cands = generate_candidates(op)
    assert {c.candidate_id for c in cands} == {FP32_TILED_E2E10_ID, INT8_SCALAR_E2E10_ID, INT8_PACKED_B_GENERIC_E2E10_ID, INT8_PACKED_B_A76DOTPROD_E2E10_ID}


def test_legality_fp32_only_dtype_rejection_for_fp32_op():
    op = _op(quantized=False, weight_layout="row_major_kx_n")
    cands = generate_candidates(op)
    legal, results = filter_legal_candidates(op, cands, target={"device_type": "cpu"})
    assert [c.candidate_id for c in legal] == [FP32_TILED_E2E10_ID]
    reasons = {r.candidate_id: r.reason_code for r in results if not r.legal}
    assert reasons[INT8_SCALAR_E2E10_ID] == ReasonCode.DTYPE_UNSUPPORTED


def test_legality_device_rejection():
    op = _op(quantized=False, weight_layout="row_major_kx_n")
    cands = generate_candidates(op)
    legal, results = filter_legal_candidates(op, cands, target={"device_type": "cuda"})
    fp32_result = next(r for r in results if r.candidate_id == FP32_TILED_E2E10_ID)
    assert fp32_result.legal  # FP32 fallback is always legal regardless of declared device
    int8_results = [r for r in results if r.candidate_id != FP32_TILED_E2E10_ID]
    assert all(r.reason_code == ReasonCode.DEVICE_UNSUPPORTED for r in int8_results)


def test_legality_layout_mismatch_rejection():
    op = _op(quantized=True, weight_layout="row_major_kx_n")  # not packed
    cands = generate_candidates(op)
    legal, results = filter_legal_candidates(op, cands, target={"device_type": "cpu"})
    legal_ids = {c.candidate_id for c in legal}
    assert INT8_SCALAR_E2E10_ID in legal_ids
    assert INT8_PACKED_B_GENERIC_E2E10_ID not in legal_ids
    packed_result = next(r for r in results if r.candidate_id == INT8_PACKED_B_GENERIC_E2E10_ID)
    assert packed_result.reason_code == ReasonCode.UNSUPPORTED_LAYOUT


def test_legality_packed_weight_required_rejected_at_descriptor_construction():
    # packed layout without packed_b_available is rejected at the earliest point
    # (descriptor construction), not deferred to candidate-level legality.
    from perf_model.matmul_bias_relu_descriptor import MatMulBiasReLUDescriptorError
    with pytest.raises(MatMulBiasReLUDescriptorError):
        _op(quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=False)


def test_legality_thread_count_rejection_for_int8():
    op = _op(quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=True, thread_count=4)
    cands = generate_candidates(op)
    legal, results = filter_legal_candidates(op, cands, target={"device_type": "cpu"})
    legal_ids = {c.candidate_id for c in legal}
    assert legal_ids == {FP32_TILED_E2E10_ID}
    int8_results = [r for r in results if r.candidate_id != FP32_TILED_E2E10_ID]
    assert all(r.reason_code == ReasonCode.UNSUPPORTED_THREAD_COUNT for r in int8_results)


def test_legality_target_isa_rejection_for_a76_dotprod_off_target():
    op = _op(quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=True,
              target_arch="x86_64", target_cpu="generic")
    cands = generate_candidates(op)
    legal, results = filter_legal_candidates(op, cands, target={"device_type": "cpu"})
    legal_ids = {c.candidate_id for c in legal}
    assert INT8_PACKED_B_GENERIC_E2E10_ID in legal_ids
    assert INT8_PACKED_B_A76DOTPROD_E2E10_ID not in legal_ids
    a76_result = next(r for r in results if r.candidate_id == INT8_PACKED_B_A76DOTPROD_E2E10_ID)
    assert a76_result.reason_code == ReasonCode.UNSUPPORTED_TARGET_ISA


def test_fallback_always_legal_fp32():
    op = _op(quantized=True, weight_layout="packed_b_transposed_nxk", packed_b_available=True)
    cands = generate_candidates(op)
    legal, results = filter_legal_candidates(op, cands, target={"device_type": "cpu"})
    fp32_result = next(r for r in results if r.candidate_id == FP32_TILED_E2E10_ID)
    assert fp32_result.legal
    assert fp32_result.reason_code == ReasonCode.FALLBACK_ALWAYS_LEGAL
