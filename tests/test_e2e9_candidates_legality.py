import pytest

import perf_model.matmul_bias_relu_candidates  # noqa: F401 -- registers MATMUL_BIAS_RELU; imported explicitly so
# test_generator_registry_covers_exactly_the_known_taxonomy_families below is deterministic
# regardless of what else pytest has collected/imported in this session.
from perf_model.implementation_candidate import (
    ImplementationKind, generate_candidates, generate_linear_candidates, generate_rmsnorm_candidates,
)
from perf_model.legality import ReasonCode, check_legality, filter_legal_candidates
from perf_model.operation_descriptor import (
    LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)


def _rmsnorm_op(tokens=16, hidden=4096, dtype="float32"):
    env = OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                             dtype=dtype, device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(tokens, hidden))
    payload = RMSNormDescriptor(token_count=tokens, hidden_size=hidden, epsilon=1e-6, has_weight=True,
                                 input_contiguous=True, output_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def _linear_op(m=4, graph_captured=False):
    env = OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                             dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(m, 151936, 896))
    payload = LinearDescriptor(M=m, N=151936, K=896, has_bias=False, decode_or_prefill="decode",
                                graph_captured=graph_captured, eager_execution=True, tensor_parallel_size=1,
                                weight_layout="row_major", input_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def test_rmsnorm_candidate_generation_covers_all_supported_block_sizes():
    cands = generate_rmsnorm_candidates(_rmsnorm_op())
    block_sizes = {c.parameters["block_size"] for c in cands if "block_size" in c.parameters}
    assert block_sizes == {64, 128, 256, 512}
    assert any(c.parameters.get("is_eager_fallback") for c in cands)


def test_linear_candidate_generation_has_default_and_gemv():
    cands = generate_linear_candidates(_linear_op())
    kinds = {c.implementation_kind for c in cands}
    assert ImplementationKind.LINEAR_DEFAULT_GEMM in kinds
    assert ImplementationKind.LINEAR_ROW_WISE_GEMV in kinds


def test_generate_candidates_dispatches_by_family():
    rms = generate_candidates(_rmsnorm_op())
    lin = generate_candidates(_linear_op())
    assert all(c.operation_family == OperationFamily.RMS_NORM for c in rms)
    assert all(c.operation_family == OperationFamily.LINEAR for c in lin)


def test_generator_registry_covers_exactly_the_known_taxonomy_families():
    # E2E-9 registered RMS_NORM and LINEAR; E2E-10 additively registered a
    # third family (MATMUL_BIAS_RELU) via the same registration mechanism
    # (perf_model.matmul_bias_relu_candidates.register_generator), without
    # touching this dict's definition or generate_candidates() itself. This
    # oracle is updated to reflect that legitimate growth -- the original
    # intent (the registry is a closed, complete set matching the taxonomy,
    # not partially wired) is preserved, not weakened.
    from perf_model.implementation_candidate import _GENERATORS
    assert set(_GENERATORS.keys()) == {OperationFamily.RMS_NORM, OperationFamily.LINEAR, OperationFamily.MATMUL_BIAS_RELU}


def test_rmsnorm_illegal_block_size_rejected():
    op = _rmsnorm_op()
    from perf_model.implementation_candidate import ImplementationCandidate
    from perf_model.operation_descriptor import DecisionKind
    bad = ImplementationCandidate(
        candidate_id="cuda_rmsnorm_fp32_bs999_v1", operation_family=OperationFamily.RMS_NORM,
        operation_subtype="rms_norm_generic", decision_kind=DecisionKind.LAUNCH_CONFIGURATION,
        implementation_kind=ImplementationKind.CUDA_RMSNORM_BLOCK_256, parameters={"block_size": 999},
        supported_dtypes=("float32",), supported_devices=("cuda",),
    )
    result = check_legality(op, bad, target={"dtype": "float32", "device_type": "cuda"})
    assert not result.legal
    assert result.reason_code == ReasonCode.BLOCK_SIZE_UNSUPPORTED


def test_rmsnorm_unsupported_dtype_rejected():
    op = _rmsnorm_op(dtype="float16")
    cands = generate_rmsnorm_candidates(op)
    cuda_cand = next(c for c in cands if c.parameters.get("block_size") == 256)
    result = check_legality(op, cuda_cand, target={"dtype": "float16", "device_type": "cuda"})
    assert not result.legal
    assert result.reason_code == ReasonCode.DTYPE_UNSUPPORTED


def test_rmsnorm_eager_fallback_always_legal():
    op = _rmsnorm_op(dtype="float16")
    cands = generate_rmsnorm_candidates(op)
    fallback = next(c for c in cands if c.parameters.get("is_eager_fallback"))
    result = check_legality(op, fallback, target={"dtype": "float16", "device_type": "cuda"})
    assert result.legal
    assert result.reason_code == ReasonCode.FALLBACK_ALWAYS_LEGAL


def test_gemv_rejected_at_m_equals_1():
    op = _linear_op(m=1)
    cands = generate_linear_candidates(op)
    gemv = next(c for c in cands if c.implementation_kind == ImplementationKind.LINEAR_ROW_WISE_GEMV)
    result = check_legality(op, gemv, target={"dtype": "float16"})
    assert not result.legal
    assert result.reason_code == ReasonCode.GEMV_REJECTED_M_TOO_SMALL


def test_gemv_rejected_above_threshold():
    op = _linear_op(m=9)
    cands = generate_linear_candidates(op)
    gemv = next(c for c in cands if c.implementation_kind == ImplementationKind.LINEAR_ROW_WISE_GEMV)
    result = check_legality(op, gemv, target={"dtype": "float16"})
    assert not result.legal
    assert result.reason_code == ReasonCode.GEMV_REJECTED_M_TOO_LARGE


def test_gemv_rejected_under_graph_capture():
    op = _linear_op(m=4, graph_captured=True)
    cands = generate_linear_candidates(op)
    gemv = next(c for c in cands if c.implementation_kind == ImplementationKind.LINEAR_ROW_WISE_GEMV)
    result = check_legality(op, gemv, target={"dtype": "float16"})
    assert not result.legal
    assert result.reason_code == ReasonCode.GEMV_REJECTED_GRAPH_CAPTURED


def test_gemv_legal_in_valid_range():
    op = _linear_op(m=4)
    cands = generate_linear_candidates(op)
    gemv = next(c for c in cands if c.implementation_kind == ImplementationKind.LINEAR_ROW_WISE_GEMV)
    result = check_legality(op, gemv, target={"dtype": "float16"})
    assert result.legal


def test_default_gemm_always_legal():
    for m in (1, 4, 9, 100):
        op = _linear_op(m=m)
        cands = generate_linear_candidates(op)
        default = next(c for c in cands if c.implementation_kind == ImplementationKind.LINEAR_DEFAULT_GEMM)
        result = check_legality(op, default, target={"dtype": "float16"})
        assert result.legal
        assert result.reason_code == ReasonCode.FALLBACK_ALWAYS_LEGAL


def test_filter_legal_candidates_returns_matching_results_length():
    op = _linear_op(m=4)
    cands = generate_linear_candidates(op)
    legal, results = filter_legal_candidates(op, cands, target={"dtype": "float16"})
    assert len(results) == len(cands)
    assert len(legal) <= len(cands)
    assert all(c.candidate_id in {r.candidate_id for r in results if r.legal} for c in legal)
