"""E2E-9: legality filtering for implementation candidates.

Rules encoded here are all DIRECTLY_MEASURED or SOURCE_PROVEN constraints
discovered during the Phase 0 audit -- not speculative:
  - RMSNorm CUDA kernel: FP32-only (rmsnorm_kernel.cu TORCH_CHECK), CUDA-only,
    block size in {64,128,256,512} (rmsnorm_extension.cpp supported_block_sizes()).
  - Linear GEMV-loop decomposition: illegal at M<=1 (no batching benefit,
    E2E-7 showed default/gemv near-identical at M=1), illegal above the
    configured threshold (E2E-7 measured GEMV-loop losing to cuBLAS batched
    GEMM at larger M), illegal under CUDA-graph capture (row-list+cat
    assembly is Python-side dynamic control flow incompatible with graph
    replay -- perf_model/tiny_m_dispatch.py is only ever invoked from the OOT
    LogitsProcessor's eager _get_logits path, never captured).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from perf_model.implementation_candidate import ImplementationCandidate, ImplementationKind
from perf_model.operation_descriptor import LinearDescriptor, OperationDescriptor, OperationFamily, RMSNormDescriptor


class ReasonCode(str, Enum):
    OK = "ok"
    DTYPE_UNSUPPORTED = "dtype_unsupported"
    DEVICE_UNSUPPORTED = "device_unsupported"
    BLOCK_SIZE_UNSUPPORTED = "block_size_unsupported"
    RUNTIME_FEATURE_MISSING = "runtime_feature_missing"
    GEMV_REJECTED_M_TOO_SMALL = "gemv_rejected_m_too_small"
    GEMV_REJECTED_M_TOO_LARGE = "gemv_rejected_m_too_large"
    GEMV_REJECTED_GRAPH_CAPTURED = "gemv_rejected_graph_captured"
    FALLBACK_ALWAYS_LEGAL = "fallback_always_legal"
    # E2E-10: the pre-existing reason-code vocabulary above cannot represent
    # these real MatMul-Bias-ReLU distinctions (see perf_model/
    # matmul_bias_relu_legality.py for where each is raised).
    PACKED_WEIGHT_REQUIRED = "packed_weight_required"
    UNSUPPORTED_LAYOUT = "unsupported_layout"
    UNSUPPORTED_THREAD_COUNT = "unsupported_thread_count"
    UNSUPPORTED_TARGET_ISA = "unsupported_target_isa"


@dataclass(frozen=True)
class LegalityResult:
    candidate_id: str
    legal: bool
    reason_code: ReasonCode
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "legal": self.legal,
                "reason_code": self.reason_code.value, "detail": self.detail}


def _check_rmsnorm(candidate: ImplementationCandidate, payload: RMSNormDescriptor, target: dict[str, Any],
                    runtime_context: dict[str, Any]) -> LegalityResult:
    if candidate.parameters.get("is_eager_fallback"):
        return LegalityResult(candidate.candidate_id, True, ReasonCode.FALLBACK_ALWAYS_LEGAL,
                               "eager torch reference is always legal")
    dtype = target.get("dtype", "float32")
    if dtype not in candidate.supported_dtypes:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.DTYPE_UNSUPPORTED,
                               f"kernel supports {candidate.supported_dtypes}, requested {dtype}")
    device = target.get("device_type", "cuda")
    if device not in candidate.supported_devices:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.DEVICE_UNSUPPORTED,
                               f"kernel supports {candidate.supported_devices}, requested {device}")
    block_size = candidate.parameters.get("block_size")
    from perf_model.implementation_candidate import RMSNORM_SUPPORTED_BLOCK_SIZES
    if block_size not in RMSNORM_SUPPORTED_BLOCK_SIZES:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.BLOCK_SIZE_UNSUPPORTED,
                               f"block_size {block_size} not in {RMSNORM_SUPPORTED_BLOCK_SIZES}")
    if not runtime_context.get("cuda_extension_compiled", True):
        return LegalityResult(candidate.candidate_id, False, ReasonCode.RUNTIME_FEATURE_MISSING,
                               "cuda_extension_compiled=False in runtime_context")
    return LegalityResult(candidate.candidate_id, True, ReasonCode.OK, "")


def _check_linear(candidate: ImplementationCandidate, payload: LinearDescriptor, target: dict[str, Any],
                   runtime_context: dict[str, Any]) -> LegalityResult:
    if candidate.implementation_kind == ImplementationKind.LINEAR_DEFAULT_GEMM:
        return LegalityResult(candidate.candidate_id, True, ReasonCode.FALLBACK_ALWAYS_LEGAL,
                               "default GEMM (F.linear) is always legal")
    dtype = target.get("dtype", "float16")
    if dtype not in candidate.supported_dtypes:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.DTYPE_UNSUPPORTED,
                               f"gemv-loop supports {candidate.supported_dtypes}, requested {dtype}")
    if payload.M <= 1:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.GEMV_REJECTED_M_TOO_SMALL,
                               f"M={payload.M} <= 1, no batching benefit (E2E-7)")
    threshold = candidate.parameters.get("threshold", 8)
    if payload.M > threshold:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.GEMV_REJECTED_M_TOO_LARGE,
                               f"M={payload.M} > threshold={threshold} (E2E-7 showed GEMV-loop losing to cuBLAS here)")
    if payload.graph_captured:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.GEMV_REJECTED_GRAPH_CAPTURED,
                               "row-list+cat assembly is not CUDA-graph-capture-safe")
    return LegalityResult(candidate.candidate_id, True, ReasonCode.OK, "")


# E2E-10: legality-checker registry, replacing the E2E-9 two-way if/elif
# that check_legality() dispatched directly. GENERIC_EXTENSION (identical
# behavior for RMS_NORM/LINEAR, registered below exactly as before) made so
# a third family's legality rules can be registered from its own family-
# specific module instead of adding a branch here.
_LEGALITY_CHECKERS: dict[OperationFamily, Any] = {}


def register_legality_checker(family: OperationFamily, checker) -> None:
    _LEGALITY_CHECKERS[family] = checker


def check_legality(operation: OperationDescriptor, candidate: ImplementationCandidate,
                    target: dict[str, Any] | None = None,
                    runtime_context: dict[str, Any] | None = None) -> LegalityResult:
    target = target or {}
    runtime_context = runtime_context or {}
    checker = _LEGALITY_CHECKERS.get(operation.common.operation_family)
    if checker is None:
        raise ValueError(f"no legality rules registered for {operation.common.operation_family}")
    return checker(candidate, operation.payload, target, runtime_context)


register_legality_checker(OperationFamily.RMS_NORM, _check_rmsnorm)
register_legality_checker(OperationFamily.LINEAR, _check_linear)


def filter_legal_candidates(operation: OperationDescriptor, candidates: list[ImplementationCandidate],
                             target: dict[str, Any] | None = None,
                             runtime_context: dict[str, Any] | None = None
                             ) -> tuple[list[ImplementationCandidate], list[LegalityResult]]:
    results = [check_legality(operation, c, target, runtime_context) for c in candidates]
    legal_ids = {r.candidate_id for r in results if r.legal}
    legal_candidates = [c for c in candidates if c.candidate_id in legal_ids]
    return legal_candidates, results
