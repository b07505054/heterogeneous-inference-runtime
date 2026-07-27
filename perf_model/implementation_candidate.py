"""E2E-9: implementation candidates + candidate generation for the two
unified decision categories (LaunchConfiguration for RMSNorm,
OperationDecomposition for Linear/LM-head).

RMSNorm candidate generation is informed by the REAL existing CUDA kernel
(`cuda_transformer_kernels/rmsnorm_kernel.cu`), which only supports FP32 and
only compiles specializations for block sizes {64,128,256,512}
(`rmsnorm_extension.cpp`'s `supported_block_sizes()`). Linear/LM-head
candidate generation is informed by the REAL E2E-7/E2E-8 evidence
(`perf_model/tiny_m_dispatch.py`): GEMV-loop decomposition is only ever
legal for 1 < M <= threshold (default 8), never at M<=1 (F.linear already
optimal there) and never above the threshold (GEMV-loop measured slower
than cuBLAS batched GEMM at large M in E2E-7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from perf_model.operation_descriptor import (
    DecisionKind, LinearDescriptor, OperationDescriptor, OperationFamily, RMSNormDescriptor,
)

RMSNORM_SUPPORTED_BLOCK_SIZES: tuple[int, ...] = (64, 128, 256, 512)
LINEAR_DEFAULT_GEMV_THRESHOLD = 8  # matches perf_model.tiny_m_dispatch.DEFAULT_THRESHOLD


class ImplementationKind(str, Enum):
    CUDA_RMSNORM_BLOCK_64 = "cuda_rmsnorm_block_64"
    CUDA_RMSNORM_BLOCK_128 = "cuda_rmsnorm_block_128"
    CUDA_RMSNORM_BLOCK_256 = "cuda_rmsnorm_block_256"
    CUDA_RMSNORM_BLOCK_512 = "cuda_rmsnorm_block_512"
    LINEAR_DEFAULT_GEMM = "linear_default_gemm"
    LINEAR_ROW_WISE_GEMV = "linear_row_wise_gemv"


_BLOCK_TO_KIND = {
    64: ImplementationKind.CUDA_RMSNORM_BLOCK_64, 128: ImplementationKind.CUDA_RMSNORM_BLOCK_128,
    256: ImplementationKind.CUDA_RMSNORM_BLOCK_256, 512: ImplementationKind.CUDA_RMSNORM_BLOCK_512,
}


@dataclass(frozen=True)
class ImplementationCandidate:
    candidate_id: str
    operation_family: OperationFamily
    operation_subtype: str
    decision_kind: DecisionKind
    implementation_kind: ImplementationKind
    parameters: dict[str, Any] = field(default_factory=dict)
    supported_dtypes: tuple[str, ...] = ()
    supported_devices: tuple[str, ...] = ()
    required_runtime_features: tuple[str, ...] = ()
    fallback_candidate_id: str | None = None
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "operation_family": self.operation_family.value,
            "operation_subtype": self.operation_subtype, "decision_kind": self.decision_kind.value,
            "implementation_kind": self.implementation_kind.value, "parameters": self.parameters,
            "supported_dtypes": list(self.supported_dtypes), "supported_devices": list(self.supported_devices),
            "required_runtime_features": list(self.required_runtime_features),
            "fallback_candidate_id": self.fallback_candidate_id, "provenance": self.provenance,
        }


def generate_rmsnorm_candidates(operation: OperationDescriptor) -> list[ImplementationCandidate]:
    assert operation.common.operation_family == OperationFamily.RMS_NORM
    fallback_id = "rmsnorm_torch_eager_fallback_v1"
    candidates = [
        ImplementationCandidate(
            candidate_id=fallback_id, operation_family=OperationFamily.RMS_NORM,
            operation_subtype=operation.common.operation_subtype.value,
            decision_kind=DecisionKind.LAUNCH_CONFIGURATION,
            implementation_kind=ImplementationKind.CUDA_RMSNORM_BLOCK_256,  # placeholder kind; real fallback is eager torch, see runtime dispatcher
            parameters={"is_eager_fallback": True},
            supported_dtypes=("float32", "float16", "bfloat16"), supported_devices=("cuda", "cpu"),
            required_runtime_features=(), fallback_candidate_id=None,
            provenance="always-legal eager torch.nn.functional RMSNorm reference; not a CUDA-kernel candidate",
        ),
    ]
    for block_size in RMSNORM_SUPPORTED_BLOCK_SIZES:
        candidates.append(ImplementationCandidate(
            candidate_id=f"cuda_rmsnorm_fp32_bs{block_size}_v1",
            operation_family=OperationFamily.RMS_NORM,
            operation_subtype=operation.common.operation_subtype.value,
            decision_kind=DecisionKind.LAUNCH_CONFIGURATION,
            implementation_kind=_BLOCK_TO_KIND[block_size],
            parameters={"block_size": block_size},
            supported_dtypes=("float32",),  # DIRECTLY_MEASURED: kernel enforces TORCH_CHECK(kFloat32) -- see rmsnorm_kernel.cu
            supported_devices=("cuda",),
            required_runtime_features=("ninja_available", "cuda_extension_compiled"),
            fallback_candidate_id=fallback_id,
            provenance="cuda_transformer_kernels/rmsnorm_kernel.cu template<kBlockSize>, "
                       "candidate_id convention matches deployment/custom_cuda_adapter.py",
        ))
    return candidates


def generate_linear_candidates(operation: OperationDescriptor) -> list[ImplementationCandidate]:
    assert operation.common.operation_family == OperationFamily.LINEAR
    fallback_id = "linear_default_gemm_v1"
    candidates = [
        ImplementationCandidate(
            candidate_id=fallback_id, operation_family=OperationFamily.LINEAR,
            operation_subtype=operation.common.operation_subtype.value,
            decision_kind=DecisionKind.OPERATION_DECOMPOSITION,
            implementation_kind=ImplementationKind.LINEAR_DEFAULT_GEMM,
            parameters={},
            supported_dtypes=("float16", "bfloat16", "float32"), supported_devices=("cuda", "cpu"),
            required_runtime_features=(), fallback_candidate_id=None,
            provenance="production default_unquantized_gemm -> torch.nn.functional.linear path",
        ),
        ImplementationCandidate(
            candidate_id="linear_row_wise_gemv_v1", operation_family=OperationFamily.LINEAR,
            operation_subtype=operation.common.operation_subtype.value,
            decision_kind=DecisionKind.OPERATION_DECOMPOSITION,
            implementation_kind=ImplementationKind.LINEAR_ROW_WISE_GEMV,
            parameters={"threshold": LINEAR_DEFAULT_GEMV_THRESHOLD},
            supported_dtypes=("float16", "bfloat16", "float32"), supported_devices=("cuda",),
            required_runtime_features=("no_cuda_graph_capture",),
            fallback_candidate_id=fallback_id,
            provenance="perf_model/tiny_m_dispatch.py tiny_m_linear row-list+cat GEMV-loop assembly, "
                       "proven in E2E-7 isolated benchmark + E2E-8 real vLLM integration",
        ),
    ]
    return candidates


_GENERATORS = {OperationFamily.RMS_NORM: generate_rmsnorm_candidates, OperationFamily.LINEAR: generate_linear_candidates}

# E2E-10: implementation_kind enum-type registry. RMSNorm and Linear share
# one ImplementationKind enum (defined in this file); discovered while
# integrating a third family that perf_model/execution_policy.py's
# ExecutionPolicy.from_dict hardcoded ImplementationKind when reconstructing
# candidates from JSON, which cannot parse a third family's distinct enum
# values (e.g. "fp32_tiled"). This registry lets execution_policy.py look up
# the right enum class per family generically instead of hardcoding one.
_IMPLEMENTATION_KIND_TYPES: dict[OperationFamily, type] = {
    OperationFamily.RMS_NORM: ImplementationKind, OperationFamily.LINEAR: ImplementationKind,
}


def register_implementation_kind_type(family: OperationFamily, kind_enum_cls: type) -> None:
    _IMPLEMENTATION_KIND_TYPES[family] = kind_enum_cls


def get_implementation_kind_type(family: OperationFamily) -> type:
    kind_cls = _IMPLEMENTATION_KIND_TYPES.get(family)
    if kind_cls is None:
        raise ValueError(f"no implementation_kind enum type registered for {family}")
    return kind_cls


def register_generator(family: OperationFamily, generator) -> None:
    """E2E-10: public registration entry point for a third family's candidate
    generator, so family-specific modules don't need to reach into the
    private _GENERATORS dict directly."""
    _GENERATORS[family] = generator


def generate_candidates(operation: OperationDescriptor, target: dict[str, Any] | None = None,
                         runtime_context: dict[str, Any] | None = None) -> list[ImplementationCandidate]:
    """target / runtime_context are accepted for interface symmetry with
    legality/cost-model calls but candidate generation itself is
    target-independent here -- both families' candidate SETS are fixed by
    what the underlying kernel/dispatch code supports, not by the specific
    GPU. Legality filtering (perf_model/legality.py) is where target- and
    runtime_context-dependent rejection happens."""
    generator = _GENERATORS.get(operation.common.operation_family)
    if generator is None:
        raise ValueError(f"no candidate generator registered for {operation.common.operation_family}")
    return generator(operation)
