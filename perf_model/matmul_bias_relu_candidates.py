"""E2E-10: candidate generation for fused MatMul-Bias-ReLU, registered into
perf_model.implementation_candidate's generator registry
(register_generator) rather than adding a branch to generate_candidates().

Candidates are DISCOVERED, not invented: this module calls the real,
existing, committed
deployment.execution_plan.slice3c_target_selection.enumerate_complete_candidates()
and wraps its "portable_cpu" backend results (the 4 real native-kernel
candidates this slice targets -- the "executorch_xnnpack" backend candidates
that same function also produces are a different runtime/dispatch path,
out of scope for this slice, see the Phase 0 audit) into E2E-9
ImplementationCandidate objects. No new candidate logic is introduced here.

Implementation-kind naming deviates from the task's illustrative example
names where reality differs: the real FP32 candidate
(portable_fused_matmul_bias_relu_bm32_bn128_bk32) is block-TILED, not a
naive scalar reference, so it is named FP32_TILED rather than
FP32_REFERENCE/FP32_SCALAR. The two INT8 packed-B variants
(generic_aarch64, cortex_a76_dotprod) share one INT8_PACKED_B
implementation kind, distinguished by a target_isa parameter -- they run
the IDENTICAL packed-B algorithm (native/cpu_kernels/
portable_fused_matmul_bias_relu_int8_packed_b.cpp), differing only in
compiler codegen flags (-O2 vs -O3 -mcpu=cortex-a76), which is a codegen/
ISA-target distinction, not a different kernel implementation.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from deployment.execution_plan.slice3c_target_selection import (
    FP32_CANDIDATE_ID,
    FP32_KERNEL_ID,
    INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,
    INT8_PACKED_GENERIC_CANDIDATE_ID,
    INT8_ROW_MAJOR_CANDIDATE_ID,
    enumerate_complete_candidates,
)
from perf_model.implementation_candidate import ImplementationCandidate, register_generator, register_implementation_kind_type
from perf_model.operation_descriptor import DecisionKind, OperationDescriptor, OperationFamily

INT8_ROW_MAJOR_KERNEL_ID = "portable_fused_matmul_bias_relu_int8_symmetric"
INT8_PACKED_KERNEL_ID = "portable_fused_matmul_bias_relu_int8_symmetric_packed_b"


class MatMulBiasReLUImplementationKind(str, Enum):
    FP32_TILED = "fp32_tiled"
    INT8_SCALAR = "int8_scalar"
    INT8_PACKED_B = "int8_packed_b"


FP32_TILED_E2E10_ID = "matmul_bias_relu_fp32_tiled_v1"
INT8_SCALAR_E2E10_ID = "matmul_bias_relu_int8_scalar_v1"
INT8_PACKED_B_GENERIC_E2E10_ID = "matmul_bias_relu_int8_packed_b_generic_v1"
INT8_PACKED_B_A76DOTPROD_E2E10_ID = "matmul_bias_relu_int8_packed_b_a76dotprod_v1"

# Discovered slice3c complete_candidate_id -> (e2e10 candidate_id, implementation_kind, extra parameters)
_SLICE3C_TO_E2E10 = {
    FP32_CANDIDATE_ID: (FP32_TILED_E2E10_ID, MatMulBiasReLUImplementationKind.FP32_TILED,
                         {"kernel_id": FP32_KERNEL_ID, "block_m": 32, "block_n": 128, "block_k": 32}),
    INT8_ROW_MAJOR_CANDIDATE_ID: (INT8_SCALAR_E2E10_ID, MatMulBiasReLUImplementationKind.INT8_SCALAR,
                                   {"kernel_id": INT8_ROW_MAJOR_KERNEL_ID, "target_isa": "generic_aarch64"}),
    INT8_PACKED_GENERIC_CANDIDATE_ID: (INT8_PACKED_B_GENERIC_E2E10_ID, MatMulBiasReLUImplementationKind.INT8_PACKED_B,
                                        {"kernel_id": INT8_PACKED_KERNEL_ID, "target_isa": "generic_aarch64"}),
    INT8_PACKED_A76_DOTPROD_CANDIDATE_ID: (INT8_PACKED_B_A76DOTPROD_E2E10_ID, MatMulBiasReLUImplementationKind.INT8_PACKED_B,
                                            {"kernel_id": INT8_PACKED_KERNEL_ID, "target_isa": "cortex_a76_dotprod"}),
}


def generate_matmul_bias_relu_candidates(operation: OperationDescriptor) -> list[ImplementationCandidate]:
    assert operation.common.operation_family == OperationFamily.MATMUL_BIAS_RELU
    payload = operation.payload

    complete_candidates = {c.candidate_id: c for c in enumerate_complete_candidates() if c.backend == "portable_cpu"}
    candidates: list[ImplementationCandidate] = []
    for slice3c_id, (e2e10_id, kind, extra) in _SLICE3C_TO_E2E10.items():
        complete = complete_candidates.get(slice3c_id)
        if complete is None:
            continue  # discovered-but-not-generated this run; never invented
        quantized = kind != MatMulBiasReLUImplementationKind.FP32_TILED
        params: dict[str, Any] = dict(extra)
        params["complete_candidate_id"] = slice3c_id
        params["codegen_target_id"] = complete.codegen_target_id
        params["required_compiler_flags"] = list(complete.required_compiler_flags)
        params["weight_layout"] = complete.weight_layout
        params["packing_scheme"] = complete.packing_scheme
        candidates.append(ImplementationCandidate(
            candidate_id=e2e10_id, operation_family=OperationFamily.MATMUL_BIAS_RELU,
            operation_subtype=operation.common.operation_subtype.value,
            decision_kind=DecisionKind.KERNEL_IMPLEMENTATION,
            implementation_kind=kind, parameters=params,
            supported_dtypes=("int8",) if quantized else ("fp32",),
            supported_devices=("cpu",),
            required_runtime_features=("packed_b_artifact",) if kind == MatMulBiasReLUImplementationKind.INT8_PACKED_B else (),
            fallback_candidate_id=None if kind == MatMulBiasReLUImplementationKind.FP32_TILED else FP32_TILED_E2E10_ID,
            provenance=f"deployment.execution_plan.slice3c_target_selection candidate '{slice3c_id}' "
                       f"(kernel_id={complete.kernel_id}); native/cpu_kernels/"
                       f"{'portable_fused_matmul_bias_relu.cpp' if not quantized else ('portable_fused_matmul_bias_relu_int8_packed_b.cpp' if 'packed' in slice3c_id else 'portable_fused_matmul_bias_relu_int8.cpp')}",
        ))
    return candidates


register_generator(OperationFamily.MATMUL_BIAS_RELU, generate_matmul_bias_relu_candidates)
register_implementation_kind_type(OperationFamily.MATMUL_BIAS_RELU, MatMulBiasReLUImplementationKind)
