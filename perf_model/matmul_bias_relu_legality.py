"""E2E-10: legality rules for fused MatMul-Bias-ReLU candidates, registered
into perf_model.legality's checker registry (register_legality_checker)
rather than adding a branch to check_legality().

Every rule here is SOURCE_PROVEN from the real dispatch contract in
deployment/execution_plan/portable_cpu_kernel_adapter.py, discovered in
Phase 0:
  - INT8 kernels (row-major and packed-B) only accept
    thread_count=1/partition_axis=none/partition_strategy=serial
    (dispatch_fused_matmul_bias_relu raises otherwise).
  - The packed-B kernel requires a materialized packed-weight artifact
    (there is no runtime repacking path).
  - cortex_a76_dotprod codegen requires target_arch=='aarch64' and
    target_cpu=='cortex-a76' (_validate_codegen_contract_static).
  - Per Phase 4's explicit instruction, shape being outside the measured
    cost-model domain is NEVER a legality rejection here -- the kernels
    themselves have no shape-divisibility requirement (confirmed by reading
    the tiling code's std::min-bounded remainder handling and by real Pi
    execution at non-multiple shapes 37x41x29, 256x256x24). Domain/
    confidence concerns live only in matmul_bias_relu_cost_model.py.
"""
from __future__ import annotations

from typing import Any

from perf_model.implementation_candidate import ImplementationCandidate
from perf_model.legality import LegalityResult, ReasonCode, register_legality_checker
from perf_model.matmul_bias_relu_candidates import MatMulBiasReLUImplementationKind
from perf_model.matmul_bias_relu_descriptor import MatMulBiasReLUDescriptor
from perf_model.operation_descriptor import OperationFamily


def _check_matmul_bias_relu(candidate: ImplementationCandidate, payload: MatMulBiasReLUDescriptor,
                             target: dict[str, Any], runtime_context: dict[str, Any]) -> LegalityResult:
    if candidate.implementation_kind == MatMulBiasReLUImplementationKind.FP32_TILED:
        # Always-legal correctness fallback: fused (bias+relu in the same
        # tile-store loop), semantically identical to the INT8 candidates'
        # dequantized output, dtype fp32 always available.
        return LegalityResult(candidate.candidate_id, True, ReasonCode.FALLBACK_ALWAYS_LEGAL,
                               "FP32 tiled fused kernel is always legal (no quantization/packing preconditions)")

    device = target.get("device_type", "cpu")
    if device != "cpu":
        return LegalityResult(candidate.candidate_id, False, ReasonCode.DEVICE_UNSUPPORTED,
                               f"candidate only supports device='cpu', requested '{device}'")

    if not payload.quantized:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.DTYPE_UNSUPPORTED,
                               "INT8 candidate requested for a non-quantized (fp32) descriptor")

    if payload.thread_count != 1:
        return LegalityResult(candidate.candidate_id, False, ReasonCode.UNSUPPORTED_THREAD_COUNT,
                               f"INT8 kernels (row-major and packed-B) only support thread_count=1, "
                               f"descriptor requested thread_count={payload.thread_count} "
                               f"(portable_cpu_kernel_adapter.dispatch_fused_matmul_bias_relu hard-rejects otherwise)")

    if candidate.implementation_kind == MatMulBiasReLUImplementationKind.INT8_SCALAR:
        if payload.weight_layout != "row_major_kx_n":
            return LegalityResult(candidate.candidate_id, False, ReasonCode.UNSUPPORTED_LAYOUT,
                                   f"int8_scalar candidate requires weight_layout='row_major_kx_n', "
                                   f"descriptor has '{payload.weight_layout}'")
        return LegalityResult(candidate.candidate_id, True, ReasonCode.OK, "")

    if candidate.implementation_kind == MatMulBiasReLUImplementationKind.INT8_PACKED_B:
        if payload.weight_layout != "packed_b_transposed_nxk":
            return LegalityResult(candidate.candidate_id, False, ReasonCode.UNSUPPORTED_LAYOUT,
                                   f"packed-B candidate requires weight_layout='packed_b_transposed_nxk', "
                                   f"descriptor has '{payload.weight_layout}'")
        if not payload.packed_b_available:
            return LegalityResult(candidate.candidate_id, False, ReasonCode.PACKED_WEIGHT_REQUIRED,
                                   "packed-B kernel requires a materialized packed-weight artifact "
                                   "(packed_b_available=False); the kernel never repacks at runtime")
        target_isa = candidate.parameters.get("target_isa")
        if target_isa == "cortex_a76_dotprod":
            if payload.target_arch != "aarch64" or payload.target_cpu != "cortex-a76":
                return LegalityResult(candidate.candidate_id, False, ReasonCode.UNSUPPORTED_TARGET_ISA,
                                       f"cortex_a76_dotprod codegen requires target_arch='aarch64' and "
                                       f"target_cpu='cortex-a76', descriptor has target_arch="
                                       f"'{payload.target_arch}' target_cpu='{payload.target_cpu}'")
        return LegalityResult(candidate.candidate_id, True, ReasonCode.OK, "")

    return LegalityResult(candidate.candidate_id, False, ReasonCode.DTYPE_UNSUPPORTED,
                           f"unrecognized implementation_kind {candidate.implementation_kind}")


register_legality_checker(OperationFamily.MATMUL_BIAS_RELU, _check_matmul_bias_relu)
