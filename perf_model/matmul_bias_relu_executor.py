"""E2E-10 Phase 9: fused MatMul-Bias-ReLU runtime executor.

Reuses, unmodified: deployment.execution_plan.portable_cpu_kernel_adapter.
dispatch_fused_matmul_bias_relu (the real subprocess dispatcher for the real
compiled native kernels) and deployment.execution_plan.int8_quantization's
quantization/packing helpers. This module's only new logic is translating an
E2E-9 ImplementationDecision (selected via the unified selector) into the
op_decision dict shape that dispatch_fused_matmul_bias_relu's contract
requires -- the same translation role perf_model/runtime_dispatcher.py plays
for RMSNorm (translating a selected block_size into a CUDA extension call)
and for LM-head (translating a selected decomposition into an assembly-
function call).

Phase 0 discovered that scripts/run_slice3a_int8_eval.py's own decision
builders are STALE against the current, committed
portable_cpu_kernel_adapter.py contract (missing the required
quantization.execution_stages block) -- confirmed by direct execution
failure, not by inspection. This module builds decisions correctly from the
adapter's real validation contract (_validate_int8_execution_stages),
verified by executing all 4 real candidates successfully on the real
Raspberry Pi 5 target (see perf_model/evidence/matmul_bias_relu_e2e10.json).

Packing/calibration-artifact preparation (MatMulBiasReLUExecutionContext) is
intentionally separated from steady_state execution: production's own
packed-B kernel self-reports "runtime_packed_weight_transform": false
(native/cpu_kernels/portable_fused_matmul_bias_relu_int8_packed_b.cpp),
confirming packing is a one-time/offline cost in the real system -- this
executor mirrors that by requiring pre-built calibration/packed artifacts as
input rather than creating them inside execute_matmul_bias_relu().

Registration into perf_model.runtime_dispatcher's executor registry is
attempted but optional: this family's real execution environment (Raspberry
Pi aarch64 CPU, plain-Python list-backed tensors) does not require or use
torch, while perf_model.runtime_dispatcher imports torch at module level for
the CUDA families. Import is wrapped in try/except so this module remains
usable standalone (e.g. on the Pi, or in any torch-less environment) --
"equivalent dispatch table" per the task's Phase 9 wording, not a required
torch dependency for a CPU-only kernel family.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deployment.execution_plan.int8_quantization import (
    PACKED_B_TRANSPOSE_LAYOUT,
    PACKED_B_TRANSPOSE_SCHEME,
    PACKED_INT8_KERNEL_CAPABILITY,
    PACKED_INT8_KERNEL_ID,
    SCHEME,
    KERNEL_CAPABILITY,
)
from deployment.execution_plan.portable_cpu_kernel_adapter import (
    PortableCpuKernelResult,
    Tensor,
    dispatch_fused_matmul_bias_relu,
)
from perf_model.implementation_decision import ImplementationDecision
from perf_model.matmul_bias_relu_candidates import MatMulBiasReLUImplementationKind

FP32_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn128_bk32"
INT8_ROW_MAJOR_KERNEL_ID = "portable_fused_matmul_bias_relu_int8_symmetric"


class MatMulBiasReLUExecutionError(ValueError):
    pass


@dataclass
class MatMulBiasReLUExecutionContext:
    """One-time artifacts, prepared ahead of the timed steady-state call.
    kernel_executables maps E2E-10 candidate_id -> compiled native binary
    Path (built once via deployment.execution_plan.slice3c_target_selection.
    materialize_build_manifest, reused across every call, never rebuilt
    inside execute_matmul_bias_relu)."""
    kernel_executables: dict[str, Path]
    calibration_artifact: dict[str, Any] | None = None
    calibration_artifact_path: Path | None = None
    packed_artifact: dict[str, Any] | None = None
    packed_artifact_path: Path | None = None
    build_identity: dict[str, dict[str, Any]] | None = None  # slice3c_complete_candidate_id -> build manifest


def _memory_placement(m: int, n: int, k: int) -> dict[str, Any]:
    return {
        "status": "selected", "compute_unit": "cpu", "selected_memory_space": "cpu_visible_host_memory",
        "transfer_operations": [], "compute_dependency_ids": [],
        "buffer_placements": [
            {"buffer_id": "input_tile", "role": "input", "memory_space": "cpu_visible_host_memory", "byte_count": m * k * 4, "alignment": 64},
            {"buffer_id": "weight_tile", "role": "weight", "memory_space": "cpu_visible_host_memory", "byte_count": k * n * 4, "alignment": 64},
            {"buffer_id": "output_tile", "role": "output", "memory_space": "cpu_visible_host_memory", "byte_count": m * n * 4, "alignment": 64},
            {"buffer_id": "scratch", "role": "scratch", "memory_space": "cpu_visible_host_memory", "byte_count": 0, "alignment": 64},
        ],
    }


def _int8_execution_stages(*, kernel_id: str, activation_scale: float, packed_ref: str | None,
                            packed_sha: str | None, binary_sha256: str | None) -> list[dict[str, Any]]:
    stages = [{
        "stage_id": "quantize_activation", "op": "hir.quantize", "scale": activation_scale, "zero_point": 0,
        "rounding_mode": "round_nearest_even", "clamp_min": -127, "clamp_max": 127,
        "source_dtype": "fp32", "destination_dtype": "int8",
    }]
    deps = ["quantized_activation_ready"]
    if kernel_id == PACKED_INT8_KERNEL_ID:
        stages.append({"stage_id": "load_packed_weight", "op": "hir.load_quantized_weight",
                        "artifact_ref": packed_ref, "artifact_sha256": packed_sha, "packed_layout": PACKED_B_TRANSPOSE_LAYOUT})
        deps.append("packed_weight_ready")
    kernel_stage = {"stage_id": "execute_int8_kernel", "op": "hir.portable_cpu_int8_fused_matmul_bias_relu",
                     "kernel_id": kernel_id, "dependency_ids": deps}
    if kernel_id == PACKED_INT8_KERNEL_ID:
        kernel_stage["fused_postprocess"] = "dequantize_bias_relu"
        if binary_sha256:
            kernel_stage["binary_sha256"] = binary_sha256
    stages.append(kernel_stage)
    stages.append({"stage_id": "return_fp32_output", "op": "runtime.return", "dependency_ids": ["fp32_output_ready"]})
    return stages


def _build_op_decision(candidate_kind: MatMulBiasReLUImplementationKind, target_isa: str | None,
                        m: int, n: int, k: int, context: MatMulBiasReLUExecutionContext) -> dict[str, Any]:
    if candidate_kind == MatMulBiasReLUImplementationKind.FP32_TILED:
        return {
            "op_name": "op_e2e10", "op_type": "hir.fused_matmul_bias_relu",
            "kernel_selection": {"contract_version": "kernel_selection_contract_v1", "status": "selected",
                                  "selected_kernel": FP32_KERNEL_ID, "source": "e2e10_unified_selector"},
            "quantization": {"selected_candidate_id": f"fused_matmul_bias_relu:quant=fp32_baseline:kernel={FP32_KERNEL_ID}",
                              "scheme": "fp32_baseline", "activation_dtype": "fp32", "weight_dtype": "fp32",
                              "accumulation_dtype": "fp32", "output_dtype": "fp32",
                              "required_kernel_capability": "quant_kernel.none",
                              "requires_calibration": False, "calibration_available": False},
            "thread_schedule": {"status": "selected", "thread_count": 1, "partition_axis": "none", "partition_strategy": "serial"},
            "memory_placement": _memory_placement(m, n, k),
        }

    if context.calibration_artifact is None or context.calibration_artifact_path is None:
        raise MatMulBiasReLUExecutionError(
            "INT8 candidate selected but no calibration artifact was prepared in the execution context "
            "-- this executor will not invent scale/zero-point values"
        )
    artifact = context.calibration_artifact
    activation_scale = artifact["activation_scale"]

    if candidate_kind == MatMulBiasReLUImplementationKind.INT8_SCALAR:
        return {
            "op_name": "op_e2e10", "op_type": "hir.fused_matmul_bias_relu",
            "kernel_selection": {"contract_version": "kernel_selection_contract_v1", "status": "selected",
                                  "selected_kernel": INT8_ROW_MAJOR_KERNEL_ID, "source": "e2e10_unified_selector"},
            "quantization": {
                "selected_candidate_id": f"fused_matmul_bias_relu:quant={SCHEME}:shape={m}x{n}x{k}:kernel={INT8_ROW_MAJOR_KERNEL_ID}",
                "scheme": SCHEME, "activation_dtype": "int8", "weight_dtype": "int8", "accumulation_dtype": "int32",
                "output_dtype": "fp32", "activation_granularity": "per_tensor", "weight_granularity": "per_tensor",
                "activation_scale": activation_scale, "weight_scale": artifact["weight_scale"],
                "activation_zero_point": 0, "weight_zero_point": 0, "required_kernel_capability": KERNEL_CAPABILITY,
                "requires_calibration": True, "calibration_available": True,
                "calibration_artifact_ref": str(context.calibration_artifact_path),
                "calibration_artifact_id": artifact["artifact_id"], "calibration_artifact_sha256": artifact["artifact_sha256"],
                "workload_id": artifact["workload_id"],
                "execution_stages": _int8_execution_stages(kernel_id=INT8_ROW_MAJOR_KERNEL_ID, activation_scale=activation_scale,
                                                            packed_ref=None, packed_sha=None, binary_sha256=None),
            },
            "thread_schedule": {"status": "selected", "thread_count": 1, "partition_axis": "none", "partition_strategy": "serial"},
            "memory_placement": _memory_placement(m, n, k),
        }

    if candidate_kind == MatMulBiasReLUImplementationKind.INT8_PACKED_B:
        if context.packed_artifact is None or context.packed_artifact_path is None:
            raise MatMulBiasReLUExecutionError(
                "packed-B candidate selected but no packed-weight artifact was prepared in the "
                "execution context -- the kernel never repacks at runtime, this executor will not either"
            )
        packed = context.packed_artifact
        build = (context.build_identity or {}).get(
            f"slice3c:portable_cpu:int8_static_symmetric:packed_b_transposed_nxk:{target_isa}", {}
        )
        return {
            "op_name": "op_e2e10", "op_type": "hir.fused_matmul_bias_relu",
            "kernel_selection": {"contract_version": "kernel_selection_contract_v1", "status": "selected",
                                  "selected_kernel": PACKED_INT8_KERNEL_ID, "source": "e2e10_unified_selector"},
            "quantization": {
                "selected_candidate_id": f"fused_matmul_bias_relu:quant={SCHEME}:shape={m}x{n}x{k}:kernel={PACKED_INT8_KERNEL_ID}:packed={packed['artifact_id']}",
                "scheme": SCHEME, "activation_dtype": "int8", "weight_dtype": "int8", "accumulation_dtype": "int32",
                "output_dtype": "fp32", "activation_granularity": "per_tensor", "weight_granularity": "per_tensor",
                "activation_scale": activation_scale, "weight_scale": artifact["weight_scale"],
                "activation_zero_point": 0, "weight_zero_point": 0, "required_kernel_capability": PACKED_INT8_KERNEL_CAPABILITY,
                "requires_calibration": True, "calibration_available": True,
                "calibration_artifact_ref": str(context.calibration_artifact_path),
                "calibration_artifact_id": artifact["artifact_id"], "calibration_artifact_sha256": artifact["artifact_sha256"],
                "workload_id": artifact["workload_id"], "kernel_requires_packed_weight": True,
                "packed_weight_artifact_ref": str(context.packed_artifact_path),
                "packed_weight_artifact_id": packed["artifact_id"], "packed_weight_sha256": packed["artifact_sha256"],
                "packed_layout": PACKED_B_TRANSPOSE_LAYOUT, "packing_scheme": PACKED_B_TRANSPOSE_SCHEME,
                "codegen_target_id": build.get("codegen_target_id", target_isa),
                "target_architecture": build.get("target_architecture", "aarch64"),
                "target_microarchitecture": build.get("target_microarchitecture", "cortex-a76" if target_isa == "cortex_a76_dotprod" else ""),
                "required_isa_features": build.get("required_isa_features", []),
                "compiler_flags": build.get("compiler_flags", []),
                "binary_sha256": build.get("binary_sha256", ""),
                "execution_stages": _int8_execution_stages(
                    kernel_id=PACKED_INT8_KERNEL_ID, activation_scale=activation_scale,
                    packed_ref=str(context.packed_artifact_path), packed_sha=packed["artifact_sha256"],
                    binary_sha256=build.get("binary_sha256"),
                ),
            },
            "thread_schedule": {"status": "selected", "thread_count": 1, "partition_axis": "none", "partition_strategy": "serial"},
            "memory_placement": _memory_placement(m, n, k),
        }

    raise MatMulBiasReLUExecutionError(f"unrecognized implementation_kind {candidate_kind}")


def execute_matmul_bias_relu(decision: ImplementationDecision, context: MatMulBiasReLUExecutionContext,
                              a: Tensor, b: Tensor, bias: Tensor, *, repeats: int = 5) -> PortableCpuKernelResult:
    """Executes the selected candidate via the real, unmodified
    dispatch_fused_matmul_bias_relu. Steady-state only: op_decision
    construction here does not create or write any artifact, only reads
    fields already prepared in `context`."""
    candidate = decision.selected_candidate
    kind = candidate.implementation_kind
    target_isa = candidate.parameters.get("target_isa")
    m, n, k = a.shape[0], b.shape[1], a.shape[1]
    op_decision = _build_op_decision(kind, target_isa, m, n, k, context)

    kwargs: dict[str, Any] = {"op_decision": op_decision, "backend": "cpu", "a": a, "b": b, "bias": bias, "repeats": repeats}
    executable = context.kernel_executables.get(candidate.candidate_id)
    if executable is None:
        raise MatMulBiasReLUExecutionError(f"no compiled kernel executable registered for candidate {candidate.candidate_id}")
    if kind == MatMulBiasReLUImplementationKind.FP32_TILED:
        kwargs["kernel_executable"] = executable
    elif kind == MatMulBiasReLUImplementationKind.INT8_SCALAR:
        kwargs["int8_kernel_executable"] = executable
    else:
        kwargs["packed_int8_kernel_executable"] = executable

    return dispatch_fused_matmul_bias_relu(**kwargs)


def _execute_via_decision(decision: ImplementationDecision, **kwargs) -> PortableCpuKernelResult:
    return execute_matmul_bias_relu(decision, kwargs["context"], kwargs["a"], kwargs["b"], kwargs["bias"],
                                     repeats=kwargs.get("repeats", 5))


try:
    from perf_model.runtime_dispatcher import register_executor as _register_executor
    _register_executor("matmul_bias_relu", _execute_via_decision)
except ImportError:
    pass  # torch not available in this environment (e.g. Raspberry Pi) -- execute_matmul_bias_relu remains usable directly
