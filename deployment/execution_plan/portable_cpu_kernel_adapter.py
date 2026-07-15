"""Phase P1B/P1C/P1D: the one CPU ExecutionPlan-driven kernel adapter.

Dispatches a compiler-selected, real, compiled native kernel from a small,
frozen candidate family (Phase P1C), plus (Phase P1D) a SEPARATE
thread-decomposition schedule decision for that same kernel:

    backend:  cpu
    op:       fused_matmul_bias_relu (hir.fused_matmul_bias_relu)
    dtype:    f32

    kernel_id in KNOWN_KERNEL_IDS (see below) -- identifies WHICH tile
    candidate runs (block_m, block_n, block_k).

    thread_count/partition_axis/partition_strategy (Phase P1D,
    thread_schedule_contract_v1) -- a SEPARATE decision identifying HOW
    MANY threads and what output partitioning that already-selected kernel
    uses. Absent op_decision["thread_schedule"] (every P1B/P1C plan) is the
    documented backward-compatible default: thread_count=1,
    partition_axis="none", partition_strategy="serial" -- exactly
    reproducing pre-P1D behavior, never silently upgraded to parallel
    execution.

This module is the last real gate before execution. It reads a per-op
decision straight out of a real compiler-generated ExecutionPlan v2 JSON
(the same dict ExecutionPlanBuilder/ExecutionPlanExporter produce -- see
ml-graph-compiler-runtime), validates it against this exact contract, and --
only if every condition holds -- invokes the compiled native executable
(native/cpu_kernels/portable_fused_matmul_bias_relu) with the caller's real
input tensors, the compiler's exact selected candidate ID, and the exact
requested thread schedule. It never falls back to PyTorch, ONNX Runtime,
NumPy, or a mock/simulated result: any mismatch (backend, kernel id, dtype,
rank, shape, thread schedule) raises PortableCpuKernelError instead. The
native kernel's own self-reported thread_count/partition_axis/
partition_strategy is cross-checked against what was requested, exactly
like the existing kernel_id self-report check.

Truth boundary: real subprocess execution of a real compiled kernel against
real caller-supplied tensors. Latency samples are real wall-clock
measurements from that one process on whatever host this runs on -- not a
performance claim, and not compared against any other backend here.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deployment.execution_plan.int8_quantization import (
    INT8_KERNEL_ID,
    KERNEL_CAPABILITY,
    PACKED_B_TRANSPOSE_LAYOUT,
    PACKED_B_TRANSPOSE_SCHEME,
    PACKED_INT8_KERNEL_CAPABILITY,
    PACKED_INT8_KERNEL_ID,
    SCHEME as INT8_SCHEME,
    load_and_validate_calibration_artifact,
    load_and_validate_packed_weight_artifact,
    sha256_bytes,
    quantize_symmetric,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KERNEL_EXECUTABLE = (
    REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu"
)
DEFAULT_INT8_KERNEL_EXECUTABLE = (
    REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu_int8"
)
DEFAULT_PACKED_INT8_KERNEL_EXECUTABLE = (
    REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu_int8_packed_b"
)

# Phase P1C frozen candidate table -- must match native/cpu_kernels/
# portable_fused_matmul_bias_relu.cpp's kCandidates exactly (cross-checked by
# tests/test_p1c_multi_candidate_contract.py). bm32_bn32_bk32 is the original
# P1B kernel, unchanged, kept for backward compatibility.
KNOWN_KERNEL_IDS = frozenset({
    "portable_fused_matmul_bias_relu_bm32_bn32_bk32",
    "portable_fused_matmul_bias_relu_bm48_bn48_bk48",
    "portable_fused_matmul_bias_relu_bm64_bn64_bk64",
    "portable_fused_matmul_bias_relu_bm128_bn128_bk32",
    "portable_fused_matmul_bias_relu_bm128_bn32_bk32",
    "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
    "portable_fused_matmul_bias_relu_bm32_bn32_bk128",
    "portable_fused_matmul_bias_relu_bm64_bn64_bk128",
})
KNOWN_INT8_KERNEL_IDS = frozenset({INT8_KERNEL_ID, PACKED_INT8_KERNEL_ID})
ALL_KNOWN_KERNEL_IDS = KNOWN_KERNEL_IDS | KNOWN_INT8_KERNEL_IDS
# Kept for P1B backward compatibility (existing callers/tests referencing a
# single default candidate name).
EXPECTED_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn32_bk32"
EXPECTED_BACKEND = "cpu"
EXPECTED_OP_TYPE_SUFFIX = "fused_matmul_bias_relu"
EXPECTED_DTYPE = "f32"
EXPECTED_DTYPE_ALIASES = frozenset({"f32", "fp32"})
CPU_VISIBLE_MEMORY_SPACES = frozenset({"cpu_visible_host_memory", "host_memory"})

# Phase P1D thread-decomposition contract (thread_schedule_contract_v1) --
# must match native/cpu_kernels/portable_fused_matmul_bias_relu.cpp's
# validation exactly (cross-checked by tests/test_p1d_thread_schedule_contract.py).
KNOWN_THREAD_COUNTS = frozenset({1, 2, 4})
KNOWN_PARTITION_AXES_MULTI_THREAD = frozenset({"m", "n"})
# The documented backward-compatible default for every P1B/P1C plan, which
# never carries a "thread_schedule" block at all.
DEFAULT_THREAD_SCHEDULE = {
    "thread_count": 1,
    "partition_axis": "none",
    "partition_strategy": "serial",
}

ADAPTER_TRUTH_BOUNDARY = (
    "portable_cpu_kernel_adapter_real_subprocess_execution_of_real_compiled_kernel_"
    "on_real_caller_supplied_tensors_not_a_performance_comparison"
)


class PortableCpuKernelError(ValueError):
    """Raised whenever the op decision, tensors, or dispatch do not satisfy
    the exact portable_fused_matmul_bias_relu_bm32_bn32_bk32 contract.
    Never caught here to fall back to a different execution path."""


@dataclass(frozen=True)
class Tensor:
    """A concrete, real (not simulated) row-major float32 tensor."""

    shape: tuple[int, ...]
    data: list[float]

    def __post_init__(self) -> None:
        expected = 1
        for dim in self.shape:
            expected *= dim
        if len(self.data) != expected:
            raise PortableCpuKernelError(
                f"tensor shape {self.shape} implies {expected} elements, "
                f"got {len(self.data)}"
            )


@dataclass(frozen=True)
class PortableCpuKernelResult:
    kernel_id: str
    backend: str
    dtype: str
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    thread_count: int
    partition_axis: str
    partition_strategy: str
    output: Tensor
    samples_ms: tuple[float, ...]
    median_latency_ms: float
    process_exit_status: int
    compiler_plan_id: str | None
    compiler_kernel_selection_source: str | None
    truth_boundary: str = ADAPTER_TRUTH_BOUNDARY
    raw_kernel_stdout: dict[str, Any] = field(default_factory=dict)


def _validate_op_decision(op_decision: dict[str, Any]) -> dict[str, Any]:
    """Validate an op decision dict from a real ExecutionPlan JSON against the
    exact contract. Returns the kernel_selection sub-dict on success; raises
    PortableCpuKernelError with a specific reason otherwise."""

    op_type = str(op_decision.get("op_type", ""))
    if not op_type.endswith(EXPECTED_OP_TYPE_SUFFIX):
        raise PortableCpuKernelError(
            f"op_type '{op_type}' is not a fused_matmul_bias_relu op -- refusing to dispatch"
        )

    kernel_selection = op_decision.get("kernel_selection")
    if not isinstance(kernel_selection, dict):
        raise PortableCpuKernelError(
            "op decision has no kernel_selection block -- compiler did not select a "
            "runtime kernel for this op; refusing to dispatch without one"
        )

    status = kernel_selection.get("status")
    if status != "selected":
        raise PortableCpuKernelError(
            f"kernel_selection.status is '{status}', not 'selected' -- compiler did not "
            "select a dispatchable kernel for this op; refusing to dispatch"
        )

    selected_kernel = kernel_selection.get("selected_kernel")
    if selected_kernel not in ALL_KNOWN_KERNEL_IDS:
        raise PortableCpuKernelError(
            f"kernel_selection.selected_kernel is '{selected_kernel}', which is not one of "
            f"this adapter's known candidates ({sorted(ALL_KNOWN_KERNEL_IDS)}) -- refusing to "
            "silently substitute"
        )

    quantization = op_decision.get("quantization")
    if isinstance(quantization, dict):
        _validate_quantization_contract(quantization, selected_kernel)

    return kernel_selection


def _validate_memory_placement_contract(
    op_decision: dict[str, Any], *, m: int, n: int, k: int
) -> None:
    """Validate the compiler-owned Slice 2 memory placement contract.

    The portable CPU adapter only executes buffers already placed in
    CPU-visible memory. It does not infer missing placement, synthesize host
    copies, run accelerator transfers, or reorder dependencies.
    """

    placement = op_decision.get("memory_placement")
    if not isinstance(placement, dict):
        raise PortableCpuKernelError(
            "op decision has no memory_placement block -- refusing to invent "
            "buffer placement or transfer ordering"
        )
    status = placement.get("status")
    if status != "selected":
        raise PortableCpuKernelError(
            f"memory_placement.status is '{status}', not 'selected' -- refusing "
            "to execute an unselected memory plan"
        )
    compute_unit = placement.get("compute_unit")
    if compute_unit != "cpu":
        raise PortableCpuKernelError(
            f"memory_placement.compute_unit is '{compute_unit}', this adapter "
            "only executes compiler plans placed on cpu"
        )
    selected_memory_space = placement.get("selected_memory_space")
    if selected_memory_space not in CPU_VISIBLE_MEMORY_SPACES:
        raise PortableCpuKernelError(
            f"memory_placement.selected_memory_space is '{selected_memory_space}', "
            "not CPU-visible host memory"
        )
    transfers = placement.get("transfer_operations")
    if not isinstance(transfers, list):
        raise PortableCpuKernelError(
            "memory_placement.transfer_operations is missing or not a list -- "
            "runtime will not reconstruct transfers"
        )
    if transfers:
        raise PortableCpuKernelError(
            "portable CPU memory plan contains transfer_operations; this adapter "
            "does not execute accelerator/device transfers or silently skip them"
        )
    buffer_placements = placement.get("buffer_placements")
    if not isinstance(buffer_placements, list):
        raise PortableCpuKernelError(
            "memory_placement.buffer_placements is missing or not a list -- "
            "runtime will not invent placements"
        )
    by_role: dict[str, dict[str, Any]] = {}
    for item in buffer_placements:
        if not isinstance(item, dict):
            raise PortableCpuKernelError("memory_placement.buffer_placements contains a non-object")
        role = str(item.get("role", ""))
        if role:
            by_role[role] = item
    expected_bytes = {
        "input": m * k * 4,
        "weight": k * n * 4,
        "output": m * n * 4,
        "scratch": 0,
    }
    for role, byte_count in expected_bytes.items():
        item = by_role.get(role)
        if item is None:
            raise PortableCpuKernelError(
                f"memory_placement missing required '{role}' buffer placement"
            )
        memory_space = item.get("memory_space")
        if memory_space != selected_memory_space:
            raise PortableCpuKernelError(
                f"memory_placement for role '{role}' uses memory_space "
                f"'{memory_space}', expected '{selected_memory_space}'"
            )
        if int(item.get("byte_count", -1)) != byte_count:
            raise PortableCpuKernelError(
                f"memory_placement for role '{role}' has byte_count "
                f"{item.get('byte_count')}, expected {byte_count}"
            )
        if int(item.get("alignment", 0) or 0) <= 0:
            raise PortableCpuKernelError(
                f"memory_placement for role '{role}' has invalid alignment "
                f"{item.get('alignment')}"
            )
    if placement.get("compute_dependency_ids") not in (None, []):
        raise PortableCpuKernelError(
            "portable CPU memory plan contains compute_dependency_ids; CPU-visible "
            "execution expects no accelerator transfer dependencies"
        )


def _validate_int8_execution_stages(
    quantization: dict[str, Any], selected_kernel: str
) -> dict[str, dict[str, Any]]:
    """Validate the compiler-provided explicit Slice 3D INT8 execution stages.

    The adapter may execute the quantize operation, packed-weight load, and
    kernel subprocess, but it must not infer that ordering from the kernel id.
    """

    stages = quantization.get("execution_stages")
    if not isinstance(stages, list):
        raise PortableCpuKernelError(
            "INT8 quantization.execution_stages is required -- runtime will "
            "not infer quantize/load/kernel ordering from dtype or kernel id"
        )
    expected_ids = ["quantize_activation"]
    if selected_kernel == PACKED_INT8_KERNEL_ID:
        expected_ids.append("load_packed_weight")
    expected_ids.extend(["execute_int8_kernel", "return_fp32_output"])
    ids = [stage.get("stage_id") if isinstance(stage, dict) else None for stage in stages]
    if ids != expected_ids:
        raise PortableCpuKernelError(
            f"INT8 execution stage order is {ids}, expected {expected_ids}; "
            "runtime will not reorder, insert, or remove quantization stages"
        )
    by_id: dict[str, dict[str, Any]] = {str(stage["stage_id"]): stage for stage in stages}

    q = by_id["quantize_activation"]
    if q.get("op") != "hir.quantize":
        raise PortableCpuKernelError("quantize_activation stage must be op=hir.quantize")
    if abs(float(q.get("scale", -1.0)) - float(quantization["activation_scale"])) > 1e-15:
        raise PortableCpuKernelError("quantize_activation stage scale does not match quantization.activation_scale")
    if int(q.get("zero_point", 999)) != int(quantization["activation_zero_point"]):
        raise PortableCpuKernelError("quantize_activation stage zero_point does not match activation_zero_point")
    if q.get("rounding_mode") != "round_nearest_even":
        raise PortableCpuKernelError("quantize_activation stage must declare rounding_mode=round_nearest_even")
    if int(q.get("clamp_min", 999)) != -127 or int(q.get("clamp_max", -999)) != 127:
        raise PortableCpuKernelError("quantize_activation stage must declare clamp range [-127, 127]")
    if q.get("source_dtype") != "fp32" or q.get("destination_dtype") != "int8":
        raise PortableCpuKernelError("quantize_activation stage must declare fp32 -> int8")

    if selected_kernel == PACKED_INT8_KERNEL_ID:
        load = by_id["load_packed_weight"]
        if load.get("op") != "hir.load_quantized_weight":
            raise PortableCpuKernelError("load_packed_weight stage must be op=hir.load_quantized_weight")
        if load.get("artifact_ref") != quantization.get("packed_weight_artifact_ref"):
            raise PortableCpuKernelError("load_packed_weight artifact_ref does not match quantization contract")
        if load.get("artifact_sha256") != quantization.get("packed_weight_sha256"):
            raise PortableCpuKernelError("load_packed_weight artifact_sha256 does not match quantization contract")
        if load.get("packed_layout") != PACKED_B_TRANSPOSE_LAYOUT:
            raise PortableCpuKernelError("load_packed_weight stage packed_layout mismatch")

    kernel = by_id["execute_int8_kernel"]
    if kernel.get("op") != "hir.portable_cpu_int8_fused_matmul_bias_relu":
        raise PortableCpuKernelError("execute_int8_kernel stage must name the lowered portable CPU INT8 op")
    if kernel.get("kernel_id") != selected_kernel:
        raise PortableCpuKernelError("execute_int8_kernel stage kernel_id does not match selected kernel")
    expected_deps = ["quantized_activation_ready"]
    if selected_kernel == PACKED_INT8_KERNEL_ID:
        expected_deps.append("packed_weight_ready")
    if kernel.get("dependency_ids") != expected_deps:
        raise PortableCpuKernelError(
            f"execute_int8_kernel dependency_ids are {kernel.get('dependency_ids')}, "
            f"expected {expected_deps}"
        )
    if selected_kernel == PACKED_INT8_KERNEL_ID:
        if kernel.get("fused_postprocess") != "dequantize_bias_relu":
            raise PortableCpuKernelError("packed INT8 kernel stage must declare fused_postprocess=dequantize_bias_relu")
        expected_binary = quantization.get("binary_sha256")
        if expected_binary and kernel.get("binary_sha256") != expected_binary:
            raise PortableCpuKernelError("execute_int8_kernel binary_sha256 does not match quantization contract")

    ret = by_id["return_fp32_output"]
    if ret.get("op") != "runtime.return" or ret.get("dependency_ids") != ["fp32_output_ready"]:
        raise PortableCpuKernelError("return_fp32_output stage must depend on fp32_output_ready")

    return by_id


def _validate_quantization_contract(quantization: dict[str, Any], selected_kernel: str) -> None:
    """Validate FP32 baseline or the one Slice 3A executable INT8 contract."""

    scheme = quantization.get("scheme") or quantization.get("strategy")
    selected_candidate_id = quantization.get("selected_candidate_id")
    if selected_candidate_id is not None and not selected_candidate_id:
        raise PortableCpuKernelError(
            "quantization.selected_candidate_id is empty -- refusing to invent a precision decision"
        )

    if scheme == INT8_SCHEME:
        if selected_kernel not in KNOWN_INT8_KERNEL_IDS:
            raise PortableCpuKernelError(
                f"INT8 contract selected kernel '{selected_kernel}', expected one of {sorted(KNOWN_INT8_KERNEL_IDS)}"
            )
        if selected_candidate_id and f"quant={INT8_SCHEME}" not in selected_candidate_id:
            raise PortableCpuKernelError("INT8 selected_candidate_id does not name int8_static_symmetric")
        expected = {
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "accumulation_dtype": "int32",
            "output_dtype": "fp32",
            "activation_granularity": "per_tensor",
            "weight_granularity": "per_tensor",
        }
        for key, value in expected.items():
            if quantization.get(key) != value:
                raise PortableCpuKernelError(f"INT8 quantization.{key} is '{quantization.get(key)}', expected '{value}'")
        expected_capability = PACKED_INT8_KERNEL_CAPABILITY if selected_kernel == PACKED_INT8_KERNEL_ID else KERNEL_CAPABILITY
        if quantization.get("required_kernel_capability") != expected_capability:
            raise PortableCpuKernelError(
                f"INT8 quantization.required_kernel_capability is '{quantization.get('required_kernel_capability')}', expected '{expected_capability}'"
            )
        for key in ("activation_scale", "weight_scale"):
            try:
                scale = float(quantization[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise PortableCpuKernelError(f"INT8 quantization.{key} is missing or invalid") from exc
            if scale <= 0.0:
                raise PortableCpuKernelError(f"INT8 quantization.{key} must be positive")
        if int(quantization.get("activation_zero_point", 999)) != 0:
            raise PortableCpuKernelError("INT8 activation_zero_point must be 0")
        if int(quantization.get("weight_zero_point", 999)) != 0:
            raise PortableCpuKernelError("INT8 weight_zero_point must be 0")
        for key in ("calibration_artifact_ref", "calibration_artifact_id", "calibration_artifact_sha256", "workload_id"):
            if not quantization.get(key):
                raise PortableCpuKernelError(f"INT8 quantization.{key} is required")
        if selected_kernel == PACKED_INT8_KERNEL_ID:
            packed_expected = {
                "packed_layout": PACKED_B_TRANSPOSE_LAYOUT,
                "packing_scheme": PACKED_B_TRANSPOSE_SCHEME,
            }
            if quantization.get("kernel_requires_packed_weight") is not True:
                raise PortableCpuKernelError("packed INT8 kernel requires quantization.kernel_requires_packed_weight=true")
            for key, value in packed_expected.items():
                if quantization.get(key) != value:
                    raise PortableCpuKernelError(f"packed INT8 quantization.{key} is '{quantization.get(key)}', expected '{value}'")
            for key in ("packed_weight_artifact_ref", "packed_weight_artifact_id", "packed_weight_sha256"):
                if not quantization.get(key):
                    raise PortableCpuKernelError(f"packed INT8 quantization.{key} is required")
            _validate_codegen_contract_static(quantization)
        elif quantization.get("kernel_requires_packed_weight") is True:
            raise PortableCpuKernelError("row-major INT8 kernel cannot consume a packed-weight contract")
        _validate_int8_execution_stages(quantization, selected_kernel)
        return

    if selected_candidate_id and "quant=fp32_baseline" not in selected_candidate_id:
        raise PortableCpuKernelError(
            "quantization.selected_candidate_id does not identify the fp32_baseline candidate -- "
            "refusing to dispatch a different precision through the FP32 portable kernel"
        )

    if scheme not in (None, "", "none", "fp32_baseline"):
        raise PortableCpuKernelError(
            f"quantization scheme '{scheme}' is not executable by this FP32 portable kernel -- "
            "runtime will not switch precision or fall back"
        )

    for key in ("activation_dtype", "weight_dtype", "accumulation_dtype", "output_dtype"):
        dtype = quantization.get(key)
        if dtype is not None and dtype not in EXPECTED_DTYPE_ALIASES:
            raise PortableCpuKernelError(
                f"quantization.{key} is '{dtype}', this kernel only supports fp32/f32 -- "
                "refusing to dispatch or substitute another precision"
            )

    required_kernel = quantization.get("required_kernel_capability")
    if required_kernel not in (None, "", "quant_kernel.none"):
        raise PortableCpuKernelError(
            f"quantization.required_kernel_capability is '{required_kernel}', but selected "
            f"kernel '{selected_kernel}' only declares quant_kernel.none"
        )

    if quantization.get("requires_calibration") is True:
        raise PortableCpuKernelError(
            "plan requires quantization calibration for the selected candidate, but this runtime "
            "adapter has no calibration materialization path"
        )


def _runtime_isa_features() -> set[str]:
    features: set[str] = set()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("Features"):
                _, value = line.split(":", 1)
                features.update(value.strip().split())
                break
    except OSError:
        pass
    return features


def _validate_codegen_contract_static(quantization: dict[str, Any]) -> None:
    codegen_target = quantization.get("codegen_target_id")
    if not codegen_target:
        return
    if codegen_target == "cortex_a76_dotprod":
        if quantization.get("target_architecture") != "aarch64":
            raise PortableCpuKernelError("codegen target requires target_architecture=aarch64")
        if quantization.get("target_microarchitecture") != "cortex-a76":
            raise PortableCpuKernelError("codegen target requires target_microarchitecture=cortex-a76")
        required = quantization.get("required_isa_features")
        if not isinstance(required, list) or "asimd" not in required or "asimddp" not in required:
            raise PortableCpuKernelError("codegen target requires required_isa_features including asimd and asimddp")
        flags = quantization.get("compiler_flags")
        if not isinstance(flags, list) or "-O3" not in flags or "-mcpu=cortex-a76" not in flags:
            raise PortableCpuKernelError("codegen target requires compiler_flags -O3 and -mcpu=cortex-a76")

def _validate_thread_schedule(op_decision: dict[str, Any]) -> dict[str, Any]:
    """Resolve and validate the thread-decomposition schedule for this op
    decision (Phase P1D, thread_schedule_contract_v1). Absent
    op_decision["thread_schedule"] -- every P1B/P1C plan -- resolves to the
    documented backward-compatible default (1 thread, serial), never
    silently upgraded to parallel execution. Present but malformed/invalid
    schedules raise PortableCpuKernelError; the Runtime never reinterprets
    or downgrades a requested thread count."""

    raw = op_decision.get("thread_schedule")
    if raw is None:
        return dict(DEFAULT_THREAD_SCHEDULE)
    if not isinstance(raw, dict):
        raise PortableCpuKernelError(
            "op decision has a malformed thread_schedule block (not an object) -- "
            "refusing to dispatch"
        )
    status = raw.get("status")
    if status != "selected":
        raise PortableCpuKernelError(
            f"thread_schedule.status is '{status}', not 'selected' -- compiler did not "
            "select a dispatchable thread schedule for this op; refusing to dispatch"
        )
    thread_count = raw.get("thread_count")
    partition_axis = raw.get("partition_axis")
    partition_strategy = raw.get("partition_strategy")
    if thread_count not in KNOWN_THREAD_COUNTS:
        raise PortableCpuKernelError(
            f"thread_schedule.thread_count is '{thread_count}', which is not one of "
            f"this adapter's known thread counts ({sorted(KNOWN_THREAD_COUNTS)}) -- "
            "refusing to silently clamp or round"
        )
    if thread_count == 1:
        if partition_axis != "none" or partition_strategy != "serial":
            raise PortableCpuKernelError(
                f"thread_count=1 requires partition_axis='none' and "
                f"partition_strategy='serial', got partition_axis='{partition_axis}' "
                f"partition_strategy='{partition_strategy}' -- refusing to silently "
                "reinterpret"
            )
    else:
        if partition_axis not in KNOWN_PARTITION_AXES_MULTI_THREAD:
            raise PortableCpuKernelError(
                f"thread_count>1 requires an explicit partition_axis of 'm' or 'n', "
                f"got '{partition_axis}' -- refusing to silently downgrade to serial"
            )
        if partition_strategy != "contiguous_chunks":
            raise PortableCpuKernelError(
                f"thread_count>1 requires partition_strategy='contiguous_chunks', "
                f"got '{partition_strategy}'"
            )
    return {
        "thread_count": thread_count,
        "partition_axis": partition_axis,
        "partition_strategy": partition_strategy,
    }


def _validate_backend(backend: str) -> None:
    if backend != EXPECTED_BACKEND:
        raise PortableCpuKernelError(
            f"requested backend '{backend}' is not '{EXPECTED_BACKEND}' -- this adapter "
            "only dispatches the cpu backend, refusing to dispatch"
        )


def _validate_tensors(a: Tensor, b: Tensor, bias: Tensor) -> tuple[int, int, int]:
    if len(a.shape) != 2:
        raise PortableCpuKernelError(f"input A must be rank 2, got shape {a.shape}")
    if len(b.shape) != 2:
        raise PortableCpuKernelError(f"input B must be rank 2, got shape {b.shape}")
    if len(bias.shape) != 1:
        raise PortableCpuKernelError(f"bias must be rank 1, got shape {bias.shape}")

    m, k_a = a.shape
    k_b, n_b = b.shape
    (n_bias,) = bias.shape

    if k_a != k_b:
        raise PortableCpuKernelError(
            f"shape mismatch: A is [{m},{k_a}], B is [{k_b},{n_b}] -- inner dimensions "
            f"{k_a} != {k_b} do not agree, refusing to dispatch"
        )
    if n_b != n_bias:
        raise PortableCpuKernelError(
            f"shape mismatch: B is [{k_b},{n_b}], bias is [{n_bias}] -- N dimensions "
            f"{n_b} != {n_bias} do not agree, refusing to dispatch"
        )
    if m <= 0 or n_b <= 0 or k_a <= 0:
        raise PortableCpuKernelError(f"non-positive shape dims: M={m} N={n_b} K={k_a}")

    return m, n_b, k_a


def _write_f32(path: Path, values: list[float]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))

def _write_i8(path: Path, values: list[int]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack(f"<{len(values)}b", *values))


def _read_f32(path: Path, expected_count: int) -> list[float]:
    data = path.read_bytes()
    if len(data) != expected_count * 4:
        raise PortableCpuKernelError(
            f"kernel output file has {len(data)} bytes, expected {expected_count * 4} "
            f"({expected_count} f32 elements) -- refusing to guess"
        )
    return list(struct.unpack(f"<{expected_count}f", data))


def dispatch_fused_matmul_bias_relu(
    *,
    op_decision: dict[str, Any],
    backend: str,
    a: Tensor,
    b: Tensor,
    bias: Tensor,
    compiler_plan_id: str | None = None,
    target_profile_id: str | None = None,
    expected_target_profile_id: str | None = None,
    repeats: int = 5,
    kernel_executable: Path = DEFAULT_KERNEL_EXECUTABLE,
    int8_kernel_executable: Path = DEFAULT_INT8_KERNEL_EXECUTABLE,
    packed_int8_kernel_executable: Path = DEFAULT_PACKED_INT8_KERNEL_EXECUTABLE,
) -> PortableCpuKernelResult:
    """Validate a real compiler op decision + real tensors against the exact
    portable_fused_matmul_bias_relu_bm32_bn32_bk32 contract, then dispatch the
    real compiled kernel via subprocess. Raises PortableCpuKernelError on any
    contract violation; never falls back to a different execution path.

    target_profile_id / expected_target_profile_id mirror the compiler-side
    precedent in apps/run_cpu_fused_schedule_discovery.cpp's
    run_use_plan_validation: when the caller supplies an expected profile id
    (e.g. the id this runtime instance was deployed for), a plan generated
    for a different target_profile_id is rejected rather than silently
    dispatched cross-target. Omitted expected_target_profile_id skips this
    check (same "empty means not provided" convention as the compiler side).
    """

    if expected_target_profile_id and target_profile_id != expected_target_profile_id:
        raise PortableCpuKernelError(
            f"plan target_profile_id '{target_profile_id}' does not match this "
            f"runtime's expected target profile '{expected_target_profile_id}' -- "
            "refusing cross-target dispatch"
        )

    _validate_backend(backend)
    kernel_selection = _validate_op_decision(op_decision)
    selected_kernel_id = kernel_selection["selected_kernel"]
    thread_schedule = _validate_thread_schedule(op_decision)
    m, n, k = _validate_tensors(a, b, bias)
    _validate_memory_placement_contract(op_decision, m=m, n=n, k=k)
    if selected_kernel_id in KNOWN_INT8_KERNEL_IDS and thread_schedule != DEFAULT_THREAD_SCHEDULE:
        raise PortableCpuKernelError("Slice 3A INT8 kernel supports only thread_count=1 partition_axis=none partition_strategy=serial")

    if selected_kernel_id not in KNOWN_INT8_KERNEL_IDS and not kernel_executable.exists():
        raise PortableCpuKernelError(
            f"kernel executable not found at {kernel_executable} -- it must be built "
            "before dispatch (g++ -O2 -std=c++17 -o portable_fused_matmul_bias_relu "
            "portable_fused_matmul_bias_relu.cpp); this adapter does not build it "
            "implicitly and will not fall back to a different execution method"
        )

    with tempfile.TemporaryDirectory(prefix="portable_cpu_kernel_") as tmp:
        tmp_path = Path(tmp)
        a_path, b_path, bias_path, out_path = (
            tmp_path / "a.bin", tmp_path / "b.bin", tmp_path / "bias.bin", tmp_path / "out.bin",
        )
        _write_f32(bias_path, bias.data)
        quantization = op_decision.get("quantization") if isinstance(op_decision.get("quantization"), dict) else {}
        if selected_kernel_id in KNOWN_INT8_KERNEL_IDS:
            stage_map = _validate_int8_execution_stages(quantization, selected_kernel_id)
            quantize_stage = stage_map["quantize_activation"]
            activation_scale = float(quantize_stage["scale"])
            weight_scale = float(quantization["weight_scale"])
            activation_zero_point = int(quantize_stage["zero_point"])
            weight_zero_point = int(quantization["weight_zero_point"])
            artifact_ref = Path(str(quantization["calibration_artifact_ref"]))
            if not artifact_ref.is_absolute():
                artifact_ref = REPO_ROOT / artifact_ref
            load_and_validate_calibration_artifact(
                artifact_ref,
                expected_artifact_id=str(quantization["calibration_artifact_id"]),
                expected_artifact_sha256=str(quantization["calibration_artifact_sha256"]),
                workload_id=str(quantization["workload_id"]),
                operator_kind="fused_matmul_bias_relu",
                m=m, n=n, k=k,
                activation_scale=activation_scale,
                weight_scale=weight_scale,
                activation_zero_point=activation_zero_point,
                weight_zero_point=weight_zero_point,
                activation_values=a.data,
                weight_values=b.data,
            )
            _write_i8(a_path, quantize_symmetric(a.data, activation_scale))
            if selected_kernel_id == PACKED_INT8_KERNEL_ID:
                packed_ref = Path(str(quantization["packed_weight_artifact_ref"]))
                if not packed_ref.is_absolute():
                    packed_ref = REPO_ROOT / packed_ref
                _, packed_data_path = load_and_validate_packed_weight_artifact(
                    packed_ref,
                    expected_artifact_id=str(quantization["packed_weight_artifact_id"]),
                    expected_artifact_sha256=str(quantization["packed_weight_sha256"]),
                    workload_id=str(quantization["workload_id"]),
                    operator_kind="fused_matmul_bias_relu",
                    m=m, n=n, k=k,
                    source_weight_values=b.data,
                    expected_layout=str(quantization["packed_layout"]),
                    expected_packing_scheme=str(quantization["packing_scheme"]),
                    expected_dtype=str(quantization["weight_dtype"]),
                )
                executable = packed_int8_kernel_executable
                expected_binary_sha256 = quantization.get("binary_sha256")
                if expected_binary_sha256 and sha256_bytes(executable.read_bytes()) != expected_binary_sha256:
                    raise PortableCpuKernelError("kernel binary_sha256 does not match compiler execution contract")
                required_isa = quantization.get("required_isa_features") or []
                if required_isa:
                    if platform.machine() != str(quantization.get("target_architecture")):
                        raise PortableCpuKernelError("runtime target architecture does not match compiler execution contract")
                    available = _runtime_isa_features()
                    missing = [f for f in required_isa if f not in available]
                    if missing:
                        raise PortableCpuKernelError(f"runtime missing required ISA features: {missing}")
                measurement_ref = quantization.get("measurement_artifact_ref")
                if measurement_ref:
                    measurement_path = Path(str(measurement_ref))
                    if not measurement_path.is_absolute():
                        measurement_path = REPO_ROOT / measurement_path
                    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
                    measured_candidate = measurement.get("candidate_id")
                    if measured_candidate is None and isinstance(measurement.get("slice3c_selection"), dict):
                        measured_candidate = measurement["slice3c_selection"].get("selected_candidate_id")
                    if measured_candidate != quantization.get("selected_complete_candidate_id"):
                        raise PortableCpuKernelError("measurement artifact candidate_id does not match selected candidate")
                    measured_binary = measurement.get("binary_sha256")
                    if measured_binary is None and isinstance(measurement.get("build_identity"), dict):
                        build = measurement["build_identity"].get(quantization.get("selected_complete_candidate_id"), {})
                        measured_binary = build.get("binary_sha256")
                    if measured_binary != expected_binary_sha256:
                        raise PortableCpuKernelError("measurement artifact binary_sha256 does not match selected binary")
                    measured_packed = measurement.get("packed_artifact_sha256")
                    if measured_packed is None:
                        measured_packed = measurement.get("packed_weight_sha256")
                    if measured_packed != quantization.get("packed_weight_sha256"):
                        raise PortableCpuKernelError("measurement artifact packed artifact does not match plan")
                argv = [
                    str(executable), "--m", str(m), "--n", str(n), "--k", str(k),
                    "--a-int8", str(a_path), "--b-packed-int8", str(packed_data_path), "--bias", str(bias_path),
                    "--out", str(out_path), "--kernel-id", selected_kernel_id,
                    "--activation-scale", str(activation_scale), "--weight-scale", str(weight_scale),
                    "--activation-zero-point", str(activation_zero_point), "--weight-zero-point", str(weight_zero_point),
                    "--packed-layout", str(quantization["packed_layout"]),
                    "--packing-scheme", str(quantization["packing_scheme"]),
                    "--repeats", str(repeats),
                ]
            else:
                _write_i8(b_path, quantize_symmetric(b.data, weight_scale))
                executable = int8_kernel_executable
                argv = [
                    str(executable), "--m", str(m), "--n", str(n), "--k", str(k),
                    "--a-int8", str(a_path), "--b-int8", str(b_path), "--bias", str(bias_path),
                    "--out", str(out_path), "--kernel-id", selected_kernel_id,
                    "--activation-scale", str(activation_scale), "--weight-scale", str(weight_scale),
                    "--activation-zero-point", str(activation_zero_point), "--weight-zero-point", str(weight_zero_point),
                    "--repeats", str(repeats),
                ]
        else:
            _write_f32(a_path, a.data)
            _write_f32(b_path, b.data)
            executable = kernel_executable
            argv = [
                str(executable), "--m", str(m), "--n", str(n), "--k", str(k),
                "--a", str(a_path), "--b", str(b_path), "--bias", str(bias_path),
                "--out", str(out_path), "--kernel-id", selected_kernel_id,
                "--thread-count", str(thread_schedule["thread_count"]),
                "--partition-axis", str(thread_schedule["partition_axis"]),
                "--partition-strategy", str(thread_schedule["partition_strategy"]),
                "--repeats", str(repeats),
            ]

        if not executable.exists():
            raise PortableCpuKernelError(f"kernel executable not found at {executable} -- it must be built before dispatch")
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise PortableCpuKernelError(
                f"kernel executable exited with status {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )

        try:
            kernel_stdout = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PortableCpuKernelError(
                f"kernel executable produced non-JSON stdout: {completed.stdout!r}"
            ) from exc

        output_values = _read_f32(out_path, m * n)

    samples_ms = tuple(float(s) for s in kernel_stdout.get("samples_ms", []))
    if not samples_ms:
        raise PortableCpuKernelError("kernel executable reported zero latency samples")

    reported_kernel_id = kernel_stdout.get("kernel_id")
    if reported_kernel_id != selected_kernel_id:
        raise PortableCpuKernelError(
            f"kernel executable reported kernel_id '{reported_kernel_id}', which does not "
            f"match the requested candidate '{selected_kernel_id}' -- refusing to trust "
            "a mismatched dispatch"
        )

    # Phase P1D: the native kernel self-reports its actual thread schedule;
    # cross-check it against what was requested, exactly like the kernel_id
    # self-report check above. A mismatch here would mean the kernel
    # silently ran a different schedule than the compiler's plan said --
    # never trusted, always a hard failure.
    reported_thread_count = kernel_stdout.get("thread_count")
    reported_partition_axis = kernel_stdout.get("partition_axis")
    reported_partition_strategy = kernel_stdout.get("partition_strategy")
    if (reported_thread_count != thread_schedule["thread_count"] or
            reported_partition_axis != thread_schedule["partition_axis"] or
            reported_partition_strategy != thread_schedule["partition_strategy"]):
        raise PortableCpuKernelError(
            f"kernel executable reported thread schedule "
            f"(thread_count={reported_thread_count}, partition_axis={reported_partition_axis!r}, "
            f"partition_strategy={reported_partition_strategy!r}), which does not match the "
            f"requested schedule {thread_schedule} -- refusing to trust a mismatched dispatch"
        )

    return PortableCpuKernelResult(
        kernel_id=selected_kernel_id,
        backend=EXPECTED_BACKEND,
        dtype=str(kernel_stdout.get("dtype", EXPECTED_DTYPE)),
        m=m, n=n, k=k,
        block_m=int(kernel_stdout.get("block_m", 0)),
        block_n=int(kernel_stdout.get("block_n", 0)),
        block_k=int(kernel_stdout.get("block_k", 0)),
        thread_count=int(reported_thread_count),
        partition_axis=str(reported_partition_axis),
        partition_strategy=str(reported_partition_strategy),
        output=Tensor(shape=(m, n), data=output_values),
        samples_ms=samples_ms,
        median_latency_ms=statistics.median(samples_ms),
        process_exit_status=completed.returncode,
        compiler_plan_id=compiler_plan_id,
        compiler_kernel_selection_source=kernel_selection.get("source"),
        raw_kernel_stdout=kernel_stdout,
    )
