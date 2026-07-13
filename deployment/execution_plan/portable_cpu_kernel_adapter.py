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
import statistics
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KERNEL_EXECUTABLE = (
    REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu"
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
# Kept for P1B backward compatibility (existing callers/tests referencing a
# single default candidate name).
EXPECTED_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn32_bk32"
EXPECTED_BACKEND = "cpu"
EXPECTED_OP_TYPE_SUFFIX = "fused_matmul_bias_relu"
EXPECTED_DTYPE = "f32"

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
    if selected_kernel not in KNOWN_KERNEL_IDS:
        raise PortableCpuKernelError(
            f"kernel_selection.selected_kernel is '{selected_kernel}', which is not one of "
            f"this adapter's known candidates ({sorted(KNOWN_KERNEL_IDS)}) -- refusing to "
            "silently substitute"
        )

    quantization = op_decision.get("quantization")
    if isinstance(quantization, dict):
        activation_dtype = quantization.get("activation_dtype")
        if activation_dtype is not None and activation_dtype != EXPECTED_DTYPE:
            raise PortableCpuKernelError(
                f"op activation_dtype is '{activation_dtype}', this kernel only "
                f"supports '{EXPECTED_DTYPE}' -- refusing to dispatch"
            )

    return kernel_selection


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

    if not kernel_executable.exists():
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
        _write_f32(a_path, a.data)
        _write_f32(b_path, b.data)
        _write_f32(bias_path, bias.data)

        completed = subprocess.run(
            [
                str(kernel_executable),
                "--m", str(m), "--n", str(n), "--k", str(k),
                "--a", str(a_path), "--b", str(b_path), "--bias", str(bias_path),
                "--out", str(out_path),
                "--kernel-id", selected_kernel_id,
                "--thread-count", str(thread_schedule["thread_count"]),
                "--partition-axis", str(thread_schedule["partition_axis"]),
                "--partition-strategy", str(thread_schedule["partition_strategy"]),
                "--repeats", str(repeats),
            ],
            capture_output=True, text=True, check=False,
        )
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
        dtype=EXPECTED_DTYPE,
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
