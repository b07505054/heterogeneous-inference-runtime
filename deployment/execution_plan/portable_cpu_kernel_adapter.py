"""Phase P1B: the one CPU ExecutionPlan-driven kernel adapter.

Dispatches exactly one compiler-selected, real, compiled native kernel:

    backend:    cpu
    kernel_id:  portable_fused_matmul_bias_relu_bm32_bn32_bk32
    op:         fused_matmul_bias_relu (hir.fused_matmul_bias_relu)

This module is the last real gate before execution. It reads a per-op
decision straight out of a real compiler-generated ExecutionPlan v2 JSON
(the same dict ExecutionPlanBuilder/ExecutionPlanExporter produce -- see
ml-graph-compiler-runtime), validates it against this exact contract, and --
only if every condition holds -- invokes the compiled native executable
(native/cpu_kernels/portable_fused_matmul_bias_relu) with the caller's real
input tensors. It never falls back to PyTorch, ONNX Runtime, NumPy, or a
mock/simulated result: any mismatch (backend, kernel id, dtype, rank, shape)
raises PortableCpuKernelError instead.

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

EXPECTED_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn32_bk32"
EXPECTED_BACKEND = "cpu"
EXPECTED_OP_TYPE_SUFFIX = "fused_matmul_bias_relu"
EXPECTED_DTYPE = "f32"

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
    if selected_kernel != EXPECTED_KERNEL_ID:
        raise PortableCpuKernelError(
            f"kernel_selection.selected_kernel is '{selected_kernel}', this adapter only "
            f"implements '{EXPECTED_KERNEL_ID}' -- refusing to silently substitute"
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
                "--kernel-id", EXPECTED_KERNEL_ID,
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

    return PortableCpuKernelResult(
        kernel_id=EXPECTED_KERNEL_ID,
        backend=EXPECTED_BACKEND,
        dtype=EXPECTED_DTYPE,
        m=m, n=n, k=k,
        block_m=int(kernel_stdout.get("block_m", 0)),
        block_n=int(kernel_stdout.get("block_n", 0)),
        block_k=int(kernel_stdout.get("block_k", 0)),
        thread_count=int(kernel_stdout.get("thread_count", 0)),
        output=Tensor(shape=(m, n), data=output_values),
        samples_ms=samples_ms,
        median_latency_ms=statistics.median(samples_ms),
        process_exit_status=completed.returncode,
        compiler_plan_id=compiler_plan_id,
        compiler_kernel_selection_source=kernel_selection.get("source"),
        raw_kernel_stdout=kernel_stdout,
    )
