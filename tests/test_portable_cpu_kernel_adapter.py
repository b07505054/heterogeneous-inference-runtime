"""Phase P1B focused tests for the portable CPU fused MatMul+Bias+ReLU adapter.

Covers: accepted valid plan, unknown-kernel rejection, wrong-backend
rejection, unsupported-dtype rejection, invalid-shape rejection,
compiler/runtime kernel-ID agreement, and output correctness against an
independent pure-Python reference implementation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from deployment.execution_plan.portable_cpu_kernel_adapter import (
    EXPECTED_KERNEL_ID,
    PortableCpuKernelError,
    Tensor,
    dispatch_fused_matmul_bias_relu,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_EXECUTABLE = REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu"
COMPILER_PROFILE_PATH = (
    REPO_ROOT.parent / "ml-graph-compiler-runtime" / "configs" / "target_profiles"
    / "raspberry_pi5_cortex_a76_cpu.json"
)


def _valid_op_decision() -> dict:
    return {
        "op_name": "op_2",
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {
            "contract_version": "kernel_selection_contract_v1",
            "status": "selected",
            "selected_kernel": EXPECTED_KERNEL_ID,
            "source": "handwritten_runtime",
        },
        "quantization": {"activation_dtype": "f32"},
    }


def _reference_matmul_bias_relu(a: Tensor, b: Tensor, bias: Tensor) -> list[float]:
    m, k = a.shape
    _, n = b.shape
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += a.data[i * k + kk] * b.data[kk * n + j]
            out[i * n + j] = max(0.0, s + bias.data[j])
    return out


def _random_tensors(m: int, n: int, k: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    rng = random.Random(seed)
    a = Tensor(shape=(m, k), data=[rng.uniform(-3, 3) for _ in range(m * k)])
    b = Tensor(shape=(k, n), data=[rng.uniform(-3, 3) for _ in range(k * n)])
    bias = Tensor(shape=(n,), data=[rng.uniform(-3, 3) for _ in range(n)])
    return a, b, bias


requires_kernel_binary = pytest.mark.skipif(
    not KERNEL_EXECUTABLE.exists(),
    reason=f"kernel executable not built at {KERNEL_EXECUTABLE}",
)


@requires_kernel_binary
def test_accepted_valid_plan_dispatches_and_returns_real_output():
    a, b, bias = _random_tensors(32, 32, 32, seed=1)
    result = dispatch_fused_matmul_bias_relu(
        op_decision=_valid_op_decision(), backend="cpu", a=a, b=b, bias=bias, repeats=2,
    )
    assert result.kernel_id == EXPECTED_KERNEL_ID
    assert result.backend == "cpu"
    assert result.m == 32 and result.n == 32 and result.k == 32
    assert result.process_exit_status == 0
    assert len(result.output.data) == 32 * 32
    assert len(result.samples_ms) == 2
    assert result.median_latency_ms >= 0.0
    # ReLU semantics: every output element must be non-negative.
    assert all(v >= 0.0 for v in result.output.data)


@requires_kernel_binary
def test_output_correctness_against_pure_python_reference():
    a, b, bias = _random_tensors(37, 41, 29, seed=2)  # deliberately not tile-aligned
    result = dispatch_fused_matmul_bias_relu(
        op_decision=_valid_op_decision(), backend="cpu", a=a, b=b, bias=bias, repeats=1,
    )
    expected = _reference_matmul_bias_relu(a, b, bias)
    max_abs_error = max(abs(got - exp) for got, exp in zip(result.output.data, expected))
    assert max_abs_error < 1e-3, f"max_abs_error={max_abs_error} exceeds f32 tolerance"


def test_unknown_kernel_id_rejected():
    op_decision = _valid_op_decision()
    op_decision["kernel_selection"]["selected_kernel"] = "some_other_kernel_id"
    a, b, bias = _random_tensors(4, 4, 4, seed=3)
    with pytest.raises(PortableCpuKernelError, match="not one of this adapter's known candidates"):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias)


def test_wrong_backend_rejected():
    a, b, bias = _random_tensors(4, 4, 4, seed=4)
    with pytest.raises(PortableCpuKernelError, match="only dispatches the cpu backend"):
        dispatch_fused_matmul_bias_relu(
            op_decision=_valid_op_decision(), backend="cuda", a=a, b=b, bias=bias,
        )


def test_unsupported_dtype_rejected():
    op_decision = _valid_op_decision()
    op_decision["quantization"]["activation_dtype"] = "fp16"
    a, b, bias = _random_tensors(4, 4, 4, seed=5)
    with pytest.raises(PortableCpuKernelError, match="only supports 'f32'"):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias)


def test_mismatched_target_profile_id_rejected():
    a, b, bias = _random_tensors(4, 4, 4, seed=8)
    with pytest.raises(PortableCpuKernelError, match="refusing cross-target dispatch"):
        dispatch_fused_matmul_bias_relu(
            op_decision=_valid_op_decision(), backend="cpu", a=a, b=b, bias=bias,
            target_profile_id="some-other-target",
            expected_target_profile_id="raspberry-pi5-cortex-a76-cpu",
        )


@requires_kernel_binary
def test_matching_target_profile_id_accepted():
    a, b, bias = _random_tensors(4, 4, 4, seed=9)
    result = dispatch_fused_matmul_bias_relu(
        op_decision=_valid_op_decision(), backend="cpu", a=a, b=b, bias=bias,
        target_profile_id="raspberry-pi5-cortex-a76-cpu",
        expected_target_profile_id="raspberry-pi5-cortex-a76-cpu",
        repeats=1,
    )
    assert result.kernel_id == EXPECTED_KERNEL_ID


def test_invalid_shape_rejected():
    a = Tensor(shape=(4, 4), data=[1.0] * 16)
    b_bad = Tensor(shape=(5, 4), data=[1.0] * 20)  # K mismatch: 4 != 5
    bias = Tensor(shape=(4,), data=[0.0] * 4)
    with pytest.raises(PortableCpuKernelError, match="do not agree"):
        dispatch_fused_matmul_bias_relu(
            op_decision=_valid_op_decision(), backend="cpu", a=a, b=b_bad, bias=bias,
        )


def test_deferred_kernel_selection_status_rejected():
    op_decision = _valid_op_decision()
    op_decision["kernel_selection"]["status"] = "deferred_no_kernel_library_declared"
    a, b, bias = _random_tensors(4, 4, 4, seed=6)
    with pytest.raises(PortableCpuKernelError, match="not 'selected'"):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias)


def test_wrong_op_type_rejected():
    op_decision = _valid_op_decision()
    op_decision["op_type"] = "linalg.conv_2d_nchw_fchw"
    a, b, bias = _random_tensors(4, 4, 4, seed=7)
    with pytest.raises(PortableCpuKernelError, match="not a fused_matmul_bias_relu op"):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias)


@pytest.mark.skipif(
    not COMPILER_PROFILE_PATH.exists(),
    reason="sibling ml-graph-compiler-runtime checkout with the Pi profile not found",
)
def test_compiler_runtime_kernel_id_agreement():
    """The compiler's declared kernelId in the Raspberry Pi target profile must
    match this adapter's EXPECTED_KERNEL_ID exactly -- proven against the real
    profile JSON, not a hardcoded string duplicated in a test."""
    profile = json.loads(COMPILER_PROFILE_PATH.read_text())
    runtime_kernels = profile.get("runtimeKernels", [])
    assert runtime_kernels, "profile declares no runtimeKernels"
    kernel_ids = [k["kernelId"] for k in runtime_kernels]
    assert EXPECTED_KERNEL_ID in kernel_ids, (
        f"adapter expects '{EXPECTED_KERNEL_ID}', profile declares {kernel_ids}"
    )
    declared = next(k for k in runtime_kernels if k["kernelId"] == EXPECTED_KERNEL_ID)
    assert declared["opName"] == "fused_matmul_bias_relu"
    assert declared["backend"] == "cpu"
    assert declared["supportedDtypes"] == ["f32"]
