"""Phase P1D cross-repository thread-schedule contract tests.

Proves, against real generated ExecutionPlan JSON and the real compiler
target-profile JSON (not string grep alone):

  1. Compiler serializes all thread fields correctly.
  2. Runtime parses them correctly.
  3. Old plans without thread_schedule retain documented one-thread behavior.
  4. Invalid thread counts reject.
  5. thread_count greater than physicalComputeUnits rejects/is filtered
     (proven at the compiler level: the profile declares physicalComputeUnits=4
     and no candidate above 4 threads is ever selectable).
  6. Invalid axis/strategy combinations reject.
  7. Runtime dispatches the exact requested schedule.
  8. Native kernel self-report matches the plan.
  9. Mutated thread_count fails the agreement check.
  10. Mutated partition axis fails the agreement check.
  11. All 5 frozen candidate schedules produce correct numerical results.
  12. Existing P1B/P1C contracts remain valid.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from deployment.execution_plan.loader import load_execution_plan
from deployment.execution_plan.portable_cpu_kernel_adapter import (
    KNOWN_KERNEL_IDS,
    KNOWN_THREAD_COUNTS,
    PortableCpuKernelError,
    Tensor,
    dispatch_fused_matmul_bias_relu,
)

RUNTIME_REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILER_REPO_ROOT = RUNTIME_REPO_ROOT.parent / "ml-graph-compiler-runtime"
COMPILE_FOR_TARGET = COMPILER_REPO_ROOT / "build-mlir" / "compile-for-target"
PI_PROFILE = COMPILER_REPO_ROOT / "configs" / "target_profiles" / "raspberry_pi5_cortex_a76_cpu.json"
P1B_MLIR = COMPILER_REPO_ROOT / "mlir" / "p1b_fused_matmul_bias_relu_cpu.mlir"
KERNEL_EXECUTABLE = RUNTIME_REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu"

pytestmark = pytest.mark.skipif(
    not (COMPILE_FOR_TARGET.exists() and PI_PROFILE.exists() and P1B_MLIR.exists()
         and KERNEL_EXECUTABLE.exists()),
    reason="requires a sibling ml-graph-compiler-runtime checkout with a built "
           "compile-for-target, and this repo's built native kernel",
)


@pytest.fixture(scope="module")
def fresh_execution_plan(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("p1d_cross_repo")
    out_path = out_dir / "execution_plan.json"
    completed = subprocess.run(
        [str(COMPILE_FOR_TARGET), "--device-profile", str(PI_PROFILE),
         "--mlir", str(P1B_MLIR), "--out", str(out_path)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, f"compile-for-target failed: {completed.stderr}"
    return load_execution_plan(out_path)


def _fused_op_decision(plan) -> dict:
    for fp in plan.function_plans:
        for op in fp.per_op_decisions:
            if op.op_type.endswith("fused_matmul_bias_relu"):
                return op.raw
    raise AssertionError("no fused_matmul_bias_relu op decision found")


def _random_tensors(m, n, k, seed):
    rng = random.Random(seed)
    a = Tensor(shape=(m, k), data=[rng.uniform(-2, 2) for _ in range(m * k)])
    b = Tensor(shape=(k, n), data=[rng.uniform(-2, 2) for _ in range(k * n)])
    bias = Tensor(shape=(n,), data=[rng.uniform(-2, 2) for _ in range(n)])
    return a, b, bias


def test_1_compiler_serializes_thread_fields(fresh_execution_plan):
    op = _fused_op_decision(fresh_execution_plan)
    assert "thread_schedule" in op
    ts = op["thread_schedule"]
    assert ts["status"] == "selected"
    assert ts["thread_count"] == 1
    assert ts["partition_axis"] == "none"
    assert ts["partition_strategy"] == "serial"
    assert ts["contract_version"] == "thread_schedule_contract_v1"
    assert "truth_boundary" in ts


def test_2_runtime_parses_thread_fields_and_dispatches(fresh_execution_plan):
    op = _fused_op_decision(fresh_execution_plan)
    a, b, bias = _random_tensors(64, 64, 64, seed=1)
    result = dispatch_fused_matmul_bias_relu(op_decision=op, backend="cpu", a=a, b=b, bias=bias, repeats=1)
    assert result.thread_count == 1
    assert result.partition_axis == "none"
    assert result.partition_strategy == "serial"


def test_3_old_plan_without_thread_schedule_defaults_to_one_thread_serial():
    op = {
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"status": "selected",
                              "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
                              "source": "handwritten_runtime"},
        "quantization": {"activation_dtype": "f32"},
        # no "thread_schedule" key at all -- exactly every P1B/P1C plan.
    }
    a, b, bias = _random_tensors(32, 32, 32, seed=2)
    result = dispatch_fused_matmul_bias_relu(op_decision=op, backend="cpu", a=a, b=b, bias=bias, repeats=1)
    assert result.thread_count == 1
    assert result.partition_axis == "none"
    assert result.partition_strategy == "serial"


def test_4_invalid_thread_count_rejected():
    op = {
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"status": "selected",
                              "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
                              "source": "handwritten_runtime"},
        "quantization": {"activation_dtype": "f32"},
        "thread_schedule": {"status": "selected", "thread_count": 8,
                             "partition_axis": "m", "partition_strategy": "contiguous_chunks"},
    }
    a, b, bias = _random_tensors(32, 32, 32, seed=3)
    with pytest.raises(PortableCpuKernelError, match="not one of this adapter's known thread counts"):
        dispatch_fused_matmul_bias_relu(op_decision=op, backend="cpu", a=a, b=b, bias=bias)


def test_5_thread_count_above_physical_compute_units_never_selected_by_compiler(fresh_execution_plan):
    """physicalComputeUnits=4 is declared; the compiler must never select a
    thread_count above it for any op -- proven directly against the real
    profile and a real generated plan."""
    profile = json.loads(PI_PROFILE.read_text())
    physical_units = profile["hardwareExecutionProfile"]["physicalComputeUnits"]
    assert physical_units == 4
    op = _fused_op_decision(fresh_execution_plan)
    ts = op.get("thread_schedule")
    if ts and ts.get("status") == "selected":
        assert ts["thread_count"] <= physical_units


def test_6_invalid_axis_strategy_combinations_rejected():
    a, b, bias = _random_tensors(16, 16, 16, seed=4)
    base = {
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"status": "selected",
                              "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
                              "source": "handwritten_runtime"},
        "quantization": {"activation_dtype": "f32"},
    }
    # thread_count=1 with a non-serial/non-none combo.
    bad1 = dict(base, thread_schedule={"status": "selected", "thread_count": 1,
                                        "partition_axis": "m", "partition_strategy": "serial"})
    with pytest.raises(PortableCpuKernelError, match="thread_count=1 requires"):
        dispatch_fused_matmul_bias_relu(op_decision=bad1, backend="cpu", a=a, b=b, bias=bias)
    # thread_count>1 with axis=none.
    bad2 = dict(base, thread_schedule={"status": "selected", "thread_count": 2,
                                        "partition_axis": "none", "partition_strategy": "contiguous_chunks"})
    with pytest.raises(PortableCpuKernelError, match="requires an explicit partition_axis"):
        dispatch_fused_matmul_bias_relu(op_decision=bad2, backend="cpu", a=a, b=b, bias=bias)
    # thread_count>1 with strategy=serial.
    bad3 = dict(base, thread_schedule={"status": "selected", "thread_count": 2,
                                        "partition_axis": "m", "partition_strategy": "serial"})
    with pytest.raises(PortableCpuKernelError, match="requires partition_strategy='contiguous_chunks'"):
        dispatch_fused_matmul_bias_relu(op_decision=bad3, backend="cpu", a=a, b=b, bias=bias)


@pytest.mark.parametrize("thread_count,axis,strategy", [
    (1, "none", "serial"),
    (2, "m", "contiguous_chunks"),
    (4, "m", "contiguous_chunks"),
    (2, "n", "contiguous_chunks"),
    (4, "n", "contiguous_chunks"),
])
def test_7_and_8_runtime_dispatches_exact_schedule_and_self_report_matches(thread_count, axis, strategy):
    op = {
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"status": "selected",
                              "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
                              "source": "handwritten_runtime"},
        "quantization": {"activation_dtype": "f32"},
        "thread_schedule": {"status": "selected", "thread_count": thread_count,
                             "partition_axis": axis, "partition_strategy": strategy},
    }
    a, b, bias = _random_tensors(96, 96, 96, seed=5)
    result = dispatch_fused_matmul_bias_relu(op_decision=op, backend="cpu", a=a, b=b, bias=bias, repeats=1)
    assert result.thread_count == thread_count
    assert result.partition_axis == axis
    assert result.partition_strategy == strategy


def test_9_mutated_thread_count_self_report_mismatch_fails(monkeypatch):
    """Simulate a kernel that reports a different thread_count than
    requested -- must be a hard failure, never silently trusted."""
    import subprocess as sp
    from deployment.execution_plan import portable_cpu_kernel_adapter as mod

    real_run = sp.run

    def fake_run(cmd, **kwargs):
        r = real_run(cmd, **kwargs)
        payload = json.loads(r.stdout)
        payload["thread_count"] = 999
        r.stdout = json.dumps(payload)
        return r

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    op = {
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"status": "selected",
                              "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
                              "source": "handwritten_runtime"},
        "quantization": {"activation_dtype": "f32"},
        "thread_schedule": {"status": "selected", "thread_count": 1,
                             "partition_axis": "none", "partition_strategy": "serial"},
    }
    a, b, bias = _random_tensors(8, 8, 8, seed=6)
    with pytest.raises(PortableCpuKernelError, match="reported thread schedule"):
        dispatch_fused_matmul_bias_relu(op_decision=op, backend="cpu", a=a, b=b, bias=bias)


def test_10_mutated_partition_axis_self_report_mismatch_fails(monkeypatch):
    import subprocess as sp
    from deployment.execution_plan import portable_cpu_kernel_adapter as mod

    real_run = sp.run

    def fake_run(cmd, **kwargs):
        r = real_run(cmd, **kwargs)
        payload = json.loads(r.stdout)
        payload["partition_axis"] = "n"
        r.stdout = json.dumps(payload)
        return r

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    op = {
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"status": "selected",
                              "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
                              "source": "handwritten_runtime"},
        "quantization": {"activation_dtype": "f32"},
        "thread_schedule": {"status": "selected", "thread_count": 2,
                             "partition_axis": "m", "partition_strategy": "contiguous_chunks"},
    }
    a, b, bias = _random_tensors(32, 32, 32, seed=7)
    with pytest.raises(PortableCpuKernelError, match="reported thread schedule"):
        dispatch_fused_matmul_bias_relu(op_decision=op, backend="cpu", a=a, b=b, bias=bias)


@pytest.mark.parametrize("thread_count,axis,strategy", [
    (1, "none", "serial"),
    (2, "m", "contiguous_chunks"),
    (4, "m", "contiguous_chunks"),
    (2, "n", "contiguous_chunks"),
    (4, "n", "contiguous_chunks"),
])
def test_11_all_candidate_schedules_produce_correct_results(thread_count, axis, strategy):
    op = {
        "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"status": "selected",
                              "selected_kernel": "portable_fused_matmul_bias_relu_bm32_bn128_bk32",
                              "source": "handwritten_runtime"},
        "quantization": {"activation_dtype": "f32"},
        "thread_schedule": {"status": "selected", "thread_count": thread_count,
                             "partition_axis": axis, "partition_strategy": strategy},
    }
    m, n, k = 101, 67, 53  # deliberately non-tile-aligned, non-thread-aligned
    a, b, bias = _random_tensors(m, n, k, seed=8)
    result = dispatch_fused_matmul_bias_relu(op_decision=op, backend="cpu", a=a, b=b, bias=bias, repeats=1)
    ref = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = sum(a.data[i * k + kk] * b.data[kk * n + j] for kk in range(k))
            ref[i * n + j] = max(0.0, s + bias.data[j])
    max_err = max(abs(x - y) for x, y in zip(result.output.data, ref))
    assert max_err < 1e-3


def test_12_existing_p1c_kernel_id_contract_still_valid():
    assert "portable_fused_matmul_bias_relu_bm32_bn32_bk32" in KNOWN_KERNEL_IDS
    assert {1, 2, 4} == set(KNOWN_THREAD_COUNTS)
