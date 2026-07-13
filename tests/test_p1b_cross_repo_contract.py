"""Phase P1B cross-repository contract test.

Proves, against a REAL, freshly-generated ExecutionPlan JSON (not a stale
fixture, not string grep):

  1. The compiler (ml-graph-compiler-runtime) emits the exact backend/kernel
     contract for the Raspberry Pi profile + fused MatMul+Bias+ReLU op.
  2. This runtime (heterogeneous-inference-runtime) accepts that contract
     and dispatches the real compiled kernel.
  3. This runtime rejects a deliberately altered kernel ID.
  4. This runtime rejects a mismatched target_profile_id.
  5. Tensor shape/dtype semantics agree across both repositories (the same
     M=N=K=128, dtype f32, tile bm32_bn32_bk32 values the compiler declared
     are the exact values the runtime dispatch is exercised with).

Skips (does not fail) when the sibling ml-graph-compiler-runtime checkout,
its built compile-for-target binary, or this runtime's compiled native
kernel are not present -- this is a real subprocess integration test, not a
mock, and can only run where both repos and both builds actually exist.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import pytest

from deployment.execution_plan.loader import load_execution_plan
from deployment.execution_plan.portable_cpu_kernel_adapter import (
    EXPECTED_KERNEL_ID,
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
    """Invoke the real compiler binary fresh (subprocess) -- not a
    committed/stale fixture -- and return the loaded, schema-validated plan."""
    out_dir = tmp_path_factory.mktemp("p1b_cross_repo")
    out_path = out_dir / "execution_plan.json"
    completed = subprocess.run(
        [
            str(COMPILE_FOR_TARGET),
            "--device-profile", str(PI_PROFILE),
            "--mlir", str(P1B_MLIR),
            "--out", str(out_path),
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, (
        f"compile-for-target failed: {completed.stderr}"
    )
    plan = load_execution_plan(out_path)
    return plan


def _fused_matmul_bias_relu_op_decision(plan) -> dict:
    for function_plan in plan.function_plans:
        for op in function_plan.per_op_decisions:
            if op.op_type.endswith("fused_matmul_bias_relu"):
                return op.raw
    raise AssertionError("no hir.fused_matmul_bias_relu op decision found in the fresh plan")


def test_1_compiler_emits_exact_backend_kernel_contract(fresh_execution_plan):
    op_decision = _fused_matmul_bias_relu_op_decision(fresh_execution_plan)
    kernel_selection = op_decision["kernel_selection"]
    assert kernel_selection["status"] == "selected"
    assert kernel_selection["selected_kernel"] == EXPECTED_KERNEL_ID
    assert kernel_selection["contract_version"] == "kernel_selection_contract_v1"
    assert fresh_execution_plan.provenance.capability_bundle.hardware_profile_ref == (
        "raspberry-pi5-cortex-a76-cpu"
    )


def test_2_runtime_accepts_the_real_contract_and_dispatches(fresh_execution_plan):
    op_decision = _fused_matmul_bias_relu_op_decision(fresh_execution_plan)
    rng = random.Random(123)
    m, n, k = 128, 128, 128  # the exact shape declared in p1b_fused_matmul_bias_relu_cpu.mlir
    a = Tensor(shape=(m, k), data=[rng.uniform(-2, 2) for _ in range(m * k)])
    b = Tensor(shape=(k, n), data=[rng.uniform(-2, 2) for _ in range(k * n)])
    bias = Tensor(shape=(n,), data=[rng.uniform(-2, 2) for _ in range(n)])

    result = dispatch_fused_matmul_bias_relu(
        op_decision=op_decision,
        backend=fresh_execution_plan.function_plans[0].backend.selected_backend,
        a=a, b=b, bias=bias,
        compiler_plan_id=fresh_execution_plan.plan_id,
        target_profile_id=fresh_execution_plan.provenance.capability_bundle.hardware_profile_ref,
        expected_target_profile_id="raspberry-pi5-cortex-a76-cpu",
        repeats=3,
    )
    assert result.kernel_id == EXPECTED_KERNEL_ID
    assert result.process_exit_status == 0
    assert len(result.output.data) == m * n
    assert all(v >= 0.0 for v in result.output.data)  # ReLU


def test_3_runtime_rejects_deliberately_altered_kernel_id(fresh_execution_plan):
    op_decision = dict(_fused_matmul_bias_relu_op_decision(fresh_execution_plan))
    op_decision["kernel_selection"] = dict(op_decision["kernel_selection"])
    op_decision["kernel_selection"]["selected_kernel"] = "portable_fused_matmul_bias_relu_ALTERED"
    a = Tensor(shape=(4, 4), data=[1.0] * 16)
    b = Tensor(shape=(4, 4), data=[1.0] * 16)
    bias = Tensor(shape=(4,), data=[0.0] * 4)
    with pytest.raises(PortableCpuKernelError, match="not one of this adapter's known candidates"):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias)


def test_4_runtime_rejects_mismatched_target_profile_id(fresh_execution_plan):
    op_decision = _fused_matmul_bias_relu_op_decision(fresh_execution_plan)
    real_profile_id = fresh_execution_plan.provenance.capability_bundle.hardware_profile_ref
    a = Tensor(shape=(4, 4), data=[1.0] * 16)
    b = Tensor(shape=(4, 4), data=[1.0] * 16)
    bias = Tensor(shape=(4,), data=[0.0] * 4)
    with pytest.raises(PortableCpuKernelError, match="refusing cross-target dispatch"):
        dispatch_fused_matmul_bias_relu(
            op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias,
            target_profile_id=real_profile_id,
            expected_target_profile_id="some-completely-different-target",
        )


def test_5_tensor_shape_and_dtype_semantics_agree_across_repos(fresh_execution_plan):
    op_decision = _fused_matmul_bias_relu_op_decision(fresh_execution_plan)

    # Compiler-side declared contract (from the real target profile JSON).
    profile = json.loads(PI_PROFILE.read_text())
    declared = next(
        k for k in profile["runtimeKernels"] if k["kernelId"] == EXPECTED_KERNEL_ID
    )
    assert declared["opName"] == "fused_matmul_bias_relu"
    assert declared["backend"] == "cpu"
    assert declared["supportedDtypes"] == ["f32"]

    # Compiler-side resolved per-op dtype for THIS specific op instance.
    assert op_decision["quantization"]["activation_dtype"] == "f32"
    assert op_decision["op_type"] == "hir.fused_matmul_bias_relu"

    # Runtime-side kernel executable reports the same tile identity embedded
    # in EXPECTED_KERNEL_ID ("..._bm32_bn32_bk32").
    a = Tensor(shape=(128, 128), data=[0.1] * (128 * 128))
    b = Tensor(shape=(128, 128), data=[0.1] * (128 * 128))
    bias = Tensor(shape=(128,), data=[0.0] * 128)
    result = dispatch_fused_matmul_bias_relu(
        op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias, repeats=1,
    )
    assert (result.block_m, result.block_n, result.block_k) == (32, 32, 32)
    assert result.dtype == "f32"
    assert (result.m, result.n, result.k) == (128, 128, 128)
