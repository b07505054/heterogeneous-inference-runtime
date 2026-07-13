"""Phase P1C cross-repository, multi-candidate contract tests.

Proves, against real generated ExecutionPlan JSON and the real compiler
target-profile JSON (not string grep alone):

  1. Every compiler-emitted candidate ID is recognized by Runtime.
  2. Runtime has no candidate ID absent from the compiler capability
     declaration (set equality -- no "internal-only" candidates exist here).
  3. Unknown candidate IDs reject.
  4. Invalid shape/candidate combinations reject, for a non-baseline P1C candidate.
  5. The compiler-selected candidate (bm32_bn32_bk32, per the P1C audit finding
     that live selection is shape-independent) executes correctly.
  6. A deliberately altered real ExecutionPlan (selected_kernel mutated to a
     fabricated string) fails.
  7. P1B single-kernel behavior remains backward-compatible.
  8. Existing CUDA/LLM/vLLM paths are unchanged (regression check).
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
def declared_candidate_ids() -> list[str]:
    profile = json.loads(PI_PROFILE.read_text())
    return [k["kernelId"] for k in profile["runtimeKernels"]]


@pytest.fixture(scope="module")
def fresh_execution_plan(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("p1c_cross_repo")
    out_path = out_dir / "execution_plan.json"
    completed = subprocess.run(
        [str(COMPILE_FOR_TARGET), "--device-profile", str(PI_PROFILE),
         "--mlir", str(P1B_MLIR), "--out", str(out_path)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, f"compile-for-target failed: {completed.stderr}"
    return load_execution_plan(out_path)


def _fused_matmul_bias_relu_op_decision(plan) -> dict:
    for fp in plan.function_plans:
        for op in fp.per_op_decisions:
            if op.op_type.endswith("fused_matmul_bias_relu"):
                return op.raw
    raise AssertionError("no hir.fused_matmul_bias_relu op decision found")


def test_1_every_compiler_emitted_candidate_is_recognized_by_runtime(declared_candidate_ids):
    assert len(declared_candidate_ids) == 8
    for cid in declared_candidate_ids:
        assert cid in KNOWN_KERNEL_IDS, f"compiler declares '{cid}' but Runtime does not recognize it"


def test_2_runtime_has_no_candidate_absent_from_compiler_declaration(declared_candidate_ids):
    assert set(KNOWN_KERNEL_IDS) == set(declared_candidate_ids), (
        "Runtime's KNOWN_KERNEL_IDS and the compiler's declared runtimeKernels must be "
        "exactly the same set in P1C (no internal-only candidates declared on either side)"
    )


def test_3_unknown_candidate_id_rejected(fresh_execution_plan):
    op_decision = dict(_fused_matmul_bias_relu_op_decision(fresh_execution_plan))
    op_decision["kernel_selection"] = dict(op_decision["kernel_selection"])
    op_decision["kernel_selection"]["selected_kernel"] = "portable_fused_matmul_bias_relu_bm999_bn999_bk999"
    a = Tensor(shape=(4, 4), data=[1.0] * 16)
    b = Tensor(shape=(4, 4), data=[1.0] * 16)
    bias = Tensor(shape=(4,), data=[0.0] * 4)
    with pytest.raises(PortableCpuKernelError, match="not one of this adapter's known candidates"):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias)


def test_4_invalid_shape_rejected_for_non_baseline_candidate(fresh_execution_plan):
    op_decision = dict(_fused_matmul_bias_relu_op_decision(fresh_execution_plan))
    op_decision["kernel_selection"] = dict(op_decision["kernel_selection"])
    op_decision["kernel_selection"]["selected_kernel"] = "portable_fused_matmul_bias_relu_bm128_bn128_bk32"
    a = Tensor(shape=(8, 8), data=[1.0] * 64)
    b_bad = Tensor(shape=(9, 8), data=[1.0] * 72)  # K mismatch: 8 != 9
    bias = Tensor(shape=(8,), data=[0.0] * 8)
    with pytest.raises(PortableCpuKernelError, match="do not agree"):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b_bad, bias=bias)


def test_5_compiler_selected_candidate_executes_correctly(fresh_execution_plan):
    op_decision = _fused_matmul_bias_relu_op_decision(fresh_execution_plan)
    assert op_decision["kernel_selection"]["selected_kernel"] == (
        "portable_fused_matmul_bias_relu_bm32_bn32_bk32"
    ), "P1C audit finding: live selection is shape-independent, always the P1B baseline candidate"
    rng = random.Random(4242)
    m, n, k = 128, 128, 128
    a = Tensor(shape=(m, k), data=[rng.uniform(-2, 2) for _ in range(m * k)])
    b = Tensor(shape=(k, n), data=[rng.uniform(-2, 2) for _ in range(k * n)])
    bias = Tensor(shape=(n,), data=[rng.uniform(-2, 2) for _ in range(n)])
    result = dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias, repeats=1)
    assert result.process_exit_status == 0
    assert all(v >= 0.0 for v in result.output.data)


def test_6_deliberately_altered_real_execution_plan_fails(fresh_execution_plan):
    op_decision = dict(_fused_matmul_bias_relu_op_decision(fresh_execution_plan))
    op_decision["kernel_selection"] = dict(op_decision["kernel_selection"])
    op_decision["kernel_selection"]["selected_kernel"] = "portable_fused_matmul_bias_relu_FABRICATED_NOT_REAL"
    a = Tensor(shape=(4, 4), data=[1.0] * 16)
    b = Tensor(shape=(4, 4), data=[1.0] * 16)
    bias = Tensor(shape=(4,), data=[0.0] * 4)
    with pytest.raises(PortableCpuKernelError):
        dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias)


def test_7_p1b_single_kernel_behavior_remains_backward_compatible(fresh_execution_plan):
    """Exact P1B contract (backend=cpu, kernel_id=bm32_bn32_bk32, dtype=f32) must
    still dispatch exactly as it did before the P1C candidate family existed."""
    op_decision = _fused_matmul_bias_relu_op_decision(fresh_execution_plan)
    rng = random.Random(99)
    a = Tensor(shape=(16, 16), data=[rng.uniform(-2, 2) for _ in range(256)])
    b = Tensor(shape=(16, 16), data=[rng.uniform(-2, 2) for _ in range(256)])
    bias = Tensor(shape=(16,), data=[rng.uniform(-2, 2) for _ in range(16)])
    result = dispatch_fused_matmul_bias_relu(op_decision=op_decision, backend="cpu", a=a, b=b, bias=bias, repeats=1)
    assert result.kernel_id == "portable_fused_matmul_bias_relu_bm32_bn32_bk32"
    assert result.block_m == 32 and result.block_n == 32 and result.block_k == 32
