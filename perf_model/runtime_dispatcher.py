"""E2E-9 Phase 10: unified runtime dispatcher. execute_with_decision() (and
the two family-specific helpers it wraps) take a selector output
(ImplementationDecision or ExecutionPolicy) and actually run the operation.

RMSNorm: loads/caches the REAL existing CUDA extension
(cuda_transformer_kernels/rmsnorm_kernel.cu + rmsnorm_extension.cpp, JIT-
compiled once per process via torch.utils.cpp_extension.load, exactly as
tests/test_rmsnorm_cuda_correctness.py and scripts/benchmark_rmsnorm_cuda.py
already do) and calls fused_rmsnorm_forward with the selected block_size.
Extension load happens on first use and is cached module-level -- it must
never occur inside a timed kernel loop.

LM-head: reuses (does not reimplement) the exact row-list+cat assembly
function from perf_model/tiny_m_dispatch.py (ASSEMBLY_VARIANTS["cat"]) for
the GEMV-decomposition branch, and torch.nn.functional.linear for the
default-GEMM branch -- the same two code paths E2E-8 already proved, now
selected via a cached ExecutionPolicy lookup instead of env-var reads. This
does NOT replace perf_model/tiny_m_dispatch.maybe_tiny_m_linear or
perf_model/tiny_m_oot_logits_processor.py's default (env-var-driven)
behavior; it is an additional, opt-in path (see execution_policy usage in
the OOT processor, gated by VLLM_TINY_M_UNIFIED_SELECTOR).
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from perf_model.execution_policy import ExecutionPolicy
from perf_model.implementation_candidate import ImplementationKind
from perf_model.implementation_decision import ImplementationDecision
from perf_model.tiny_m_dispatch import ASSEMBLY_VARIANTS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RMSNORM_KERNEL_DIR = _REPO_ROOT / "cuda_transformer_kernels"


def torch_rmsnorm_eager(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Standard weighted-RMSNorm eager reference, used only for the
    always-legal fallback candidate. Independently implemented (not
    imported from the existing test's reference function); numerically
    validated against the real CUDA kernel output in
    tests/test_runtime_dispatcher_e2e9.py, not merely assumed correct."""
    orig_dtype = x.dtype
    x32 = x.float()
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    normed = x32 * torch.rsqrt(variance + eps)
    return (normed * weight.float()).to(orig_dtype)


@functools.lru_cache(maxsize=1)
def _load_rmsnorm_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="fused_rmsnorm_cuda_ext_e2e9",
        sources=[str(_RMSNORM_KERNEL_DIR / "rmsnorm_extension.cpp"), str(_RMSNORM_KERNEL_DIR / "rmsnorm_kernel.cu")],
        extra_cuda_cflags=["-O3"],
    )


def execute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                     decision: ImplementationDecision) -> torch.Tensor:
    candidate = decision.selected_candidate
    if candidate.parameters.get("is_eager_fallback"):
        return torch_rmsnorm_eager(x, weight, eps)
    block_size = candidate.parameters["block_size"]
    ext = _load_rmsnorm_extension()
    return ext.fused_rmsnorm_forward(x, weight, eps, block_size)


def execute_rmsnorm_static_policy(x: torch.Tensor, weight: torch.Tensor, eps: float,
                                   policy: ExecutionPolicy) -> torch.Tensor:
    candidate = policy.resolve_candidate()
    if candidate.parameters.get("is_eager_fallback"):
        return torch_rmsnorm_eager(x, weight, eps)
    block_size = candidate.parameters["block_size"]
    ext = _load_rmsnorm_extension()
    return ext.fused_rmsnorm_forward(x, weight, eps, block_size)


def execute_lm_head_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                            policy: ExecutionPolicy) -> torch.Tensor:
    original_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1])
    m = x2d.shape[0]
    candidate = policy.resolve_candidate({"M": m})
    if candidate.implementation_kind == ImplementationKind.LINEAR_ROW_WISE_GEMV:
        assemble = ASSEMBLY_VARIANTS["cat"]
        out2d = assemble(x2d, weight, bias)
        n = weight.shape[0]
        return out2d.reshape(*original_shape[:-1], n)
    return F.linear(x, weight, bias)


# E2E-10: executor registry, replacing the E2E-9 single-family if-statement
# that execute_with_decision() dispatched directly. GENERIC_EXTENSION
# (identical behavior for rms_norm, registered below exactly as before) made
# so a third family's executor can be registered from its own family-
# specific module instead of adding a branch here. LM-head does not go
# through this registry at all (it uses execute_lm_head_linear directly with
# a cached ExecutionPolicy) so it is unaffected either way.
_EXECUTORS: dict[str, Any] = {}


def register_executor(operation_family: str, executor) -> None:
    _EXECUTORS[operation_family] = executor


def execute_with_decision(operation_family: str, decision: ImplementationDecision, **kwargs) -> torch.Tensor:
    executor = _EXECUTORS.get(operation_family)
    if executor is None:
        raise ValueError(f"execute_with_decision: no executor registered for operation_family {operation_family!r}")
    return executor(decision, **kwargs)


def _execute_rmsnorm_via_decision(decision: ImplementationDecision, *, x, weight, eps) -> torch.Tensor:
    return execute_rmsnorm(x, weight, eps, decision)


register_executor("rms_norm", _execute_rmsnorm_via_decision)
