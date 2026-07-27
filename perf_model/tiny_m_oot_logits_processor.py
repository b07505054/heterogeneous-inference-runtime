"""E2E-8 Phase 3 (extended, E2E-9 Phase 10): LM-head-only integration via
vLLM's first-class PluggableLayer.register_oot extension point -- NO vLLM
source file is modified. Importing this module before the engine constructs
its model registers TinyMLogitsProcessor as the out-of-tree replacement for
vllm.model_executor.layers.logits_processor.LogitsProcessor; vLLM's own
PluggableLayer.__new__ dispatches to it transparently.

Behavior is byte-identical to stock vLLM whenever VLLM_TINY_M_GEMV_ENABLE is
unset/false -- the override only changes the LM-head matmul call site
(lm_head.quant_method.apply -> maybe_tiny_m_linear), nothing else in
_get_logits (gather, vocab-padding slice, soft-cap, scale) is touched.

E2E-9 adds ONE additional, opt-in branch (VLLM_TINY_M_UNIFIED_SELECTOR=1):
when set, the matmul call site instead dispatches through a cached
perf_model.execution_policy.ExecutionPolicy (built once, lazily, on first
call -- never per-token) via perf_model.runtime_dispatcher.execute_lm_head_linear.
This policy's runtime_piecewise rules are themselves derived by calling the
unified selector once per M at build time (see
execution_policy.build_runtime_piecewise_policy), so the resulting dispatch
decision matches maybe_tiny_m_linear's for the M range covered by the
policy. The original env-var-driven path (VLLM_TINY_M_GEMV_ENABLE) is
UNCHANGED and remains the default; VLLM_TINY_M_UNIFIED_SELECTOR is off by
default and mutually exclusive with it at each call.
"""
from __future__ import annotations

import os

import torch

from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from perf_model.tiny_m_dispatch import maybe_tiny_m_linear

UNIFIED_SELECTOR_ENV = "VLLM_TINY_M_UNIFIED_SELECTOR"
_unified_policy_cache = {}


def _unified_selector_enabled() -> bool:
    val = os.environ.get(UNIFIED_SELECTOR_ENV)
    return val is not None and val.strip().lower() in ("1", "true", "yes", "on")


def _get_or_build_unified_policy(lm_head: VocabParallelEmbedding):
    """Lazily builds (once) and caches the LM-head ExecutionPolicy. Building
    calls select_implementation() a handful of times (one per M in the
    covered range) -- strictly a one-time, out-of-band cost, never incurred
    inside the per-token decode loop after the first call."""
    cache_key = "lm_head_policy"
    if cache_key in _unified_policy_cache:
        return _unified_policy_cache[cache_key]

    from perf_model.cost_model_registry import CostModelRegistry
    from perf_model.execution_policy import build_runtime_piecewise_policy
    from perf_model.operation_descriptor import (
        LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype,
    )
    from perf_model.tiny_m_dispatch import DEFAULT_THRESHOLD
    from perf_model.tiny_m_linear_cost_model import TinyMLinearCostModel

    registry = CostModelRegistry()
    registry.register(OperationFamily.LINEAR, TinyMLinearCostModel())

    n, k = lm_head.weight.shape[0], lm_head.weight.shape[1]
    template = OperationDescriptor(
        common=OperationEnvelope(
            operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
            dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode",
            logical_shape=(1, n, k),
        ),
        payload=LinearDescriptor(
            M=1, N=n, K=k, has_bias=False, decode_or_prefill="decode", graph_captured=False,
            eager_execution=True, tensor_parallel_size=1, weight_layout="row_major", input_contiguous=True,
        ),
    )
    m_values = list(range(1, DEFAULT_THRESHOLD + 5))
    policy = build_runtime_piecewise_policy(template, registry, m_values)
    _unified_policy_cache[cache_key] = policy
    return policy


@PluggableLayer.register_oot(name="LogitsProcessor")
class TinyMLogitsProcessor(LogitsProcessor):
    def _get_logits(
        self, hidden_states: torch.Tensor, lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Identical to LogitsProcessor._get_logits except the matmul call
        # site: lm_head.quant_method.apply(...) -> maybe_tiny_m_linear(...).
        # production: default_unquantized_gemm(layer, x, weight, bias) ==
        # F.linear(x, layer.weight, bias) with bias=embedding_bias passed
        # through UNCHANGED (never falls back to lm_head.bias -- confirmed
        # via vllm/model_executor/layers/utils.py:default_unquantized_gemm
        # and layer.py:UnquantizedLinearMethod.apply). Replicated exactly.
        if _unified_selector_enabled():
            from perf_model.runtime_dispatcher import execute_lm_head_linear
            policy = _get_or_build_unified_policy(lm_head)
            logits = execute_lm_head_linear(hidden_states, lm_head.weight, embedding_bias, policy)
        else:
            logits = maybe_tiny_m_linear(hidden_states, lm_head.weight, embedding_bias, op_family="lm_head")

        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits


print("[tiny_m_oot] TinyMLogitsProcessor registered as OOT replacement for LogitsProcessor")
