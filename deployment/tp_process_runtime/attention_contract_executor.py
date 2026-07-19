"""D4A Part D-3/F/I: full per-layer attention block TP contract.

Attention is validated as one composed contract, not independent linear
tests: hidden_state -> Q/K/V column-parallel projections -> local head
reshape -> rotary position embedding (replicated, applied to each rank's
LOCAL Q/K heads) -> GQA repeat (rank-local) -> attention computation
(head-parallel) -> local head concatenation -> o_proj row-parallel
reduction -> attention block output.

All math here is a direct numpy reimplementation of the exact formulas
read from transformers/models/qwen2/modeling_qwen2.py
(rotate_half/apply_rotary_pos_emb/repeat_kv/eager_attention_forward) --
see whole_model_inventory.py for the paired vLLM contract facts this
validates against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deployment.execution_plan.schema import DistributedTensorShard
from deployment.tp_process_runtime.column_parallel_executor import (
    build_column_rank_shards,
    rank_local_column_output,
)


class AttentionContractError(ValueError):
    """Fail-closed: the attention TP contract could not be validated."""


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray,
                          *, unsqueeze_dim: int = 1) -> tuple[np.ndarray, np.ndarray]:
    cos = np.expand_dims(cos, unsqueeze_dim)
    sin = np.expand_dims(sin, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: np.ndarray, n_rep: int) -> np.ndarray:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = np.repeat(hidden_states[:, :, None, :, :], n_rep, axis=2)
    return expanded.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, *, scaling: float,
                     causal_mask: np.ndarray, num_key_value_groups: int) -> np.ndarray:
    """Returns attn_output shaped (batch, seq, num_heads, head_dim) -- matches
    transformers.eager_attention_forward's post-transpose convention."""
    key_states = repeat_kv(k, num_key_value_groups)
    value_states = repeat_kv(v, num_key_value_groups)
    attn_weights = np.matmul(q, np.swapaxes(key_states, -1, -2)) * scaling
    attn_weights = attn_weights + causal_mask
    attn_weights = attn_weights.astype(np.float64)
    attn_weights = attn_weights - attn_weights.max(axis=-1, keepdims=True)
    exp = np.exp(attn_weights)
    attn_weights = (exp / exp.sum(axis=-1, keepdims=True)).astype(q.dtype)
    attn_output = np.matmul(attn_weights, value_states)
    return np.swapaxes(attn_output, 1, 2)


def build_causal_mask(seq_len: int, dtype: np.dtype) -> np.ndarray:
    mask = np.triu(np.full((seq_len, seq_len), np.finfo(np.float32).min, dtype=np.float32), k=1)
    return mask.astype(dtype)[None, None, :, :]


@dataclass(frozen=True)
class AttentionRankTrace:
    rank_id: int
    q_local_shape: tuple[int, ...]
    k_local_shape: tuple[int, ...]
    v_local_shape: tuple[int, ...]
    attn_score_shape: tuple[int, ...]
    kv_repetition_factor: int
    local_context_shape: tuple[int, ...]
    o_proj_partial_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank_id": self.rank_id,
            "q_local_shape": list(self.q_local_shape), "k_local_shape": list(self.k_local_shape),
            "v_local_shape": list(self.v_local_shape), "attn_score_shape": list(self.attn_score_shape),
            "kv_repetition_factor": self.kv_repetition_factor,
            "local_context_shape": list(self.local_context_shape),
            "o_proj_partial_shape": list(self.o_proj_partial_shape),
        }


@dataclass(frozen=True)
class AttentionBlockResult:
    reconstructed_output: np.ndarray
    final_shape: tuple[int, ...]
    rank_traces: tuple[AttentionRankTrace, ...]


def run_serialized_tp_attention_block(
    *, hidden_states: np.ndarray, q_weight: np.ndarray, q_bias: np.ndarray | None,
    k_weight: np.ndarray, k_bias: np.ndarray | None, v_weight: np.ndarray, v_bias: np.ndarray | None,
    o_weight: np.ndarray, cos: np.ndarray, sin: np.ndarray,
    num_heads_per_rank: int, num_kv_heads_per_rank: int, head_dim: int, world_size: int,
    q_shards: tuple[DistributedTensorShard, ...], k_shards: tuple[DistributedTensorShard, ...],
    v_shards: tuple[DistributedTensorShard, ...], o_shards: tuple[DistributedTensorShard, ...],
) -> AttentionBlockResult:
    """Serialized (one-rank-at-a-time) whole attention-block TP replay for
    one real captured layer input. hidden_states is [batch, seq, hidden]
    (already input_layernorm-normalized, matching the real model's
    self_attn call site). cos/sin are the REAL rotary_emb module's output
    (replicated -- identical for every rank), converted to numpy.
    """
    batch, seq, hidden = hidden_states.shape
    x2d = hidden_states.reshape(batch * seq, hidden)

    q_rank_shards = build_column_rank_shards(q_weight, q_bias, q_shards)
    k_rank_shards = build_column_rank_shards(k_weight, k_bias, k_shards)
    v_rank_shards = build_column_rank_shards(v_weight, v_bias, v_shards)

    if any(s.partition_axis != 1 for s in o_shards):
        raise AttentionContractError("o_proj shards must partition axis 1 (input/contraction features)")
    o_ranges_by_rank = {s.shard_index: (s.range_start, s.range_end) for s in o_shards}
    expected_o_input = num_heads_per_rank * head_dim * world_size
    covered = 0
    for idx in sorted(o_ranges_by_rank):
        start, end = o_ranges_by_rank[idx]
        if start != covered:
            raise AttentionContractError(f"o_proj shard coverage gap/overlap at {start} (expected {covered})")
        covered = end
    if covered != expected_o_input or covered != o_weight.shape[1]:
        raise AttentionContractError(
            f"o_proj shard coverage ({covered}) does not match expected input width "
            f"({expected_o_input}) or real o_proj weight input dim ({o_weight.shape[1]})"
        )

    causal_mask = build_causal_mask(seq, hidden_states.dtype)
    num_key_value_groups = num_heads_per_rank // num_kv_heads_per_rank
    scaling = head_dim ** -0.5

    partials: dict[int, np.ndarray] = {}
    traces: list[AttentionRankTrace] = []
    for rank_id in sorted(q_rank_shards):
        q_local = rank_local_column_output(x2d, q_rank_shards[rank_id]).reshape(batch, seq, num_heads_per_rank, head_dim)
        k_local = rank_local_column_output(x2d, k_rank_shards[rank_id]).reshape(batch, seq, num_kv_heads_per_rank, head_dim)
        v_local = rank_local_column_output(x2d, v_rank_shards[rank_id]).reshape(batch, seq, num_kv_heads_per_rank, head_dim)

        q_local = np.swapaxes(q_local, 1, 2)  # (batch, heads, seq, head_dim)
        k_local = np.swapaxes(k_local, 1, 2)
        v_local = np.swapaxes(v_local, 1, 2)

        q_local, k_local = apply_rotary_pos_emb(q_local, k_local, cos, sin)

        attn_out = eager_attention(
            q_local, k_local, v_local, scaling=scaling, causal_mask=causal_mask,
            num_key_value_groups=num_key_value_groups,
        )  # (batch, seq, num_heads_per_rank, head_dim)
        local_context = attn_out.reshape(batch * seq, num_heads_per_rank * head_dim)

        o_start, o_end = o_ranges_by_rank[rank_id]
        w_shard_o = o_weight[:, o_start:o_end]
        partial = local_context @ w_shard_o.T
        partials[rank_id] = partial

        traces.append(AttentionRankTrace(
            rank_id=rank_id, q_local_shape=q_local.shape, k_local_shape=k_local.shape,
            v_local_shape=v_local.shape, attn_score_shape=(batch, num_heads_per_rank, seq, seq),
            kv_repetition_factor=num_key_value_groups, local_context_shape=local_context.shape,
            o_proj_partial_shape=partial.shape,
        ))

    reduced = np.sum([partials[i] for i in sorted(partials)], axis=0)
    reconstructed = reduced.reshape(batch, seq, hidden)
    return AttentionBlockResult(
        reconstructed_output=reconstructed, final_shape=reconstructed.shape, rank_traces=tuple(traces),
    )
