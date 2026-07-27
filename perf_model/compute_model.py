"""Level 1: analytical operation-count and memory-traffic estimates for the
major transformer primitives.

This module NEVER produces milliseconds. It produces FLOP counts (multiply-
add = 2 FLOPs, standard convention) and memory-traffic byte counts only.
Converting these to time requires an effective-throughput constant, which
lives in phase_model.py and must be tagged "derived_from_phase_measurement"
or "unavailable" -- never "measured_microbenchmark" (no kernel profiler is
used in this slice) and never a bare unlabeled constant.

Batching interaction (how concurrent sequences share compute/bandwidth) is
NOT modeled here -- every count below is per one forward pass over the given
token count, single stream. Concurrency effects are an explicit, named gap
(see BATCHING_INTERACTION_ERROR in calibration_row.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from perf_model.schema import ModelFeatures

CAUSAL_ATTENTION_FACTOR = 0.5  # causal mask means ~half of the T x T score/value matmul is skipped


@dataclass(frozen=True)
class OpCountBreakdown:
    qkv_proj_flops: int
    attention_score_flops: int
    attention_value_flops: int
    output_proj_flops: int
    mlp_flops: int
    vocab_proj_flops: int
    source: str = "analytical_flop_bandwidth"

    @property
    def total_flops(self) -> int:
        return (
            self.qkv_proj_flops + self.attention_score_flops + self.attention_value_flops
            + self.output_proj_flops + self.mlp_flops + self.vocab_proj_flops
        )

    def to_dict(self) -> dict:
        return {
            "qkv_proj_flops": self.qkv_proj_flops,
            "attention_score_flops": self.attention_score_flops,
            "attention_value_flops": self.attention_value_flops,
            "output_proj_flops": self.output_proj_flops,
            "mlp_flops": self.mlp_flops,
            "vocab_proj_flops": self.vocab_proj_flops,
            "total_flops": self.total_flops,
            "source": self.source,
        }


def _qkv_flops(model: ModelFeatures, tokens: int) -> int:
    h = model.hidden_size
    kv_dim = model.kv_head_count * model.head_dimension
    q = 2 * tokens * h * h
    k = 2 * tokens * h * kv_dim
    v = 2 * tokens * h * kv_dim
    return q + k + v


def _mlp_flops(model: ModelFeatures, tokens: int) -> int:
    # SwiGLU-style MLP (gate + up + down), matches Qwen2/most modern decoder-only archs.
    h, i = model.hidden_size, model.intermediate_size
    gate = 2 * tokens * h * i
    up = 2 * tokens * h * i
    down = 2 * tokens * i * h
    return gate + up + down


def prefill_op_counts(model: ModelFeatures, prompt_tokens: int) -> OpCountBreakdown:
    """One full forward pass over `prompt_tokens`, all layers, causal self-attention."""
    per_layer_qkv = _qkv_flops(model, prompt_tokens)
    per_layer_score = int(
        2 * model.attention_head_count * prompt_tokens * prompt_tokens
        * model.head_dimension * CAUSAL_ATTENTION_FACTOR
    )
    per_layer_value = per_layer_score  # same shape as score matmul
    per_layer_out = 2 * prompt_tokens * model.hidden_size * model.hidden_size
    per_layer_mlp = _mlp_flops(model, prompt_tokens)

    vocab = 2 * prompt_tokens * model.hidden_size * model.vocabulary_size

    layers = model.layer_count
    return OpCountBreakdown(
        qkv_proj_flops=per_layer_qkv * layers,
        attention_score_flops=per_layer_score * layers,
        attention_value_flops=per_layer_value * layers,
        output_proj_flops=per_layer_out * layers,
        mlp_flops=per_layer_mlp * layers,
        vocab_proj_flops=vocab,
    )


def decode_step_op_counts(model: ModelFeatures, kv_context_tokens: int) -> OpCountBreakdown:
    """One new token, attending over `kv_context_tokens` prior tokens (no causal factor
    needed -- the new token attends to the full existing context, one row of scores)."""
    tokens = 1
    per_layer_qkv = _qkv_flops(model, tokens)
    per_layer_score = 2 * model.attention_head_count * kv_context_tokens * model.head_dimension
    per_layer_value = per_layer_score
    per_layer_out = 2 * tokens * model.hidden_size * model.hidden_size
    per_layer_mlp = _mlp_flops(model, tokens)
    vocab = 2 * tokens * model.hidden_size * model.vocabulary_size

    layers = model.layer_count
    return OpCountBreakdown(
        qkv_proj_flops=per_layer_qkv * layers,
        attention_score_flops=per_layer_score * layers,
        attention_value_flops=per_layer_value * layers,
        output_proj_flops=per_layer_out * layers,
        mlp_flops=per_layer_mlp * layers,
        vocab_proj_flops=vocab,
    )


def decode_step_memory_traffic_bytes(
    model: ModelFeatures, *, weight_bytes: int, kv_bytes_per_token: int, kv_context_tokens: int
) -> int:
    """Single-sequence (batch=1) decode is memory-bandwidth bound: each step must
    effectively move the full weight set plus the live KV cache once. This is the
    textbook single-stream decode argument, not a measurement. Batched decode
    amortizes the weight-read term across the batch -- not modeled here; see
    module docstring.
    """
    return weight_bytes + kv_bytes_per_token * kv_context_tokens


def prefill_memory_traffic_bytes(*, weight_bytes: int) -> int:
    """Prefill over many tokens is compute-bound; the dominant traffic term is a
    single weight read (activations stay resident). KV-cache writes are counted
    separately by the memory model, not as compute-phase traffic.
    """
    return weight_bytes
