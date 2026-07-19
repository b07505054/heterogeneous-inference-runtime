"""D4A Part D-4/K: vocabulary-parallel executors for embedding and lm_head.

Embedding (VocabParallelEmbedding, per whole_model_inventory's verified
vLLM source excerpt): each rank masks token IDs outside its owned vocab
range to a sentinel, does a local nn.Embedding-style lookup (masked rows
are then zeroed), and the two rank outputs are all_reduce(sum)-ed to
reconstruct the correct row for every token regardless of which rank owns
it.

lm_head (tied to embed_tokens for this model): each rank computes local
logits = hidden_states @ W_vocab_shard^T, and the two local logit shards
are concatenated (all_gather along the vocab dimension, rank order) to
reconstruct the full logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deployment.execution_plan.schema import DistributedTensorShard


class VocabParallelError(ValueError):
    """Fail-closed: the plan's vocab shard metadata does not describe a
    valid partition of the real embedding table."""


@dataclass(frozen=True)
class VocabRankShard:
    rank_id: int
    vocab_start: int
    vocab_end: int
    weight_shard: np.ndarray  # [vocab_shard_size, hidden_size]

    @property
    def shard_size(self) -> int:
        return self.vocab_end - self.vocab_start


def build_vocab_rank_shards(
    embedding_weight: np.ndarray, tensor_shards: tuple[DistributedTensorShard, ...],
) -> dict[int, VocabRankShard]:
    if embedding_weight.ndim != 2:
        raise VocabParallelError(f"expected 2-D embedding weight, got ndim={embedding_weight.ndim}")
    if any(s.partition_axis != 0 for s in tensor_shards):
        raise VocabParallelError("vocab-parallel shards must partition axis 0 (vocab dimension)")

    vocab_size = embedding_weight.shape[0]
    covered = 0
    shards: dict[int, VocabRankShard] = {}
    for s in sorted(tensor_shards, key=lambda s: s.range_start):
        if s.range_start != covered:
            raise VocabParallelError(f"vocab shard coverage gap/overlap at {s.range_start} (expected {covered})")
        if s.range_end > vocab_size:
            raise VocabParallelError(f"vocab shard range_end {s.range_end} exceeds vocab_size {vocab_size}")
        shards[s.shard_index] = VocabRankShard(
            rank_id=s.shard_index, vocab_start=s.range_start, vocab_end=s.range_end,
            weight_shard=np.ascontiguousarray(embedding_weight[s.range_start:s.range_end, :]),
        )
        covered = s.range_end
    if covered != vocab_size:
        raise VocabParallelError(f"vocab shard coverage ends at {covered}, not full vocab_size {vocab_size}")
    return shards


def rank_local_masked_embedding(token_ids: np.ndarray, shard: VocabRankShard) -> np.ndarray:
    """Masked local embedding lookup: tokens outside [vocab_start, vocab_end)
    contribute an all-zero row; owned tokens are looked up normally."""
    owned = (token_ids >= shard.vocab_start) & (token_ids < shard.vocab_end)
    local_ids = np.where(owned, token_ids - shard.vocab_start, 0)
    looked_up = shard.weight_shard[local_ids]
    return looked_up * owned[..., None]


def reconstruct_embedding(rank_outputs: dict[int, np.ndarray]) -> np.ndarray:
    """all_reduce(sum) reconstruction -- exactly one rank contributes a
    nonzero row per token, so summation reproduces the true embedding row."""
    ordered = [rank_outputs[i] for i in sorted(rank_outputs)]
    return np.sum(ordered, axis=0)


def rank_local_lm_head_logits(hidden_states: np.ndarray, shard: VocabRankShard) -> np.ndarray:
    """local_logits = hidden_states @ W_vocab_shard^T -- hidden_states is
    the FULL replicated final hidden state (never sharded for this op)."""
    return hidden_states @ shard.weight_shard.T


def reconstruct_lm_head_logits(rank_outputs: dict[int, np.ndarray], *, org_vocab_size: int) -> np.ndarray:
    """all_gather (concat along the vocab/last axis, rank order), then trim
    to org_vocab_size to drop any padding (a no-op here since vocab_size
    divides evenly by world_size for this model -- verified, not assumed,
    by the caller)."""
    ordered = [rank_outputs[i] for i in sorted(rank_outputs)]
    full = np.concatenate(ordered, axis=-1)
    return full[..., :org_vocab_size]
