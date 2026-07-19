"""D4A Part D-1/F: ColumnParallelLinearExecutor.

Y = X @ W^T (+ b). W is partitioned along its OUTPUT (dim 0) axis:
    W = concat(W_rank0, W_rank1) along dim 0
Each rank receives the FULL replicated X and only its own local W/b shard:
    Y_rank = X @ W_rank^T (+ b_rank)
Reconstruction (validation-only -- real vLLM keeps this sharded and feeds
it directly into the next rank-local consumer, see whole_model_inventory
and whole_model_plan_builder for why no collective is required here):
    Y = concat(Y_rank0, Y_rank1) along the output axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deployment.execution_plan.schema import DistributedTensorShard


class ColumnParallelError(ValueError):
    """Fail-closed: the plan's shard metadata does not describe a valid
    column-parallel partition of the real captured tensors."""


@dataclass(frozen=True)
class ColumnRankShard:
    rank_id: int
    range_start: int
    range_end: int
    w_shard: np.ndarray  # [shard_out_features, in_features] -- ONLY this rank's rows of W
    b_shard: np.ndarray | None  # [shard_out_features] -- ONLY this rank's slice of bias, if any

    @property
    def shard_width(self) -> int:
        return self.range_end - self.range_start


def build_column_rank_shards(
    w: np.ndarray, b: np.ndarray | None, tensor_shards: tuple[DistributedTensorShard, ...],
) -> dict[int, ColumnRankShard]:
    """Splits the real W ([out_features, in_features]) and optional bias
    ([out_features]) into per-rank shards using ONLY the compiler-declared
    tensor_shards ranges over the OUTPUT axis (dim 0).
    """
    if w.ndim != 2:
        raise ColumnParallelError(f"expected 2-D W, got W.ndim={w.ndim}")
    if not tensor_shards:
        raise ColumnParallelError("plan declares no tensor_shards; cannot derive a partition")
    if any(s.partition_axis != 0 for s in tensor_shards):
        raise ColumnParallelError("column-parallel shards must partition axis 0 (output features)")

    out_features = w.shape[0]
    covered = 0
    shards: dict[int, ColumnRankShard] = {}
    for s in sorted(tensor_shards, key=lambda s: s.range_start):
        if s.range_start != covered:
            raise ColumnParallelError(f"shard coverage gap/overlap at offset {s.range_start} (expected {covered})")
        if s.range_end > out_features:
            raise ColumnParallelError(f"shard range_end {s.range_end} exceeds W output dimension {out_features}")
        w_shard = np.ascontiguousarray(w[s.range_start:s.range_end, :])
        b_shard = np.ascontiguousarray(b[s.range_start:s.range_end]) if b is not None else None
        shards[s.shard_index] = ColumnRankShard(
            rank_id=s.shard_index, range_start=s.range_start, range_end=s.range_end,
            w_shard=w_shard, b_shard=b_shard,
        )
        covered = s.range_end
    if covered != out_features:
        raise ColumnParallelError(f"shard coverage ends at {covered}, not the full W output dimension {out_features}")
    return shards


def rank_local_column_output(x: np.ndarray, shard: ColumnRankShard) -> np.ndarray:
    """Y_rank = X @ W_rank^T (+ b_rank). X is the FULL replicated input --
    never sliced for column-parallel (only the weight is sharded)."""
    y = x @ shard.w_shard.T
    if shard.b_shard is not None:
        y = y + shard.b_shard
    return y


def reconstruct_column_output(rank_outputs: dict[int, np.ndarray]) -> np.ndarray:
    """Validation-only reconstruction: concatenate rank outputs along the
    output (last) axis, in rank order. Real vLLM execution never performs
    this concatenation for q/k/v/gate/up -- it keeps them sharded (see
    whole_model_plan_builder.WorkItem.reconstructed_output_contract for
    each family)."""
    ordered = [rank_outputs[i] for i in sorted(rank_outputs)]
    return np.concatenate(ordered, axis=-1)
