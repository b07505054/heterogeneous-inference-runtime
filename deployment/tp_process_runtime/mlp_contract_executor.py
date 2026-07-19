"""D4A Part D-1/F/J: full per-layer MLP block TP contract.

hidden_state -> gate_proj/up_proj (column-parallel, matching shard
ownership) -> SiLU(gate_rank) * up_rank (rank-local elementwise product,
valid only because gate and up share the same output-shard ownership,
verified explicitly here) -> down_proj (row-parallel) -> all_reduce -> MLP
output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deployment.execution_plan.schema import DistributedTensorShard
from deployment.tp_process_runtime.column_parallel_executor import (
    build_column_rank_shards,
    rank_local_column_output,
)


class MLPContractError(ValueError):
    """Fail-closed: the MLP TP contract could not be validated."""


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class MLPRankTrace:
    rank_id: int
    gate_local_shape: tuple[int, ...]
    up_local_shape: tuple[int, ...]
    intermediate_local_shape: tuple[int, ...]
    down_partial_shape: tuple[int, ...]
    gate_shard_range: tuple[int, int]
    up_shard_range: tuple[int, int]
    shard_ownership_matches: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank_id": self.rank_id, "gate_local_shape": list(self.gate_local_shape),
            "up_local_shape": list(self.up_local_shape),
            "intermediate_local_shape": list(self.intermediate_local_shape),
            "down_partial_shape": list(self.down_partial_shape),
            "gate_shard_range": list(self.gate_shard_range), "up_shard_range": list(self.up_shard_range),
            "shard_ownership_matches": self.shard_ownership_matches,
        }


@dataclass(frozen=True)
class MLPBlockResult:
    reconstructed_output: np.ndarray
    rank_traces: tuple[MLPRankTrace, ...]


def run_serialized_tp_mlp_block(
    *, hidden_states: np.ndarray, gate_weight: np.ndarray, up_weight: np.ndarray, down_weight: np.ndarray,
    gate_shards: tuple[DistributedTensorShard, ...], up_shards: tuple[DistributedTensorShard, ...],
    down_shards: tuple[DistributedTensorShard, ...],
) -> MLPBlockResult:
    batch, seq, hidden = hidden_states.shape
    x2d = hidden_states.reshape(batch * seq, hidden)

    gate_rank_shards = build_column_rank_shards(gate_weight, None, gate_shards)
    up_rank_shards = build_column_rank_shards(up_weight, None, up_shards)

    if any(s.partition_axis != 1 for s in down_shards):
        raise MLPContractError("down_proj shards must partition axis 1 (input/contraction features)")
    down_ranges = {s.shard_index: (s.range_start, s.range_end) for s in down_shards}
    covered = 0
    for idx in sorted(down_ranges):
        start, end = down_ranges[idx]
        if start != covered:
            raise MLPContractError(f"down_proj shard coverage gap/overlap at {start} (expected {covered})")
        covered = end
    if covered != down_weight.shape[1]:
        raise MLPContractError(
            f"down_proj shard coverage ({covered}) does not match real down_proj input dim ({down_weight.shape[1]})"
        )

    partials: dict[int, np.ndarray] = {}
    traces: list[MLPRankTrace] = []
    for rank_id in sorted(gate_rank_shards):
        gate_shard = gate_rank_shards[rank_id]
        up_shard = up_rank_shards[rank_id]
        ownership_matches = (gate_shard.range_start, gate_shard.range_end) == (up_shard.range_start, up_shard.range_end)
        if not ownership_matches:
            raise MLPContractError(
                f"rank {rank_id}: gate_proj shard range {(gate_shard.range_start, gate_shard.range_end)} "
                f"does not match up_proj shard range {(up_shard.range_start, up_shard.range_end)} -- "
                "the elementwise SiLU(gate)*up product would combine mismatched intermediate columns"
            )

        gate_local = rank_local_column_output(x2d, gate_shard)
        up_local = rank_local_column_output(x2d, up_shard)
        intermediate_local = silu(gate_local) * up_local

        down_start, down_end = down_ranges[rank_id]
        w_shard_down = down_weight[:, down_start:down_end]
        partial = intermediate_local @ w_shard_down.T
        partials[rank_id] = partial

        traces.append(MLPRankTrace(
            rank_id=rank_id, gate_local_shape=gate_local.shape, up_local_shape=up_local.shape,
            intermediate_local_shape=intermediate_local.shape, down_partial_shape=partial.shape,
            gate_shard_range=(gate_shard.range_start, gate_shard.range_end),
            up_shard_range=(up_shard.range_start, up_shard.range_end),
            shard_ownership_matches=ownership_matches,
        ))

    reduced = np.sum([partials[i] for i in sorted(partials)], axis=0)
    reconstructed = reduced.reshape(batch, seq, hidden)
    return MLPBlockResult(reconstructed_output=reconstructed, rank_traces=tuple(traces))
