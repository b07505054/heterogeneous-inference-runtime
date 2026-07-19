"""D3A Part E/F: TP decomposition for a real nn.Linear o_proj operator.

nn.Linear computes Y = X @ W^T (+ b), with weight shape
[out_features, in_features] (PyTorch convention, verified against the real
captured module -- never assumed). Row-parallel decomposition partitions the
*input*-feature (contraction) axis -- W's column axis / X's column axis --
exactly as declared by the compiler's D2 tensor_shards:

    X = concat(X0, X1) along dim=1 (in_features)
    W = concat(W0, W1) along dim=1 (in_features)   # W stays [out, in_shard]
    Y_partial_r = X_r @ W_r^T                       # shape [seq, out_features]
    Y = all_reduce_sum(Y_partial_0, Y_partial_1)
    if bias is not None: Y = Y + bias                # applied exactly once,
                                                       # only after reduction
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deployment.execution_plan.schema import DistributedTensorShard


class TPDecompositionError(ValueError):
    """Fail-closed: the plan's shard metadata does not describe a valid
    partition of the real captured tensors."""


@dataclass(frozen=True)
class RankShard:
    rank_id: int
    range_start: int
    range_end: int
    x_shard: np.ndarray  # [seq_len, shard_width] -- ONLY this rank's columns of X
    w_shard: np.ndarray  # [out_features, shard_width] -- ONLY this rank's columns of W

    @property
    def shard_width(self) -> int:
        return self.range_end - self.range_start

    def x_checksum(self) -> float:
        return float(np.sum(self.x_shard))

    def w_checksum(self) -> float:
        return float(np.sum(self.w_shard))


def build_rank_shards(
    x: np.ndarray, w: np.ndarray, tensor_shards: tuple[DistributedTensorShard, ...],
) -> dict[int, RankShard]:
    """Splits the real captured X ([seq, in_features]) and W
    ([out_features, in_features]) into per-rank shards using ONLY the
    compiler-declared tensor_shards ranges -- shard boundaries are never
    invented independently of the plan.
    """
    if x.ndim != 2 or w.ndim != 2:
        raise TPDecompositionError(f"expected 2-D X and W, got X.ndim={x.ndim} W.ndim={w.ndim}")
    if x.shape[1] != w.shape[1]:
        raise TPDecompositionError(
            f"captured input hidden dimension {x.shape[1]} differs from module weight "
            f"in_features {w.shape[1]}"
        )
    if not tensor_shards:
        raise TPDecompositionError("plan declares no tensor_shards; cannot derive a partition")

    hidden_dim = x.shape[1]
    covered = 0
    shards: dict[int, RankShard] = {}
    for s in sorted(tensor_shards, key=lambda s: s.range_start):
        if s.range_start != covered:
            raise TPDecompositionError(
                f"shard coverage gap/overlap at offset {s.range_start} (expected {covered})"
            )
        if s.range_end > hidden_dim:
            raise TPDecompositionError(
                f"shard range_end {s.range_end} exceeds captured hidden dimension {hidden_dim}"
            )
        x_shard = x[:, s.range_start:s.range_end]
        w_shard = w[:, s.range_start:s.range_end]
        shards[s.shard_index] = RankShard(
            rank_id=s.shard_index, range_start=s.range_start, range_end=s.range_end,
            x_shard=np.ascontiguousarray(x_shard), w_shard=np.ascontiguousarray(w_shard),
        )
        covered = s.range_end
    if covered != hidden_dim:
        raise TPDecompositionError(
            f"shard coverage ends at {covered}, not the full captured hidden dimension {hidden_dim}"
        )
    return shards


def rank_local_partial_output(shard: RankShard) -> np.ndarray:
    """Y_partial_r = X_r @ W_r^T -- computed from ONLY this rank's shard."""
    return shard.x_shard @ shard.w_shard.T


def apply_bias_contract(reduced: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    """Bias is applied exactly once, after the collective reduction."""
    if bias is None:
        return reduced
    return reduced + bias


# -- Negative-path helpers, used only by Part M negative tests -------------

def apply_bias_twice_incorrectly(partials: list[np.ndarray], bias: np.ndarray) -> np.ndarray:
    """Deliberately wrong: applies bias independently on every rank's
    partial output before summation, double- (or N-times-) counting it.
    Exists only to prove the correct contract's negative-test coverage."""
    biased_partials = [p + bias for p in partials]
    return np.sum(biased_partials, axis=0)
