"""CollectiveCoordinator: a central-coordinator implementation of
all_reduce(sum) over localhost IPC.

Not an efficient ring/tree collective -- every rank sends its full
contribution to the parent coordinator process, which reduces and
broadcasts the result back to every rank. That is O(N) point-to-point
messages through the parent, not a peer-to-peer collective topology. This
is intentional for D1: the goal is to prove real cross-process data
movement and correct collective semantics (ordering, participation,
timeout), not communication efficiency.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field

import numpy as np

from deployment.tp_process_runtime.messages import MSG_CONTRIBUTION, payload_to_array


@dataclass
class CollectiveOutcome:
    collective_id: str
    sequence_id: int
    status: str  # "completed" | "timeout" | "shape_mismatch"
    contributions: dict[int, dict] = field(default_factory=dict)
    missing_ranks: set[int] = field(default_factory=set)
    duplicate_events: list[dict] = field(default_factory=list)
    unexpected_events: list[dict] = field(default_factory=list)
    sequence_mismatch_events: list[dict] = field(default_factory=list)
    passthrough_events: list[dict] = field(default_factory=list)
    reduced: np.ndarray | None = None
    bytes_contributed: int = 0
    start_ts: float = 0.0
    end_ts: float = 0.0


class CollectiveCoordinator:
    """Runs entirely in the parent process. Owns the shared from-rank queue
    for the duration of one collective step."""

    def run_all_reduce_sum(
        self,
        *,
        collective_id: str,
        sequence_id: int,
        expected_ranks: set[int],
        from_rank_queue,
        timeout_s: float,
    ) -> CollectiveOutcome:
        outcome = CollectiveOutcome(
            collective_id=collective_id, sequence_id=sequence_id, status="completed"
        )
        outcome.start_ts = time.time()
        deadline = outcome.start_ts + timeout_s

        while len(outcome.contributions) < len(expected_ranks):
            remaining = deadline - time.time()
            if remaining <= 0:
                outcome.status = "timeout"
                break
            try:
                msg = from_rank_queue.get(timeout=remaining)
            except queue.Empty:
                outcome.status = "timeout"
                break

            if msg.get("type") != MSG_CONTRIBUTION:
                outcome.passthrough_events.append(msg)
                continue
            if msg.get("collective_id") != collective_id or msg.get("sequence_id") != sequence_id:
                outcome.sequence_mismatch_events.append(msg)
                continue
            rank_id = msg["rank_id"]
            if rank_id not in expected_ranks:
                outcome.unexpected_events.append(msg)
                continue
            if rank_id in outcome.contributions:
                outcome.duplicate_events.append(msg)
                continue
            outcome.contributions[rank_id] = msg
            outcome.bytes_contributed += msg["byte_count"]

        outcome.missing_ranks = expected_ranks - set(outcome.contributions)
        outcome.end_ts = time.time()

        if outcome.status == "completed" and not outcome.missing_ranks:
            arrays = [payload_to_array(c) for c in outcome.contributions.values()]
            shapes = {a.shape for a in arrays}
            if len(shapes) != 1:
                outcome.status = "shape_mismatch"
            else:
                outcome.reduced = np.sum(arrays, axis=0)
        return outcome
