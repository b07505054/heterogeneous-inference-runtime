"""Rank-local worker: runs inside a real, separate OS process (spawned via
multiprocessing's "spawn" context so the child starts a fresh interpreter and
receives data only through the queues it is handed -- never through
fork()-inherited memory).

Rank-local operation (D1 Part G): sharded matmul partial output.
  A is partitioned along K, B is partitioned along K.
  C_partial_rank = A_rank @ B_rank
  all_reduce(sum): C = sum_rank(C_partial_rank)

A rank never sees the full A/B, only its own K-slice, and never sees another
rank's partial output except through the coordinator-broadcast reduced
result (never directly from another rank).
"""

from __future__ import annotations

import os
import queue
import time
import traceback
from dataclasses import dataclass

import numpy as np

from deployment.tp_process_runtime.messages import (
    EVT_COLLECTIVE_RESULT_RECEIVED,
    EVT_COLLECTIVE_SKIPPED_INTENTIONAL,
    EVT_ENTERED_COLLECTIVE,
    EVT_ERROR,
    EVT_LOCAL_COMPUTE_DONE,
    EVT_PROCESS_STARTED,
    EVT_RANK_DONE,
    EVT_SHARD_RECEIVED,
    EVT_SHUTTING_DOWN,
    MSG_COLLECTIVE_RESULT,
    MSG_SHARD,
    MSG_SHUTDOWN,
    array_to_payload,
    make_event,
    payload_to_array,
)


@dataclass(frozen=True)
class RankProcessSpec:
    """Metadata planned for one rank -- no tensor bytes. Tensor bytes only
    ever travel as a queue message payload (see messages.py)."""

    rank_id: int
    world_size: int
    tensor_id: str
    collective_id: str | None
    sequence_id: int | None
    participants: tuple[int, ...]
    reduction: str
    skip_collective: bool = False
    shard_wait_timeout_s: float = 30.0
    collective_result_timeout_s: float = 30.0
    shutdown_wait_timeout_s: float = 30.0


def rank_worker_main(rank_id: int, to_rank, from_rank, spec: RankProcessSpec) -> None:
    """Entry point for the child process. Module-level (spawn-picklable)."""
    try:
        from_rank.put(make_event(rank_id, EVT_PROCESS_STARTED, pid=os.getpid()))

        try:
            shard_msg = to_rank.get(timeout=spec.shard_wait_timeout_s)
        except queue.Empty:
            from_rank.put(make_event(rank_id, EVT_ERROR,
                                      message="timed out waiting for shard"))
            return
        if shard_msg.get("type") != MSG_SHARD:
            from_rank.put(make_event(rank_id, EVT_ERROR,
                                      message=f"unexpected message: {shard_msg.get('type')}"))
            return

        a = payload_to_array(shard_msg["a"])
        b = payload_to_array(shard_msg["b"])
        from_rank.put(make_event(rank_id, EVT_SHARD_RECEIVED,
                                  a_shape=a.shape, b_shape=b.shape))

        t0 = time.perf_counter()
        partial = a @ b
        compute_ms = (time.perf_counter() - t0) * 1000.0
        from_rank.put(make_event(rank_id, EVT_LOCAL_COMPUTE_DONE,
                                  compute_ms=compute_ms, partial_shape=partial.shape))

        if spec.skip_collective:
            from_rank.put(make_event(rank_id, EVT_COLLECTIVE_SKIPPED_INTENTIONAL))
            # Intentionally does not contribute and does not wait for a
            # result -- this rank will be terminated by the coordinator's
            # deadlock/timeout path (D1 Part J negative test).
            to_rank.get(timeout=spec.shutdown_wait_timeout_s)
            return

        from_rank.put(make_event(rank_id, EVT_ENTERED_COLLECTIVE,
                                  collective_id=spec.collective_id,
                                  sequence_id=spec.sequence_id))
        contribution = {
            "type": "contribution",
            "rank_id": rank_id,
            "collective_id": spec.collective_id,
            "sequence_id": spec.sequence_id,
            "tensor_id": spec.tensor_id,
            "ts": time.time(),
            **array_to_payload(partial),
        }
        from_rank.put(contribution)

        try:
            result_msg = to_rank.get(timeout=spec.collective_result_timeout_s)
        except queue.Empty:
            from_rank.put(make_event(rank_id, EVT_ERROR,
                                      message="timed out waiting for collective result"))
            return
        if result_msg.get("type") != MSG_COLLECTIVE_RESULT:
            from_rank.put(make_event(rank_id, EVT_ERROR,
                                      message=f"unexpected message: {result_msg.get('type')}"))
            return
        reduced = payload_to_array(result_msg)
        from_rank.put(make_event(rank_id, EVT_COLLECTIVE_RESULT_RECEIVED,
                                  shape=reduced.shape))
        from_rank.put(make_event(rank_id, EVT_RANK_DONE))

        shutdown_msg = to_rank.get(timeout=spec.shutdown_wait_timeout_s)
        if shutdown_msg.get("type") != MSG_SHUTDOWN:
            from_rank.put(make_event(rank_id, EVT_ERROR,
                                      message=f"expected shutdown, got {shutdown_msg.get('type')}"))
            return
        from_rank.put(make_event(rank_id, EVT_SHUTTING_DOWN))
    except Exception as exc:  # child-process exceptions must propagate to parent
        from_rank.put(make_event(rank_id, EVT_ERROR, message=str(exc),
                                  traceback=traceback.format_exc()))
        raise
