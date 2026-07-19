"""D3A Part G: serialized (single-process, sequential) replay of the D1
all_reduce(sum) collective contract.

This reuses D1's CollectiveCoordinator unmodified -- not a reimplementation
-- fed through a synchronous stdlib queue.Queue instead of
multiprocessing.Queue. queue.Queue exposes the same .put()/.get(timeout=...)
surface (including raising queue.Empty on timeout), so CollectiveCoordinator
runs identically without any code change.

"Serialized" is explicit: rank 0's and rank 1's contributions are placed on
the queue sequentially, in one process, one physical device. This is
distinct from (and, when both are exercised, is reported alongside) D1's
real multi-process/IPC collective, which D3A also exercises separately in
deployment/tp_process_runtime/runtime.py against the same real captured
tensors -- see run_distributed_d3a_pipeline.py's "multiprocess_ipc_replay"
section.
"""

from __future__ import annotations

import queue
import time

import numpy as np

from deployment.tp_process_runtime.collective import CollectiveCoordinator, CollectiveOutcome
from deployment.tp_process_runtime.messages import array_to_payload


def run_serialized_all_reduce(
    *, collective_id: str, sequence_id: int, tensor_id: str,
    contributions: dict[int, np.ndarray], timeout_s: float = 5.0,
) -> CollectiveOutcome:
    q: queue.Queue = queue.Queue()
    for rank_id in sorted(contributions):
        arr = contributions[rank_id]
        msg = {
            "type": "contribution", "rank_id": rank_id,
            "collective_id": collective_id, "sequence_id": sequence_id,
            "tensor_id": tensor_id, "ts": time.time(),
            **array_to_payload(arr),
        }
        q.put(msg)

    coordinator = CollectiveCoordinator()
    return coordinator.run_all_reduce_sum(
        collective_id=collective_id, sequence_id=sequence_id,
        expected_ranks=set(contributions.keys()), from_rank_queue=q, timeout_s=timeout_s,
    )
