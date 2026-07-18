"""Wire message vocabulary for the D1 multi-process TP simulation.

Every message is a plain dict so it can move through a real
multiprocessing.Queue (backed by an OS pipe + pickling on POSIX) without any
custom reducers. Tensor payloads are always raw contiguous bytes
(`ndarray.tobytes()`) plus an explicit shape/dtype -- never a shared-memory
handle and never a pickled ndarray object (which would obscure how many
bytes actually moved).

Truth boundary: this is a localhost, single-host CPU simulation of a
collective. It is not NCCL, not GPU-to-GPU communication.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

MSG_SHARD = "shard"
MSG_COLLECTIVE_RESULT = "collective_result"
MSG_SHUTDOWN = "shutdown"
MSG_EVENT = "event"
MSG_CONTRIBUTION = "contribution"

EVT_PROCESS_STARTED = "process_started"
EVT_SHARD_RECEIVED = "shard_received"
EVT_LOCAL_COMPUTE_DONE = "local_compute_done"
EVT_ENTERED_COLLECTIVE = "entered_collective"
EVT_COLLECTIVE_SKIPPED_INTENTIONAL = "collective_skipped_intentional"
EVT_COLLECTIVE_RESULT_RECEIVED = "collective_result_received"
EVT_RANK_DONE = "rank_done"
EVT_SHUTTING_DOWN = "shutting_down"
EVT_ERROR = "error"


def array_to_payload(arr: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(arr)
    payload_bytes = contiguous.tobytes()
    return {
        "shape": contiguous.shape,
        "dtype": str(contiguous.dtype),
        "payload": payload_bytes,
        "byte_count": len(payload_bytes),
    }


def payload_to_array(payload: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(payload["payload"], dtype=payload["dtype"]).reshape(
        payload["shape"]
    )


def make_event(rank_id: int, event: str, **fields: Any) -> dict[str, Any]:
    return {"type": MSG_EVENT, "rank_id": rank_id, "event": event, "ts": time.time(), **fields}
