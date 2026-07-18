"""DistributedProcessRuntime: D1 orchestrator.

Launches real OS processes from a compiler-planned DistributedPlan,
coordinates one all_reduce(sum) collective over real localhost IPC, and
reconstructs the distributed output explicitly from the bytes that moved
through the queues -- never from process-inherited shared memory.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deployment.execution_plan.schema import DistributedPlan
from deployment.tp_process_runtime.collective import CollectiveCoordinator, CollectiveOutcome
from deployment.tp_process_runtime.messages import (
    MSG_COLLECTIVE_RESULT,
    MSG_EVENT,
    MSG_SHARD,
    MSG_SHUTDOWN,
    array_to_payload,
    make_event,
)
from deployment.tp_process_runtime.rank_worker import RankProcessSpec, rank_worker_main


class DistributedRuntimeError(RuntimeError):
    """Fail-closed error: malformed/unsupported distributed plan, or a
    child-process exception propagated to the parent."""


@dataclass
class RankMailbox:
    rank_id: int
    to_rank: Any    # multiprocessing.Queue -- this rank's private input IPC endpoint
    from_rank: Any  # multiprocessing.Queue -- the shared output IPC endpoint


@dataclass
class DistributedExecutionTrace:
    """Ordered event log for one run -- the provenance source of truth."""

    events: list[dict] = field(default_factory=list)

    def append(self, event: dict) -> None:
        self.events.append(event)

    def to_jsonable(self) -> list[dict]:
        return [_jsonable(e) for e in self.events]


@dataclass
class ProcessRecord:
    rank_id: int
    pid: int | None = None
    exitcode: int | None = None
    start_ts: float = 0.0
    join_ts: float | None = None
    alive_after_teardown: bool = True


@dataclass
class DistributedExecutionResult:
    status: str  # "completed" | "timeout" | "error"
    world_size: int
    distributed_output: np.ndarray | None
    trace: DistributedExecutionTrace
    processes: dict[int, ProcessRecord]
    collective_outcomes: list[CollectiveOutcome]
    provenance: dict[str, int]
    timings: dict[str, float]
    deadlock: dict | None = None
    error_message: str | None = None

    @property
    def all_ranks_completed(self) -> bool:
        return self.status == "completed" and all(
            p.exitcode == 0 for p in self.processes.values()
        )

    @property
    def all_collectives_completed(self) -> bool:
        return self.status == "completed" and all(
            o.status == "completed" for o in self.collective_outcomes
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if k != "payload"}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class DistributedProcessRuntime:
    """D1 multi-process rank runtime. Uses Python multiprocessing with the
    "spawn" start method (real OS processes, fresh interpreters -- not
    threads, not fork()-inherited shared memory)."""

    def __init__(self, *, mp_context: str = "spawn") -> None:
        self._ctx = mp.get_context(mp_context)

    def run(
        self,
        distributed_plan: DistributedPlan,
        a: np.ndarray,
        b: np.ndarray,
        *,
        shard_wait_timeout_s: float = 30.0,
        collective_timeout_s: float = 10.0,
        shutdown_timeout_s: float = 10.0,
        force_skip_collective_rank: int | None = None,
    ) -> DistributedExecutionResult:
        _validate_plan_for_execution(distributed_plan)
        world_size = distributed_plan.world_size
        ranks = sorted(r.rank_id for r in distributed_plan.ranks)
        shard_map = _shard_ranges(distributed_plan)
        _validate_problem_shapes(a, b, shard_map, world_size)

        step = distributed_plan.collectives[0] if distributed_plan.collectives else None
        collective_id = step.collective_id if step else "tp1_single_rank_passthrough"
        sequence_id = step.sequence_id if step else 0
        participants = set(step.participants) if step else set(ranks)

        trace = DistributedExecutionTrace()
        timings: dict[str, float] = {}
        t_run_start = time.perf_counter()

        from_rank_queue = self._ctx.Queue()
        mailboxes: dict[int, RankMailbox] = {
            r: RankMailbox(r, self._ctx.Queue(), from_rank_queue) for r in ranks
        }
        processes: dict[int, ProcessRecord] = {r: ProcessRecord(rank_id=r) for r in ranks}
        procs: dict[int, Any] = {}

        t0 = time.perf_counter()
        for r in ranks:
            spec = RankProcessSpec(
                rank_id=r,
                world_size=world_size,
                tensor_id=step.tensor_id if step else shard_map[r][2],
                collective_id=collective_id,
                sequence_id=sequence_id,
                participants=tuple(sorted(participants)),
                reduction=step.reduction if step else "sum",
                skip_collective=(r == force_skip_collective_rank),
                shard_wait_timeout_s=shard_wait_timeout_s,
                collective_result_timeout_s=collective_timeout_s,
                shutdown_wait_timeout_s=shutdown_timeout_s,
            )
            proc = self._ctx.Process(
                target=rank_worker_main,
                args=(r, mailboxes[r].to_rank, mailboxes[r].from_rank, spec),
                name=f"d1-rank-{r}",
            )
            proc.start()
            procs[r] = proc
            processes[r].pid = proc.pid
            processes[r].start_ts = time.perf_counter()
        pids = [p.pid for p in procs.values()]
        if len(set(pids)) != len(pids):
            raise DistributedRuntimeError(f"non-unique child process IDs observed: {pids}")
        timings["process_startup_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for r in ranks:
            a_start, a_end, _tensor_id = shard_map[r]
            a_slice = a[:, a_start:a_end]
            b_slice = b[a_start:a_end, :]
            mailboxes[r].to_rank.put({
                "type": MSG_SHARD,
                "a": array_to_payload(a_slice),
                "b": array_to_payload(b_slice),
            })
        timings["shard_dispatch_s"] = time.perf_counter() - t0

        coordinator = CollectiveCoordinator()
        outcome = coordinator.run_all_reduce_sum(
            collective_id=collective_id,
            sequence_id=sequence_id,
            expected_ranks=set(ranks),
            from_rank_queue=from_rank_queue,
            timeout_s=collective_timeout_s,
        )
        for ev in outcome.passthrough_events:
            trace.append(ev)
        timings["collective_latency_s"] = outcome.end_ts - outcome.start_ts

        status = "completed" if outcome.status == "completed" and not outcome.missing_ranks else "timeout"
        deadlock_record = None
        distributed_output = None

        if status == "timeout":
            deadlock_record = {
                "collective_id": collective_id,
                "sequence_id": sequence_id,
                "missing_ranks": sorted(outcome.missing_ranks),
                "elapsed_s": outcome.end_ts - outcome.start_ts,
                "status": "timeout_detected_deadlock",
            }
            self._terminate_all(procs, processes, trace)
        else:
            distributed_output = outcome.reduced
            t0 = time.perf_counter()
            for r in ranks:
                mailboxes[r].to_rank.put({
                    "type": MSG_COLLECTIVE_RESULT,
                    **array_to_payload(distributed_output),
                })
            self._drain_until(from_rank_queue, trace, ranks_pending=set(ranks),
                               event_name="rank_done", timeout_s=collective_timeout_s)
            timings["broadcast_and_ack_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            for r in ranks:
                mailboxes[r].to_rank.put({"type": MSG_SHUTDOWN})
            self._join_all(procs, processes, trace, timeout_s=shutdown_timeout_s)
            timings["process_shutdown_s"] = time.perf_counter() - t0

        self._drain_remaining(from_rank_queue, trace)

        error_message = None
        for ev in trace.events:
            if ev.get("type") == MSG_EVENT and ev.get("event") == "error":
                error_message = ev.get("message")
                status = "error"

        timings["end_to_end_s"] = time.perf_counter() - t_run_start

        provenance = _compute_provenance(
            trace=trace, planned_ranks=set(ranks), shard_map=shard_map,
            outcome=outcome, processes=processes,
        )

        return DistributedExecutionResult(
            status=status,
            world_size=world_size,
            distributed_output=distributed_output,
            trace=trace,
            processes=processes,
            collective_outcomes=[outcome],
            provenance=provenance,
            timings=timings,
            deadlock=deadlock_record,
            error_message=error_message,
        )

    def _drain_until(self, from_rank_queue, trace, *, ranks_pending: set[int],
                      event_name: str, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        seen: set[int] = set()
        while seen != ranks_pending and time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                msg = from_rank_queue.get(timeout=remaining)
            except queue.Empty:
                break
            trace.append(msg)
            if msg.get("type") == MSG_EVENT and msg.get("event") in (event_name, "error"):
                seen.add(msg["rank_id"])

    def _join_all(self, procs, processes: dict[int, ProcessRecord], trace,
                   *, timeout_s: float) -> None:
        for r, proc in procs.items():
            proc.join(timeout=timeout_s)
            alive = proc.is_alive()
            if alive:
                proc.kill()
                proc.join(timeout=5.0)
                alive = proc.is_alive()
            processes[r].join_ts = time.perf_counter()
            processes[r].exitcode = proc.exitcode
            processes[r].alive_after_teardown = alive
            trace.append(make_event(r, "process_exit_checked", alive=alive,
                                     exitcode=proc.exitcode))

    def _terminate_all(self, procs, processes: dict[int, ProcessRecord], trace) -> None:
        for proc in procs.values():
            proc.terminate()
        for r, proc in procs.items():
            proc.join(timeout=5.0)
            alive = proc.is_alive()
            if alive:
                proc.kill()
                proc.join(timeout=5.0)
                alive = proc.is_alive()
            processes[r].join_ts = time.perf_counter()
            processes[r].exitcode = proc.exitcode
            processes[r].alive_after_teardown = alive
            trace.append(make_event(r, "process_exit_checked", alive=alive,
                                     exitcode=proc.exitcode, terminated=True))

    def _drain_remaining(self, from_rank_queue, trace) -> None:
        while True:
            try:
                msg = from_rank_queue.get(timeout=0.2)
            except queue.Empty:
                break
            trace.append(msg)


def _shard_ranges(plan: DistributedPlan) -> dict[int, tuple]:
    by_rank: dict[int, tuple] = {}
    for shard in plan.tensor_shards:
        by_rank[shard.shard_index] = (shard.range_start, shard.range_end, shard.tensor_id)
    if not by_rank:
        # world_size == 1, no explicit shard declared: whole-K single shard.
        by_rank[0] = (0, None, "partial_output")  # range_end filled in by caller
    return by_rank


def _validate_plan_for_execution(plan: DistributedPlan) -> None:
    if plan.world_size < 1 or plan.tensor_parallel_size != plan.world_size:
        raise DistributedRuntimeError(
            "distributed plan fails D1 execution precondition: "
            f"world_size={plan.world_size} tensor_parallel_size={plan.tensor_parallel_size}"
        )
    if plan.pipeline_parallel_size != 1:
        raise DistributedRuntimeError("D1 runtime supports pipeline_parallel_size == 1 only")
    declared_ranks = sorted(r.rank_id for r in plan.ranks)
    if declared_ranks != list(range(plan.world_size)):
        raise DistributedRuntimeError(
            f"distributed plan rank ids are not contiguous 0..world_size-1: {declared_ranks}"
        )
    declared_tensor_ids = {s.tensor_id for s in plan.tensor_shards}
    for step in plan.collectives:
        if step.kind != "all_reduce":
            raise DistributedRuntimeError(
                f"D1 runtime does not implement collective kind {step.kind!r}; "
                "refusing to silently ignore it"
            )
        # D2: a collective must reference a tensor_id this plan actually
        # shards -- never an unknown operator/tensor (Part J negative test).
        if plan.tensor_shards and step.tensor_id not in declared_tensor_ids:
            raise DistributedRuntimeError(
                f"collective {step.collective_id!r} references unknown tensor_id "
                f"{step.tensor_id!r}; declared tensor_shards cover {sorted(declared_tensor_ids)!r}"
            )
    if plan.world_size > 1 and not plan.collectives:
        raise DistributedRuntimeError(
            "distributed plan with world_size > 1 must declare at least one "
            "collective step -- refusing to silently execute ranks in isolation"
        )


def _validate_problem_shapes(a: np.ndarray, b: np.ndarray, shard_map: dict[int, tuple],
                              world_size: int) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise DistributedRuntimeError(f"incompatible problem shapes: {a.shape} x {b.shape}")
    k = a.shape[1]
    if world_size == 1:
        start, _end, tensor_id = shard_map[0]
        shard_map[0] = (start, k, tensor_id)
        return
    covered = 0
    for r in range(world_size):
        start, end, _ = shard_map[r]
        if start != covered:
            raise DistributedRuntimeError(
                f"shard coverage gap/overlap at rank {r}: expected start {covered}, got {start}"
            )
        covered = end
    if covered != k:
        raise DistributedRuntimeError(
            f"shard coverage {covered} does not match problem K dimension {k}"
        )


def _compute_provenance(*, trace: DistributedExecutionTrace, planned_ranks: set[int],
                         shard_map: dict[int, tuple], outcome: CollectiveOutcome,
                         processes: dict[int, ProcessRecord]) -> dict[str, int]:
    launched_ranks: set[int] = set()
    shard_received_ranks: dict[int, tuple] = {}
    unexpected_rank_events = 0
    rank_mismatch = 0

    for ev in trace.events:
        if ev.get("type") != "event":
            continue
        rid = ev.get("rank_id")
        if not isinstance(rid, int):
            rank_mismatch += 1
            continue
        if rid not in planned_ranks:
            unexpected_rank_events += 1
            continue
        if ev.get("event") == "process_started":
            launched_ranks.add(rid)
        if ev.get("event") == "shard_received":
            shard_received_ranks[rid] = (ev.get("a_shape"), ev.get("b_shape"))

    shard_mismatch_count = 0
    for rid, (a_shape, _b_shape) in shard_received_ranks.items():
        start, end, _ = shard_map.get(rid, (None, None, None))
        expected_k = (end - start) if start is not None and end is not None else None
        if expected_k is not None and a_shape is not None and tuple(a_shape)[-1] != expected_k:
            shard_mismatch_count += 1

    orphan_process_count = sum(1 for p in processes.values() if p.alive_after_teardown)

    return {
        "rank_mismatch_count": rank_mismatch,
        "missing_rank_count": len(planned_ranks - launched_ranks),
        "unexpected_rank_count": unexpected_rank_events,
        "shard_mismatch_count": shard_mismatch_count,
        "collective_sequence_mismatch_count": len(outcome.sequence_mismatch_events),
        "missing_collective_participant_count": len(outcome.missing_ranks),
        "unexpected_collective_participant_count": len(outcome.unexpected_events),
        "fallback_count": sum(1 for e in trace.events if e.get("event") == "fallback_used"),
        "silent_downgrade_count": sum(1 for e in trace.events if e.get("event") == "silent_downgrade"),
        "orphan_process_count": orphan_process_count,
    }
