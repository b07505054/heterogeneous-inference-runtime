"""D2: cross-layer provenance -- planned (compiler ExecutionPlan.distributed)
versus executed (D1 DistributedProcessRuntime result).

Every check compares a value the compiler declared against a value the
runtime actually observed via real events; nothing here is hardcoded to
"pass".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deployment.execution_plan.schema import DistributedPlan
from deployment.tp_process_runtime.runtime import DistributedExecutionResult


@dataclass(frozen=True)
class CrossLayerProvenanceReport:
    operator_id_match: bool
    world_size_match: bool
    rank_ids_match: bool
    shard_ranges_match: bool
    collective_id_match: bool
    sequence_id_match: bool
    participant_set_match: bool
    no_silent_downgrade: bool
    no_synthetic_fallback_dimensions: bool
    mismatch_count: int
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def all_match(self) -> bool:
        return self.mismatch_count == 0


def verify_cross_layer_provenance(
    plan: DistributedPlan, result: DistributedExecutionResult, workload_hidden_dim: int,
) -> CrossLayerProvenanceReport:
    details: dict[str, Any] = {}

    planned_operator_id = plan.tensor_shards[0].tensor_id if plan.tensor_shards else None
    # Contribution messages are consumed by CollectiveCoordinator into
    # outcome.contributions -- they are not appended to trace.events (only
    # non-contribution passthrough events are), so this is the correct real
    # source for "what tensor_id did the runtime actually execute against".
    executed_operator_ids = {
        c.get("tensor_id")
        for o in result.collective_outcomes
        for c in o.contributions.values()
    }
    operator_id_match = bool(planned_operator_id) and executed_operator_ids == {planned_operator_id}
    details["planned_operator_id"] = planned_operator_id
    details["executed_operator_ids"] = sorted(x for x in executed_operator_ids if x)

    world_size_match = result.world_size == plan.world_size == len(result.processes)
    details["planned_world_size"] = plan.world_size
    details["executed_process_count"] = len(result.processes)

    planned_rank_ids = {r.rank_id for r in plan.ranks}
    executed_rank_ids = set(result.processes.keys())
    rank_ids_match = planned_rank_ids == executed_rank_ids
    details["planned_rank_ids"] = sorted(planned_rank_ids)
    details["executed_rank_ids"] = sorted(executed_rank_ids)

    planned_shard_widths = {s.shard_index: s.range_end - s.range_start for s in plan.tensor_shards}
    executed_shard_widths: dict[int, int] = {}
    for e in result.trace.events:
        if e.get("event") == "shard_received":
            a_shape = e.get("a_shape")
            if a_shape:
                executed_shard_widths[e["rank_id"]] = a_shape[-1]
    shard_ranges_match = bool(planned_shard_widths) and all(
        executed_shard_widths.get(idx) == width for idx, width in planned_shard_widths.items()
    )
    details["planned_shard_widths"] = planned_shard_widths
    details["executed_shard_widths"] = executed_shard_widths

    planned_collective = plan.collectives[0] if plan.collectives else None
    executed_collective_ids = {o.collective_id for o in result.collective_outcomes}
    executed_sequence_ids = {o.sequence_id for o in result.collective_outcomes}
    collective_id_match = (
        planned_collective is not None
        and executed_collective_ids == {planned_collective.collective_id}
    )
    sequence_id_match = (
        planned_collective is not None
        and executed_sequence_ids == {planned_collective.sequence_id}
    )
    planned_participants = set(planned_collective.participants) if planned_collective else set()
    executed_participants: set[int] = set()
    for o in result.collective_outcomes:
        executed_participants |= set(o.contributions.keys())
    participant_set_match = bool(planned_participants) and planned_participants == executed_participants
    details["planned_collective_id"] = planned_collective.collective_id if planned_collective else None
    details["executed_collective_ids"] = sorted(executed_collective_ids)
    details["planned_sequence_id"] = planned_collective.sequence_id if planned_collective else None
    details["executed_sequence_ids"] = sorted(executed_sequence_ids)
    details["planned_participants"] = sorted(planned_participants)
    details["executed_participants"] = sorted(executed_participants)

    no_silent_downgrade = result.world_size == plan.world_size and plan.world_size > 1

    plan_declared_hidden_dim = max((s.range_end for s in plan.tensor_shards), default=0)
    no_synthetic_fallback_dimensions = (
        workload_hidden_dim == plan_declared_hidden_dim and plan_declared_hidden_dim > 0
    )
    details["plan_declared_hidden_dim"] = plan_declared_hidden_dim
    details["workload_hidden_dim_used"] = workload_hidden_dim

    checks = [
        operator_id_match, world_size_match, rank_ids_match, shard_ranges_match,
        collective_id_match, sequence_id_match, participant_set_match,
        no_silent_downgrade, no_synthetic_fallback_dimensions,
    ]
    mismatch_count = sum(1 for c in checks if not c)

    return CrossLayerProvenanceReport(
        operator_id_match=operator_id_match,
        world_size_match=world_size_match,
        rank_ids_match=rank_ids_match,
        shard_ranges_match=shard_ranges_match,
        collective_id_match=collective_id_match,
        sequence_id_match=sequence_id_match,
        participant_set_match=participant_set_match,
        no_silent_downgrade=no_silent_downgrade,
        no_synthetic_fallback_dimensions=no_synthetic_fallback_dimensions,
        mismatch_count=mismatch_count,
        details=details,
    )
