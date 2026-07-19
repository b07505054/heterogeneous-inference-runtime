"""D3B Part O: cross-layer provenance counters.

Every counter here is COMPUTED by comparing the materialized launch spec
back against its declared sources (the D2/D3A compiler plan, the D3B
candidate selection, the installed vLLM argument registry, and process
launch bookkeeping) -- never hardcoded to zero. A successful materialization
run requires every counter to equal zero; a preflight rejection is not
itself a provenance mismatch and must not be counted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deployment.vllm_adapter.distributed_launch_spec import FieldSource


@dataclass(frozen=True)
class ProvenanceCounters:
    source_plan_mismatch_count: int
    candidate_mismatch_count: int
    model_mismatch_count: int
    tp_mismatch_count: int
    pp_mismatch_count: int
    world_size_mismatch_count: int
    rank_placement_mismatch_count: int
    unsupported_argument_count: int
    silent_default_count: int
    silent_downgrade_count: int
    preflight_bypass_count: int
    unexpected_process_launch_count: int
    orphan_process_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_plan_mismatch_count": self.source_plan_mismatch_count,
            "candidate_mismatch_count": self.candidate_mismatch_count,
            "model_mismatch_count": self.model_mismatch_count,
            "tp_mismatch_count": self.tp_mismatch_count,
            "pp_mismatch_count": self.pp_mismatch_count,
            "world_size_mismatch_count": self.world_size_mismatch_count,
            "rank_placement_mismatch_count": self.rank_placement_mismatch_count,
            "unsupported_argument_count": self.unsupported_argument_count,
            "silent_default_count": self.silent_default_count,
            "silent_downgrade_count": self.silent_downgrade_count,
            "preflight_bypass_count": self.preflight_bypass_count,
            "unexpected_process_launch_count": self.unexpected_process_launch_count,
            "orphan_process_count": self.orphan_process_count,
            "all_zero": self.all_zero(),
        }

    def all_zero(self) -> bool:
        return all(
            v == 0
            for v in (
                self.source_plan_mismatch_count,
                self.candidate_mismatch_count,
                self.model_mismatch_count,
                self.tp_mismatch_count,
                self.pp_mismatch_count,
                self.world_size_mismatch_count,
                self.rank_placement_mismatch_count,
                self.unsupported_argument_count,
                self.silent_default_count,
                self.silent_downgrade_count,
                self.preflight_bypass_count,
                self.unexpected_process_launch_count,
                self.orphan_process_count,
            )
        )


def compute_provenance_counters(
    *,
    plan_id: str,
    spec_source_execution_plan_id: str,
    selected_candidate_id: str,
    spec_source_candidate_id: str,
    expected_model_id: str,
    spec_model: str,
    plan_tensor_parallel_size: int,
    spec_tensor_parallel_size: int,
    plan_pipeline_parallel_size: int,
    spec_pipeline_parallel_size: int,
    plan_world_size: int,
    spec_world_size: int,
    plan_rank_ids: tuple[int, ...],
    spec_rank_ids: tuple[int, ...],
    unsupported_arguments: tuple[str, ...],
    field_provenance: dict[str, Any],
    execution_readiness_state: str,
    preflight_passed: bool,
    subprocess_launch_attempts_for_rejected_specs: int,
    tracked_pids_still_alive: tuple[int, ...],
) -> ProvenanceCounters:
    source_plan_mismatch_count = 0 if spec_source_execution_plan_id == plan_id else 1
    candidate_mismatch_count = 0 if spec_source_candidate_id == selected_candidate_id else 1
    model_mismatch_count = 0 if spec_model == expected_model_id else 1
    tp_mismatch_count = 0 if spec_tensor_parallel_size == plan_tensor_parallel_size else 1
    pp_mismatch_count = 0 if spec_pipeline_parallel_size == plan_pipeline_parallel_size else 1
    world_size_mismatch_count = 0 if spec_world_size == plan_world_size else 1
    rank_placement_mismatch_count = 0 if tuple(sorted(plan_rank_ids)) == tuple(sorted(spec_rank_ids)) else 1

    unsupported_argument_count = len(unsupported_arguments)

    silent_default_count = sum(
        1
        for entry in field_provenance.values()
        if entry.source == FieldSource.EXPLICIT_D3B_DEFAULT and not entry.reason.strip()
    )

    # A downgrade would be the materializer emitting a smaller TP than the
    # plan declared (e.g. plan says TP=2, spec says TP=1). Equal or absent
    # is not a downgrade; the tp_mismatch_count above already flags any
    # difference generally, this one specifically flags the downgrade
    # direction so it reads unambiguously in the report.
    silent_downgrade_count = 1 if spec_tensor_parallel_size < plan_tensor_parallel_size else 0

    # A bypass would mean the readiness state advanced past what preflight
    # allows. Fail-closed: any state other than PREFLIGHT_REJECTED while
    # preflight did not pass counts as a bypass.
    preflight_bypass_count = 1 if (not preflight_passed and execution_readiness_state != "PREFLIGHT_REJECTED") else 0
    if execution_readiness_state in {"EXECUTION_READY", "EXECUTION_STARTED"}:
        preflight_bypass_count += 1

    unexpected_process_launch_count = subprocess_launch_attempts_for_rejected_specs
    orphan_process_count = len(tracked_pids_still_alive)

    return ProvenanceCounters(
        source_plan_mismatch_count=source_plan_mismatch_count,
        candidate_mismatch_count=candidate_mismatch_count,
        model_mismatch_count=model_mismatch_count,
        tp_mismatch_count=tp_mismatch_count,
        pp_mismatch_count=pp_mismatch_count,
        world_size_mismatch_count=world_size_mismatch_count,
        rank_placement_mismatch_count=rank_placement_mismatch_count,
        unsupported_argument_count=unsupported_argument_count,
        silent_default_count=silent_default_count,
        silent_downgrade_count=silent_downgrade_count,
        preflight_bypass_count=preflight_bypass_count,
        unexpected_process_launch_count=unexpected_process_launch_count,
        orphan_process_count=orphan_process_count,
    )
