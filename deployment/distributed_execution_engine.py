"""Simulated stage-by-stage execution of a DistributedRuntimePlan.

DistributedExecutionEngine.execute() walks the pre-computed stage timings in
DistributedRuntimePlan and produces a DistributedRuntimeResult containing one
DistributedStageResult per executed stage.

This is deterministic simulation only:
- No real sleep, wall clock, or time.sleep calls.
- No RPC, NCCL, or network I/O.
- No mutation of the input plan.
- All timings come from the plan; none are re-derived here.

Stage execution order:
  prefill_queue_wait → prefill_compute → kv_transfer → handoff
  → decode_queue_wait → decode_compute

decision_summary is not included in stage_results; it is a planner
artifact captured in the plan itself and exposed via cost_summary in the
trace, not a runtime execution stage.

Truth boundary:
  "distributed_runtime_execution_simulated_not_real_cluster_execution"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from deployment.distributed_runtime_plan import DistributedRuntimePlan
from deployment.speculative_execution import SpeculativeCoordinator

_TB_ENGINE = "distributed_runtime_execution_simulated_not_real_cluster_execution"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistributedStageResult:
    """Result for a single executed stage in the distributed pipeline.

    end_ms = start_ms + duration_ms.
    worker_id and backend are None for infrastructure stages (kv_transfer,
    handoff) and queue-wait stages (no compute is running during a wait).
    success is always True for simulated execution; it is included so
    downstream code can pattern-match on real vs simulated results uniformly.
    """

    stage_name: str
    category: str
    start_ms: float
    duration_ms: float
    end_ms: float
    worker_id: Optional[str]
    backend: Optional[str]
    success: bool
    truth_boundary: str


@dataclass(frozen=True)
class DistributedRuntimeResult:
    """Simulated execution result for a full prefill/decode distributed pipeline.

    total_latency_ms = stage_results[-1].end_ms.
    ttft_ms and tpot_ms are taken directly from the selected policy in
    decision_comparison; they are not re-derived from the stage timeline.
    stage_results contains exactly 6 entries (no decision_summary stage).

    truth_boundary = "distributed_runtime_execution_simulated_not_real_cluster_execution"
    """

    model_name: str
    target_profile_id: str
    selected_policy: str
    total_latency_ms: float
    ttft_ms: float
    tpot_ms: float
    stage_results: tuple[DistributedStageResult, ...]
    truth_boundary: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DistributedExecutionEngine:
    """Simulates stage-by-stage execution of a DistributedRuntimePlan.

    Reads timings from the plan and produces a DistributedRuntimeResult.
    Stages are walked in order; each stage's start_ms is the previous
    stage's end_ms. Does not sleep, use the wall clock, call RPC, or
    mutate the input plan.
    """

    def execute(self, plan: DistributedRuntimePlan) -> DistributedRuntimeResult:
        """Execute plan stages and return a DistributedRuntimeResult.

        Does not mutate plan. Returns the same result for the same plan.
        """
        cmp = plan.decision_comparison
        handoff_ms = cmp.pd_split.handoff_overhead_ms

        stages: list[DistributedStageResult] = []
        cursor = 0.0

        def _run(
            name: str,
            category: str,
            duration: float,
            worker_id: Optional[str],
            backend: Optional[str],
            truth_boundary: str,
        ) -> None:
            nonlocal cursor
            start = cursor
            end = start + duration
            stages.append(DistributedStageResult(
                stage_name=name,
                category=category,
                start_ms=start,
                duration_ms=duration,
                end_ms=end,
                worker_id=worker_id,
                backend=backend,
                success=True,
                truth_boundary=truth_boundary,
            ))
            cursor = end

        _run(
            "prefill_queue_wait", "prefill",
            plan.prefill.queue_wait_ms,
            plan.prefill.worker_id, None,
            plan.prefill.truth_boundary,
        )
        _run(
            "prefill_compute", "prefill",
            plan.prefill.service_ms,
            plan.prefill.worker_id, plan.prefill.backend,
            plan.prefill.truth_boundary,
        )
        _run(
            "kv_transfer", "kv_transfer",
            plan.kv_transfer.transfer_cost_ms,
            None, None,
            plan.kv_transfer.truth_boundary,
        )
        _run(
            "handoff", "handoff",
            handoff_ms,
            None, None,
            plan.truth_boundary,
        )
        _run(
            "decode_queue_wait", "decode",
            plan.decode.queue_wait_ms,
            plan.decode.worker_id, None,
            plan.decode.truth_boundary,
        )
        speculative_decision = plan.decode.speculative_decoding
        if SpeculativeCoordinator.should_use_speculative(speculative_decision):
            for speculative_stage in SpeculativeCoordinator.build_simulated_stages(
                speculative_decision
            ):
                _run(
                    speculative_stage.stage_name, speculative_stage.category,
                    speculative_stage.duration_ms,
                    plan.decode.worker_id, plan.decode.backend,
                    speculative_stage.truth_boundary,
                )
        else:
            _run(
                "decode_compute", "decode",
                plan.decode.service_ms,
                plan.decode.worker_id, plan.decode.backend,
                plan.decode.truth_boundary,
            )

        total_latency_ms = stages[-1].end_ms

        if cmp.selected_policy == "pd_split":
            ttft_ms = cmp.pd_split.ttft_ms
            tpot_ms = cmp.pd_split.tpot_ms
        else:
            ttft_ms = cmp.colocated.ttft_ms
            tpot_ms = cmp.colocated.tpot_ms

        return DistributedRuntimeResult(
            model_name=plan.model_name,
            target_profile_id=plan.target_profile_id,
            selected_policy=cmp.selected_policy,
            total_latency_ms=total_latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            stage_results=tuple(stages),
            truth_boundary=_TB_ENGINE,
        )
