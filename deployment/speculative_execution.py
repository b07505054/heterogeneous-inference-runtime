"""Simulated speculative decoding execution lifecycle helpers.

This module turns a selected speculative decoding decision into deterministic
runtime sub-stage simulations. It does not execute draft or target models, does
not mutate KV cache, and does not produce observed speedup metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from deployment.speculative_decoding import (
    TB_SPECULATIVE_SIMULATION,
    SpeculativeDecodingDecision,
    SpeculativeDecodingState,
)


@dataclass(frozen=True)
class SpeculativeStageSimulation:
    """One simulated speculative decode sub-stage."""

    stage_name: str
    category: str
    duration_ms: float
    truth_boundary: str


class SpeculativeExecutionSimulator:
    """Builds simulated speculative decode sub-stages from a selected decision."""

    @staticmethod
    def build_stages(
        decision: SpeculativeDecodingDecision,
    ) -> tuple[SpeculativeStageSimulation, ...]:
        if not decision.selected or decision.plan is None:
            raise ValueError(
                "SpeculativeExecutionSimulator requires a selected decision with a plan"
            )

        plan = decision.plan
        stages = [
            SpeculativeStageSimulation(
                stage_name="speculative_draft",
                category="decode",
                duration_ms=plan.draft_generation_cost_ms + plan.draft_kv_overhead_ms,
                truth_boundary=TB_SPECULATIVE_SIMULATION,
            ),
            SpeculativeStageSimulation(
                stage_name="speculative_verify",
                category="decode",
                duration_ms=plan.target_verification_cost_ms,
                truth_boundary=TB_SPECULATIVE_SIMULATION,
            ),
            SpeculativeStageSimulation(
                stage_name="speculative_accept",
                category="decode",
                duration_ms=plan.acceptance_check_ms,
                truth_boundary=TB_SPECULATIVE_SIMULATION,
            ),
        ]

        has_recovery = SpeculativeDecodingState.RECOVERING in plan.state_sequence
        if has_recovery:
            stages.append(
                SpeculativeStageSimulation(
                    stage_name="speculative_recover",
                    category="decode",
                    duration_ms=plan.correction_cost_ms + plan.rollback_cost_ms,
                    truth_boundary=TB_SPECULATIVE_SIMULATION,
                )
            )

        stages.append(
            SpeculativeStageSimulation(
                stage_name="speculative_commit",
                category="decode",
                duration_ms=(
                    plan.target_kv_commit_ms
                    + plan.output_commit_ms
                    + plan.coordinator_overhead_ms
                    + (0.0 if has_recovery else plan.rollback_cost_ms)
                ),
                truth_boundary=TB_SPECULATIVE_SIMULATION,
            )
        )

        total_ms = sum(stage.duration_ms for stage in stages)
        if abs(total_ms - plan.estimated_cycle_cost_ms) > 1e-9:
            raise ValueError(
                "speculative sub-stage durations do not sum to estimated cycle cost"
            )

        return tuple(stages)


class SpeculativeCoordinator:
    """Owns the decision boundary for simulated speculative decode execution."""

    @staticmethod
    def should_use_speculative(
        decision: SpeculativeDecodingDecision | None,
    ) -> bool:
        return decision is not None and decision.selected

    @staticmethod
    def build_simulated_stages(
        decision: SpeculativeDecodingDecision,
    ) -> tuple[SpeculativeStageSimulation, ...]:
        return SpeculativeExecutionSimulator.build_stages(decision)
