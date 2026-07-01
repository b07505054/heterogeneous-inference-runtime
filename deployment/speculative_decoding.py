"""Runtime-side speculative decoding planning types and decisions.

Speculative decoding is a runtime decode strategy. This module defines the
typed planning surface for one speculative cycle, but it does not execute draft
or target models and does not mutate distributed runtime plans or traces.

Truth boundaries:
  "speculative_decoding_cost_model_not_measured_acceptance"
  "speculative_runtime_context_static_simulation_state"
  "speculative_decoding_uses_existing_runtime_decode_timing"
  "speculative_decoding_simulated_not_real_model_execution"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

TB_SPECULATIVE_COST_MODEL = "speculative_decoding_cost_model_not_measured_acceptance"
TB_SPECULATIVE_CONFIG = "speculative_decoding_config_static_runtime_policy"
TB_SPECULATIVE_CONTEXT = "speculative_runtime_context_static_simulation_state"
TB_SPECULATIVE_EXISTING_DECODE_TIMING = (
    "speculative_decoding_uses_existing_runtime_decode_timing"
)
TB_SPECULATIVE_SIMULATION = "speculative_decoding_simulated_not_real_model_execution"

BASELINE_DECODE_STAGE_SERVICE_MS = "decode_stage_service_ms"
_EPSILON = 1e-6


class SpeculativeDecodingState(Enum):
    """States for one speculative decoding cycle."""

    IDLE = "idle"
    DRAFTING = "drafting"
    VERIFYING = "verifying"
    ACCEPTING = "accepting"
    RECOVERING = "recovering"
    COMMITTING = "committing"
    NEXT_CYCLE = "next_cycle"


@dataclass(frozen=True)
class SpeculativeRuntimeContext:
    """Runtime state available when planning a speculative decode cycle."""

    request_id: str
    decode_iteration: int
    current_sequence_length: int
    generated_tokens_so_far: int
    remaining_token_budget: int
    target_kv_ready: bool
    draft_kv_ready: bool
    backend: str
    device_load_estimate: float | None = None
    scheduler_budget_ms: float | None = None
    truth_boundary: str = TB_SPECULATIVE_CONTEXT


@dataclass(frozen=True)
class SpeculativeDecodingConfig:
    """Static runtime policy for speculative decoding estimates."""

    enabled: bool
    draft_model_name: str
    target_model_name: str
    draft_length: int
    draft_model_decode_ms_per_token: float
    target_verify_ms_per_draft_token: float
    expected_acceptance_rate: float
    draft_kv_overhead_ms: float = 0.0
    target_kv_commit_ms: float = 0.0
    acceptance_check_ms: float = 0.0
    correction_token_ms: float = 0.0
    rollback_cost_ms: float = 0.0
    output_commit_ms: float = 0.0
    coordinator_overhead_ms: float = 0.0
    truth_boundary: str = TB_SPECULATIVE_CONFIG


@dataclass(frozen=True)
class SpeculativeCyclePlan:
    """Estimated plan for one speculative decoding cycle."""

    state_sequence: tuple[SpeculativeDecodingState, ...]
    runtime_context: SpeculativeRuntimeContext
    draft_length: int
    expected_acceptance_rate: float
    expected_accepted_tokens: float
    expected_rejected_tokens: float
    expected_correction_token_after_reject: float
    expected_progress_tokens: float
    baseline_decode_service_ms: float
    baseline_source: str
    normal_cost_for_same_progress_ms: float
    draft_generation_cost_ms: float
    draft_kv_overhead_ms: float
    target_verification_cost_ms: float
    target_kv_commit_ms: float
    acceptance_check_ms: float
    correction_cost_ms: float
    rollback_cost_ms: float
    output_commit_ms: float
    coordinator_overhead_ms: float
    estimated_cycle_cost_ms: float
    estimated_speedup: float
    truth_boundary: str


@dataclass(frozen=True)
class SpeculativeCycleResult:
    """Future result type for simulated or real speculative execution.

    The evaluator intentionally does not create this object. Observed metrics
    should only be populated by a future coordinator that actually runs or
    simulates a speculative cycle.
    """

    cycle_id: str
    draft_tokens: tuple[int, ...]
    accepted_tokens: tuple[int, ...]
    rejected_tokens: tuple[int, ...]
    correction_token: int | None
    accepted_length: int
    reject_position: int | None
    draft_latency_ms: float
    verify_latency_ms: float
    recovery_latency_ms: float
    commit_latency_ms: float
    total_cycle_ms: float
    observed_acceptance_rate: float
    observed_speedup: float | None
    fallback_reason: str | None
    next_state: SpeculativeDecodingState
    truth_boundary: str = TB_SPECULATIVE_SIMULATION


@dataclass(frozen=True)
class SpeculativeDecodingDecision:
    """Runtime decision for whether speculative decoding should be selected."""

    enabled: bool
    selected: bool
    reason: str
    plan: SpeculativeCyclePlan | None
    truth_boundary: str


class SpeculativeDecodingEvaluator:
    """Pure runtime-side speculative decoding selector."""

    @staticmethod
    def evaluate(
        config: SpeculativeDecodingConfig,
        *,
        baseline_decode_service_ms: float,
        runtime_context: SpeculativeRuntimeContext,
    ) -> SpeculativeDecodingDecision:
        """Return a speculative decoding decision from static estimates.

        baseline_decode_service_ms is the normal DecodeStage.service_ms value.
        It must not include queue wait, KV transfer, handoff, or total request
        latency. The plan is selected only when the estimated speculative cycle
        cost is lower than normal target-model decode for the same expected
        progress.
        """
        if not config.enabled:
            return _disabled("disabled_by_config")

        if baseline_decode_service_ms <= 0.0:
            return _disabled("invalid_baseline_decode_service_ms")

        if config.draft_length <= 0:
            return _disabled("invalid_draft_length")

        if config.expected_acceptance_rate <= 0.0:
            return _disabled("non_positive_acceptance_rate")

        expected_accepted = config.draft_length * config.expected_acceptance_rate
        expected_rejected = max(config.draft_length - expected_accepted, 0.0)
        expected_correction = 1.0 if expected_rejected > 0.0 else 0.0
        expected_progress = expected_accepted + expected_correction

        normal_cost = baseline_decode_service_ms * max(expected_progress, 1.0)
        draft_generation = (
            config.draft_length * config.draft_model_decode_ms_per_token
        )
        target_verification = (
            config.draft_length * config.target_verify_ms_per_draft_token
        )
        draft_kv_overhead = max(config.draft_kv_overhead_ms, 0.0)
        target_kv_commit = max(config.target_kv_commit_ms, 0.0)
        acceptance_check = max(config.acceptance_check_ms, 0.0)
        correction_cost = expected_correction * max(config.correction_token_ms, 0.0)
        rollback_cost = max(config.rollback_cost_ms, 0.0)
        output_commit = max(config.output_commit_ms, 0.0)
        coordinator_overhead = max(config.coordinator_overhead_ms, 0.0)
        estimated_cycle_cost = (
            draft_generation
            + draft_kv_overhead
            + target_verification
            + target_kv_commit
            + acceptance_check
            + correction_cost
            + rollback_cost
            + output_commit
            + coordinator_overhead
        )
        estimated_speedup = normal_cost / max(estimated_cycle_cost, _EPSILON)

        plan = SpeculativeCyclePlan(
            state_sequence=_state_sequence(expected_rejected),
            runtime_context=runtime_context,
            draft_length=config.draft_length,
            expected_acceptance_rate=config.expected_acceptance_rate,
            expected_accepted_tokens=expected_accepted,
            expected_rejected_tokens=expected_rejected,
            expected_correction_token_after_reject=expected_correction,
            expected_progress_tokens=expected_progress,
            baseline_decode_service_ms=baseline_decode_service_ms,
            baseline_source=BASELINE_DECODE_STAGE_SERVICE_MS,
            normal_cost_for_same_progress_ms=normal_cost,
            draft_generation_cost_ms=draft_generation,
            draft_kv_overhead_ms=draft_kv_overhead,
            target_verification_cost_ms=target_verification,
            target_kv_commit_ms=target_kv_commit,
            acceptance_check_ms=acceptance_check,
            correction_cost_ms=correction_cost,
            rollback_cost_ms=rollback_cost,
            output_commit_ms=output_commit,
            coordinator_overhead_ms=coordinator_overhead,
            estimated_cycle_cost_ms=estimated_cycle_cost,
            estimated_speedup=estimated_speedup,
            truth_boundary=TB_SPECULATIVE_EXISTING_DECODE_TIMING,
        )

        if estimated_cycle_cost >= normal_cost:
            return SpeculativeDecodingDecision(
                enabled=True,
                selected=False,
                reason="speculative_cost_not_lower_than_baseline",
                plan=plan,
                truth_boundary=TB_SPECULATIVE_COST_MODEL,
            )

        return SpeculativeDecodingDecision(
            enabled=True,
            selected=True,
            reason="speculative_cost_model_profitable",
            plan=plan,
            truth_boundary=TB_SPECULATIVE_COST_MODEL,
        )


def _state_sequence(
    expected_rejected_tokens: float,
) -> tuple[SpeculativeDecodingState, ...]:
    states = [
        SpeculativeDecodingState.IDLE,
        SpeculativeDecodingState.DRAFTING,
        SpeculativeDecodingState.VERIFYING,
        SpeculativeDecodingState.ACCEPTING,
    ]
    if expected_rejected_tokens > 0.0:
        states.append(SpeculativeDecodingState.RECOVERING)
    states.extend(
        [
            SpeculativeDecodingState.COMMITTING,
            SpeculativeDecodingState.NEXT_CYCLE,
        ]
    )
    return tuple(states)


def _disabled(reason: str) -> SpeculativeDecodingDecision:
    return SpeculativeDecodingDecision(
        enabled=False,
        selected=False,
        reason=reason,
        plan=None,
        truth_boundary=TB_SPECULATIVE_COST_MODEL,
    )
