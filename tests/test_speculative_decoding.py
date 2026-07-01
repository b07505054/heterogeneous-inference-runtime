"""Tests for standalone speculative decoding planning and decisions."""

from dataclasses import FrozenInstanceError

import pytest

from deployment.speculative_decoding import (
    BASELINE_DECODE_STAGE_SERVICE_MS,
    TB_SPECULATIVE_CONFIG,
    TB_SPECULATIVE_CONTEXT,
    TB_SPECULATIVE_COST_MODEL,
    TB_SPECULATIVE_EXISTING_DECODE_TIMING,
    TB_SPECULATIVE_SIMULATION,
    SpeculativeCycleResult,
    SpeculativeDecodingConfig,
    SpeculativeDecodingEvaluator,
    SpeculativeDecodingState,
    SpeculativeRuntimeContext,
)


def _config(**overrides):
    fields = {
        "enabled": True,
        "draft_model_name": "tiny-draft",
        "target_model_name": "target-llm",
        "draft_length": 4,
        "draft_model_decode_ms_per_token": 0.2,
        "target_verify_ms_per_draft_token": 0.2,
        "expected_acceptance_rate": 0.75,
        "draft_kv_overhead_ms": 0.1,
        "target_kv_commit_ms": 0.1,
        "acceptance_check_ms": 0.05,
        "correction_token_ms": 0.5,
        "rollback_cost_ms": 0.1,
        "output_commit_ms": 0.05,
        "coordinator_overhead_ms": 0.1,
    }
    fields.update(overrides)
    return SpeculativeDecodingConfig(**fields)


def _context(**overrides):
    fields = {
        "request_id": "req-1",
        "decode_iteration": 7,
        "current_sequence_length": 128,
        "generated_tokens_so_far": 16,
        "remaining_token_budget": 32,
        "target_kv_ready": True,
        "draft_kv_ready": True,
        "backend": "metal",
        "device_load_estimate": 0.4,
        "scheduler_budget_ms": 8.0,
    }
    fields.update(overrides)
    return SpeculativeRuntimeContext(**fields)


def _evaluate(config=None, **kwargs):
    return SpeculativeDecodingEvaluator.evaluate(
        config or _config(),
        baseline_decode_service_ms=kwargs.pop("baseline_decode_service_ms", 1.0),
        runtime_context=kwargs.pop("runtime_context", _context()),
    )


def test_disabled_config_returns_normal_decode_decision():
    decision = _evaluate(_config(enabled=False))

    assert decision.enabled is False
    assert decision.selected is False
    assert decision.reason == "disabled_by_config"
    assert decision.plan is None
    assert decision.truth_boundary == TB_SPECULATIVE_COST_MODEL


def test_invalid_draft_length_rejects_speculative_decoding():
    decision = _evaluate(_config(draft_length=0))

    assert decision.selected is False
    assert decision.reason == "invalid_draft_length"
    assert decision.plan is None


def test_non_positive_acceptance_rejects_speculative_decoding():
    decision = _evaluate(_config(expected_acceptance_rate=0.0))

    assert decision.selected is False
    assert decision.reason == "non_positive_acceptance_rate"
    assert decision.plan is None


def test_missing_or_zero_baseline_rejects_speculative_decoding():
    decision = _evaluate(baseline_decode_service_ms=0.0)

    assert decision.selected is False
    assert decision.reason == "invalid_baseline_decode_service_ms"
    assert decision.plan is None


def test_lower_estimated_cycle_cost_selects_speculative():
    decision = _evaluate()

    assert decision.enabled is True
    assert decision.selected is True
    assert decision.reason == "speculative_cost_model_profitable"
    assert decision.plan is not None

    plan = decision.plan
    assert plan.draft_generation_cost_ms == pytest.approx(0.8)
    assert plan.draft_kv_overhead_ms == pytest.approx(0.1)
    assert plan.target_verification_cost_ms == pytest.approx(0.8)
    assert plan.target_kv_commit_ms == pytest.approx(0.1)
    assert plan.acceptance_check_ms == pytest.approx(0.05)
    assert plan.correction_cost_ms == pytest.approx(0.5)
    assert plan.rollback_cost_ms == pytest.approx(0.1)
    assert plan.output_commit_ms == pytest.approx(0.05)
    assert plan.coordinator_overhead_ms == pytest.approx(0.1)
    assert plan.estimated_cycle_cost_ms == pytest.approx(2.6)
    assert plan.expected_accepted_tokens == pytest.approx(3.0)
    assert plan.expected_rejected_tokens == pytest.approx(1.0)
    assert plan.expected_correction_token_after_reject == pytest.approx(1.0)
    assert plan.expected_progress_tokens == pytest.approx(4.0)
    assert plan.normal_cost_for_same_progress_ms == pytest.approx(4.0)
    assert plan.estimated_speedup == pytest.approx(4.0 / 2.6)


def test_higher_estimated_cycle_cost_rejects_speculative():
    decision = _evaluate(
        _config(
            draft_model_decode_ms_per_token=0.8,
            target_verify_ms_per_draft_token=0.8,
        )
    )

    assert decision.enabled is True
    assert decision.selected is False
    assert decision.reason == "speculative_cost_not_lower_than_baseline"
    assert decision.plan is not None
    assert (
        decision.plan.estimated_cycle_cost_ms
        >= decision.plan.normal_cost_for_same_progress_ms
    )


def test_state_sequence_without_expected_rejection_skips_recovering():
    decision = _evaluate(_config(expected_acceptance_rate=1.0))

    assert decision.plan is not None
    assert decision.plan.state_sequence == (
        SpeculativeDecodingState.IDLE,
        SpeculativeDecodingState.DRAFTING,
        SpeculativeDecodingState.VERIFYING,
        SpeculativeDecodingState.ACCEPTING,
        SpeculativeDecodingState.COMMITTING,
        SpeculativeDecodingState.NEXT_CYCLE,
    )


def test_state_sequence_with_expected_rejection_includes_recovering():
    decision = _evaluate(_config(expected_acceptance_rate=0.5))

    assert decision.plan is not None
    assert decision.plan.state_sequence == (
        SpeculativeDecodingState.IDLE,
        SpeculativeDecodingState.DRAFTING,
        SpeculativeDecodingState.VERIFYING,
        SpeculativeDecodingState.ACCEPTING,
        SpeculativeDecodingState.RECOVERING,
        SpeculativeDecodingState.COMMITTING,
        SpeculativeDecodingState.NEXT_CYCLE,
    )


def test_runtime_context_is_carried_into_plan_and_frozen():
    context = _context(request_id="req-frozen", decode_iteration=3)
    decision = _evaluate(runtime_context=context)

    assert decision.plan is not None
    assert decision.plan.runtime_context is context
    assert decision.plan.runtime_context.truth_boundary == TB_SPECULATIVE_CONTEXT

    with pytest.raises(FrozenInstanceError):
        context.decode_iteration = 4


def test_baseline_uses_decode_stage_service_ms_only():
    decision = _evaluate(baseline_decode_service_ms=1.25)

    assert decision.plan is not None
    assert decision.plan.baseline_decode_service_ms == pytest.approx(1.25)
    assert decision.plan.baseline_source == BASELINE_DECODE_STAGE_SERVICE_MS
    assert decision.plan.truth_boundary == TB_SPECULATIVE_EXISTING_DECODE_TIMING


def test_truth_boundaries_present_on_config_decision_context_and_plan():
    cfg = _config()
    context = _context()

    assert cfg.truth_boundary == TB_SPECULATIVE_CONFIG
    assert context.truth_boundary == TB_SPECULATIVE_CONTEXT

    decision = _evaluate(cfg, runtime_context=context)
    assert decision.truth_boundary == TB_SPECULATIVE_COST_MODEL
    assert decision.plan is not None
    assert decision.plan.truth_boundary == TB_SPECULATIVE_EXISTING_DECODE_TIMING
    assert SpeculativeCycleResult.__dataclass_fields__["truth_boundary"].default == (
        TB_SPECULATIVE_SIMULATION
    )


def test_negative_optional_costs_are_clamped_to_zero():
    decision = _evaluate(
        _config(
            draft_kv_overhead_ms=-1.0,
            target_kv_commit_ms=-1.0,
            acceptance_check_ms=-1.0,
            correction_token_ms=-1.0,
            rollback_cost_ms=-1.0,
            output_commit_ms=-1.0,
            coordinator_overhead_ms=-1.0,
        )
    )

    assert decision.plan is not None
    assert decision.plan.draft_kv_overhead_ms == pytest.approx(0.0)
    assert decision.plan.target_kv_commit_ms == pytest.approx(0.0)
    assert decision.plan.acceptance_check_ms == pytest.approx(0.0)
    assert decision.plan.correction_cost_ms == pytest.approx(0.0)
    assert decision.plan.rollback_cost_ms == pytest.approx(0.0)
    assert decision.plan.output_commit_ms == pytest.approx(0.0)
    assert decision.plan.coordinator_overhead_ms == pytest.approx(0.0)


def test_evaluator_does_not_produce_cycle_result_or_observed_speedup():
    decision = _evaluate()

    assert decision.plan is not None
    assert not hasattr(decision, "result")
    assert not hasattr(decision.plan, "observed_speedup")
    assert decision.plan.estimated_speedup > 1.0


def test_phase_two_step_one_has_no_distributed_plan_dependency():
    import deployment.speculative_decoding as speculative_decoding

    assert "distributed_runtime_plan" not in speculative_decoding.__dict__
    assert "DecodeStage" not in speculative_decoding.__dict__
    assert "DistributedRuntimePlan" not in speculative_decoding.__dict__
