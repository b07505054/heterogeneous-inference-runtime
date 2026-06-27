"""Tests for SchedulingDecision, MemoryDecision, ReplayDecision and their evaluators."""

import math

import pytest

from deployment.runtime_decisions import (
    MemoryDecisionEvaluator,
    ReplayDecisionEvaluator,
    SchedulingDecisionEvaluator,
)
from deployment.runtime_execution_plan import RuntimeExecutionPlanAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DECODE_ELIGIBLE: dict = {
    "function_name": "decode",
    "execution_mode": "colocated",
    "cost_summary": {
        "colocated_total_ms": 4.8,
        "pd_split_total_ms": 0.0,
        "confidence": "low",
        "policy": "colocated",
        "cost_source": "formula_synthetic",
    },
    "kv_plan": {
        "layout": "contiguous",
        "kv_byte_estimate_mb": 6.75,
        "layout_reason": "",
        "truth_boundary": "static_formula_estimate_not_measured_memory",
    },
    "replay_plan": {
        "replay_eligible": True,
        "cuda_graph_bucket": "decode_static",
        "override_reason": "",
        "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
    },
    "backend_execution_plan": {
        "primary_backend": "coreml",
        "fallback_chain": ["metal", "cpu"],
        "decision_source": "target_preferred",
        "required_precision": "fp16",
        "required_kv_layout": "contiguous",
        "requires_replay": True,
    },
    "provenance": {
        "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
        "cost_source": "formula_synthetic",
    },
    "source_passes": [
        "serving-phase-analysis",
        "kv-layout-planning",
        "replay-eligibility",
        "execution-provider-planning",
    ],
}

_PREFILL_NOT_ELIGIBLE: dict = {
    "function_name": "prefill",
    "execution_mode": "colocated",
    "cost_summary": {
        "colocated_total_ms": 31.2,
        "pd_split_total_ms": 0.0,
        "confidence": "low",
        "policy": "colocated",
        "cost_source": "formula_synthetic",
    },
    "kv_plan": {
        "layout": "paged",
        "kv_byte_estimate_mb": 6.75,
        "layout_reason": "",
        "truth_boundary": "static_formula_estimate_not_measured_memory",
    },
    "replay_plan": {
        "replay_eligible": False,
        "cuda_graph_bucket": "",
        "override_reason": "",
        "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
    },
    "backend_execution_plan": {
        "primary_backend": "cpu",
        "fallback_chain": [],
        "decision_source": "constraint_conflict",
        "required_precision": "fp16",
        "required_kv_layout": "paged",
        "requires_replay": False,
    },
    "provenance": {
        "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
        "cost_source": "formula_synthetic",
    },
    "source_passes": [
        "serving-phase-analysis",
        "kv-layout-planning",
        "replay-eligibility",
        "execution-provider-planning",
    ],
}

_UNKNOWN_LAYOUT: dict = {
    **_PREFILL_NOT_ELIGIBLE,
    "kv_plan": {
        "layout": "striped",
        "kv_byte_estimate_mb": 4.0,
        "layout_reason": "",
        "truth_boundary": "static_formula_estimate_not_measured_memory",
    },
}


# ---------------------------------------------------------------------------
# SchedulingDecision tests
# ---------------------------------------------------------------------------

def test_scheduling_decision_uses_compiler_cost_and_confidence():
    plan = RuntimeExecutionPlanAdapter.from_dict(_PREFILL_NOT_ELIGIBLE)
    decision = SchedulingDecisionEvaluator.evaluate(plan)
    assert decision.compiler_cost_ms == pytest.approx(31.2)
    assert decision.confidence == "low"
    assert decision.priority == "conservative"
    assert decision.execution_policy == "colocated"


def test_scheduling_decision_truth_boundary():
    plan = RuntimeExecutionPlanAdapter.from_dict(_DECODE_ELIGIBLE)
    decision = SchedulingDecisionEvaluator.evaluate(plan)
    assert decision.truth_boundary == "compiler_cost_estimate_not_measured_latency"


# ---------------------------------------------------------------------------
# MemoryDecision tests
# ---------------------------------------------------------------------------

def test_memory_decision_copies_kv_layout():
    plan = RuntimeExecutionPlanAdapter.from_dict(_DECODE_ELIGIBLE)
    decision = MemoryDecisionEvaluator.evaluate(plan)
    assert decision.kv_layout_used == "contiguous"
    assert decision.estimated_mb_from_compiler == pytest.approx(6.75)
    assert decision.allocator_kind == "contiguous"
    assert decision.admitted is True
    assert decision.rejection_reason == ""


def test_memory_decision_estimates_page_budget():
    plan = RuntimeExecutionPlanAdapter.from_dict(_DECODE_ELIGIBLE)
    decision = MemoryDecisionEvaluator.evaluate(plan)
    # ceil(6.75 / 1.0) = 7
    assert decision.page_budget_estimate == math.ceil(6.75 / 1.0)


def test_memory_decision_truth_boundary_verbatim():
    plan = RuntimeExecutionPlanAdapter.from_dict(_PREFILL_NOT_ELIGIBLE)
    decision = MemoryDecisionEvaluator.evaluate(plan)
    assert decision.truth_boundary == "static_formula_estimate_not_measured_memory"


# ---------------------------------------------------------------------------
# ReplayDecision tests
# ---------------------------------------------------------------------------

def test_replay_decision_does_not_claim_capture():
    plan = RuntimeExecutionPlanAdapter.from_dict(_DECODE_ELIGIBLE)
    decision = ReplayDecisionEvaluator.evaluate(plan)
    assert decision.replay_eligible_from_compiler is True
    assert decision.replay_requested is True
    assert decision.bucket == "decode_static"
    assert decision.capture_attempted is False
    assert decision.captured is False
    assert decision.skipped_reason == "capture_not_implemented"


def test_replay_decision_not_eligible():
    plan = RuntimeExecutionPlanAdapter.from_dict(_PREFILL_NOT_ELIGIBLE)
    decision = ReplayDecisionEvaluator.evaluate(plan)
    assert decision.replay_eligible_from_compiler is False
    assert decision.replay_requested is False
    assert decision.captured is False
    assert decision.skipped_reason == "not_eligible_per_compiler_plan"


def test_replay_decision_truth_boundary_verbatim():
    plan = RuntimeExecutionPlanAdapter.from_dict(_DECODE_ELIGIBLE)
    decision = ReplayDecisionEvaluator.evaluate(plan)
    assert decision.truth_boundary == "static_shape_replay_eligibility_not_cuda_graph_capture"


# ---------------------------------------------------------------------------
# allocator_kind edge case
# ---------------------------------------------------------------------------

def test_unknown_kv_layout_maps_to_unknown_allocator():
    plan = RuntimeExecutionPlanAdapter.from_dict(_UNKNOWN_LAYOUT)
    decision = MemoryDecisionEvaluator.evaluate(plan)
    assert decision.kv_layout_used == "striped"
    assert decision.allocator_kind == "unknown"
