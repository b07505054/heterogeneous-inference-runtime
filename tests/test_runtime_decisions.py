"""Tests for SchedulingDecision, MemoryDecision, ReplayDecision and their evaluators."""

import math

import pytest

from deployment.execution_plan_v2.loader import parse_execution_plan_v2
from deployment.runtime_decisions import (
    MemoryDecisionEvaluator,
    ReplayDecisionEvaluator,
    SchedulingDecisionEvaluator,
)

# ---------------------------------------------------------------------------
# V2 fixtures
# ---------------------------------------------------------------------------

_V2_DECODE: dict = {
    "schema": "execution_plan",
    "schema_version": "2.0.0",
    "plan_id": "decode-decisions-plan",
    "provenance": {
        "compiler_tool": "test",
        "model_spec_ref": "profiles/models/test_model.json",
        "capability_bundle": {"hardware_profile_ref": "hardware/test_device.json"},
        "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
    },
    "model_identity": {"model_id": "test-model"},
    "global_decisions": {
        "quantization": {"strategy": "none", "dtype": "float16"},
        "memory": {
            "kv_cache_layout": "contiguous",
            "estimated_kv_peak_mb": 6.75,
            "memory_budget_fraction": 0.75,
            "truth_boundary": "static_formula_estimate_not_measured_memory",
        },
        "serving": {
            "topology": "colocated",
            "colocated_cost_estimate_ms": 4.8,
            "replay_eligible": True,
            "token_budget_per_step": 2048,
            "prefix_reuse_eligible": False,
            "chunked_prefill_eligible": False,
            "parallelism_kind": "none",
            "parallelism_degree": 1,
        },
    },
    "function_plans": [
        {
            "function_name": "decode",
            "serving_phase": "decode",
            "backend": {
                "decision_type": "BackendDecision",
                "scope": "Function",
                "selected_backend": "coreml_ane",
                "fallback_backends": ["arm_compute", "cpu"],
                "reason": "target_preferred",
            },
            "per_op_decisions": [],
        }
    ],
}

# 31.2ms cost, prefill, replay not eligible
_V2_PREFILL: dict = {
    "schema": "execution_plan",
    "schema_version": "2.0.0",
    "plan_id": "prefill-decisions-plan",
    "provenance": {
        "compiler_tool": "test",
        "model_spec_ref": "profiles/models/test_model.json",
        "capability_bundle": {"hardware_profile_ref": "hardware/test_device.json"},
        "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
    },
    "model_identity": {"model_id": "test-model"},
    "global_decisions": {
        "quantization": {"strategy": "none", "dtype": "float16"},
        "memory": {
            "kv_cache_layout": "paged",
            "estimated_kv_peak_mb": 6.75,
            "memory_budget_fraction": 0.0,
            "truth_boundary": "static_formula_estimate_not_measured_memory",
        },
        "serving": {
            "topology": "colocated",
            "colocated_cost_estimate_ms": 31.2,
            "replay_eligible": False,
            "token_budget_per_step": 0,
            "prefix_reuse_eligible": False,
            "chunked_prefill_eligible": False,
            "parallelism_kind": "none",
            "parallelism_degree": 1,
        },
    },
    "function_plans": [
        {
            "function_name": "prefill",
            "serving_phase": "prefill",
            "backend": {
                "decision_type": "BackendDecision",
                "scope": "Function",
                "selected_backend": "cpu",
                "fallback_backends": [],
                "reason": "constraint_conflict",
            },
            "per_op_decisions": [],
        }
    ],
}

_V2_UNKNOWN_LAYOUT: dict = {
    **_V2_PREFILL,
    "plan_id": "unknown-layout-plan",
    "global_decisions": {
        **_V2_PREFILL["global_decisions"],
        "memory": {
            "kv_cache_layout": "striped",
            "estimated_kv_peak_mb": 4.0,
            "memory_budget_fraction": 0.0,
            "truth_boundary": "static_formula_estimate_not_measured_memory",
        },
    },
}


def _decode_plan():
    return parse_execution_plan_v2(_V2_DECODE)


def _prefill_plan():
    return parse_execution_plan_v2(_V2_PREFILL)


def _unknown_layout_plan():
    return parse_execution_plan_v2(_V2_UNKNOWN_LAYOUT)


# ---------------------------------------------------------------------------
# SchedulingDecision tests
# ---------------------------------------------------------------------------

def test_scheduling_decision_uses_compiler_cost_and_confidence():
    plan = _prefill_plan()
    fp = plan.function_plans[0]
    serving = plan.global_decisions.serving
    decision = SchedulingDecisionEvaluator.evaluate(fp, serving)
    assert decision.compiler_cost_ms == pytest.approx(31.2)
    assert decision.confidence == "unknown"
    assert decision.priority == "normal"
    assert decision.execution_policy == "colocated"


def test_scheduling_decision_truth_boundary():
    plan = _decode_plan()
    fp = plan.function_plans[0]
    serving = plan.global_decisions.serving
    decision = SchedulingDecisionEvaluator.evaluate(fp, serving)
    assert decision.truth_boundary == "compiler_cost_estimate_not_measured_latency"


# ---------------------------------------------------------------------------
# MemoryDecision tests
# ---------------------------------------------------------------------------

def test_memory_decision_copies_kv_layout():
    plan = _decode_plan()
    decision = MemoryDecisionEvaluator.evaluate(plan.global_decisions.memory)
    assert decision.kv_layout_used == "contiguous"
    assert decision.estimated_mb_from_compiler == pytest.approx(6.75)
    assert decision.allocator_kind == "contiguous"
    assert decision.admitted is True
    assert decision.rejection_reason == ""


def test_memory_decision_estimates_page_budget():
    plan = _decode_plan()
    decision = MemoryDecisionEvaluator.evaluate(plan.global_decisions.memory)
    # ceil(6.75 / 1.0) = 7
    assert decision.page_budget_estimate == math.ceil(6.75 / 1.0)


def test_memory_decision_truth_boundary_verbatim():
    plan = _prefill_plan()
    decision = MemoryDecisionEvaluator.evaluate(plan.global_decisions.memory)
    assert decision.truth_boundary == "static_formula_estimate_not_measured_memory"


# ---------------------------------------------------------------------------
# ReplayDecision tests
# ---------------------------------------------------------------------------

def test_replay_decision_does_not_claim_capture():
    plan = _decode_plan()
    fp = plan.function_plans[0]
    serving = plan.global_decisions.serving
    decision = ReplayDecisionEvaluator.evaluate(fp, serving)
    assert decision.replay_eligible_from_compiler is True
    assert decision.replay_requested is True
    assert decision.bucket == "decode_static"
    assert decision.capture_attempted is False
    assert decision.captured is False
    assert decision.skipped_reason == "capture_not_implemented"


def test_replay_decision_not_eligible():
    plan = _prefill_plan()
    fp = plan.function_plans[0]
    serving = plan.global_decisions.serving
    decision = ReplayDecisionEvaluator.evaluate(fp, serving)
    assert decision.replay_eligible_from_compiler is False
    assert decision.replay_requested is False
    assert decision.captured is False
    assert decision.skipped_reason == "not_eligible_per_compiler_plan"


def test_replay_decision_truth_boundary_verbatim():
    plan = _decode_plan()
    fp = plan.function_plans[0]
    serving = plan.global_decisions.serving
    decision = ReplayDecisionEvaluator.evaluate(fp, serving)
    assert decision.truth_boundary == "static_shape_replay_eligibility_not_cuda_graph_capture"


# ---------------------------------------------------------------------------
# allocator_kind edge case
# ---------------------------------------------------------------------------

def test_unknown_kv_layout_maps_to_unknown_allocator():
    plan = _unknown_layout_plan()
    decision = MemoryDecisionEvaluator.evaluate(plan.global_decisions.memory)
    assert decision.kv_layout_used == "striped"
    assert decision.allocator_kind == "unknown"
