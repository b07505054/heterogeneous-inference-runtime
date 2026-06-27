"""Tests for RuntimeExecutionPlan runtime IR and its adapter."""

import pytest

from deployment.runtime_execution_plan import (
    RuntimeExecutionPlan,
    RuntimeExecutionPlanAdapter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAGED_PREFILL_FIXTURE: dict = {
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

CONSTRAINED_DECODE_FIXTURE: dict = {
    "function_name": "decode_constrained",
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parse_complete_compiler_plan_dict():
    plan = RuntimeExecutionPlanAdapter.from_dict(PAGED_PREFILL_FIXTURE)
    assert plan.function_name == "prefill"
    assert plan.execution_policy == "colocated"
    assert plan.scheduling_policy.compiler_cost_ms == pytest.approx(31.2)
    assert plan.scheduling_policy.confidence == "low"
    assert plan.scheduling_policy.policy == "colocated"
    assert plan.memory_policy.kv_layout == "paged"
    assert plan.memory_policy.kv_byte_estimate_mb == pytest.approx(6.75)
    assert plan.memory_policy.truth_boundary == "static_formula_estimate_not_measured_memory"
    assert plan.replay_policy.eligible is False
    assert plan.replay_policy.cuda_graph_bucket == ""
    assert plan.replay_policy.truth_boundary == "static_shape_replay_eligibility_not_cuda_graph_capture"
    assert plan.backend_policy.primary_backend == "cpu"
    assert plan.backend_policy.compiler_decision_source == "constraint_conflict"
    assert plan.backend_policy.required_precision == "fp16"
    assert plan.backend_policy.required_kv_layout == "paged"
    assert plan.backend_policy.requires_replay is False
    assert plan.compiler_provenance.truth_boundary == "compiler_execution_provider_plan_not_runtime_dispatch"
    assert plan.compiler_provenance.cost_source == "formula_synthetic"
    assert len(plan.compiler_provenance.source_passes) == 4


def test_missing_optional_fields_default_safely():
    minimal = {"function_name": "fn"}
    plan = RuntimeExecutionPlanAdapter.from_dict(minimal)
    assert plan.function_name == "fn"
    assert plan.backend_policy.primary_backend == "cpu"
    assert plan.backend_policy.selected_backend == "cpu"
    assert plan.replay_policy.capture_claimed is False
    assert plan.scheduling_policy.compiler_cost_ms == pytest.approx(0.0)
    assert plan.memory_policy.kv_layout == "unknown"


def test_backend_primary_fallback_mapping():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    assert plan.backend_policy.primary_backend == "coreml"
    assert plan.backend_policy.fallback_chain == ["metal", "cpu"]
    assert plan.backend_policy.selected_backend == "coreml"
    assert plan.backend_policy.backend_state == "planned"
    assert plan.backend_policy.runtime_override_reason == ""


def test_low_confidence_maps_to_conservative_priority():
    plan = RuntimeExecutionPlanAdapter.from_dict(PAGED_PREFILL_FIXTURE)
    assert plan.scheduling_policy.confidence == "low"
    assert plan.scheduling_policy.priority == "conservative"


def test_replay_eligible_stored_but_no_capture_claim():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    assert plan.replay_policy.eligible is True
    assert plan.replay_policy.cuda_graph_bucket == "decode_static"
    assert plan.replay_policy.capture_claimed is False


def test_provenance_preserved_verbatim():
    plan = RuntimeExecutionPlanAdapter.from_dict(PAGED_PREFILL_FIXTURE)
    assert plan.compiler_provenance.truth_boundary == (
        "compiler_execution_provider_plan_not_runtime_dispatch"
    )
    assert plan.compiler_provenance.cost_source == "formula_synthetic"
    assert "serving-phase-analysis" in plan.compiler_provenance.source_passes
    assert "kv-layout-planning" in plan.compiler_provenance.source_passes
    assert "replay-eligibility" in plan.compiler_provenance.source_passes
    assert "execution-provider-planning" in plan.compiler_provenance.source_passes


def test_selected_backend_defaults_to_primary():
    for fixture in (PAGED_PREFILL_FIXTURE, CONSTRAINED_DECODE_FIXTURE):
        plan = RuntimeExecutionPlanAdapter.from_dict(fixture)
        assert plan.backend_policy.selected_backend == plan.backend_policy.primary_backend
        assert plan.backend_policy.runtime_override_reason == ""
        assert plan.backend_policy.has_runtime_override() is False


def test_runtime_override_requires_reason():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    assert plan.validate_runtime_override() is True

    plan.backend_policy.selected_backend = "metal"
    assert plan.backend_policy.has_runtime_override() is True
    assert plan.validate_runtime_override() is False  # override without reason

    plan.backend_policy.runtime_override_reason = "primary_backend_unavailable"
    assert plan.validate_runtime_override() is True


def test_pd_split_uses_pd_split_cost():
    pd_fixture = {
        "function_name": "prefill_pd",
        "execution_mode": "pd_split",
        "cost_summary": {
            "colocated_total_ms": 31.2,
            "pd_split_total_ms": 18.5,
            "confidence": "low",
            "policy": "pd_split",
            "cost_source": "formula_synthetic",
        },
        "backend_execution_plan": {"primary_backend": "cpu"},
    }
    plan = RuntimeExecutionPlanAdapter.from_dict(pd_fixture)
    assert plan.execution_policy == "pd_split"
    assert plan.scheduling_policy.compiler_cost_ms == pytest.approx(18.5)


def test_runtime_ir_not_runtime_result():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    assert plan.replay_policy.capture_claimed is False
    assert not hasattr(plan, "measured_latency_ms")
    assert not hasattr(plan, "measured_memory_mb")
    assert not hasattr(plan, "cuda_graph_captured")
