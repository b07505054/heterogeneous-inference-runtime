"""Tests for ExecutionEngine, BackendDispatcher, and RuntimeResult."""

import pytest

from deployment.backend_dispatcher import BackendDispatcher
from deployment.execution_engine import ExecutionEngine
from deployment.runtime_execution_plan import RuntimeExecutionPlanAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONSTRAINED_DECODE_FIXTURE: dict = {
    "function_name": "decode_constrained",
    "execution_mode": "colocated",
    "cost_summary": {
        "colocated_total_ms": 4.8,
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

CONFLICT_FIXTURE: dict = {
    "function_name": "prefill_conflict",
    "execution_mode": "colocated",
    "cost_summary": {
        "colocated_total_ms": 0.0,
        "confidence": "low",
        "policy": "colocated",
        "cost_source": "formula_synthetic",
    },
    "kv_plan": {
        "layout": "paged",
        "kv_byte_estimate_mb": 0.0,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_primary_backend_available_selects_primary():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result = ExecutionEngine().execute(plan)
    assert result.backend_decision.selected_backend == "coreml"
    assert result.backend_decision.override_reason == ""
    assert result.compiler_vs_runtime_backend == "match"


def test_primary_unavailable_selects_first_fallback():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    engine = ExecutionEngine(backend_dispatcher=BackendDispatcher(unavailable={"coreml"}))
    result = engine.execute(plan)
    assert result.backend_decision.selected_backend == "metal"
    assert result.backend_decision.override_reason == "primary_backend_unavailable"
    assert result.backend_decision.attempted_backends == ["coreml", "metal"]
    assert result.compiler_vs_runtime_backend == "override"


def test_all_backends_unavailable_uses_cpu_emergency():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    engine = ExecutionEngine(
        backend_dispatcher=BackendDispatcher(unavailable={"coreml", "metal", "cpu"})
    )
    result = engine.execute(plan)
    assert result.backend_decision.selected_backend == "cpu"
    assert result.backend_decision.override_reason == "all_backends_unavailable_cpu_emergency"
    assert result.compiler_vs_runtime_backend == "override"


def test_constraint_conflict_uses_constraint_conflict_reason():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONFLICT_FIXTURE)
    engine = ExecutionEngine(backend_dispatcher=BackendDispatcher(unavailable={"cpu"}))
    result = engine.execute(plan)
    assert result.backend_decision.selected_backend == "cpu"
    assert result.backend_decision.override_reason == "constraint_conflict_emergency_cpu"


def test_execution_context_is_mutable_but_plan_is_not_modified():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    original_selected = plan.backend_policy.selected_backend
    engine = ExecutionEngine(backend_dispatcher=BackendDispatcher(unavailable={"coreml"}))
    result = engine.execute(plan)
    assert plan.backend_policy.selected_backend == original_selected  # plan unchanged
    assert result.backend_decision.selected_backend == "metal"        # result has runtime selection


def test_compiler_summary_echoes_plan_fields():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result = ExecutionEngine().execute(plan)
    cs = result.compiler_summary
    assert cs.function_name == plan.function_name
    assert cs.compiler_primary_backend == plan.backend_policy.primary_backend
    assert cs.compiler_decision_source == plan.backend_policy.compiler_decision_source
    assert cs.compiler_cost_ms == pytest.approx(plan.scheduling_policy.compiler_cost_ms)
    assert cs.compiler_kv_layout == plan.memory_policy.kv_layout
    assert cs.compiler_truth_boundary == plan.compiler_provenance.truth_boundary


def test_runtime_result_truth_boundary():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result = ExecutionEngine().execute(plan)
    assert result.runtime_truth_boundary == "runtime_result_not_compiler_plan"


def test_runtime_result_has_no_measurement_claims():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result = ExecutionEngine().execute(plan)
    assert result.replay_result is None
    assert result.memory_result is None
    assert result.execution_statistics is None
    assert not hasattr(result, "actual_latency_ms")
    assert not hasattr(result, "cuda_graph_captured")
