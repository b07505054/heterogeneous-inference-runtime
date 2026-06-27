"""Tests for CompilerRuntimeAdapter."""

import pytest

from deployment.compiler_runtime_adapter import CompilerRuntimeAdapter
from deployment.execution_engine import ExecutionEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PREFILL_PLAN: dict = {
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

_DECODE_PLAN: dict = {
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

_FULL_SERVING_PLAN: dict = {
    "target_profile_id": "apple-a17pro-mobile",
    "model_name": "tiny-gpt",
    "num_layers": 12,
    "hidden_size": 768,
    "function_plans": [_PREFILL_PLAN, _DECODE_PLAN],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_from_serving_plan_converts_multiple_functions():
    plans = CompilerRuntimeAdapter.from_serving_plan(_FULL_SERVING_PLAN)
    assert len(plans) == 2
    # Preserves order: prefill first, decode second.
    assert plans[0].function_name == "prefill"
    assert plans[1].function_name == "decode"


def test_target_profile_id_propagates_from_module():
    plans = CompilerRuntimeAdapter.from_serving_plan(_FULL_SERVING_PLAN)
    for plan in plans:
        assert plan.target_profile_id == "apple-a17pro-mobile"


def test_function_level_target_profile_id_overrides_module():
    decode_with_override = {**_DECODE_PLAN, "target_profile_id": "function-override"}
    serving_plan = {**_FULL_SERVING_PLAN, "function_plans": [_PREFILL_PLAN, decode_with_override]}
    plans = CompilerRuntimeAdapter.from_serving_plan(serving_plan)
    assert plans[0].target_profile_id == "apple-a17pro-mobile"    # module-level
    assert plans[1].target_profile_id == "function-override"       # function-level wins


def test_as_function_map_keys_by_function_name():
    plan_map = CompilerRuntimeAdapter.as_function_map(_FULL_SERVING_PLAN)
    assert set(plan_map.keys()) == {"prefill", "decode"}
    assert plan_map["prefill"].function_name == "prefill"
    assert plan_map["decode"].function_name == "decode"


def test_as_function_map_rejects_duplicate_function_names():
    dup_plan = {
        **_FULL_SERVING_PLAN,
        "function_plans": [_PREFILL_PLAN, {**_PREFILL_PLAN}],  # prefill appears twice
    }
    with pytest.raises(ValueError, match="Duplicate function_name"):
        CompilerRuntimeAdapter.as_function_map(dup_plan)


def test_from_function_plan_still_works():
    plan = CompilerRuntimeAdapter.from_function_plan(_DECODE_PLAN)
    assert plan.function_name == "decode"
    assert plan.backend_policy.primary_backend == "coreml"


def test_missing_function_plans_raises():
    no_function_plans = {
        "target_profile_id": "apple-a17pro-mobile",
        "model_name": "tiny-gpt",
    }
    with pytest.raises(ValueError, match="function_plans"):
        CompilerRuntimeAdapter.from_serving_plan(no_function_plans)


def test_function_plans_must_be_list():
    bad_type = {**_FULL_SERVING_PLAN, "function_plans": "not-a-list"}
    with pytest.raises(ValueError, match="list"):
        CompilerRuntimeAdapter.from_serving_plan(bad_type)


def test_missing_backend_execution_plan_raises():
    fn_dict = {k: v for k, v in _PREFILL_PLAN.items() if k != "backend_execution_plan"}
    with pytest.raises(ValueError, match="backend_execution_plan"):
        CompilerRuntimeAdapter.from_function_plan(fn_dict)


def test_missing_primary_backend_raises():
    bep_no_primary = {k: v for k, v in _PREFILL_PLAN["backend_execution_plan"].items()
                      if k != "primary_backend"}
    fn_dict = {**_PREFILL_PLAN, "backend_execution_plan": bep_no_primary}
    with pytest.raises(ValueError, match="primary_backend"):
        CompilerRuntimeAdapter.from_function_plan(fn_dict)


def test_rejects_selected_backend_contamination():
    contaminated = {**_DECODE_PLAN, "selected_backend": "coreml"}
    with pytest.raises(ValueError, match="selected_backend"):
        CompilerRuntimeAdapter.from_function_plan(contaminated)


def test_rejects_measured_latency_contamination():
    contaminated = {**_DECODE_PLAN, "measured_latency_ms": 3.2}
    with pytest.raises(ValueError, match="measured_latency_ms"):
        CompilerRuntimeAdapter.from_function_plan(contaminated)


def test_rejects_cuda_graph_captured_contamination():
    contaminated = {**_DECODE_PLAN, "cuda_graph_captured": True}
    with pytest.raises(ValueError, match="cuda_graph_captured"):
        CompilerRuntimeAdapter.from_function_plan(contaminated)


def test_truth_boundary_preserved():
    plans = CompilerRuntimeAdapter.from_serving_plan(_FULL_SERVING_PLAN)
    for plan in plans:
        assert plan.compiler_provenance.truth_boundary == (
            "compiler_execution_provider_plan_not_runtime_dispatch"
        )
        assert plan.memory_policy.truth_boundary == (
            "static_formula_estimate_not_measured_memory"
        )
        assert plan.replay_policy.truth_boundary == (
            "static_shape_replay_eligibility_not_cuda_graph_capture"
        )


def test_output_feeds_execution_engine():
    plans = CompilerRuntimeAdapter.from_serving_plan(_FULL_SERVING_PLAN)
    engine = ExecutionEngine()
    result = engine.execute(plans[0])
    assert result.runtime_truth_boundary == "runtime_result_not_compiler_plan"
    assert result.function_name == "prefill"
