"""Tests for ExecutionEngine, BackendDispatcher, and RuntimeResult."""

import pytest

from deployment.backend_dispatcher import BackendDispatcher
from deployment.execution_engine import ExecutionEngine
from deployment.execution_plan_v2.loader import parse_execution_plan_v2
from deployment.execution_plan_v2.schema import ExecutionPlanV2, FunctionPlan
from deployment.execution_trace_recorder import ExecutionTraceRecorder

# ---------------------------------------------------------------------------
# V2 fixtures
# ---------------------------------------------------------------------------

_V2_DECODE: dict = {
    "schema": "execution_plan",
    "schema_version": "2.0.0",
    "plan_id": "decode-plan",
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
            "function_name": "decode_constrained",
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

_V2_PREFILL: dict = {
    "schema": "execution_plan",
    "schema_version": "2.0.0",
    "plan_id": "prefill-plan",
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
            "estimated_kv_peak_mb": 0.0,
            "memory_budget_fraction": 0.0,
            "truth_boundary": "static_formula_estimate_not_measured_memory",
        },
        "serving": {
            "topology": "colocated",
            "colocated_cost_estimate_ms": 0.0,
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
            "function_name": "prefill_conflict",
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


def _decode() -> tuple[ExecutionPlanV2, FunctionPlan]:
    plan = parse_execution_plan_v2(_V2_DECODE)
    return plan, plan.function_plans[0]


def _prefill() -> tuple[ExecutionPlanV2, FunctionPlan]:
    plan = parse_execution_plan_v2(_V2_PREFILL)
    return plan, plan.function_plans[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_primary_backend_available_selects_primary():
    plan, fp = _decode()
    result = ExecutionEngine().execute(plan, fp.function_name)
    assert result.backend_decision.selected_backend == "coreml_ane"
    assert result.backend_decision.override_reason == ""
    assert result.compiler_vs_runtime_backend == "match"


def test_primary_unavailable_selects_first_fallback():
    plan, fp = _decode()
    engine = ExecutionEngine(backend_dispatcher=BackendDispatcher(unavailable={"coreml_ane"}))
    result = engine.execute(plan, fp.function_name)
    assert result.backend_decision.selected_backend == "arm_compute"
    assert result.backend_decision.override_reason == "primary_backend_unavailable"
    assert result.backend_decision.attempted_backends == ["coreml_ane", "arm_compute"]
    assert result.compiler_vs_runtime_backend == "override"


def test_all_backends_unavailable_uses_cpu_emergency():
    plan, fp = _decode()
    engine = ExecutionEngine(
        backend_dispatcher=BackendDispatcher(unavailable={"coreml_ane", "arm_compute", "cpu"})
    )
    result = engine.execute(plan, fp.function_name)
    assert result.backend_decision.selected_backend == "cpu"
    assert result.backend_decision.override_reason == "all_backends_unavailable_cpu_emergency"
    assert result.compiler_vs_runtime_backend == "override"


def test_constraint_conflict_uses_constraint_conflict_reason():
    plan, fp = _prefill()
    engine = ExecutionEngine(backend_dispatcher=BackendDispatcher(unavailable={"cpu"}))
    result = engine.execute(plan, fp.function_name)
    assert result.backend_decision.selected_backend == "cpu"
    assert result.backend_decision.override_reason == "constraint_conflict_emergency_cpu"


def test_execution_context_is_mutable_but_plan_is_not_modified():
    plan, fp = _decode()
    original_backend = fp.backend.selected_backend  # "coreml_ane" — frozen
    engine = ExecutionEngine(backend_dispatcher=BackendDispatcher(unavailable={"coreml_ane"}))
    result = engine.execute(plan, fp.function_name)
    assert fp.backend.selected_backend == original_backend   # FunctionPlan is frozen
    assert result.backend_decision.selected_backend == "arm_compute"


def test_compiler_summary_echoes_plan_fields():
    plan, fp = _decode()
    result = ExecutionEngine().execute(plan, fp.function_name)
    cs = result.compiler_summary
    assert cs.function_name == fp.function_name
    assert cs.compiler_primary_backend == fp.backend.selected_backend
    assert cs.compiler_decision_source == fp.backend.reason
    assert cs.compiler_cost_ms == pytest.approx(
        plan.global_decisions.serving.colocated_cost_estimate_ms
    )
    assert cs.compiler_kv_layout == plan.global_decisions.memory.kv_layout
    assert cs.compiler_truth_boundary == plan.provenance.truth_boundary


def test_runtime_result_truth_boundary():
    plan, fp = _decode()
    result = ExecutionEngine().execute(plan, fp.function_name)
    assert result.runtime_truth_boundary == "runtime_result_not_compiler_plan"


def test_runtime_result_has_no_measurement_claims():
    plan, fp = _decode()
    result = ExecutionEngine().execute(plan, fp.function_name)
    assert result.replay_decision.captured is False
    assert result.replay_decision.capture_attempted is False
    assert result.memory_decision.admitted is True
    assert result.execution_statistics is None
    assert not hasattr(result, "actual_latency_ms")
    assert not hasattr(result, "cuda_graph_captured")


def test_runtime_result_contains_all_typed_decisions():
    from deployment.backend_dispatcher import BackendDecision
    from deployment.runtime_decisions import MemoryDecision, ReplayDecision, SchedulingDecision

    plan, fp = _decode()
    result = ExecutionEngine().execute(plan, fp.function_name)
    assert isinstance(result.scheduling_decision, SchedulingDecision)
    assert isinstance(result.memory_decision, MemoryDecision)
    assert isinstance(result.replay_decision, ReplayDecision)
    assert isinstance(result.backend_decision, BackendDecision)


def test_decision_trace_lists_all_stages_in_order():
    plan, fp = _decode()
    result = ExecutionEngine().execute(plan, fp.function_name)
    expected = [
        "execution_plan_v2",
        "scheduling_decision_evaluator",
        "memory_decision_evaluator",
        "replay_decision_evaluator",
        "backend_dispatcher",
        "execution_engine",
    ]
    assert result.decision_trace == expected


def test_backend_runs_after_replay_in_decision_trace():
    plan, fp = _decode()
    result = ExecutionEngine().execute(plan, fp.function_name)
    trace = result.decision_trace
    replay_idx = trace.index("replay_decision_evaluator")
    backend_idx = trace.index("backend_dispatcher")
    assert backend_idx > replay_idx


# ---------------------------------------------------------------------------
# ExecutionTraceRecorder integration
# ---------------------------------------------------------------------------

def test_recorder_does_not_change_runtime_result():
    """execute() and execute(recorder=...) must return identical RuntimeResult."""
    plan, fp = _decode()
    result_plain = ExecutionEngine().execute(plan, fp.function_name)
    result_traced = ExecutionEngine().execute(plan, fp.function_name, recorder=ExecutionTraceRecorder())
    assert result_plain == result_traced


def test_recorder_none_is_backward_compatible():
    """Explicitly passing recorder=None must be identical to omitting it."""
    plan, fp = _decode()
    result_omitted = ExecutionEngine().execute(plan, fp.function_name)
    result_explicit_none = ExecutionEngine().execute(plan, fp.function_name, recorder=None)
    assert result_omitted == result_explicit_none


def test_recorder_receives_five_events_per_execute():
    """One execute() call produces exactly 5 trace events:
    scheduler, memory, replay, backend, compute."""
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    assert len(rec.events()) == 5


def test_recorder_event_categories_in_pipeline_order():
    """Events must appear in decision-pipeline order."""
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    categories = [e.category for e in rec.events()]
    assert categories == ["scheduler", "memory", "replay", "backend", "compute"]


def test_recorder_scheduling_event_metadata():
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    sched_ev = rec.events()[0]
    assert sched_ev.category == "scheduler"
    assert sched_ev.name == "scheduling_decision"
    assert sched_ev.lane == "scheduler"
    assert sched_ev.metadata["priority"] == "normal"
    assert sched_ev.metadata["admitted"] == "True"
    assert sched_ev.metadata["confidence"] == "unknown"
    assert sched_ev.truth_boundary == "compiler_cost_estimate_not_measured_latency"


def test_recorder_memory_event_metadata():
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    mem_ev = rec.events()[1]
    assert mem_ev.category == "memory"
    assert mem_ev.name == "memory_decision"
    assert mem_ev.lane == "kv_cache"
    assert mem_ev.metadata["kv_layout_used"] == "contiguous"
    assert mem_ev.metadata["allocator_kind"] == "contiguous"
    assert mem_ev.metadata["admitted"] == "True"


def test_recorder_replay_event_metadata():
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    replay_ev = rec.events()[2]
    assert replay_ev.category == "replay"
    assert replay_ev.name == "replay_decision"
    assert replay_ev.lane == "runtime"
    assert replay_ev.duration_ms == pytest.approx(0.0)
    assert replay_ev.metadata["eligible"] == "True"
    assert replay_ev.metadata["bucket"] == "decode_static"
    # V2: replay_requested = replay_eligible (no separate requires_replay)
    assert replay_ev.metadata["replay_requested"] == "True"
    assert replay_ev.truth_boundary == "static_shape_replay_eligibility_not_cuda_graph_capture"


def test_recorder_backend_event_metadata():
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    backend_ev = rec.events()[3]
    assert backend_ev.category == "backend"
    assert backend_ev.name == "backend_dispatch"
    assert backend_ev.lane == "runtime"
    assert backend_ev.duration_ms == pytest.approx(0.0)
    assert backend_ev.metadata["selected_backend"] == "coreml_ane"
    assert backend_ev.metadata["override_reason"] == ""
    assert backend_ev.truth_boundary == "runtime_backend_dispatch_not_compiler_plan"


def test_recorder_compute_event_duration_equals_compiler_cost():
    """Compute event duration must equal the compiler's cost estimate."""
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.category == "compute"
    assert compute_ev.duration_ms == pytest.approx(
        plan.global_decisions.serving.colocated_cost_estimate_ms
    )
    assert compute_ev.duration_ms == pytest.approx(4.8)


def test_recorder_compute_event_lane_gpu_for_coreml_ane():
    """coreml_ane backend → compute lane must be 'gpu'."""
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.lane == "gpu"


def test_recorder_compute_event_lane_cpu_for_cpu_backend():
    """cpu-only backend → compute lane must be 'cpu'."""
    plan, fp = _prefill()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.category == "compute"
    assert compute_ev.lane == "cpu"


def test_recorder_compute_event_name_is_function_name():
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.name == fp.function_name  # "decode_constrained"


def test_recorder_compute_event_backend_in_metadata():
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.metadata["backend"] == "coreml_ane"


def test_recorder_accumulates_across_multiple_execute_calls():
    """Two execute() calls produce 10 events total (5 per call)."""
    plan1, fp1 = _prefill()
    plan2, fp2 = _decode()
    rec = ExecutionTraceRecorder()
    engine = ExecutionEngine()
    engine.execute(plan1, fp1.function_name, recorder=rec)
    engine.execute(plan2, fp2.function_name, recorder=rec)
    assert len(rec.events()) == 10


def test_recorder_clock_advances_across_execute_calls():
    """Clock must be strictly greater after each execute() call."""
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    engine = ExecutionEngine()
    engine.execute(plan, fp.function_name, recorder=rec)
    time_after_first = rec.current_time_ms()
    engine.execute(plan, fp.function_name, recorder=rec)
    time_after_second = rec.current_time_ms()
    assert time_after_second > time_after_first


def test_recorder_event_timestamps_monotonically_non_decreasing():
    """No event may start before the preceding event started."""
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    events = rec.events()
    for i in range(1, len(events)):
        assert events[i].start_ms >= events[i - 1].start_ms


def test_recorder_request_id_matches_function_name():
    """All events for one execute() call carry the function_name as request_id."""
    plan, fp = _decode()
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, fp.function_name, recorder=rec)
    for ev in rec.events():
        assert ev.request_id == fp.function_name
