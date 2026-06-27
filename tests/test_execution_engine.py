"""Tests for ExecutionEngine, BackendDispatcher, and RuntimeResult."""

import pytest

from deployment.backend_dispatcher import BackendDispatcher
from deployment.execution_engine import ExecutionEngine
from deployment.execution_trace_recorder import ExecutionTraceRecorder
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
    assert result.replay_decision.captured is False
    assert result.replay_decision.capture_attempted is False
    assert result.memory_decision.admitted is True
    assert result.execution_statistics is None
    assert not hasattr(result, "actual_latency_ms")
    assert not hasattr(result, "cuda_graph_captured")


def test_runtime_result_contains_all_typed_decisions():
    from deployment.backend_dispatcher import BackendDecision
    from deployment.runtime_decisions import MemoryDecision, ReplayDecision, SchedulingDecision

    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result = ExecutionEngine().execute(plan)
    assert isinstance(result.scheduling_decision, SchedulingDecision)
    assert isinstance(result.memory_decision, MemoryDecision)
    assert isinstance(result.replay_decision, ReplayDecision)
    assert isinstance(result.backend_decision, BackendDecision)


def test_decision_trace_lists_all_stages_in_order():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result = ExecutionEngine().execute(plan)
    expected = [
        "compiler_runtime_adapter",
        "scheduling_decision_evaluator",
        "memory_decision_evaluator",
        "replay_decision_evaluator",
        "backend_dispatcher",
        "execution_engine",
    ]
    assert result.decision_trace == expected


def test_backend_runs_after_replay_in_decision_trace():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result = ExecutionEngine().execute(plan)
    trace = result.decision_trace
    replay_idx = trace.index("replay_decision_evaluator")
    backend_idx = trace.index("backend_dispatcher")
    assert backend_idx > replay_idx


# ---------------------------------------------------------------------------
# ExecutionTraceRecorder integration
# ---------------------------------------------------------------------------

def test_recorder_does_not_change_runtime_result():
    """execute(plan) and execute(plan, recorder) must return identical RuntimeResult."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result_plain = ExecutionEngine().execute(plan)
    result_traced = ExecutionEngine().execute(plan, recorder=ExecutionTraceRecorder())
    assert result_plain == result_traced


def test_recorder_none_is_backward_compatible():
    """Explicitly passing recorder=None must be identical to omitting it."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    result_omitted = ExecutionEngine().execute(plan)
    result_explicit_none = ExecutionEngine().execute(plan, recorder=None)
    assert result_omitted == result_explicit_none


def test_recorder_receives_five_events_per_execute():
    """One execute() call produces exactly 5 trace events:
    scheduler, memory, replay, backend, compute."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    assert len(rec.events()) == 5


def test_recorder_event_categories_in_pipeline_order():
    """Events must appear in decision-pipeline order."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    categories = [e.category for e in rec.events()]
    assert categories == ["scheduler", "memory", "replay", "backend", "compute"]


def test_recorder_scheduling_event_metadata():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    sched_ev = rec.events()[0]
    assert sched_ev.category == "scheduler"
    assert sched_ev.name == "scheduling_decision"
    assert sched_ev.lane == "scheduler"
    assert sched_ev.metadata["priority"] == "conservative"   # low confidence → conservative
    assert sched_ev.metadata["admitted"] == "True"
    assert sched_ev.metadata["confidence"] == "low"
    assert sched_ev.truth_boundary == "compiler_cost_estimate_not_measured_latency"


def test_recorder_memory_event_metadata():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    mem_ev = rec.events()[1]
    assert mem_ev.category == "memory"
    assert mem_ev.name == "memory_decision"
    assert mem_ev.lane == "kv_cache"
    assert mem_ev.metadata["kv_layout_used"] == "contiguous"
    assert mem_ev.metadata["allocator_kind"] == "contiguous"
    assert mem_ev.metadata["admitted"] == "True"


def test_recorder_replay_event_metadata():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    replay_ev = rec.events()[2]
    assert replay_ev.category == "replay"
    assert replay_ev.name == "replay_decision"
    assert replay_ev.lane == "runtime"
    assert replay_ev.duration_ms == pytest.approx(0.0)
    assert replay_ev.metadata["eligible"] == "True"
    assert replay_ev.metadata["bucket"] == "decode_static"
    # replay_requested=True because eligible=True AND requires_replay=True
    assert replay_ev.metadata["replay_requested"] == "True"
    assert replay_ev.truth_boundary == "static_shape_replay_eligibility_not_cuda_graph_capture"


def test_recorder_backend_event_metadata():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    backend_ev = rec.events()[3]
    assert backend_ev.category == "backend"
    assert backend_ev.name == "backend_dispatch"
    assert backend_ev.lane == "runtime"
    assert backend_ev.duration_ms == pytest.approx(0.0)
    assert backend_ev.metadata["selected_backend"] == "coreml"
    assert backend_ev.metadata["override_reason"] == ""
    assert backend_ev.truth_boundary == "runtime_backend_dispatch_not_compiler_plan"


def test_recorder_compute_event_duration_equals_compiler_cost():
    """Compute event duration must equal the compiler's cost estimate."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.category == "compute"
    assert compute_ev.duration_ms == pytest.approx(plan.scheduling_policy.compiler_cost_ms)
    assert compute_ev.duration_ms == pytest.approx(4.8)


def test_recorder_compute_event_lane_gpu_for_coreml():
    """coreml backend → compute lane must be 'gpu'."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.lane == "gpu"


def test_recorder_compute_event_lane_cpu_for_cpu_backend():
    """cpu-only backend → compute lane must be 'cpu'."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONFLICT_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.category == "compute"
    assert compute_ev.lane == "cpu"


def test_recorder_compute_event_name_is_function_name():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.name == plan.function_name  # "decode_constrained"


def test_recorder_compute_event_backend_in_metadata():
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    compute_ev = rec.events()[4]
    assert compute_ev.metadata["backend"] == "coreml"


def test_recorder_accumulates_across_multiple_execute_calls():
    """Two execute() calls produce 10 events total (5 per call)."""
    prefill_plan = RuntimeExecutionPlanAdapter.from_dict(CONFLICT_FIXTURE)
    decode_plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    engine = ExecutionEngine()
    engine.execute(prefill_plan, recorder=rec)
    engine.execute(decode_plan, recorder=rec)
    assert len(rec.events()) == 10


def test_recorder_clock_advances_across_execute_calls():
    """Clock must be strictly greater after each execute() call."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    engine = ExecutionEngine()
    engine.execute(plan, recorder=rec)
    time_after_first = rec.current_time_ms()
    engine.execute(plan, recorder=rec)
    time_after_second = rec.current_time_ms()
    assert time_after_second > time_after_first


def test_recorder_event_timestamps_monotonically_non_decreasing():
    """No event may start before the preceding event started."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    events = rec.events()
    for i in range(1, len(events)):
        assert events[i].start_ms >= events[i - 1].start_ms


def test_recorder_request_id_matches_function_name():
    """All events for one execute() call carry the plan's function_name as request_id."""
    plan = RuntimeExecutionPlanAdapter.from_dict(CONSTRAINED_DECODE_FIXTURE)
    rec = ExecutionTraceRecorder()
    ExecutionEngine().execute(plan, recorder=rec)
    for ev in rec.events():
        assert ev.request_id == plan.function_name
