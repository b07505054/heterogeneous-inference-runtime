"""Tests for ExecutionTraceRecorder.

All tests are pure unit tests: no ExecutionEngine, no JSON, no file I/O.
Tests cover begin/end semantics, instant events, clock, metadata,
truth boundaries, snapshots, latency samples, and accessor isolation.
"""

import pytest

from deployment.execution_trace_recorder import (
    MEMORY_PHASE_GAP_MS,
    SCHEDULING_PHASE_GAP_MS,
    ExecutionTraceRecorder,
    TraceEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rec() -> ExecutionTraceRecorder:
    return ExecutionTraceRecorder()


# ---------------------------------------------------------------------------
# begin / end — happy path
# ---------------------------------------------------------------------------

def test_begin_end_creates_exactly_one_event():
    rec = _rec()
    rec.begin_stage("compute", "prefill", "gpu")
    rec.end_stage()
    assert len(rec.events()) == 1


def test_begin_end_event_category_name_lane():
    rec = _rec()
    rec.begin_stage("scheduler", "scheduling_decision", "scheduler")
    rec.end_stage()
    ev = rec.events()[0]
    assert ev.category == "scheduler"
    assert ev.name == "scheduling_decision"
    assert ev.lane == "scheduler"


def test_begin_end_start_ms_captured_at_begin_time():
    rec = _rec()
    rec.advance_clock(7.0)
    rec.begin_stage("compute", "decode", "cpu")
    rec.end_stage()
    assert rec.events()[0].start_ms == pytest.approx(7.0)


def test_begin_end_end_ms_captured_at_end_time():
    rec = _rec()
    rec.begin_stage("compute", "prefill", "gpu")
    rec.advance_clock(31.2)
    rec.end_stage()
    ev = rec.events()[0]
    assert ev.end_ms == pytest.approx(31.2)
    assert ev.duration_ms == pytest.approx(31.2)


def test_begin_end_duration_zero_without_advance():
    rec = _rec()
    rec.begin_stage("scheduler", "scheduling_decision", "scheduler")
    rec.end_stage()
    ev = rec.events()[0]
    assert ev.start_ms == pytest.approx(0.0)
    assert ev.end_ms == pytest.approx(0.0)
    assert ev.duration_ms == pytest.approx(0.0)


def test_begin_end_request_id_preserved():
    rec = _rec()
    rec.begin_stage("memory", "memory_decision", "kv_cache", request_id="decode_constrained")
    rec.end_stage()
    assert rec.events()[0].request_id == "decode_constrained"


def test_begin_end_request_id_none_by_default():
    rec = _rec()
    rec.begin_stage("memory", "memory_decision", "kv_cache")
    rec.end_stage()
    assert rec.events()[0].request_id is None


# ---------------------------------------------------------------------------
# begin / end — error cases
# ---------------------------------------------------------------------------

def test_nested_begin_raises_runtime_error():
    rec = _rec()
    rec.begin_stage("compute", "prefill", "gpu")
    with pytest.raises(RuntimeError, match="prefill"):
        rec.begin_stage("scheduler", "scheduling_decision", "scheduler")


def test_nested_begin_error_names_the_active_stage():
    rec = _rec()
    rec.begin_stage("compute", "my_stage", "gpu")
    with pytest.raises(RuntimeError, match="my_stage"):
        rec.begin_stage("memory", "other_stage", "kv_cache")


def test_end_without_begin_raises_runtime_error():
    rec = _rec()
    with pytest.raises(RuntimeError, match="begin_stage"):
        rec.end_stage()


def test_double_end_raises_runtime_error():
    rec = _rec()
    rec.begin_stage("compute", "prefill", "gpu")
    rec.end_stage()
    with pytest.raises(RuntimeError):
        rec.end_stage()


# ---------------------------------------------------------------------------
# Instant events
# ---------------------------------------------------------------------------

def test_instant_event_duration_is_zero():
    rec = _rec()
    rec.instant_event("replay", "replay_decision", "runtime")
    assert rec.events()[0].duration_ms == pytest.approx(0.0)


def test_instant_event_start_equals_end():
    rec = _rec()
    rec.instant_event("backend", "backend_dispatch", "runtime")
    ev = rec.events()[0]
    assert ev.start_ms == ev.end_ms


def test_instant_event_positioned_at_current_time():
    rec = _rec()
    rec.advance_clock(12.5)
    rec.instant_event("replay", "replay_decision", "runtime")
    ev = rec.events()[0]
    assert ev.start_ms == pytest.approx(12.5)
    assert ev.end_ms == pytest.approx(12.5)


def test_instant_event_category_name_lane():
    rec = _rec()
    rec.instant_event("backend", "backend_dispatch", "runtime")
    ev = rec.events()[0]
    assert ev.category == "backend"
    assert ev.name == "backend_dispatch"
    assert ev.lane == "runtime"


def test_instant_event_request_id_preserved():
    rec = _rec()
    rec.instant_event("replay", "replay_decision", "runtime", request_id="decode_fn")
    assert rec.events()[0].request_id == "decode_fn"


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_passed_to_begin_is_in_event():
    rec = _rec()
    rec.begin_stage("scheduler", "scheduling_decision", "scheduler",
                    metadata={"priority": "normal", "confidence": "low"})
    rec.end_stage()
    ev = rec.events()[0]
    assert ev.metadata["priority"] == "normal"
    assert ev.metadata["confidence"] == "low"


def test_metadata_update_in_end_stage_is_merged():
    rec = _rec()
    rec.begin_stage("memory", "memory_decision", "kv_cache",
                    metadata={"allocator_kind": "paged"})
    rec.end_stage(metadata_update={"admitted": "True", "page_budget_estimate": "7"})
    ev = rec.events()[0]
    assert ev.metadata["allocator_kind"] == "paged"
    assert ev.metadata["admitted"] == "True"
    assert ev.metadata["page_budget_estimate"] == "7"


def test_metadata_update_overwrites_key_from_begin():
    rec = _rec()
    rec.begin_stage("memory", "memory_decision", "kv_cache",
                    metadata={"kv_layout_used": "unknown"})
    rec.end_stage(metadata_update={"kv_layout_used": "contiguous"})
    assert rec.events()[0].metadata["kv_layout_used"] == "contiguous"


def test_metadata_default_is_empty_dict():
    rec = _rec()
    rec.begin_stage("compute", "decode", "cpu")
    rec.end_stage()
    assert rec.events()[0].metadata == {}


def test_instant_event_metadata_preserved():
    rec = _rec()
    rec.instant_event("replay", "replay_decision", "runtime",
                      metadata={"eligible": "True", "bucket": "decode_static"})
    ev = rec.events()[0]
    assert ev.metadata["eligible"] == "True"
    assert ev.metadata["bucket"] == "decode_static"


def test_metadata_dict_is_independent_copy_from_input():
    src = {"key": "original"}
    rec = _rec()
    rec.begin_stage("compute", "prefill", "gpu", metadata=src)
    src["key"] = "mutated"          # mutate caller's dict after begin
    rec.end_stage()
    assert rec.events()[0].metadata["key"] == "original"


# ---------------------------------------------------------------------------
# Truth boundary
# ---------------------------------------------------------------------------

def test_truth_boundary_preserved_in_begin_end():
    rec = _rec()
    rec.begin_stage("compute", "prefill", "gpu",
                    truth_boundary="compiler_cost_estimate_not_measured_latency")
    rec.end_stage()
    assert rec.events()[0].truth_boundary == "compiler_cost_estimate_not_measured_latency"


def test_truth_boundary_preserved_in_instant_event():
    rec = _rec()
    rec.instant_event("replay", "replay_decision", "runtime",
                      truth_boundary="static_shape_replay_eligibility_not_cuda_graph_capture")
    assert rec.events()[0].truth_boundary == "static_shape_replay_eligibility_not_cuda_graph_capture"


def test_truth_boundary_default_is_empty_string():
    rec = _rec()
    rec.begin_stage("compute", "decode", "cpu")
    rec.end_stage()
    assert rec.events()[0].truth_boundary == ""


def test_truth_boundary_instant_default_is_empty_string():
    rec = _rec()
    rec.instant_event("backend", "backend_dispatch", "runtime")
    assert rec.events()[0].truth_boundary == ""


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

def test_initial_clock_is_zero():
    rec = _rec()
    assert rec.current_time_ms() == pytest.approx(0.0)


def test_advance_clock_increments_time():
    rec = _rec()
    rec.advance_clock(5.0)
    assert rec.current_time_ms() == pytest.approx(5.0)


def test_advance_clock_accumulates():
    rec = _rec()
    rec.advance_clock(3.0)
    rec.advance_clock(2.0)
    assert rec.current_time_ms() == pytest.approx(5.0)


def test_advance_clock_zero_is_noop():
    rec = _rec()
    rec.advance_clock(10.0)
    rec.advance_clock(0.0)
    assert rec.current_time_ms() == pytest.approx(10.0)


def test_advance_clock_between_begin_and_end_drives_duration():
    rec = _rec()
    rec.begin_stage("compute", "prefill", "gpu")
    rec.advance_clock(10.0)
    rec.advance_clock(5.0)          # two advances within one stage
    rec.end_stage()
    ev = rec.events()[0]
    assert ev.start_ms == pytest.approx(0.0)
    assert ev.end_ms == pytest.approx(15.0)
    assert ev.duration_ms == pytest.approx(15.0)


def test_module_constants_are_positive():
    assert SCHEDULING_PHASE_GAP_MS > 0.0
    assert MEMORY_PHASE_GAP_MS > 0.0


def test_stage_gap_constants_produce_expected_clock_progression():
    rec = _rec()
    # Simulate the scheduling → memory inter-stage gap pattern.
    rec.begin_stage("scheduler", "scheduling_decision", "scheduler")
    rec.end_stage()
    rec.advance_clock(SCHEDULING_PHASE_GAP_MS)
    after_scheduling = rec.current_time_ms()

    rec.begin_stage("memory", "memory_decision", "kv_cache")
    rec.end_stage()
    rec.advance_clock(MEMORY_PHASE_GAP_MS)
    after_memory = rec.current_time_ms()

    assert after_scheduling == pytest.approx(SCHEDULING_PHASE_GAP_MS)
    assert after_memory == pytest.approx(SCHEDULING_PHASE_GAP_MS + MEMORY_PHASE_GAP_MS)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def test_record_snapshot_appends_entry():
    rec = _rec()
    rec.record_snapshot(queue_depth=3, memory_mb=500.0, active_requests=2)
    assert len(rec.snapshots()) == 1


def test_record_snapshot_captures_correct_values():
    rec = _rec()
    rec.record_snapshot(queue_depth=5, memory_mb=1200.0, active_requests=4)
    snap = rec.snapshots()[0]
    assert snap["queue_depth"] == 5
    assert snap["memory_mb"] == pytest.approx(1200.0)
    assert snap["active_requests"] == 4


def test_record_snapshot_uses_current_clock():
    rec = _rec()
    rec.advance_clock(100.0)
    rec.record_snapshot(queue_depth=1, memory_mb=200.0, active_requests=1)
    assert rec.snapshots()[0]["time_ms"] == pytest.approx(100.0)


def test_record_snapshot_initial_clock_is_zero():
    rec = _rec()
    rec.record_snapshot(queue_depth=0, memory_mb=0.0, active_requests=0)
    assert rec.snapshots()[0]["time_ms"] == pytest.approx(0.0)


def test_multiple_snapshots_ordered_by_insertion():
    rec = _rec()
    rec.record_snapshot(1, 100.0, 1)
    rec.advance_clock(50.0)
    rec.record_snapshot(2, 200.0, 2)
    snaps = rec.snapshots()
    assert len(snaps) == 2
    assert snaps[0]["time_ms"] == pytest.approx(0.0)
    assert snaps[1]["time_ms"] == pytest.approx(50.0)
    assert snaps[1]["queue_depth"] == 2


# ---------------------------------------------------------------------------
# Latency samples
# ---------------------------------------------------------------------------

def test_record_request_latency_appends_sample():
    rec = _rec()
    rec.record_request_latency(45.2)
    assert rec.latency_samples() == pytest.approx([45.2])


def test_multiple_latency_samples_preserved_in_order():
    rec = _rec()
    rec.record_request_latency(10.0)
    rec.record_request_latency(20.0)
    rec.record_request_latency(15.0)
    assert rec.latency_samples() == pytest.approx([10.0, 20.0, 15.0])


# ---------------------------------------------------------------------------
# Accessor isolation (returned lists are copies)
# ---------------------------------------------------------------------------

def test_events_returns_copy_not_internal_list():
    rec = _rec()
    rec.instant_event("replay", "replay_decision", "runtime")
    events = rec.events()
    events.clear()
    assert len(rec.events()) == 1


def test_snapshots_returns_copy_not_internal_list():
    rec = _rec()
    rec.record_snapshot(1, 100.0, 1)
    snaps = rec.snapshots()
    snaps.clear()
    assert len(rec.snapshots()) == 1


def test_latency_samples_returns_copy_not_internal_list():
    rec = _rec()
    rec.record_request_latency(5.0)
    samples = rec.latency_samples()
    samples.clear()
    assert len(rec.latency_samples()) == 1


# ---------------------------------------------------------------------------
# Event ordering across multiple calls
# ---------------------------------------------------------------------------

def test_events_append_in_call_order():
    rec = _rec()
    rec.begin_stage("scheduler", "scheduling_decision", "scheduler")
    rec.end_stage()
    rec.instant_event("replay", "replay_decision", "runtime")
    rec.begin_stage("compute", "decode", "cpu")
    rec.end_stage()
    categories = [e.category for e in rec.events()]
    assert categories == ["scheduler", "replay", "compute"]


def test_event_timestamps_monotonically_non_decreasing():
    rec = _rec()
    rec.begin_stage("scheduler", "scheduling_decision", "scheduler")
    rec.end_stage()
    rec.advance_clock(SCHEDULING_PHASE_GAP_MS)
    rec.begin_stage("memory", "memory_decision", "kv_cache")
    rec.end_stage()
    rec.advance_clock(MEMORY_PHASE_GAP_MS)
    rec.instant_event("replay", "replay_decision", "runtime")
    rec.instant_event("backend", "backend_dispatch", "runtime")
    rec.begin_stage("compute", "decode", "cpu")
    rec.advance_clock(4.8)
    rec.end_stage()

    events = rec.events()
    for i in range(1, len(events)):
        assert events[i].start_ms >= events[i - 1].start_ms
