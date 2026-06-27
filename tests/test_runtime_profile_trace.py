"""Tests for RuntimeProfileTrace schema and RuntimeProfileTraceBuilder.

All tests are pure unit tests. No ExecutionEngine, no file I/O (except the
write_json roundtrip test which uses tmp_path). Recorder instances are built
manually so tests remain independent of ExecutionEngine internals.
"""

from __future__ import annotations

import json
import math

import pytest

from deployment.execution_trace_recorder import ExecutionTraceRecorder
from deployment.runtime_profile_trace import (
    ComparisonSummary,
    RuntimeProfileTrace,
    RuntimeProfileTraceBuilder,
    RuntimeTimeSeries,
    TraceVariant,
    VariantSummary,
    _percentile,
    _safe_delta_pct,
)


# ---------------------------------------------------------------------------
# Recorder factories — build populated recorders without ExecutionEngine
# ---------------------------------------------------------------------------

def _recorder_with_latency(latency_samples: list[float]) -> ExecutionTraceRecorder:
    """Recorder with latency samples and one compute event per sample."""
    rec = ExecutionTraceRecorder()
    for lat in latency_samples:
        rec.begin_stage("compute", "decode", "gpu")
        rec.advance_clock(lat)
        rec.end_stage()
        rec.record_request_latency(lat)
    return rec


def _recorder_with_snapshots(
    snapshots: list[tuple[int, float, int]],
) -> ExecutionTraceRecorder:
    """Recorder with explicit (queue_depth, memory_mb, active_requests) snapshots."""
    rec = ExecutionTraceRecorder()
    for i, (qd, mem, ar) in enumerate(snapshots):
        rec.record_snapshot(queue_depth=qd, memory_mb=mem, active_requests=ar)
        rec.advance_clock(10.0)
    return rec


def _empty_recorder() -> ExecutionTraceRecorder:
    return ExecutionTraceRecorder()


def _make_variant(
    variant_id: str = "baseline",
    runtime_mode: str = "fcfs_fixed_batch",
    optimizer_features: list[str] | None = None,
    recorder: ExecutionTraceRecorder | None = None,
    truth_boundary: str = "offline_runtime_simulation_not_iphone_execution",
) -> TraceVariant:
    rec = recorder if recorder is not None else _empty_recorder()
    return RuntimeProfileTraceBuilder.from_recorder(
        rec,
        variant_id=variant_id,
        runtime_mode=runtime_mode,
        optimizer_features=optimizer_features or [],
        truth_boundary=truth_boundary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0


def test_percentile_single_element():
    assert _percentile([42.0], 50) == pytest.approx(42.0)
    assert _percentile([42.0], 95) == pytest.approx(42.0)


def test_percentile_p50_median_of_even_list():
    # nearest-rank: for [1,2,3,4] p50 → k = max(0, int(4*50/100) - 1) = 1 → sorted[1] = 2
    result = _percentile([3.0, 1.0, 4.0, 2.0], 50)
    assert result == pytest.approx(2.0)


def test_percentile_p95_of_100_elements():
    samples = [float(i) for i in range(1, 101)]  # 1..100
    # k = max(0, int(100*95/100) - 1) = 94 → sorted[94] = 95.0
    assert _percentile(samples, 95) == pytest.approx(95.0)


def test_safe_delta_pct_zero_baseline_returns_zero():
    assert _safe_delta_pct(100.0, 0.0) == pytest.approx(0.0)


def test_safe_delta_pct_improvement():
    # optimized=500, baseline=1000 → -50%
    assert _safe_delta_pct(500.0, 1000.0) == pytest.approx(-50.0)


def test_safe_delta_pct_regression():
    # optimized=1200, baseline=1000 → +20%
    assert _safe_delta_pct(1200.0, 1000.0) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# 1. from_recorder_preserves_events
# ---------------------------------------------------------------------------

def test_from_recorder_preserves_events():
    rec = ExecutionTraceRecorder()
    rec.begin_stage("scheduler", "scheduling_decision", "scheduler")
    rec.end_stage()
    rec.instant_event("replay", "replay_decision", "runtime")
    rec.begin_stage("compute", "decode", "gpu")
    rec.advance_clock(5.0)
    rec.end_stage()

    variant = _make_variant(recorder=rec)
    assert len(variant.events) == 3
    assert variant.events[0].category == "scheduler"
    assert variant.events[1].category == "replay"
    assert variant.events[2].category == "compute"


def test_from_recorder_events_are_not_shared_with_recorder():
    """Mutating variant.events must not affect a subsequent recorder.events() call."""
    rec = ExecutionTraceRecorder()
    rec.instant_event("replay", "replay_decision", "runtime")
    variant = _make_variant(recorder=rec)
    variant.events.clear()           # type: ignore[attr-defined]  # frozen list, but list itself is mutable
    assert len(rec.events()) == 1    # recorder unaffected


def test_from_recorder_event_fields_intact():
    rec = ExecutionTraceRecorder()
    rec.begin_stage(
        "compute", "prefill", "gpu",
        request_id="req_0",
        metadata={"backend": "coreml"},
        truth_boundary="compiler_cost_estimate_not_measured_latency",
    )
    rec.advance_clock(12.3)
    rec.end_stage()

    ev = _make_variant(recorder=rec).events[0]
    assert ev.category == "compute"
    assert ev.name == "prefill"
    assert ev.lane == "gpu"
    assert ev.request_id == "req_0"
    assert ev.duration_ms == pytest.approx(12.3)
    assert ev.metadata["backend"] == "coreml"
    assert ev.truth_boundary == "compiler_cost_estimate_not_measured_latency"


# ---------------------------------------------------------------------------
# 2. from_recorder uses latency samples for p50/p95
# ---------------------------------------------------------------------------

def test_from_recorder_uses_latency_samples_for_p50_p95():
    samples = [float(i * 10) for i in range(1, 21)]  # 10, 20, ..., 200
    rec = _recorder_with_latency(samples)
    variant = _make_variant(recorder=rec)
    # p50: k = int(20*50/100)-1 = 9 → sorted[9] = 100.0
    assert variant.summary.p50_latency_ms == pytest.approx(100.0)
    # p95: k = int(20*95/100)-1 = 18 → sorted[18] = 190.0
    assert variant.summary.p95_latency_ms == pytest.approx(190.0)


def test_from_recorder_single_latency_sample():
    rec = _recorder_with_latency([77.0])
    variant = _make_variant(recorder=rec)
    assert variant.summary.p50_latency_ms == pytest.approx(77.0)
    assert variant.summary.p95_latency_ms == pytest.approx(77.0)


# ---------------------------------------------------------------------------
# 3. from_recorder handles empty latency samples
# ---------------------------------------------------------------------------

def test_from_recorder_handles_empty_latency_samples():
    variant = _make_variant(recorder=_empty_recorder())
    assert variant.summary.p50_latency_ms == pytest.approx(0.0)
    assert variant.summary.p95_latency_ms == pytest.approx(0.0)
    assert variant.summary.total_requests == 0


# ---------------------------------------------------------------------------
# 4. from_recorder builds timeseries from snapshots
# ---------------------------------------------------------------------------

def test_from_recorder_builds_timeseries_from_snapshots():
    rec = _recorder_with_snapshots([(3, 512.0, 2), (5, 768.0, 4)])
    variant = _make_variant(recorder=rec)
    ts = variant.timeseries
    assert ts.queue_depth == [3, 5]
    assert ts.memory_mb == pytest.approx([512.0, 768.0])
    assert ts.active_requests == [2, 4]
    assert len(ts.timestamps_ms) == 2


def test_from_recorder_timeseries_empty_when_no_snapshots():
    variant = _make_variant(recorder=_empty_recorder())
    ts = variant.timeseries
    assert ts.timestamps_ms == []
    assert ts.queue_depth == []
    assert ts.memory_mb == []
    assert ts.active_requests == []


def test_from_recorder_timeseries_not_interpolated():
    """Snapshots must be stored verbatim; no gap-filling."""
    rec = ExecutionTraceRecorder()
    rec.record_snapshot(queue_depth=1, memory_mb=100.0, active_requests=1)
    rec.advance_clock(99.0)   # large gap — no automatic fill
    rec.record_snapshot(queue_depth=9, memory_mb=900.0, active_requests=9)

    ts = _make_variant(recorder=rec).timeseries
    assert len(ts.timestamps_ms) == 2
    assert ts.timestamps_ms[1] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# 5. variant summary — peak memory
# ---------------------------------------------------------------------------

def test_variant_summary_peak_memory():
    rec = _recorder_with_snapshots([(1, 200.0, 1), (2, 500.0, 2), (1, 300.0, 1)])
    variant = _make_variant(recorder=rec)
    assert variant.summary.peak_memory_mb == pytest.approx(500.0)


def test_variant_summary_peak_memory_no_snapshots():
    variant = _make_variant(recorder=_empty_recorder())
    assert variant.summary.peak_memory_mb == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. variant summary — avg queue depth
# ---------------------------------------------------------------------------

def test_variant_summary_avg_queue_depth():
    rec = _recorder_with_snapshots([(2, 0.0, 0), (4, 0.0, 0), (6, 0.0, 0)])
    variant = _make_variant(recorder=rec)
    assert variant.summary.avg_queue_depth == pytest.approx(4.0)


def test_variant_summary_avg_queue_depth_no_snapshots():
    variant = _make_variant(recorder=_empty_recorder())
    assert variant.summary.avg_queue_depth == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. build_comparison_summary computes deltas
# ---------------------------------------------------------------------------

def test_build_comparison_summary_computes_deltas():
    baseline_rec = _recorder_with_latency([1000.0] * 10)
    baseline_rec_snap = _recorder_with_snapshots([(10, 800.0, 5)])
    # We need a single recorder for both; build manually.
    baseline_rec2 = ExecutionTraceRecorder()
    for _ in range(10):
        baseline_rec2.record_request_latency(1000.0)
    baseline_rec2.record_snapshot(queue_depth=10, memory_mb=800.0, active_requests=5)

    opt_rec = ExecutionTraceRecorder()
    for _ in range(10):
        opt_rec.record_request_latency(500.0)
    opt_rec.record_snapshot(queue_depth=5, memory_mb=600.0, active_requests=3)

    baseline = _make_variant("baseline", recorder=baseline_rec2)
    optimized = _make_variant("optimized", recorder=opt_rec)

    comp = RuntimeProfileTraceBuilder.build_comparison_summary(baseline, optimized)

    # p95: baseline=1000, optimized=500 → -50%
    assert comp.latency_delta_pct == pytest.approx(-50.0, rel=1e-3)
    # memory: baseline=800, optimized=600 → -25%
    assert comp.peak_memory_delta_pct == pytest.approx(-25.0, rel=1e-3)
    # queue: baseline=10, optimized=5 → -50%
    assert comp.queue_depth_delta_pct == pytest.approx(-50.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 8. build_comparison_summary handles zero baseline
# ---------------------------------------------------------------------------

def test_build_comparison_summary_handles_zero_baseline():
    baseline = _make_variant("baseline", recorder=_empty_recorder())
    optimized = _make_variant("optimized", recorder=_empty_recorder())
    comp = RuntimeProfileTraceBuilder.build_comparison_summary(baseline, optimized)
    assert comp.latency_delta_pct == pytest.approx(0.0)
    assert comp.peak_memory_delta_pct == pytest.approx(0.0)
    assert comp.queue_depth_delta_pct == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. build_trace contains baseline and optimized
# ---------------------------------------------------------------------------

def test_build_trace_contains_baseline_and_optimized():
    baseline = _make_variant("baseline")
    optimized = _make_variant("optimized")
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="apple_a17pro_mobile",
        model_name="llama_3_8b",
        compiler_plan_ref="artifacts/apple_demo/serving_execution_plan_iphone.json",
        baseline=baseline,
        optimized=optimized,
    )
    assert "baseline" in trace.variants
    assert "optimized" in trace.variants
    assert trace.variants["baseline"] is baseline
    assert trace.variants["optimized"] is optimized


def test_build_trace_metadata_fields():
    baseline = _make_variant("baseline")
    optimized = _make_variant("optimized")
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="apple_a17pro_mobile",
        model_name="llama_3_8b",
        compiler_plan_ref="artifacts/apple_demo/plan.json",
        baseline=baseline,
        optimized=optimized,
    )
    assert trace.schema_version == "1"
    assert trace.artifact_type == "runtime_profile_trace"
    assert trace.target_profile_id == "apple_a17pro_mobile"
    assert trace.model_name == "llama_3_8b"
    assert trace.compiler_plan_ref == "artifacts/apple_demo/plan.json"


def test_build_trace_comparison_summary_populated():
    baseline_rec = ExecutionTraceRecorder()
    for _ in range(5):
        baseline_rec.record_request_latency(2000.0)

    opt_rec = ExecutionTraceRecorder()
    for _ in range(5):
        opt_rec.record_request_latency(1000.0)

    baseline = _make_variant("baseline", recorder=baseline_rec)
    optimized = _make_variant("optimized", recorder=opt_rec)
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="test_profile",
        model_name="test_model",
        compiler_plan_ref="",
        baseline=baseline,
        optimized=optimized,
    )
    assert trace.comparison_summary.latency_delta_pct == pytest.approx(-50.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 10. to_dict schema
# ---------------------------------------------------------------------------

def test_runtime_profile_trace_to_dict_schema():
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="apple_a17pro_mobile",
        model_name="llama_3_8b",
        compiler_plan_ref="plan.json",
        baseline=_make_variant("baseline"),
        optimized=_make_variant("optimized"),
    )
    d = trace.to_dict()
    assert d["schema_version"] == "1"
    assert d["artifact_type"] == "runtime_profile_trace"
    assert d["target_profile_id"] == "apple_a17pro_mobile"
    assert d["model_name"] == "llama_3_8b"
    assert "trace_truth_boundary" in d
    assert "variants" in d
    assert "baseline" in d["variants"]
    assert "optimized" in d["variants"]
    assert "comparison_summary" in d


def test_trace_variant_to_dict_schema():
    rec = ExecutionTraceRecorder()
    rec.instant_event("replay", "replay_decision", "runtime",
                      metadata={"eligible": "True"})
    rec.record_snapshot(queue_depth=2, memory_mb=400.0, active_requests=1)
    rec.record_request_latency(55.0)

    variant = _make_variant(recorder=rec)
    d = variant.to_dict()

    assert d["variant_id"] == "baseline"
    assert "events" in d
    assert len(d["events"]) == 1
    assert d["events"][0]["category"] == "replay"
    assert d["events"][0]["metadata"]["eligible"] == "True"
    assert "timeseries" in d
    assert "summary" in d
    assert d["summary"]["p50_latency_ms"] == pytest.approx(55.0)
    assert d["summary"]["total_requests"] == 1


def test_timeseries_to_dict_schema():
    ts = RuntimeTimeSeries(
        timestamps_ms=[0.0, 10.0],
        queue_depth=[1, 2],
        memory_mb=[100.0, 200.0],
        active_requests=[1, 2],
    )
    d = ts.to_dict()
    assert d["timestamps_ms"] == [0.0, 10.0]
    assert d["queue_depth"] == [1, 2]
    assert d["memory_mb"] == [100.0, 200.0]
    assert d["active_requests"] == [1, 2]


def test_variant_summary_to_dict_schema():
    summary = VariantSummary(
        p50_latency_ms=100.0,
        p95_latency_ms=500.0,
        peak_memory_mb=800.0,
        avg_queue_depth=4.5,
        total_events=50,
        total_requests=20,
        truth_boundary="offline_runtime_simulation_not_iphone_execution",
    )
    d = summary.to_dict()
    assert d["p50_latency_ms"] == pytest.approx(100.0)
    assert d["p95_latency_ms"] == pytest.approx(500.0)
    assert d["peak_memory_mb"] == pytest.approx(800.0)
    assert d["total_events"] == 50
    assert d["truth_boundary"] == "offline_runtime_simulation_not_iphone_execution"


def test_comparison_summary_to_dict_schema():
    comp = ComparisonSummary(
        latency_delta_pct=-55.0,
        peak_memory_delta_pct=-20.0,
        queue_depth_delta_pct=-30.0,
        headline="Compiler-guided runtime reduces p95 latency by 55.0%",
    )
    d = comp.to_dict()
    assert d["latency_delta_pct"] == pytest.approx(-55.0)
    assert d["headline"].startswith("Compiler-guided")


# ---------------------------------------------------------------------------
# 11. write_json roundtrip
# ---------------------------------------------------------------------------

def test_write_json_roundtrip(tmp_path):
    rec = ExecutionTraceRecorder()
    rec.begin_stage("compute", "decode", "gpu", request_id="req_0",
                    metadata={"backend": "coreml"})
    rec.advance_clock(4.8)
    rec.end_stage()
    rec.record_request_latency(4.8)
    rec.record_snapshot(queue_depth=1, memory_mb=256.0, active_requests=1)

    baseline = _make_variant("baseline", recorder=rec)
    optimized = _make_variant("optimized", recorder=_recorder_with_latency([2.0]))
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="test_profile",
        model_name="test_model",
        compiler_plan_ref="plan.json",
        baseline=baseline,
        optimized=optimized,
    )

    out_path = tmp_path / "subdir" / "trace.json"
    trace.write_json(out_path)

    assert out_path.exists()
    with out_path.open() as f:
        data = json.load(f)

    assert data["schema_version"] == "1"
    assert data["artifact_type"] == "runtime_profile_trace"
    assert "baseline" in data["variants"]
    assert data["variants"]["baseline"]["events"][0]["category"] == "compute"
    assert data["variants"]["baseline"]["timeseries"]["memory_mb"] == pytest.approx([256.0])
    assert data["comparison_summary"]["latency_delta_pct"] < 0


def test_write_json_creates_parent_dirs(tmp_path):
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="test",
        model_name="test",
        compiler_plan_ref="",
        baseline=_make_variant("baseline"),
        optimized=_make_variant("optimized"),
    )
    deep_path = tmp_path / "a" / "b" / "c" / "trace.json"
    trace.write_json(deep_path)
    assert deep_path.exists()


# ---------------------------------------------------------------------------
# 12. truth_boundary present everywhere
# ---------------------------------------------------------------------------

def test_truth_boundary_present_everywhere():
    rec = ExecutionTraceRecorder()
    rec.instant_event("replay", "replay_decision", "runtime",
                      truth_boundary="static_shape_replay_eligibility_not_cuda_graph_capture")
    rec.record_request_latency(10.0)

    variant = _make_variant(
        "baseline",
        recorder=rec,
        truth_boundary="offline_runtime_simulation_not_iphone_execution",
    )
    baseline = variant
    optimized = _make_variant("optimized", recorder=_recorder_with_latency([5.0]))

    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="test_profile",
        model_name="test_model",
        compiler_plan_ref="",
        baseline=baseline,
        optimized=optimized,
    )

    # Trace level
    assert trace.trace_truth_boundary == "offline_runtime_simulation_not_iphone_execution"
    # Variant summary level
    assert trace.variants["baseline"].summary.truth_boundary == (
        "offline_runtime_simulation_not_iphone_execution"
    )
    # Event level (from recorder)
    replay_ev = trace.variants["baseline"].events[0]
    assert replay_ev.truth_boundary == "static_shape_replay_eligibility_not_cuda_graph_capture"


def test_default_trace_truth_boundary_is_offline_simulation():
    """Default truth boundary must propagate when callers omit it."""
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id="",
        model_name="",
        compiler_plan_ref="",
        baseline=_make_variant("baseline"),
        optimized=_make_variant("optimized"),
    )
    assert "offline_runtime_simulation" in trace.trace_truth_boundary
    for variant in trace.variants.values():
        assert "offline_runtime_simulation" in variant.summary.truth_boundary


# ---------------------------------------------------------------------------
# Headline copy tests
# ---------------------------------------------------------------------------

def test_comparison_headline_reduction():
    baseline_rec = ExecutionTraceRecorder()
    baseline_rec.record_request_latency(1000.0)
    opt_rec = ExecutionTraceRecorder()
    opt_rec.record_request_latency(500.0)

    comp = RuntimeProfileTraceBuilder.build_comparison_summary(
        _make_variant("b", recorder=baseline_rec),
        _make_variant("o", recorder=opt_rec),
    )
    assert "reduces" in comp.headline
    assert "50.0%" in comp.headline


def test_comparison_headline_regression():
    baseline_rec = ExecutionTraceRecorder()
    baseline_rec.record_request_latency(500.0)
    opt_rec = ExecutionTraceRecorder()
    opt_rec.record_request_latency(1000.0)

    comp = RuntimeProfileTraceBuilder.build_comparison_summary(
        _make_variant("b", recorder=baseline_rec),
        _make_variant("o", recorder=opt_rec),
    )
    assert "increases" in comp.headline


def test_comparison_headline_no_change():
    baseline_rec = ExecutionTraceRecorder()
    baseline_rec.record_request_latency(500.0)
    opt_rec = ExecutionTraceRecorder()
    opt_rec.record_request_latency(500.0)

    comp = RuntimeProfileTraceBuilder.build_comparison_summary(
        _make_variant("b", recorder=baseline_rec),
        _make_variant("o", recorder=opt_rec),
    )
    assert "no" in comp.headline.lower()


# ---------------------------------------------------------------------------
# Variant fields
# ---------------------------------------------------------------------------

def test_variant_fields_preserved():
    variant = RuntimeProfileTraceBuilder.from_recorder(
        _empty_recorder(),
        variant_id="cost_aware",
        runtime_mode="cost_aware_paged",
        optimizer_features=["page_prefetch", "cost_based_scheduling"],
    )
    assert variant.variant_id == "cost_aware"
    assert variant.runtime_mode == "cost_aware_paged"
    assert variant.optimizer_features == ["page_prefetch", "cost_based_scheduling"]


def test_variant_total_duration_ms_equals_recorder_clock():
    rec = ExecutionTraceRecorder()
    rec.advance_clock(42.5)
    variant = _make_variant(recorder=rec)
    assert variant.total_duration_ms == pytest.approx(42.5)


def test_variant_total_events_count():
    rec = ExecutionTraceRecorder()
    for _ in range(7):
        rec.instant_event("backend", "backend_dispatch", "runtime")
    variant = _make_variant(recorder=rec)
    assert variant.summary.total_events == 7
