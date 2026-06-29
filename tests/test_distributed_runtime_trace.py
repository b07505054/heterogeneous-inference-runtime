"""Tests for build_distributed_runtime_trace."""

import pytest

from deployment.distributed_runtime_plan import PDSplitPlanner, QueueState, ServiceTimeModel
from deployment.distributed_runtime_trace import build_distributed_runtime_trace
from deployment.runtime_execution_plan import RuntimeExecutionPlanAdapter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PREFILL_DICT = {
    "function_name": "prefill",
    "execution_mode": "pd_split",
    "target_profile_id": "test-profile",
    "cost_summary": {
        "pd_split_total_ms": 31.2,
        "colocated_total_ms": 31.2,
        "confidence": "high",
        "policy": "pd_split",
    },
    "kv_plan": {
        "layout": "paged",
        "kv_byte_estimate_mb": 4.0,
        "layout_reason": "paged_kv_preferred",
        "truth_boundary": "static_kv_layout_plan_not_runtime_allocation",
    },
    "replay_plan": {
        "replay_eligible": False,
        "cuda_graph_bucket": "",
        "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
    },
    "backend_execution_plan": {
        "primary_backend": "cuda",
        "fallback_chain": ["cpu"],
        "decision_source": "default_policy",
        "required_precision": "fp16",
        "required_kv_layout": "paged",
        "requires_replay": False,
    },
}

_DECODE_DICT = {
    "function_name": "decode",
    "execution_mode": "pd_split",
    "target_profile_id": "test-profile",
    "cost_summary": {
        "pd_split_total_ms": 8.5,
        "colocated_total_ms": 8.5,
        "confidence": "high",
        "policy": "pd_split",
    },
    "kv_plan": {
        "layout": "paged",
        "kv_byte_estimate_mb": 0.5,
        "layout_reason": "paged_kv_preferred",
        "truth_boundary": "static_kv_layout_plan_not_runtime_allocation",
    },
    "replay_plan": {
        "replay_eligible": True,
        "cuda_graph_bucket": "decode_static",
        "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
    },
    "backend_execution_plan": {
        "primary_backend": "cuda",
        "fallback_chain": ["cpu"],
        "decision_source": "default_policy",
        "required_precision": "fp16",
        "required_kv_layout": "paged",
        "requires_replay": True,
    },
}


def _make_plan(**kwargs):
    p = RuntimeExecutionPlanAdapter.from_dict(_PREFILL_DICT)
    d = RuntimeExecutionPlanAdapter.from_dict(_DECODE_DICT)
    return PDSplitPlanner.plan([p, d], **kwargs)


# ---------------------------------------------------------------------------
# Event order and structure
# ---------------------------------------------------------------------------

def test_trace_event_order_is_queue_prefill_transfer_handoff_queue_decode():
    trace = build_distributed_runtime_trace(_make_plan())
    names = [e["name"] for e in trace["events"]]
    assert names == [
        "prefill_queue_wait",
        "prefill_compute",
        "kv_transfer",
        "handoff",
        "decode_queue_wait",
        "decode_compute",
        "decision_summary",
    ]


def test_trace_has_prefill_kv_handoff_decode_order():
    trace = build_distributed_runtime_trace(_make_plan())
    names = [e["name"] for e in trace["events"]]
    assert names[1] == "prefill_compute"
    assert names[2] == "kv_transfer"
    assert names[3] == "handoff"
    assert names[5] == "decode_compute"


def test_trace_has_decision_summary_last():
    trace = build_distributed_runtime_trace(_make_plan())
    assert trace["events"][-1]["name"] == "decision_summary"


def test_trace_event_count():
    trace = build_distributed_runtime_trace(_make_plan())
    # prefill_queue_wait, prefill_compute, kv_transfer, handoff,
    # decode_queue_wait, decode_compute, decision_summary
    assert len(trace["events"]) == 7


def test_trace_event_names_complete():
    trace = build_distributed_runtime_trace(_make_plan())
    names = {e["name"] for e in trace["events"]}
    assert names == {
        "prefill_queue_wait",
        "prefill_compute",
        "kv_transfer",
        "handoff",
        "decode_queue_wait",
        "decode_compute",
        "decision_summary",
    }


def test_trace_contains_queue_wait_events():
    trace = build_distributed_runtime_trace(_make_plan())
    names = [e["name"] for e in trace["events"]]
    assert "prefill_queue_wait" in names
    assert "decode_queue_wait" in names


# ---------------------------------------------------------------------------
# Durations
# ---------------------------------------------------------------------------

def test_trace_durations_match_plan_costs():
    plan = _make_plan(kv_bandwidth_mb_per_ms=24.0, handoff_overhead_ms=0.2)
    trace = build_distributed_runtime_trace(plan)
    ev = {e["name"]: e for e in trace["events"]}

    assert ev["prefill_queue_wait"]["duration_ms"] == pytest.approx(0.0)
    assert ev["prefill_compute"]["duration_ms"] == pytest.approx(31.2)
    assert ev["kv_transfer"]["duration_ms"] == pytest.approx(4.0 / 24.0)
    assert ev["handoff"]["duration_ms"] == pytest.approx(0.2)
    assert ev["decode_queue_wait"]["duration_ms"] == pytest.approx(0.0)
    assert ev["decode_compute"]["duration_ms"] == pytest.approx(8.5)
    assert ev["decision_summary"]["duration_ms"] == pytest.approx(0.0)


def test_trace_queue_wait_duration_when_worker_delayed():
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=5.0,
        decode_worker_available_at_ms=0.0,
    )
    trace = build_distributed_runtime_trace(_make_plan(queue_state=qs))
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["prefill_queue_wait"]["duration_ms"] == pytest.approx(5.0)
    assert ev["decode_queue_wait"]["duration_ms"] == pytest.approx(0.0)


def test_trace_decode_queue_wait_duration_nonzero():
    kv_ms = 4.0 / 24.0
    decode_ready = 31.2 + kv_ms + 0.2
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=decode_ready + 4.0,
    )
    trace = build_distributed_runtime_trace(
        _make_plan(kv_bandwidth_mb_per_ms=24.0, handoff_overhead_ms=0.2, queue_state=qs)
    )
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["decode_queue_wait"]["duration_ms"] == pytest.approx(4.0)


def test_trace_zero_kv_transfer_duration_when_no_kv():
    no_kv = {**_PREFILL_DICT, "kv_plan": {
        "layout": "unknown",
        "kv_byte_estimate_mb": 0.0,
        "layout_reason": "",
        "truth_boundary": "",
    }}
    p = RuntimeExecutionPlanAdapter.from_dict(no_kv)
    d = RuntimeExecutionPlanAdapter.from_dict(_DECODE_DICT)
    plan = PDSplitPlanner.plan([p, d])
    trace = build_distributed_runtime_trace(plan)
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["kv_transfer"]["duration_ms"] == pytest.approx(0.0)


def test_trace_handoff_duration_zero_when_no_overhead():
    plan = _make_plan(handoff_overhead_ms=0.0)
    trace = build_distributed_runtime_trace(plan)
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["handoff"]["duration_ms"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def test_trace_prefill_queue_wait_starts_at_zero():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["prefill_queue_wait"]["start_ms"] == pytest.approx(0.0)


def test_trace_event_timestamps_are_sequential():
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=5.0,
        decode_worker_available_at_ms=0.0,
    )
    plan = _make_plan(kv_bandwidth_mb_per_ms=24.0, handoff_overhead_ms=0.2, queue_state=qs)
    trace = build_distributed_runtime_trace(plan)
    events = trace["events"][:-1]   # exclude decision_summary
    for i in range(len(events) - 1):
        cur = events[i]
        nxt = events[i + 1]
        expected = cur["start_ms"] + cur["duration_ms"]
        assert nxt["start_ms"] == pytest.approx(expected), (
            f"'{nxt['name']}' should start at {expected:.4f}ms after '{cur['name']}', "
            f"got {nxt['start_ms']:.4f}ms"
        )


def test_trace_decision_summary_starts_after_decode_compute():
    plan = _make_plan(kv_bandwidth_mb_per_ms=24.0, handoff_overhead_ms=0.2)
    trace = build_distributed_runtime_trace(plan)
    ev = {e["name"]: e for e in trace["events"]}
    expected_end = (
        ev["decode_compute"]["start_ms"] + ev["decode_compute"]["duration_ms"]
    )
    assert ev["decision_summary"]["start_ms"] == pytest.approx(expected_end)


def test_trace_prefill_compute_starts_after_prefill_queue_wait():
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=3.0,
        decode_worker_available_at_ms=0.0,
    )
    trace = build_distributed_runtime_trace(_make_plan(queue_state=qs))
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["prefill_compute"]["start_ms"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

def test_trace_event_categories():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["prefill_queue_wait"]["category"] == "prefill"
    assert ev["prefill_compute"]["category"] == "prefill"
    assert ev["kv_transfer"]["category"] == "kv_transfer"
    assert ev["handoff"]["category"] == "handoff"
    assert ev["decode_queue_wait"]["category"] == "decode"
    assert ev["decode_compute"]["category"] == "decode"
    assert ev["decision_summary"]["category"] == "decision"


# ---------------------------------------------------------------------------
# Worker IDs and backends
# ---------------------------------------------------------------------------

def test_trace_worker_ids_on_compute_events():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["prefill_compute"]["worker_id"] == "prefill-worker-0"
    assert ev["decode_compute"]["worker_id"] == "decode-worker-0"


def test_trace_worker_ids_on_queue_wait_events():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}
    # Queue wait events are on the same worker — the request is waiting in that worker's queue
    assert ev["prefill_queue_wait"]["worker_id"] == "prefill-worker-0"
    assert ev["decode_queue_wait"]["worker_id"] == "decode-worker-0"


def test_trace_backends_on_compute_events():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}
    assert ev["prefill_compute"]["backend"] == "cuda"
    assert ev["decode_compute"]["backend"] == "cuda"


def test_trace_backend_none_on_queue_wait_events():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}
    # No backend computation during queue wait
    assert ev["prefill_queue_wait"]["backend"] is None
    assert ev["decode_queue_wait"]["backend"] is None


def test_trace_non_compute_events_have_no_worker_or_backend():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}
    for name in ("kv_transfer", "handoff", "decision_summary"):
        assert ev[name]["worker_id"] is None, f"{name} should have worker_id=None"
        assert ev[name]["backend"] is None, f"{name} should have backend=None"


# ---------------------------------------------------------------------------
# Truth boundaries
# ---------------------------------------------------------------------------

def test_trace_contains_truth_boundaries():
    trace = build_distributed_runtime_trace(_make_plan())
    ev = {e["name"]: e for e in trace["events"]}

    _tb_plan = "pd_split_schedule_static_plan_not_live_cluster_execution"
    _tb_kv = "kv_transfer_cost_model_not_measured_network"
    _tb_goodput = "goodput_proxy_not_cluster_throughput"

    assert ev["prefill_queue_wait"]["truth_boundary"] == _tb_plan
    assert ev["prefill_compute"]["truth_boundary"] == _tb_plan
    assert ev["kv_transfer"]["truth_boundary"] == _tb_kv
    assert ev["handoff"]["truth_boundary"] == _tb_plan
    assert ev["decode_queue_wait"]["truth_boundary"] == _tb_plan
    assert ev["decode_compute"]["truth_boundary"] == _tb_plan
    assert ev["decision_summary"]["truth_boundary"] == _tb_goodput
    assert trace["truth_boundary"] == _tb_plan


# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------

def test_trace_contains_decision_summary():
    trace = build_distributed_runtime_trace(_make_plan())
    summary = trace["cost_summary"]
    assert "colocated" in summary
    assert "pd_split" in summary
    assert "selected_policy" in summary
    assert "decision_reason" in summary
    assert "goodput_proxy" in summary
    assert "truth_boundary" in summary


def test_trace_cost_summary_colocated_fields():
    plan = _make_plan()
    trace = build_distributed_runtime_trace(plan)
    col = trace["cost_summary"]["colocated"]
    cmp = plan.decision_comparison.colocated
    assert col["total_ms"] == pytest.approx(cmp.total_ms)
    assert col["ttft_ms"] == pytest.approx(cmp.ttft_ms)
    assert col["tpot_ms"] == pytest.approx(cmp.tpot_ms)
    assert col["truth_boundary"] == cmp.truth_boundary


def test_trace_cost_summary_pd_split_fields():
    plan = _make_plan()
    trace = build_distributed_runtime_trace(plan)
    pd = trace["cost_summary"]["pd_split"]
    cmp = plan.decision_comparison.pd_split
    assert pd["total_ms"] == pytest.approx(cmp.total_ms)
    assert pd["ttft_ms"] == pytest.approx(cmp.ttft_ms)
    assert pd["tpot_ms"] == pytest.approx(cmp.tpot_ms)
    assert pd["truth_boundary"] == cmp.truth_boundary


def test_trace_cost_summary_selected_policy_matches_plan():
    plan = _make_plan()
    trace = build_distributed_runtime_trace(plan)
    assert trace["cost_summary"]["selected_policy"] == plan.decision_comparison.selected_policy


def test_trace_cost_summary_goodput_proxy_matches_plan():
    plan = _make_plan()
    trace = build_distributed_runtime_trace(plan)
    assert trace["cost_summary"]["goodput_proxy"] == pytest.approx(
        plan.decision_comparison.goodput_proxy
    )


def test_trace_cost_summary_decision_reason_matches_plan():
    plan = _make_plan()
    trace = build_distributed_runtime_trace(plan)
    assert trace["cost_summary"]["decision_reason"] == plan.decision_comparison.decision_reason


def test_trace_cost_summary_truth_boundary():
    plan = _make_plan()
    trace = build_distributed_runtime_trace(plan)
    assert trace["cost_summary"]["truth_boundary"] == "goodput_proxy_not_cluster_throughput"


# ---------------------------------------------------------------------------
# Top-level trace fields
# ---------------------------------------------------------------------------

def test_trace_artifact_type():
    trace = build_distributed_runtime_trace(_make_plan())
    assert trace["artifact_type"] == "distributed_runtime_trace"


def test_trace_target_profile_id():
    trace = build_distributed_runtime_trace(_make_plan())
    assert trace["target_profile_id"] == "test-profile"


def test_trace_total_compiler_cost_ms_matches_plan():
    plan = _make_plan(kv_bandwidth_mb_per_ms=24.0, handoff_overhead_ms=0.2)
    trace = build_distributed_runtime_trace(plan)
    assert trace["total_compiler_cost_ms"] == pytest.approx(plan.total_compiler_cost_ms)


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------

def test_trace_does_not_mutate_plan():
    plan = _make_plan()
    pre_prefill = plan.prefill.service_ms
    pre_decode = plan.decode.service_ms
    pre_kv = plan.kv_transfer.transfer_cost_ms
    pre_policy = plan.decision_comparison.selected_policy
    pre_total = plan.total_compiler_cost_ms
    pre_prefill_q = plan.prefill.queue_wait_ms
    pre_decode_q = plan.decode.queue_wait_ms

    build_distributed_runtime_trace(plan)

    assert plan.prefill.service_ms == pre_prefill
    assert plan.decode.service_ms == pre_decode
    assert plan.kv_transfer.transfer_cost_ms == pre_kv
    assert plan.decision_comparison.selected_policy == pre_policy
    assert plan.total_compiler_cost_ms == pre_total
    assert plan.prefill.queue_wait_ms == pre_prefill_q
    assert plan.decode.queue_wait_ms == pre_decode_q


# ---------------------------------------------------------------------------
# Varied planner params reflected in trace
# ---------------------------------------------------------------------------

def test_trace_reflects_pd_split_selection():
    fast_decode = {**_DECODE_DICT, "cost_summary": {
        **_DECODE_DICT["cost_summary"],
        "pd_split_total_ms": 1.0,
        "colocated_total_ms": 1.0,
    }}
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=0.0,
        colocated_worker_available_at_ms=50.0,   # colocated busy with prior request
    )
    p = RuntimeExecutionPlanAdapter.from_dict(_PREFILL_DICT)
    d = RuntimeExecutionPlanAdapter.from_dict(fast_decode)
    plan = PDSplitPlanner.plan(
        [p, d],
        kv_bandwidth_mb_per_ms=10000.0,
        handoff_overhead_ms=0.0,
        queue_state=qs,
        slo_ttft_ms=500.0,
        slo_tpot_ms=500.0,
    )
    trace = build_distributed_runtime_trace(plan)
    assert trace["cost_summary"]["selected_policy"] == "pd_split"


def test_trace_service_adjustment_reflected_in_compute_duration():
    stm = ServiceTimeModel(
        source="compiler_estimate",
        prefill_service_adjustment_ms=3.0,
    )
    plan = _make_plan(service_time_model=stm)
    trace = build_distributed_runtime_trace(plan)
    ev = {e["name"]: e for e in trace["events"]}
    # prefill_compute duration = service_ms = 31.2 + 3.0
    assert ev["prefill_compute"]["duration_ms"] == pytest.approx(31.2 + 3.0)
