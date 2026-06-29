"""Tests for DistributedExecutionEngine."""

import inspect

import pytest

from deployment.distributed_execution_engine import (
    DistributedExecutionEngine,
    DistributedRuntimeResult,
    DistributedStageResult,
)
from deployment.distributed_runtime_plan import PDSplitPlanner, QueueState
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


def _engine():
    return DistributedExecutionEngine()


# ---------------------------------------------------------------------------
# Core result checks
# ---------------------------------------------------------------------------

def test_execute_returns_result():
    result = _engine().execute(_make_plan())
    assert isinstance(result, DistributedRuntimeResult)


def test_result_contains_stage_results():
    result = _engine().execute(_make_plan())
    assert all(isinstance(s, DistributedStageResult) for s in result.stage_results)


def test_stage_count_is_six():
    # decision_summary is not an execution stage; 6 stages only.
    result = _engine().execute(_make_plan())
    assert len(result.stage_results) == 6


def test_stage_order_matches_plan():
    result = _engine().execute(_make_plan())
    names = [s.stage_name for s in result.stage_results]
    assert names == [
        "prefill_queue_wait",
        "prefill_compute",
        "kv_transfer",
        "handoff",
        "decode_queue_wait",
        "decode_compute",
    ]


# ---------------------------------------------------------------------------
# Latency rollup
# ---------------------------------------------------------------------------

def test_total_latency_matches_final_stage_end():
    result = _engine().execute(_make_plan())
    assert result.total_latency_ms == pytest.approx(result.stage_results[-1].end_ms)


def test_total_latency_matches_sum_of_durations():
    result = _engine().execute(_make_plan())
    total = sum(s.duration_ms for s in result.stage_results)
    assert result.total_latency_ms == pytest.approx(total)


# ---------------------------------------------------------------------------
# ttft / tpot from selected policy
# ---------------------------------------------------------------------------

def test_ttft_tpot_match_selected_policy():
    plan = _make_plan()
    result = _engine().execute(plan)
    cmp = plan.decision_comparison
    if cmp.selected_policy == "pd_split":
        assert result.ttft_ms == pytest.approx(cmp.pd_split.ttft_ms)
        assert result.tpot_ms == pytest.approx(cmp.pd_split.tpot_ms)
    else:
        assert result.ttft_ms == pytest.approx(cmp.colocated.ttft_ms)
        assert result.tpot_ms == pytest.approx(cmp.colocated.tpot_ms)


def test_ttft_tpot_use_pd_split_when_selected():
    fast_decode = {**_DECODE_DICT, "cost_summary": {
        **_DECODE_DICT["cost_summary"],
        "pd_split_total_ms": 1.0,
        "colocated_total_ms": 1.0,
    }}
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=0.0,
        colocated_worker_available_at_ms=50.0,
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
    assert plan.decision_comparison.selected_policy == "pd_split"
    result = _engine().execute(plan)
    cmp = plan.decision_comparison
    assert result.ttft_ms == pytest.approx(cmp.pd_split.ttft_ms)
    assert result.tpot_ms == pytest.approx(cmp.pd_split.tpot_ms)
    assert result.selected_policy == "pd_split"


def test_ttft_tpot_use_colocated_when_selected():
    # Default scenario: colocated is typically selected (no worker delay advantage).
    plan = _make_plan(slo_ttft_ms=1.0)   # tight TTFT SLO forces colocated
    assert plan.decision_comparison.selected_policy == "colocated"
    result = _engine().execute(plan)
    cmp = plan.decision_comparison
    assert result.ttft_ms == pytest.approx(cmp.colocated.ttft_ms)
    assert result.tpot_ms == pytest.approx(cmp.colocated.tpot_ms)
    assert result.selected_policy == "colocated"


# ---------------------------------------------------------------------------
# Stage geometry
# ---------------------------------------------------------------------------

def test_stage_end_is_start_plus_duration():
    result = _engine().execute(_make_plan())
    for s in result.stage_results:
        assert s.end_ms == pytest.approx(s.start_ms + s.duration_ms), (
            f"Stage '{s.stage_name}': end_ms={s.end_ms} "
            f"!= start_ms={s.start_ms} + duration_ms={s.duration_ms}"
        )


def test_stages_are_sequential():
    result = _engine().execute(_make_plan())
    stages = result.stage_results
    for i in range(len(stages) - 1):
        assert stages[i + 1].start_ms == pytest.approx(stages[i].end_ms), (
            f"'{stages[i + 1].stage_name}'.start_ms={stages[i + 1].start_ms} "
            f"!= '{stages[i].stage_name}'.end_ms={stages[i].end_ms}"
        )


def test_prefill_queue_wait_starts_at_zero():
    result = _engine().execute(_make_plan())
    assert result.stage_results[0].start_ms == pytest.approx(0.0)


def test_stage_durations_match_plan():
    plan = _make_plan(kv_bandwidth_mb_per_ms=24.0, handoff_overhead_ms=0.2)
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_queue_wait"].duration_ms == pytest.approx(plan.prefill.queue_wait_ms)
    assert by["prefill_compute"].duration_ms == pytest.approx(plan.prefill.service_ms)
    assert by["kv_transfer"].duration_ms == pytest.approx(plan.kv_transfer.transfer_cost_ms)
    assert by["handoff"].duration_ms == pytest.approx(
        plan.decision_comparison.pd_split.handoff_overhead_ms
    )
    assert by["decode_queue_wait"].duration_ms == pytest.approx(plan.decode.queue_wait_ms)
    assert by["decode_compute"].duration_ms == pytest.approx(plan.decode.service_ms)


# ---------------------------------------------------------------------------
# Zero-duration queue waits kept
# ---------------------------------------------------------------------------

def test_zero_duration_queue_wait_kept():
    plan = _make_plan(queue_state=QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=0.0,
    ))
    result = _engine().execute(plan)
    names = [s.stage_name for s in result.stage_results]
    assert "prefill_queue_wait" in names
    assert "decode_queue_wait" in names
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_queue_wait"].duration_ms == pytest.approx(0.0)
    assert by["decode_queue_wait"].duration_ms == pytest.approx(0.0)


def test_zero_duration_stages_do_not_advance_cursor():
    # With zero queue waits, prefill_compute must start at 0.0.
    plan = _make_plan(queue_state=QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=0.0,
    ))
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_compute"].start_ms == pytest.approx(0.0)


def test_nonzero_prefill_queue_wait_delays_compute():
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=5.0,
        decode_worker_available_at_ms=0.0,
    )
    plan = _make_plan(queue_state=qs)
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_queue_wait"].duration_ms == pytest.approx(5.0)
    assert by["prefill_compute"].start_ms == pytest.approx(5.0)


def test_nonzero_decode_queue_wait_delays_decode_compute():
    kv_ms = 4.0 / 24.0
    decode_ready = 31.2 + kv_ms + 0.2
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=decode_ready + 3.0,
    )
    plan = _make_plan(kv_bandwidth_mb_per_ms=24.0, handoff_overhead_ms=0.2, queue_state=qs)
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    assert by["decode_queue_wait"].duration_ms == pytest.approx(3.0)
    assert by["decode_compute"].start_ms == pytest.approx(
        by["decode_queue_wait"].start_ms + 3.0
    )


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------

def test_does_not_mutate_plan():
    plan = _make_plan()
    snapshot = (
        plan.prefill.queue_wait_ms,
        plan.prefill.service_ms,
        plan.decode.queue_wait_ms,
        plan.decode.service_ms,
        plan.kv_transfer.transfer_cost_ms,
        plan.decision_comparison.selected_policy,
        plan.total_compiler_cost_ms,
    )
    _engine().execute(plan)
    after = (
        plan.prefill.queue_wait_ms,
        plan.prefill.service_ms,
        plan.decode.queue_wait_ms,
        plan.decode.service_ms,
        plan.kv_transfer.transfer_cost_ms,
        plan.decision_comparison.selected_policy,
        plan.total_compiler_cost_ms,
    )
    assert snapshot == after


def test_result_is_immutable():
    result = _engine().execute(_make_plan())
    with pytest.raises((AttributeError, TypeError)):
        result.total_latency_ms = 0.0  # type: ignore[misc]


def test_stage_result_is_immutable():
    result = _engine().execute(_make_plan())
    with pytest.raises((AttributeError, TypeError)):
        result.stage_results[0].duration_ms = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Truth boundaries
# ---------------------------------------------------------------------------

def test_stage_truth_boundaries_preserved():
    plan = _make_plan()
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_queue_wait"].truth_boundary == plan.prefill.truth_boundary
    assert by["prefill_compute"].truth_boundary == plan.prefill.truth_boundary
    assert by["kv_transfer"].truth_boundary == plan.kv_transfer.truth_boundary
    assert by["handoff"].truth_boundary == plan.truth_boundary
    assert by["decode_queue_wait"].truth_boundary == plan.decode.truth_boundary
    assert by["decode_compute"].truth_boundary == plan.decode.truth_boundary


def test_result_truth_boundary():
    result = _engine().execute(_make_plan())
    assert result.truth_boundary == (
        "distributed_runtime_execution_simulated_not_real_cluster_execution"
    )


def test_stage_truth_boundaries_not_engine_boundary():
    # Stage truth boundaries come from the plan, not from the engine.
    plan = _make_plan()
    result = _engine().execute(plan)
    engine_tb = "distributed_runtime_execution_simulated_not_real_cluster_execution"
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_compute"].truth_boundary != engine_tb
    assert by["kv_transfer"].truth_boundary != engine_tb


# ---------------------------------------------------------------------------
# Worker IDs and backends
# ---------------------------------------------------------------------------

def test_worker_ids_on_compute_stages():
    plan = _make_plan()
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_compute"].worker_id == plan.prefill.worker_id
    assert by["decode_compute"].worker_id == plan.decode.worker_id


def test_worker_ids_on_queue_wait_stages():
    plan = _make_plan()
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    # Queue waits belong to the same worker as the subsequent compute stage.
    assert by["prefill_queue_wait"].worker_id == plan.prefill.worker_id
    assert by["decode_queue_wait"].worker_id == plan.decode.worker_id


def test_backends_on_compute_stages():
    plan = _make_plan()
    result = _engine().execute(plan)
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_compute"].backend == plan.prefill.backend
    assert by["decode_compute"].backend == plan.decode.backend


def test_backends_none_on_non_compute_stages():
    result = _engine().execute(_make_plan())
    by = {s.stage_name: s for s in result.stage_results}
    for name in ("prefill_queue_wait", "decode_queue_wait", "kv_transfer", "handoff"):
        assert by[name].backend is None, f"Expected backend=None on '{name}'"


def test_all_stages_succeed():
    result = _engine().execute(_make_plan())
    assert all(s.success for s in result.stage_results)


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

def test_stage_categories():
    result = _engine().execute(_make_plan())
    by = {s.stage_name: s for s in result.stage_results}
    assert by["prefill_queue_wait"].category == "prefill"
    assert by["prefill_compute"].category == "prefill"
    assert by["kv_transfer"].category == "kv_transfer"
    assert by["handoff"].category == "handoff"
    assert by["decode_queue_wait"].category == "decode"
    assert by["decode_compute"].category == "decode"


# ---------------------------------------------------------------------------
# Model name and profile ID propagation
# ---------------------------------------------------------------------------

def test_result_carries_model_name_and_profile_id():
    plan = _make_plan()
    result = _engine().execute(plan)
    assert result.model_name == plan.model_name
    assert result.target_profile_id == plan.target_profile_id


# ---------------------------------------------------------------------------
# No wall clock / sleep
# ---------------------------------------------------------------------------

def test_no_wall_clock_sleep():
    import deployment.distributed_execution_engine as mod
    src = inspect.getsource(mod)
    # Check for actual call expressions, not docstring mentions.
    assert "time.sleep(" not in src, "Engine must not call time.sleep()"
    assert "time.time(" not in src, "Engine must not use wall-clock time.time()"
    assert "datetime.now(" not in src, "Engine must not use datetime.now()"
    assert "import time" not in src, "Engine must not import the time module"
