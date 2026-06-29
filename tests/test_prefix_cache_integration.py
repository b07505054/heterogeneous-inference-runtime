"""Integration tests: PrefixCacheResult applied to PDSplitPlanner."""

import inspect

import pytest

from deployment.distributed_runtime_plan import (
    PDSplitPlanner,
    PrefixCacheAdjustment,
    QueueState,
)
from deployment.prefix_cache_simulator import PrefixCacheResult
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

# Baseline values derived from fixtures (no queue wait, default 32 MB/ms bandwidth):
#   prefill_service_ms = 31.2
#   kv_transfer_ms     = 4.0 / 32.0 = 0.125
#   handoff_overhead   = 0.2
#   decode_service_ms  = 8.5
#   pd_ttft_baseline   = 31.2 + 0.125 + 0.2 + 0 = 31.525
_PREFILL_SERVICE_MS = 31.2
_DECODE_SERVICE_MS = 8.5
_KV_TRANSFER_MS = 4.0 / 32.0          # 0.125  (4 MB at 32 MB/ms)
_HANDOFF_MS = 0.2
_PD_TTFT_BASELINE = _PREFILL_SERVICE_MS + _KV_TRANSFER_MS + _HANDOFF_MS  # 31.525


def _make_plans():
    p = RuntimeExecutionPlanAdapter.from_dict(_PREFILL_DICT)
    d = RuntimeExecutionPlanAdapter.from_dict(_DECODE_DICT)
    return [p, d]


def _cache(
    hit_type: str,
    hit_tokens: int,
    saved_ms: float,
    remote_bytes: float = 0.0,
    prompt_tokens: int = 100,
) -> PrefixCacheResult:
    """Build a PrefixCacheResult directly without going through the simulator."""
    miss_tokens = max(0, prompt_tokens - hit_tokens)
    hit_ratio = hit_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
    return PrefixCacheResult(
        cache_key="test-key",
        hit_type=hit_type,
        hit_tokens=hit_tokens,
        miss_tokens=miss_tokens,
        hit_ratio=hit_ratio,
        saved_prefill_ms=saved_ms,
        remote_transfer_bytes=remote_bytes,
        evicted_cache_keys=(),
        truth_boundary="prefix_cache_simulated_not_real_kv_cache",
    )


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------

def test_miss_preserves_existing_plan():
    plans = _make_plans()
    plan_no_cache = PDSplitPlanner.plan(plans)
    plan_with_miss = PDSplitPlanner.plan(plans, prefix_cache_result=_cache("miss", 0, 0.0))

    assert plan_with_miss.prefill.service_ms == pytest.approx(plan_no_cache.prefill.service_ms)
    assert plan_with_miss.kv_transfer.transfer_cost_ms == pytest.approx(
        plan_no_cache.kv_transfer.transfer_cost_ms
    )
    assert plan_with_miss.decision_comparison.pd_split.ttft_ms == pytest.approx(
        plan_no_cache.decision_comparison.pd_split.ttft_ms
    )
    assert plan_with_miss.decision_comparison.pd_split.tpot_ms == pytest.approx(
        plan_no_cache.decision_comparison.pd_split.tpot_ms
    )
    assert plan_with_miss.decision_comparison.selected_policy == (
        plan_no_cache.decision_comparison.selected_policy
    )


def test_local_hit_reduces_prefill_service_time():
    saved_ms = 10.0
    plan = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 50, saved_ms)
    )
    expected_prefill = _PREFILL_SERVICE_MS - saved_ms  # 21.2
    assert plan.prefill.service_ms == pytest.approx(expected_prefill)


def test_remote_hit_reduces_prefill_but_adds_transfer():
    saved_ms = 10.0
    remote_bytes = 2.0 * 1024 * 1024   # 2 MB
    plan = PDSplitPlanner.plan(
        _make_plans(),
        prefix_cache_result=_cache("remote_hit", 50, saved_ms, remote_bytes),
    )
    expected_prefill = _PREFILL_SERVICE_MS - saved_ms           # 21.2
    remote_cost_ms = (remote_bytes / (1024 * 1024)) / 32.0     # 2 / 32 = 0.0625
    expected_kv_ms = _KV_TRANSFER_MS + remote_cost_ms           # 0.125 + 0.0625

    assert plan.prefill.service_ms == pytest.approx(expected_prefill)
    assert plan.kv_transfer.transfer_cost_ms == pytest.approx(expected_kv_ms)
    assert plan.prefix_cache_adjustment is not None
    assert plan.prefix_cache_adjustment.remote_transfer_cost_ms == pytest.approx(remote_cost_ms)


def test_ttft_reflects_prefix_cache_savings():
    saved_ms = 10.0
    plan_no_cache = PDSplitPlanner.plan(_make_plans())
    plan_cached = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 50, saved_ms)
    )
    # TTFT should decrease by exactly saved_ms.
    assert plan_cached.decision_comparison.pd_split.ttft_ms == pytest.approx(
        plan_no_cache.decision_comparison.pd_split.ttft_ms - saved_ms
    )
    assert plan_cached.decision_comparison.colocated.ttft_ms == pytest.approx(
        plan_no_cache.decision_comparison.colocated.ttft_ms - saved_ms
    )
    # And must be strictly lower.
    assert plan_cached.decision_comparison.pd_split.ttft_ms < (
        plan_no_cache.decision_comparison.pd_split.ttft_ms
    )


def test_tpot_is_not_artificially_changed_by_prefix_hit():
    plan_no_cache = PDSplitPlanner.plan(_make_plans())
    plan_cached = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 50, 10.0)
    )
    # TPOT = decode_service_ms, unaffected by prefix cache.
    assert plan_cached.decision_comparison.pd_split.tpot_ms == pytest.approx(
        plan_no_cache.decision_comparison.pd_split.tpot_ms
    )
    assert plan_cached.decision_comparison.colocated.tpot_ms == pytest.approx(
        plan_no_cache.decision_comparison.colocated.tpot_ms
    )
    assert plan_cached.decision_comparison.pd_split.tpot_ms == pytest.approx(_DECODE_SERVICE_MS)


def test_selected_policy_can_change_due_to_cache_hit():
    # Colocated worker is busy (50ms queue wait) so pd_split is cheaper overall,
    # but pd_ttft=31.525ms just exceeds slo_ttft=31.0ms → colocated selected.
    # A 1ms prefix hit reduces pd_ttft to 30.525ms ≤ 31.0ms → pd_split wins.
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=0.0,
        colocated_worker_available_at_ms=50.0,
    )
    plan_no_cache = PDSplitPlanner.plan(_make_plans(), slo_ttft_ms=31.0, queue_state=qs)
    assert plan_no_cache.decision_comparison.selected_policy == "colocated"
    assert plan_no_cache.decision_comparison.pd_split.ttft_ms > 31.0

    plan_cached = PDSplitPlanner.plan(
        _make_plans(),
        slo_ttft_ms=31.0,
        queue_state=qs,
        prefix_cache_result=_cache("local_hit", 10, 1.0),
    )
    assert plan_cached.decision_comparison.selected_policy == "pd_split"
    assert plan_cached.decision_comparison.pd_split.ttft_ms <= 31.0


def test_adjusted_prefill_service_time_clamped_at_zero():
    plan = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 100, 9999.0)
    )
    assert plan.prefill.service_ms == pytest.approx(0.0)
    assert plan.prefix_cache_adjustment is not None
    assert plan.prefix_cache_adjustment.adjusted_prefill_service_ms == pytest.approx(0.0)


def test_prefix_cache_fields_are_exposed_in_plan():
    cache_result = _cache("local_hit", 50, 10.0)
    plan = PDSplitPlanner.plan(_make_plans(), prefix_cache_result=cache_result)

    adj = plan.prefix_cache_adjustment
    assert adj is not None
    assert isinstance(adj, PrefixCacheAdjustment)
    # All six required fields are accessible.
    _ = adj.hit_type
    _ = adj.hit_tokens
    _ = adj.saved_prefill_ms
    _ = adj.remote_transfer_bytes
    _ = adj.adjusted_prefill_service_ms
    _ = adj.truth_boundary

    assert adj.hit_type == "local_hit"
    assert adj.hit_tokens == 50
    assert adj.saved_prefill_ms == pytest.approx(10.0)
    assert adj.remote_transfer_bytes == pytest.approx(0.0)
    assert adj.adjusted_prefill_service_ms == pytest.approx(_PREFILL_SERVICE_MS - 10.0)


def test_does_not_mutate_inputs():
    plans = _make_plans()
    cache_result = _cache("local_hit", 50, 10.0)

    snapshot_p = (
        plans[0].function_name,
        plans[0].scheduling_policy.compiler_cost_ms,
        plans[0].memory_policy.kv_byte_estimate_mb,
    )
    snapshot_d = (
        plans[1].function_name,
        plans[1].scheduling_policy.compiler_cost_ms,
    )
    snapshot_cache = (
        cache_result.hit_type,
        cache_result.hit_tokens,
        cache_result.saved_prefill_ms,
        cache_result.remote_transfer_bytes,
    )

    PDSplitPlanner.plan(plans, prefix_cache_result=cache_result)

    assert (
        plans[0].function_name,
        plans[0].scheduling_policy.compiler_cost_ms,
        plans[0].memory_policy.kv_byte_estimate_mb,
    ) == snapshot_p
    assert (
        plans[1].function_name,
        plans[1].scheduling_policy.compiler_cost_ms,
    ) == snapshot_d
    assert (
        cache_result.hit_type,
        cache_result.hit_tokens,
        cache_result.saved_prefill_ms,
        cache_result.remote_transfer_bytes,
    ) == snapshot_cache


def test_truth_boundary_mentions_prefix_cache_simulation():
    plan = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 50, 5.0)
    )
    adj = plan.prefix_cache_adjustment
    assert adj is not None
    assert "prefix_cache" in adj.truth_boundary


def test_no_wall_clock_sleep():
    import deployment.distributed_runtime_plan as mod
    src = inspect.getsource(mod)
    assert "time.sleep(" not in src, "Planner must not call time.sleep()"
    assert "time.time(" not in src, "Planner must not use time.time()"
    assert "datetime.now(" not in src, "Planner must not use datetime.now()"


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------

def test_no_cache_result_leaves_adjustment_none():
    plan = PDSplitPlanner.plan(_make_plans())
    assert plan.prefix_cache_adjustment is None


def test_miss_adjustment_has_zero_savings():
    plan = PDSplitPlanner.plan(_make_plans(), prefix_cache_result=_cache("miss", 0, 0.0))
    adj = plan.prefix_cache_adjustment
    assert adj is not None
    assert adj.hit_type == "miss"
    assert adj.saved_prefill_ms == pytest.approx(0.0)
    assert adj.remote_transfer_bytes == pytest.approx(0.0)
    assert adj.remote_transfer_cost_ms == pytest.approx(0.0)
    assert adj.adjusted_prefill_service_ms == pytest.approx(_PREFILL_SERVICE_MS)


def test_local_hit_has_zero_remote_transfer_cost():
    plan = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 50, 10.0, remote_bytes=9999.0)
    )
    adj = plan.prefix_cache_adjustment
    assert adj is not None
    # remote_bytes is non-zero but hit_type is local_hit, so no transfer cost.
    assert adj.remote_transfer_cost_ms == pytest.approx(0.0)
    # kv_transfer is unchanged from baseline.
    assert plan.kv_transfer.transfer_cost_ms == pytest.approx(_KV_TRANSFER_MS)


def test_remote_hit_total_kv_transfer_includes_remote_cost():
    remote_bytes = 4.0 * 1024 * 1024   # 4 MB
    plan = PDSplitPlanner.plan(
        _make_plans(),
        prefix_cache_result=_cache("remote_hit", 50, 0.0, remote_bytes),
    )
    remote_cost_ms = (4.0) / 32.0   # 4 MB / 32 MB/ms = 0.125 ms
    assert plan.kv_transfer.transfer_cost_ms == pytest.approx(
        _KV_TRANSFER_MS + remote_cost_ms
    )


def test_prefill_adjustment_propagates_to_cost_breakdowns():
    saved_ms = 5.0
    plan = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 25, saved_ms)
    )
    expected_prefill = _PREFILL_SERVICE_MS - saved_ms
    assert plan.decision_comparison.pd_split.prefill_service_ms == pytest.approx(expected_prefill)
    assert plan.decision_comparison.colocated.prefill_service_ms == pytest.approx(expected_prefill)


def test_remote_hit_kv_transfer_propagates_to_pd_split_breakdown():
    remote_bytes = 2.0 * 1024 * 1024
    plan = PDSplitPlanner.plan(
        _make_plans(),
        prefix_cache_result=_cache("remote_hit", 50, 0.0, remote_bytes),
    )
    remote_cost_ms = 2.0 / 32.0   # 0.0625 ms
    assert plan.decision_comparison.pd_split.kv_transfer_ms == pytest.approx(
        _KV_TRANSFER_MS + remote_cost_ms
    )


def test_ttft_decreases_by_savings_minus_remote_cost_for_remote_hit():
    saved_ms = 10.0
    remote_bytes = 2.0 * 1024 * 1024   # 2 MB → 0.0625 ms extra
    remote_cost_ms = 2.0 / 32.0

    plan_no_cache = PDSplitPlanner.plan(_make_plans())
    plan_cached = PDSplitPlanner.plan(
        _make_plans(),
        prefix_cache_result=_cache("remote_hit", 50, saved_ms, remote_bytes),
    )
    expected_delta = -(saved_ms - remote_cost_ms)  # net saving on TTFT
    assert plan_cached.decision_comparison.pd_split.ttft_ms == pytest.approx(
        plan_no_cache.decision_comparison.pd_split.ttft_ms + expected_delta
    )


def test_adjustment_is_immutable():
    plan = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 50, 5.0)
    )
    adj = plan.prefix_cache_adjustment
    assert adj is not None
    with pytest.raises((AttributeError, TypeError)):
        adj.saved_prefill_ms = 0.0  # type: ignore[misc]


def test_total_compiler_cost_reflects_adjustment():
    saved_ms = 10.0
    plan_no_cache = PDSplitPlanner.plan(_make_plans())
    plan_cached = PDSplitPlanner.plan(
        _make_plans(), prefix_cache_result=_cache("local_hit", 50, saved_ms)
    )
    # total_compiler_cost_ms includes prefill_service_ms; it must decrease.
    assert plan_cached.total_compiler_cost_ms == pytest.approx(
        plan_no_cache.total_compiler_cost_ms - saved_ms
    )
