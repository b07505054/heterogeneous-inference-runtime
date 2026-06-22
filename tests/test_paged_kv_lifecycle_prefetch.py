import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.llm_runtime_decision import PagedKVLifecycle


def test_true_boundary_hit_after_prior_step_prefetch():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    first_access = kv.access_current_page("r1", step=0)
    assert first_access["hit"] is False

    prefetch = kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    assert prefetch["event"] == "page_prefetch"
    assert len(kv.request_pages["r1"]) == 2

    access = kv.access_current_page("r1", step=1)
    assert access["hit"] is True
    assert kv.prefetch_hits == 1
    assert kv.prefetch_misses == 1


def test_no_double_allocation_at_boundary():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r1", step=0)
    kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )

    assert kv.allocated_pages == 2
    assert kv.has_page_for_token("r1", 4) is True

    if not kv.has_page_for_token("r1", 4):
        kv.allocate_range(request_id="r1", token_begin=4, token_end=8, step=1)

    assert kv.allocated_pages == 2
    assert len(kv.request_pages["r1"]) == 2


def test_pressure_guard_skips_speculative_prefetch():
    kv = PagedKVLifecycle(total_pages=2, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r1", step=0)

    prefetch = kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=0.4,
        token_index=3,
        request_token_budget=12,
    )
    assert prefetch["event"] == "page_prefetch_skipped"
    assert prefetch["reason"] == "memory_pressure_above_prefetch_budget"
    assert kv.pressure_prefetch_skips == 1
    assert kv.allocated_pages == 1
    assert len(kv.request_pages["r1"]) == 1


def test_speculative_memory_error_does_not_reject_request():
    kv = PagedKVLifecycle(total_pages=2, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    kv.allocate_range(request_id="r2", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r1", step=0)

    prefetch = kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=1.5,
        token_index=3,
        request_token_budget=12,
    )
    assert prefetch["event"] == "page_prefetch_skipped"
    assert prefetch["reason"] == "insufficient_free_pages"
    assert kv.pressure_prefetch_skips == 1
    assert len(kv.request_pages["r1"]) == 1
    assert kv.pages[kv.request_pages["r1"][0]].state == "resident"


def test_usefulness_score_zero_denominator_defaults_to_zero():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    summary = kv.summary()
    assert summary["prefetch_attempts"] == 0
    assert summary["usefulness_score"] == 0.0


def test_usefulness_score_is_one_when_all_prefetches_are_consumed():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r1", step=0)
    kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    kv.access_current_page("r1", step=1)

    summary = kv.summary()
    assert summary["prefetch_hits"] == 1
    assert summary["prefetch_waste"] == 0
    assert summary["usefulness_score"] == 1.0


def test_usefulness_score_reflects_hits_and_waste_and_excludes_misses():
    kv = PagedKVLifecycle(total_pages=20, page_size_tokens=4, kv_mb_per_page=1.0)

    # r1: speculative page is consumed -> 1 hit
    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r1", step=0)
    kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    kv.access_current_page("r1", step=1)

    # r2: speculative page is never consumed and request is released -> 1 waste
    kv.allocate_range(request_id="r2", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r2", step=0)
    kv.prefetch_next_decode_page(
        request_id="r2",
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    kv.release_request("r2")

    summary = kv.summary()
    assert summary["prefetch_hits"] == 1
    assert summary["prefetch_waste"] == 1
    assert summary["prefetch_misses"] == 2
    assert summary["usefulness_score"] == 0.5


def test_usefulness_score_is_read_only_and_does_not_mutate_counters():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r1", step=0)
    kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    kv.access_current_page("r1", step=1)

    first = kv.summary()
    second = kv.summary()
    assert first == second
    assert kv.prefetch_hits == 1
    assert kv.prefetch_waste == 0


def _resolve_hit(kv: PagedKVLifecycle, request_id: str) -> None:
    kv.allocate_range(request_id=request_id, token_begin=0, token_end=4, step=0)
    kv.access_current_page(request_id, step=0)
    kv.prefetch_next_decode_page(
        request_id=request_id,
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    kv.access_current_page(request_id, step=1)


def _resolve_waste(kv: PagedKVLifecycle, request_id: str) -> None:
    kv.allocate_range(request_id=request_id, token_begin=0, token_end=4, step=0)
    kv.access_current_page(request_id, step=0)
    kv.prefetch_next_decode_page(
        request_id=request_id,
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    kv.release_request(request_id)


def test_usefulness_ema_zero_denominator_defaults_to_zero():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    summary = kv.summary()
    assert summary["usefulness_score_ema"] == 0.0
    assert summary["usefulness_ema_alpha"] == 0.2


def test_usefulness_ema_cold_start_initializes_directly_from_first_sample():
    kv_hit_first = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    _resolve_hit(kv_hit_first, "r1")
    assert kv_hit_first.summary()["usefulness_score_ema"] == 1.0

    kv_waste_first = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    _resolve_waste(kv_waste_first, "r1")
    assert kv_waste_first.summary()["usefulness_score_ema"] == 0.0


def test_usefulness_ema_matches_manual_formula_across_sequence():
    kv = PagedKVLifecycle(total_pages=30, page_size_tokens=4, kv_mb_per_page=1.0)
    _resolve_hit(kv, "r1")
    _resolve_hit(kv, "r2")
    _resolve_waste(kv, "r3")

    assert kv.summary()["usefulness_score_ema"] == 0.8


def test_usefulness_ema_is_order_sensitive_unlike_cumulative_score():
    hit_first = PagedKVLifecycle(total_pages=30, page_size_tokens=4, kv_mb_per_page=1.0)
    _resolve_hit(hit_first, "r1")
    _resolve_hit(hit_first, "r2")
    _resolve_waste(hit_first, "r3")

    waste_first = PagedKVLifecycle(total_pages=30, page_size_tokens=4, kv_mb_per_page=1.0)
    _resolve_waste(waste_first, "r1")
    _resolve_hit(waste_first, "r2")
    _resolve_hit(waste_first, "r3")

    hit_first_summary = hit_first.summary()
    waste_first_summary = waste_first.summary()

    assert hit_first_summary["usefulness_score"] == waste_first_summary["usefulness_score"]
    assert hit_first_summary["usefulness_score_ema"] != waste_first_summary["usefulness_score_ema"]
    assert hit_first_summary["usefulness_score_ema"] == 0.8
    assert waste_first_summary["usefulness_score_ema"] == 0.36


def test_usefulness_ema_tracking_alone_does_not_block_prefetch_when_guard_inactive():
    kv = PagedKVLifecycle(total_pages=30, page_size_tokens=4, kv_mb_per_page=1.0)
    for idx in range(5):
        _resolve_hit(kv, f"hit-{idx}")
    assert kv.summary()["usefulness_score_ema"] == 1.0
    assert kv.adaptive_guard_active is False

    kv.allocate_range(request_id="r-final", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r-final", step=0)
    prefetch = kv.prefetch_next_decode_page(
        request_id="r-final",
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )
    assert prefetch["event"] == "page_prefetch"


def _attempt_prefetch_with_fresh_page(kv: PagedKVLifecycle, request_id: str) -> dict:
    kv.allocate_range(request_id=request_id, token_begin=0, token_end=4, step=0)
    kv.access_current_page(request_id, step=0)
    return kv.prefetch_next_decode_page(
        request_id=request_id,
        step=0,
        pressure_disable_threshold=0.82,
        token_index=3,
        request_token_budget=12,
    )


def test_adaptive_guard_summary_fields_default_values():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    summary = kv.summary()
    assert summary["usefulness_min_samples"] == 5
    assert summary["usefulness_disable_threshold"] == 0.3
    assert summary["usefulness_reenable_threshold"] == 0.5
    assert summary["adaptive_guard_active"] is False
    assert summary["adaptive_prefetch_skips"] == 0


def test_adaptive_guard_inactive_before_min_samples_reached():
    kv = PagedKVLifecycle(total_pages=50, page_size_tokens=4, kv_mb_per_page=1.0)
    for idx in range(4):
        _resolve_waste(kv, f"r{idx}")
    assert kv.prefetch_waste == 4
    assert kv.adaptive_guard_active is False

    attempt = _attempt_prefetch_with_fresh_page(kv, "r-check")
    assert attempt["event"] == "page_prefetch"
    assert kv.adaptive_guard_active is False


def test_adaptive_guard_activates_after_min_samples_with_low_ema():
    kv = PagedKVLifecycle(total_pages=50, page_size_tokens=4, kv_mb_per_page=1.0)
    for idx in range(5):
        _resolve_waste(kv, f"r{idx}")
    assert kv.prefetch_waste == 5
    assert kv.usefulness_ema == 0.0
    assert kv.adaptive_guard_active is False  # not yet evaluated with resolved == 5

    attempt = _attempt_prefetch_with_fresh_page(kv, "r-trigger")
    assert attempt["event"] == "page_prefetch_skipped"
    assert attempt["reason"] == "usefulness_below_adaptive_guard_threshold"
    assert kv.adaptive_guard_active is True
    assert kv.adaptive_prefetch_skips == 1
    # guard only skips; it must not have allocated a second page
    assert len(kv.request_pages["r-trigger"]) == 1


def test_adaptive_guard_hysteresis_avoids_flapping_and_recovers():
    kv = PagedKVLifecycle(total_pages=50, page_size_tokens=4, kv_mb_per_page=1.0)

    # Build 5 wastes (guard stays inactive at each check, since resolved < 5
    # at the start of every one of these calls) while leaving 4 speculative
    # pages pending (not yet accessed) to resolve as hits afterward.
    _resolve_waste(kv, "waste-0")
    _attempt_prefetch_with_fresh_page(kv, "pending-0")
    _resolve_waste(kv, "waste-1")
    _attempt_prefetch_with_fresh_page(kv, "pending-1")
    _resolve_waste(kv, "waste-2")
    _attempt_prefetch_with_fresh_page(kv, "pending-2")
    _resolve_waste(kv, "waste-3")
    _attempt_prefetch_with_fresh_page(kv, "pending-3")
    _resolve_waste(kv, "waste-4")

    assert kv.prefetch_waste == 5
    assert kv.adaptive_guard_active is False

    # 6th distinct call crosses the min-sample floor with ema == 0.0 -> activates.
    blocked = _attempt_prefetch_with_fresh_page(kv, "trigger")
    assert blocked["reason"] == "usefulness_below_adaptive_guard_threshold"
    assert kv.adaptive_guard_active is True
    assert len(kv.request_pages["trigger"]) == 1

    # Resolve the 4 pending pages as hits one at a time. ema rises but must
    # not cross the reenable threshold (0.5) until the 4th resolution, so the
    # guard must stay active (no flapping) for the first three probes.
    kv.access_current_page("pending-0", step=1)
    assert round(kv.usefulness_ema, 4) == 0.2
    probe = _attempt_prefetch_with_fresh_page(kv, "probe-1")
    assert probe["reason"] == "usefulness_below_adaptive_guard_threshold"
    assert kv.adaptive_guard_active is True

    kv.access_current_page("pending-1", step=1)
    assert round(kv.usefulness_ema, 4) == 0.36
    probe = _attempt_prefetch_with_fresh_page(kv, "probe-2")
    assert probe["reason"] == "usefulness_below_adaptive_guard_threshold"
    assert kv.adaptive_guard_active is True

    kv.access_current_page("pending-2", step=1)
    assert round(kv.usefulness_ema, 4) == 0.488
    probe = _attempt_prefetch_with_fresh_page(kv, "probe-3")
    assert probe["reason"] == "usefulness_below_adaptive_guard_threshold"
    assert kv.adaptive_guard_active is True

    kv.access_current_page("pending-3", step=1)
    assert round(kv.usefulness_ema, 4) == 0.5904
    recovered = _attempt_prefetch_with_fresh_page(kv, "recovered")
    assert recovered["event"] == "page_prefetch"
    assert kv.adaptive_guard_active is False


def test_pressure_guard_checked_before_and_takes_priority_over_adaptive_guard():
    kv = PagedKVLifecycle(total_pages=2, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.prefetch_waste = 5
    kv.usefulness_ema = 0.0
    kv._update_adaptive_guard_state()
    assert kv.adaptive_guard_active is True

    kv.allocate_range(request_id="r1", token_begin=0, token_end=4, step=0)
    kv.allocate_range(request_id="r2", token_begin=0, token_end=4, step=0)
    kv.access_current_page("r1", step=0)

    attempt = kv.prefetch_next_decode_page(
        request_id="r1",
        step=0,
        pressure_disable_threshold=0.5,
        token_index=3,
        request_token_budget=12,
    )
    assert attempt["reason"] == "memory_pressure_above_prefetch_budget"
    assert kv.pressure_prefetch_skips == 1
    # the adaptive check never ran because the pressure guard returned first
    assert kv.adaptive_prefetch_skips == 0


def test_kv_internal_fragmentation_ratio_zero_when_pages_fully_packed():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=8, step=0)
    assert kv.summary()["kv_internal_fragmentation_ratio"] == 0.0


def test_kv_internal_fragmentation_ratio_reflects_partial_last_page():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=16, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=20, step=0)
    # 2 pages allocated (32 tokens capacity), only 20 tokens written
    assert kv.summary()["kv_internal_fragmentation_ratio"] == round(1 - 20 / 32, 4)


def test_kv_internal_fragmentation_ratio_zero_denominator_defaults_to_zero():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=4, kv_mb_per_page=1.0)
    assert kv.summary()["kv_internal_fragmentation_ratio"] == 0.0


def test_kv_internal_fragmentation_ratio_is_lifetime_not_snapshot():
    kv = PagedKVLifecycle(total_pages=10, page_size_tokens=16, kv_mb_per_page=1.0)
    kv.allocate_range(request_id="r1", token_begin=0, token_end=20, step=0)
    kv.release_request("r1")
    assert kv.pages == {}

    kv.allocate_range(request_id="r2", token_begin=0, token_end=16, step=1)
    # lifetime: 32 + 16 = 48 capacity, 20 + 16 = 36 written, despite r1's
    # pages having been released and popped from self.pages
    summary = kv.summary()
    assert kv.tokens_capacity_allocated == 48
    assert kv.tokens_written == 36
    assert summary["kv_internal_fragmentation_ratio"] == round(1 - 36 / 48, 4)


def test_contiguous_free_run_ratio_full_pool_free():
    kv = PagedKVLifecycle(total_pages=8, page_size_tokens=4, kv_mb_per_page=1.0)
    assert kv.summary()["contiguous_free_run_ratio"] == 1.0


def test_contiguous_free_run_ratio_reflects_scattered_releases():
    kv = PagedKVLifecycle(total_pages=8, page_size_tokens=4, kv_mb_per_page=1.0)
    for i in range(8):
        kv.allocate_range(request_id=f"r{i}", token_begin=0, token_end=4, step=0)
    assert kv.summary()["contiguous_free_run_ratio"] == 0.0

    # release alternating pages -> free indices {0, 2, 4, 6}, longest run is 1
    for i in (0, 2, 4, 6):
        kv.release_request(f"r{i}")
    assert kv.free_pages == [0, 2, 4, 6]
    assert kv.summary()["contiguous_free_run_ratio"] == round(1 / 8, 4)

    # release the remaining pages too -> free indices {0..7}, longest run is 8
    for i in (1, 3, 5, 7):
        kv.release_request(f"r{i}")
    assert kv.summary()["contiguous_free_run_ratio"] == 1.0
