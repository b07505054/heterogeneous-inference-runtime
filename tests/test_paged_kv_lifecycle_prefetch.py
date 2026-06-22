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


def test_usefulness_ema_does_not_affect_prefetch_decisions():
    kv = PagedKVLifecycle(total_pages=30, page_size_tokens=4, kv_mb_per_page=1.0)
    for idx in range(5):
        _resolve_waste(kv, f"waste-{idx}")
    assert kv.summary()["usefulness_score_ema"] == 0.0

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
