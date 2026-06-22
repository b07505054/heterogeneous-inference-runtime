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
