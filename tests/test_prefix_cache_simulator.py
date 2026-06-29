"""Tests for PrefixCacheSimulator."""

import inspect

import pytest

from deployment.prefix_cache_simulator import (
    PrefixCacheEntry,
    PrefixCacheRequest,
    PrefixCacheResult,
    PrefixCacheSimulator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(
    cache_key: str,
    cached_tokens: int,
    worker_id: str,
    kv_bytes: float = 1024.0,
    last_access_ms: float = 0.0,
) -> PrefixCacheEntry:
    return PrefixCacheEntry(
        cache_key=cache_key,
        cached_tokens=cached_tokens,
        worker_id=worker_id,
        kv_bytes=kv_bytes,
        last_access_ms=last_access_ms,
    )


def _request(
    cache_key: str,
    prompt_tokens: int,
    cacheable_tokens: int,
    preferred_worker_id: str = "w0",
) -> PrefixCacheRequest:
    return PrefixCacheRequest(
        cache_key=cache_key,
        prompt_tokens=prompt_tokens,
        cacheable_tokens=cacheable_tokens,
        preferred_worker_id=preferred_worker_id,
    )


def _sim() -> PrefixCacheSimulator:
    return PrefixCacheSimulator()


_LARGE_BUDGET = 1_000_000.0   # effectively no eviction
_BASE_PREFILL = 100.0          # ms; convenient for ratio checks


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------

def test_miss_when_key_absent():
    entries = [_entry("other_key", 50, "w0")]
    result = _sim().evaluate(_request("missing", 100, 100), entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "miss"
    assert result.hit_tokens == 0
    assert result.miss_tokens == 100


def test_local_hit_when_key_on_preferred_worker():
    entries = [_entry("k1", 80, "w0")]
    result = _sim().evaluate(_request("k1", 100, 100, preferred_worker_id="w0"),
                              entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "local_hit"
    assert result.hit_tokens > 0


def test_remote_hit_when_key_on_other_worker():
    entries = [_entry("k1", 80, "w1")]  # not on w0
    result = _sim().evaluate(_request("k1", 100, 100, preferred_worker_id="w0"),
                              entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "remote_hit"
    assert result.hit_tokens > 0


def test_hit_tokens_are_capped():
    # hit_tokens = min(cached_tokens=30, cacheable_tokens=50, prompt_tokens=100)
    entries = [_entry("k1", 30, "w0")]
    result = _sim().evaluate(_request("k1", 100, 50, preferred_worker_id="w0"),
                              entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_tokens == 30  # capped by cached_tokens

    # hit_tokens = min(cached_tokens=80, cacheable_tokens=40, prompt_tokens=100)
    entries2 = [_entry("k2", 80, "w0")]
    result2 = _sim().evaluate(_request("k2", 100, 40, preferred_worker_id="w0"),
                               entries2, _BASE_PREFILL, _LARGE_BUDGET)
    assert result2.hit_tokens == 40  # capped by cacheable_tokens

    # hit_tokens = min(cached_tokens=200, cacheable_tokens=200, prompt_tokens=60)
    entries3 = [_entry("k3", 200, "w0")]
    result3 = _sim().evaluate(_request("k3", 60, 200, preferred_worker_id="w0"),
                               entries3, _BASE_PREFILL, _LARGE_BUDGET)
    assert result3.hit_tokens == 60  # capped by prompt_tokens


def test_saved_prefill_scales_with_hit_tokens():
    # Full hit: all 100 tokens cached on preferred worker.
    entries = [_entry("k1", 100, "w0")]
    result = _sim().evaluate(_request("k1", 100, 100), entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.saved_prefill_ms == pytest.approx(_BASE_PREFILL)

    # Partial hit: 50/100 tokens saved → 50% of base_prefill_ms.
    entries2 = [_entry("k2", 50, "w0")]
    result2 = _sim().evaluate(_request("k2", 100, 100), entries2, _BASE_PREFILL, _LARGE_BUDGET)
    assert result2.saved_prefill_ms == pytest.approx(_BASE_PREFILL * 0.5)

    # Miss: 0 saved.
    result3 = _sim().evaluate(_request("k3", 100, 100), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert result3.saved_prefill_ms == pytest.approx(0.0)


def test_remote_hit_adds_transfer_bytes():
    entries = [_entry("k1", 100, "w1", kv_bytes=4096.0)]  # w1 != w0
    result = _sim().evaluate(_request("k1", 100, 100, preferred_worker_id="w0"),
                              entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "remote_hit"
    assert result.remote_transfer_bytes > 0.0
    # Full hit: all kv_bytes transferred.
    assert result.remote_transfer_bytes == pytest.approx(4096.0)


def test_local_hit_has_zero_transfer_bytes():
    entries = [_entry("k1", 100, "w0", kv_bytes=4096.0)]
    result = _sim().evaluate(_request("k1", 100, 100, preferred_worker_id="w0"),
                              entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "local_hit"
    assert result.remote_transfer_bytes == pytest.approx(0.0)


def test_does_not_mutate_entries():
    entries = [
        _entry("k1", 80, "w0", kv_bytes=512.0, last_access_ms=1.0),
        _entry("k2", 40, "w1", kv_bytes=256.0, last_access_ms=2.0),
    ]
    snapshot = [(e.cache_key, e.cached_tokens, e.worker_id, e.kv_bytes, e.last_access_ms)
                for e in entries]
    _sim().evaluate(_request("k1", 100, 100), entries, _BASE_PREFILL, 100.0)
    after = [(e.cache_key, e.cached_tokens, e.worker_id, e.kv_bytes, e.last_access_ms)
             for e in entries]
    assert snapshot == after


def test_eviction_uses_lru():
    # Three entries totalling 1500 bytes; budget is 700 bytes.
    # LRU order: a (oldest, t=10) → b (t=20) → c (newest, t=30).
    # Evict a (1500-500=1000 > 700), then b (1000-500=500 ≤ 700). Stop.
    entries = [
        _entry("a", 100, "w0", kv_bytes=500.0, last_access_ms=10.0),
        _entry("b", 100, "w0", kv_bytes=500.0, last_access_ms=20.0),
        _entry("c", 100, "w0", kv_bytes=500.0, last_access_ms=30.0),
    ]
    result = _sim().evaluate(_request("x", 10, 10), entries, _BASE_PREFILL, 700.0)
    assert "a" in result.evicted_cache_keys
    assert "b" in result.evicted_cache_keys
    assert "c" not in result.evicted_cache_keys


def test_zero_prompt_tokens_safe():
    entries = [_entry("k1", 100, "w0")]
    result = _sim().evaluate(_request("k1", 0, 0), entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_tokens == 0
    assert result.miss_tokens == 0
    assert result.hit_ratio == pytest.approx(0.0)
    assert result.saved_prefill_ms == pytest.approx(0.0)


def test_truth_boundary_is_explicit():
    result = _sim().evaluate(_request("k", 10, 10), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert result.truth_boundary == "prefix_cache_simulated_not_real_kv_cache"


def test_no_wall_clock_sleep():
    import deployment.prefix_cache_simulator as mod
    src = inspect.getsource(mod)
    assert "time.sleep(" not in src, "Simulator must not call time.sleep()"
    assert "time.time(" not in src, "Simulator must not use time.time()"
    assert "datetime.now(" not in src, "Simulator must not use datetime.now()"
    assert "import time" not in src, "Simulator must not import the time module"


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------

def test_result_is_prefix_cache_result():
    result = _sim().evaluate(_request("k", 10, 10), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert isinstance(result, PrefixCacheResult)


def test_miss_has_zero_saved_prefill_ms():
    result = _sim().evaluate(_request("absent", 50, 50), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert result.saved_prefill_ms == pytest.approx(0.0)


def test_miss_has_zero_remote_transfer_bytes():
    result = _sim().evaluate(_request("absent", 50, 50), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert result.remote_transfer_bytes == pytest.approx(0.0)


def test_miss_tokens_plus_hit_tokens_equals_prompt_tokens():
    entries = [_entry("k1", 40, "w0")]
    result = _sim().evaluate(_request("k1", 100, 100), entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_tokens + result.miss_tokens == 100


def test_hit_ratio_is_one_for_full_hit():
    entries = [_entry("k1", 100, "w0")]
    result = _sim().evaluate(_request("k1", 100, 100), entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_ratio == pytest.approx(1.0)


def test_hit_ratio_zero_for_miss():
    result = _sim().evaluate(_request("absent", 100, 100), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_ratio == pytest.approx(0.0)


def test_local_hit_preferred_over_remote():
    # Both local and remote entries exist; local must win.
    entries = [
        _entry("k1", 80, "w1"),   # remote
        _entry("k1", 80, "w0"),   # local (preferred)
    ]
    result = _sim().evaluate(_request("k1", 100, 100, preferred_worker_id="w0"),
                              entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "local_hit"


def test_no_eviction_when_within_budget():
    entries = [_entry("a", 100, "w0", kv_bytes=200.0)]
    result = _sim().evaluate(_request("x", 10, 10), entries, _BASE_PREFILL, 500.0)
    assert result.evicted_cache_keys == ()


def test_eviction_lru_order_is_ascending_last_access():
    # Youngest entry should survive when only one eviction is needed.
    entries = [
        _entry("old",  100, "w0", kv_bytes=600.0, last_access_ms=1.0),
        _entry("new",  100, "w0", kv_bytes=600.0, last_access_ms=99.0),
    ]
    result = _sim().evaluate(_request("x", 10, 10), entries, _BASE_PREFILL, 700.0)
    assert "old" in result.evicted_cache_keys
    assert "new" not in result.evicted_cache_keys


def test_eviction_respects_exact_budget_boundary():
    # Total = 1000; budget = 1000 → no eviction needed.
    entries = [
        _entry("a", 100, "w0", kv_bytes=500.0, last_access_ms=1.0),
        _entry("b", 100, "w0", kv_bytes=500.0, last_access_ms=2.0),
    ]
    result = _sim().evaluate(_request("x", 10, 10), entries, _BASE_PREFILL, 1000.0)
    assert result.evicted_cache_keys == ()


def test_remote_transfer_bytes_proportional_to_hit_tokens():
    # Entry has 200 tokens and 4000 kv_bytes.
    # Request only reuses 100 tokens → 50% of kv_bytes.
    entries = [_entry("k1", 200, "w1", kv_bytes=4000.0)]
    result = _sim().evaluate(_request("k1", 200, 100, preferred_worker_id="w0"),
                              entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "remote_hit"
    assert result.hit_tokens == 100
    assert result.remote_transfer_bytes == pytest.approx(2000.0)


def test_result_cache_key_matches_request():
    result = _sim().evaluate(_request("my_key", 10, 10), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert result.cache_key == "my_key"


def test_result_is_immutable():
    result = _sim().evaluate(_request("k", 10, 10), [], _BASE_PREFILL, _LARGE_BUDGET)
    with pytest.raises((AttributeError, TypeError)):
        result.hit_type = "local_hit"  # type: ignore[misc]


def test_entry_is_immutable():
    e = _entry("k", 100, "w0")
    with pytest.raises((AttributeError, TypeError)):
        e.cached_tokens = 0  # type: ignore[misc]


def test_empty_entries_list():
    result = _sim().evaluate(_request("k", 50, 50), [], _BASE_PREFILL, _LARGE_BUDGET)
    assert result.hit_type == "miss"
    assert result.evicted_cache_keys == ()


def test_multiple_calls_are_independent():
    # Two calls with different requests must not interfere.
    entries = [_entry("k1", 100, "w0")]
    r1 = _sim().evaluate(_request("k1", 100, 100), entries, _BASE_PREFILL, _LARGE_BUDGET)
    r2 = _sim().evaluate(_request("k2", 100, 100), entries, _BASE_PREFILL, _LARGE_BUDGET)
    assert r1.hit_type == "local_hit"
    assert r2.hit_type == "miss"
