"""Deterministic prefix-cache simulator for distributed runtime planning.

PrefixCacheSimulator.evaluate() checks a PrefixCacheRequest against a list of
PrefixCacheEntry objects and returns a PrefixCacheResult describing the hit
type, token savings, and any LRU evictions that would be required to admit the
new entry.

This is simulation only:
- No wall clock, no time.sleep, no real KV memory.
- No mutation of input entries.
- All decisions are deterministic given the inputs.

Hit-type precedence:
  local_hit  — cache_key found on preferred_worker_id
  remote_hit — cache_key found on a different worker (local_hit takes priority)
  miss       — cache_key not found anywhere

Token and latency formulas:
  hit_tokens        = min(entry.cached_tokens, request.cacheable_tokens,
                          request.prompt_tokens)    [0 on miss]
  miss_tokens       = prompt_tokens - hit_tokens
  hit_ratio         = hit_tokens / prompt_tokens    [0.0 when prompt_tokens == 0]
  saved_prefill_ms  = base_prefill_ms * hit_tokens / prompt_tokens
                                                    [0.0 when prompt_tokens == 0]
  remote_transfer_bytes = entry.kv_bytes * hit_tokens / entry.cached_tokens
                          for remote_hit, else 0.0  [0.0 when cached_tokens == 0]

Eviction:
  When sum(e.kv_bytes for e in entries) > max_cache_bytes, entries are evicted
  in LRU order (ascending last_access_ms) until the budget is met. Evicted
  cache_keys are reported in the result. Input entries are never mutated.

Truth boundary:
  "prefix_cache_simulated_not_real_kv_cache"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

_TB = "prefix_cache_simulated_not_real_kv_cache"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixCacheEntry:
    """A single cached prefix record held by one worker.

    kv_bytes is the KV-cache footprint of the cached prefix, used for both
    eviction budget accounting and remote-transfer-cost estimation.
    last_access_ms is the simulated timestamp of the last access; used only
    for LRU eviction ordering — it does not reflect wall-clock time.
    """

    cache_key: str
    cached_tokens: int
    worker_id: str
    kv_bytes: float
    last_access_ms: float


@dataclass(frozen=True)
class PrefixCacheRequest:
    """A lookup request for a single prefix key.

    cacheable_tokens is the upper bound on how many tokens of the prefix can
    actually be reused (e.g. constrained by sequence length, alignment, or
    model-level policy). It caps hit_tokens together with cached_tokens and
    prompt_tokens.
    preferred_worker_id names the worker whose local cache should be checked
    first; a match there is a local_hit.
    """

    cache_key: str
    prompt_tokens: int
    cacheable_tokens: int
    preferred_worker_id: str


@dataclass(frozen=True)
class PrefixCacheResult:
    """Outcome of a prefix-cache lookup, including eviction information.

    hit_type is one of "miss", "local_hit", "remote_hit".
    evicted_cache_keys contains the cache_keys (in LRU order) of entries that
    would need to be evicted to stay within max_cache_bytes. An empty tuple
    means no eviction is required.
    truth_boundary is always "prefix_cache_simulated_not_real_kv_cache".
    """

    cache_key: str
    hit_type: str
    hit_tokens: int
    miss_tokens: int
    hit_ratio: float
    saved_prefill_ms: float
    remote_transfer_bytes: float
    evicted_cache_keys: tuple[str, ...]
    truth_boundary: str


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class PrefixCacheSimulator:
    """Deterministic prefix-cache simulator.

    All methods are pure functions of their arguments. No state is stored
    between calls. No wall clock is used. No input is mutated.
    """

    def evaluate(
        self,
        request: PrefixCacheRequest,
        entries: list[PrefixCacheEntry],
        base_prefill_ms: float,
        max_cache_bytes: float,
    ) -> PrefixCacheResult:
        """Evaluate a cache request and return a PrefixCacheResult.

        entries is treated as read-only. max_cache_bytes is the total KV-byte
        budget; any excess triggers simulated LRU eviction.
        """
        # Resolve hit type: prefer local over remote.
        local_entry: Optional[PrefixCacheEntry] = None
        remote_entry: Optional[PrefixCacheEntry] = None

        for entry in entries:
            if entry.cache_key == request.cache_key:
                if entry.worker_id == request.preferred_worker_id:
                    local_entry = entry
                elif remote_entry is None:
                    remote_entry = entry

        if local_entry is not None:
            hit_type = "local_hit"
            matching_entry: Optional[PrefixCacheEntry] = local_entry
        elif remote_entry is not None:
            hit_type = "remote_hit"
            matching_entry = remote_entry
        else:
            hit_type = "miss"
            matching_entry = None

        # Token accounting.
        if matching_entry is not None and request.prompt_tokens > 0:
            hit_tokens = min(
                matching_entry.cached_tokens,
                request.cacheable_tokens,
                request.prompt_tokens,
            )
        else:
            hit_tokens = 0

        miss_tokens = request.prompt_tokens - hit_tokens

        if request.prompt_tokens > 0:
            hit_ratio = hit_tokens / request.prompt_tokens
            saved_prefill_ms = base_prefill_ms * hit_tokens / request.prompt_tokens
        else:
            hit_ratio = 0.0
            saved_prefill_ms = 0.0

        # Remote transfer cost.
        if (
            hit_type == "remote_hit"
            and matching_entry is not None
            and matching_entry.cached_tokens > 0
        ):
            remote_transfer_bytes = (
                matching_entry.kv_bytes * hit_tokens / matching_entry.cached_tokens
            )
        else:
            remote_transfer_bytes = 0.0

        # LRU eviction simulation (does not mutate entries).
        evicted_keys = _simulate_lru_eviction(entries, max_cache_bytes)

        return PrefixCacheResult(
            cache_key=request.cache_key,
            hit_type=hit_type,
            hit_tokens=hit_tokens,
            miss_tokens=miss_tokens,
            hit_ratio=hit_ratio,
            saved_prefill_ms=saved_prefill_ms,
            remote_transfer_bytes=remote_transfer_bytes,
            evicted_cache_keys=tuple(evicted_keys),
            truth_boundary=_TB,
        )


def _simulate_lru_eviction(
    entries: list[PrefixCacheEntry],
    max_cache_bytes: float,
) -> list[str]:
    """Return cache_keys to evict (LRU order) so total kv_bytes <= max_cache_bytes.

    Does not mutate entries. Returns an empty list if no eviction is needed.
    """
    total = sum(e.kv_bytes for e in entries)
    if total <= max_cache_bytes:
        return []

    sorted_by_lru = sorted(entries, key=lambda e: e.last_access_ms)
    evicted: list[str] = []
    remaining = total
    for entry in sorted_by_lru:
        if remaining <= max_cache_bytes:
            break
        evicted.append(entry.cache_key)
        remaining -= entry.kv_bytes
    return evicted
