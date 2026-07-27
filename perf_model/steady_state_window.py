"""E2E-6: extract a common, validated steady-state decode window from N
simultaneously-launched, per-token timelines, and compute the derived batch-
shape quantities (batch-step latency, per-request TPOT, aggregate throughput,
scaling efficiency, latency inflation) from it.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WindowValidation:
    valid: bool
    reason: str
    running_count_samples_in_window: list[float] = field(default_factory=list)


def validate_running_count_window(
    samples: list[dict], window_start_time: float, window_end_time: float, expected_n: int, tolerance_s: float = 0.2,
) -> WindowValidation:
    """Confirms num_requests_running == expected_n throughout [start,end], per
    the nearest sample within tolerance. Any deviation invalidates the window
    (reject, not silently truncate the requirement away)."""
    # No padding on the boundaries: a sample just before window_start can
    # legitimately show a lower running count (request not yet admitted) and
    # must not be treated as a deviation inside the window.
    in_window = [s for s in samples if window_start_time <= s["time"] <= window_end_time]
    if not in_window:
        return WindowValidation(False, "no_running_count_samples_in_window", [])
    values = [s["num_requests_running"] for s in in_window if s.get("num_requests_running") is not None]
    if not values:
        return WindowValidation(False, "no_valid_running_count_values", [])
    if any(v != expected_n for v in values):
        return WindowValidation(False, f"running_count_deviated_from_{expected_n}:_observed_{sorted(set(values))}", values)
    return WindowValidation(True, "ok", values)


def extract_common_window_gaps(
    token_arrival_times: list[float], window_start_token: int, window_end_token: int,
) -> list[float]:
    """gaps (ms) strictly between token indices [window_start_token, window_end_token)."""
    if len(token_arrival_times) <= window_end_token:
        return []
    gaps = []
    for i in range(window_start_token, window_end_token):
        if i + 1 >= len(token_arrival_times):
            break
        gaps.append((token_arrival_times[i + 1] - token_arrival_times[i]) * 1000.0)
    return gaps


def per_request_tpot_ms(gaps_ms: list[float]) -> float | None:
    return statistics.median(gaps_ms) if gaps_ms else None


def batch_step_latency_ms(per_request_tpots_ms: list[float]) -> float | None:
    """If every request in the batch advances one token per shared scheduler
    step (continuous batching, all N requests in `running`), each request's
    own inter-token gap already equals the step latency -- so batch-step
    latency is estimated as the median across requests' own median TPOT,
    not divided or multiplied by N. Only valid when the window passed
    validate_running_count_window."""
    return statistics.median(per_request_tpots_ms) if per_request_tpots_ms else None


def aggregate_throughput_tokens_per_s(n_requests: int, window_wall_time_s: float) -> float | None:
    """n_requests tokens emitted (one per request) over the common window wall time."""
    if window_wall_time_s <= 0:
        return None
    return n_requests / window_wall_time_s


def scaling_efficiency(throughput_n: float | None, throughput_1: float | None, n: int) -> float | None:
    if throughput_n is None or throughput_1 is None or throughput_1 <= 0 or n <= 0:
        return None
    return throughput_n / (n * throughput_1)


def latency_inflation(tpot_n: float | None, tpot_1: float | None) -> float | None:
    if tpot_n is None or tpot_1 is None or tpot_1 <= 0:
        return None
    return tpot_n / tpot_1


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return statistics.stdev(values) / mean
