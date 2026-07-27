"""E2E-4: derived interference labels from an anchor request's per-token
timeline, split into a pre-admission baseline window and a post-admission
window by the real observed admission timestamp (not assumed).

All functions take plain gap/timestamp lists (seconds, from
perf_model.token_timeline) and return milliseconds, matching the rest of the
perf_model package's units convention.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

DEFAULT_RECOVERY_TOLERANCE = 1.5  # gap must be within 1.5x baseline
DEFAULT_RECOVERY_CONSECUTIVE = 3  # for this many consecutive tokens to count as "recovered"


@dataclass(frozen=True)
class SplitTimeline:
    """Anchor token_arrival_times split into pre/post-admission gap lists,
    using the real admission_time (a submit_time observed from the admitted
    requests, not an assumption)."""
    pre_gaps_ms: list[float]
    post_gaps_ms: list[float]
    post_gap_end_times: list[float]  # wall-clock end time of each post-admission gap, for recovery search
    admission_time: float


def split_timeline(token_arrival_times: list[float], admission_time: float) -> SplitTimeline:
    gaps = [(token_arrival_times[i], token_arrival_times[i + 1], (token_arrival_times[i + 1] - token_arrival_times[i]) * 1000.0)
            for i in range(len(token_arrival_times) - 1)]
    pre = [g[2] for g in gaps if g[1] <= admission_time]
    post = [g for g in gaps if g[1] > admission_time]
    return SplitTimeline(pre_gaps_ms=pre, post_gaps_ms=[g[2] for g in post],
                          post_gap_end_times=[g[1] for g in post], admission_time=admission_time)


def baseline_gap_ms(split: SplitTimeline) -> float | None:
    if not split.pre_gaps_ms:
        return None
    return statistics.median(split.pre_gaps_ms)


def peak_stall_ms(split: SplitTimeline, baseline: float | None) -> float | None:
    if baseline is None or not split.post_gaps_ms:
        return None
    return max(split.post_gaps_ms) - baseline


def total_stall_area_ms(split: SplitTimeline, baseline: float | None) -> float | None:
    if baseline is None or not split.post_gaps_ms:
        return None
    return sum(max(g - baseline, 0.0) for g in split.post_gaps_ms)


def affected_token_count(split: SplitTimeline, baseline: float | None, threshold_ratio: float = 1.2) -> int:
    if baseline is None or baseline <= 0:
        return 0
    return sum(1 for g in split.post_gaps_ms if g > baseline * threshold_ratio)


def recovery_time_ms(
    split: SplitTimeline, baseline: float | None, *,
    tolerance: float = DEFAULT_RECOVERY_TOLERANCE, consecutive: int = DEFAULT_RECOVERY_CONSECUTIVE,
) -> float | None:
    """Time from admission until `consecutive` back-to-back post-admission
    gaps all fall within `tolerance` x baseline. None if never recovered
    within the captured window (explicit, not silently zero)."""
    if baseline is None or not split.post_gaps_ms:
        return None
    threshold = baseline * tolerance
    run = 0
    for gap, end_time in zip(split.post_gaps_ms, split.post_gap_end_times):
        if gap <= threshold:
            run += 1
            if run >= consecutive:
                return (end_time - split.admission_time) * 1000.0
        else:
            run = 0
    return None  # never recovered within the captured window


def sustained_slowdown_ratio(split: SplitTimeline, baseline: float | None) -> float | None:
    if baseline is None or baseline <= 0 or not split.post_gaps_ms:
        return None
    return statistics.median(split.post_gaps_ms) / baseline


def post_admission_percentiles(split: SplitTimeline) -> dict:
    if not split.post_gaps_ms:
        return {"p50": None, "p95": None, "max": None}
    ordered = sorted(split.post_gaps_ms)
    p95_idx = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return {"p50": statistics.median(ordered), "p95": ordered[p95_idx], "max": ordered[-1]}


def interference_e2e_ms(
    measured_anchor_e2e_ms: float | None, ttft_ms: float | None, baseline_gap: float | None, output_tokens: int,
) -> float | None:
    """estimated_isolated_anchor_e2e uses the anchor's OWN pre-admission
    baseline gap (self-referential, not a cross-experiment assumption) to
    project what its E2E would have been with no interference at all."""
    if measured_anchor_e2e_ms is None or ttft_ms is None or baseline_gap is None:
        return None
    estimated_isolated_e2e = ttft_ms + max(output_tokens - 1, 0) * baseline_gap
    return measured_anchor_e2e_ms - estimated_isolated_e2e
