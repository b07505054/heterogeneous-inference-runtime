"""E2E-5: align anchor inter-token gaps against the OBSERVED num_requests_running
timeseries (sampled at ~120ms resolution during the round), rather than
assuming everything after admission is "interfered". This is the key
methodological correction this slice makes over E2E-4: a naive post-admission
median can be diluted by short-lived admitted requests finishing early and
the anchor recovering for the remainder of the window, understating the true
concurrent-decode rate. Aligning against the real sampled running count
avoids that dilution.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class AlignedGaps:
    solo_gaps_ms: list[float]        # gaps while nearest sample shows running <= 1
    concurrent_gaps_ms: list[float]  # gaps while nearest sample shows running >= 2
    unmatched_gaps_ms: list[float]   # no sample within tolerance of the gap's end time


def _nearest_sample(samples: list[dict], t: float, tolerance_s: float) -> dict | None:
    best = None
    best_dt = tolerance_s
    for s in samples:
        dt = abs(s["time"] - t)
        if dt <= best_dt:
            best = s
            best_dt = dt
    return best


def align_gaps_to_running_count(
    token_arrival_times: list[float], samples: list[dict], *, tolerance_s: float = 0.25,
) -> AlignedGaps:
    solo, concurrent, unmatched = [], [], []
    for i in range(len(token_arrival_times) - 1):
        start, end = token_arrival_times[i], token_arrival_times[i + 1]
        gap_ms = (end - start) * 1000.0
        sample = _nearest_sample(samples, end, tolerance_s)
        if sample is None or sample.get("num_requests_running") is None:
            unmatched.append(gap_ms)
        elif sample["num_requests_running"] >= 2:
            concurrent.append(gap_ms)
        else:
            solo.append(gap_ms)
    return AlignedGaps(solo_gaps_ms=solo, concurrent_gaps_ms=concurrent, unmatched_gaps_ms=unmatched)


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def peak_running_waiting(samples: list[dict]) -> dict:
    running = [s["num_requests_running"] for s in samples if s.get("num_requests_running") is not None]
    waiting = [s["num_requests_waiting"] for s in samples if s.get("num_requests_waiting") is not None]
    return {
        "peak_requests_running": max(running) if running else None,
        "peak_requests_waiting": max(waiting) if waiting else None,
        "n_samples": len(samples),
    }
