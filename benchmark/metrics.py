from __future__ import annotations

import statistics
from typing import Iterable


def percentile(values: Iterable[float], p: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = round((len(ordered) - 1) * p / 100.0)
    index = min(len(ordered) - 1, max(0, int(index)))
    return ordered[index]


def latency_summary_ms(values: Iterable[float]) -> dict:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(samples),
        "mean": round(statistics.mean(samples), 6),
        "p50": round(percentile(samples, 50), 6),
        "p95": round(percentile(samples, 95), 6),
        "p99": round(percentile(samples, 99), 6),
        "min": round(min(samples), 6),
        "max": round(max(samples), 6),
    }


def token_throughput(total_tokens: int, elapsed_ms: float) -> float | None:
    if total_tokens <= 0 or elapsed_ms <= 0:
        return None
    return round(total_tokens / (elapsed_ms / 1000.0), 6)


def openai_latency_metrics(results: Iterable[dict]) -> dict:
    rows = list(results)
    successes = [row for row in rows if row.get("ok")]
    errors = [row for row in rows if not row.get("ok")]
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in successes)
    elapsed_ms = sum(float(row.get("e2e_latency_ms") or 0.0) for row in successes)
    return {
        "ttft_ms": latency_summary_ms(
            row["ttft_ms"] for row in successes if row.get("ttft_ms") is not None
        ),
        "tpot_ms": latency_summary_ms(
            row["tpot_ms"] for row in successes if row.get("tpot_ms") is not None
        ),
        "e2e_latency_ms": latency_summary_ms(
            row["e2e_latency_ms"] for row in successes if row.get("e2e_latency_ms") is not None
        ),
        "tokens_per_second": token_throughput(output_tokens, elapsed_ms),
        "success_count": len(successes),
        "error_count": len(errors),
        "total_output_tokens": output_tokens,
    }

