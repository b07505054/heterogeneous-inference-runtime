from __future__ import annotations

from typing import Any


CONSTRAINT_METRICS = {
    "max_latency_ms": ("latency_ms", "latency_exceeds_max", "missing_latency_ms", "max"),
    "max_p95_ms": ("latency_ms", "latency_exceeds_max", "missing_latency_ms", "max"),
    "max_package_mb": ("package_mb", "package_exceeds_max", "missing_package_mb", "max"),
    "max_memory_mb": ("memory_mb", "memory_exceeds_max", "missing_memory_mb", "max"),
    "max_rss_mb": ("memory_mb", "memory_exceeds_max", "missing_memory_mb", "max"),
    "max_drift": ("drift", "drift_exceeds_max", "missing_drift", "max"),
    "min_throughput": ("throughput", "throughput_below_min", "missing_throughput", "min"),
    "min_tokens_per_second": ("throughput", "throughput_below_min", "missing_throughput", "min"),
}


def evaluate_constraints(candidate: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    """Evaluate runtime-neutral constraints against a normalized candidate."""

    metrics = candidate.get("metrics", {})
    reasons = list(candidate.get("reasons", []))
    for constraint_name, constraint_value in constraints.items():
        if constraint_value is None or constraint_name not in CONSTRAINT_METRICS:
            continue
        metric_name, fail_reason, missing_reason, mode = CONSTRAINT_METRICS[constraint_name]
        metric_value = metrics.get(metric_name)
        if not _is_number(metric_value):
            reasons.append(missing_reason)
            continue
        limit = float(constraint_value)
        if mode == "max" and float(metric_value) > limit:
            reasons.append(fail_reason)
        if mode == "min" and float(metric_value) < limit:
            reasons.append(fail_reason)

    return {
        "eligible": not reasons,
        "reasons": reasons,
    }


def filter_candidates(candidates: list[dict[str, Any]], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    filtered = []
    for candidate in candidates:
        result = evaluate_constraints(candidate, constraints)
        filtered.append({**candidate, **result})
    return filtered


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
