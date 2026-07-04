from __future__ import annotations

from typing import Any


VALID_OBJECTIVES = {"latency", "throughput", "memory", "package_size", "size", "balanced"}
BALANCED_WEIGHTS = {
    "latency_ms": 0.4,
    "throughput": 0.3,
    "memory_mb": 0.2,
    "package_mb": 0.1,
}


def validate_objective(objective: str) -> str:
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(VALID_OBJECTIVES)}")
    return objective


def objective_sort_key(candidate: dict[str, Any], candidates: list[dict[str, Any]], objective: str):
    objective = validate_objective(objective)
    metrics = candidate.get("metrics", {})
    if objective == "latency":
        return (_metric(metrics, "latency_ms"), -_metric(metrics, "throughput"), candidate["source_artifact"])
    if objective == "throughput":
        return (-_metric(metrics, "throughput"), _metric(metrics, "latency_ms"), candidate["source_artifact"])
    if objective == "memory":
        return (_metric(metrics, "memory_mb"), _metric(metrics, "latency_ms"), candidate["source_artifact"])
    if objective in {"package_size", "size"}:
        return (_metric(metrics, "package_mb"), _metric(metrics, "latency_ms"), candidate["source_artifact"])
    return (balanced_score(candidate, candidates), candidate["source_artifact"])


def balanced_score(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> float:
    """Weighted score: latency 0.4, throughput 0.3, memory 0.2, package 0.1."""

    metrics = candidate.get("metrics", {})
    score = 0.0
    used_weight = 0.0
    for metric_name, weight in BALANCED_WEIGHTS.items():
        values = [
            other.get("metrics", {}).get(metric_name)
            for other in candidates
            if _is_number(other.get("metrics", {}).get(metric_name))
        ]
        value = metrics.get(metric_name)
        if not values or not _is_number(value):
            continue
        if metric_name == "throughput":
            normalized = _normalize_higher_is_better(float(value), values)
        else:
            normalized = _normalize_lower_is_better(float(value), values)
        score += normalized * weight
        used_weight += weight
    if used_weight == 0.0:
        return float("inf")
    return round(score / used_weight, 9)


def _metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if _is_number(value):
        return float(value)
    return float("inf")


def _normalize_lower_is_better(value: float, values: list[float]) -> float:
    low = min(values)
    high = max(values)
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def _normalize_higher_is_better(value: float, values: list[float]) -> float:
    low = min(values)
    high = max(values)
    if high == low:
        return 0.0
    return (high - value) / (high - low)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
