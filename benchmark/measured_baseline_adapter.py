from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_STATUSES = {"ok", "partial", "unavailable"}
LOWER_IS_BETTER_TOKENS = (
    "latency",
    "ttft",
    "tpot",
    "package_size",
    "drift",
    "max_abs",
    "mean_abs",
)
HIGHER_IS_BETTER_TOKENS = (
    "tokens_per_second",
    "fps",
    "qps",
    "throughput",
)


def load_measured_baseline(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_measured_baseline(payload)
    metrics = extract_comparable_metrics(payload.get("metrics", {}))
    return {
        "source_path": str(path),
        "artifact_type": payload["artifact_type"],
        "evidence_type": payload["evidence_type"],
        "status": payload["status"],
        "benchmark_target": payload["benchmark_target"],
        "metrics": metrics,
    }


def validate_measured_baseline(payload: dict) -> None:
    if payload.get("artifact_type") != "measured_baseline":
        raise ValueError("artifact_type must be measured_baseline")
    if payload.get("evidence_type") != "measured":
        raise ValueError("evidence_type must be measured")
    if payload.get("status") not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    if not isinstance(payload.get("benchmark_target"), dict):
        raise ValueError("benchmark_target must be an object")
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError("metrics must be an object")


def extract_comparable_metrics(metrics: dict) -> dict[str, float]:
    extracted: dict[str, float] = {}
    _extract_recursive(metrics, prefix="", output=extracted)
    return extracted


def compare_measured_baselines(before_path: str | Path, after_path: str | Path) -> dict:
    before = load_measured_baseline(before_path)
    after = load_measured_baseline(after_path)
    warnings = []
    if before["benchmark_target"] != after["benchmark_target"]:
        warnings.append("benchmark_target differs between before and after artifacts")

    before_metrics = before["metrics"]
    after_metrics = after["metrics"]
    common = sorted(set(before_metrics) & set(after_metrics))
    deltas = {}
    for name in common:
        direction = _metric_direction(name)
        before_value = before_metrics[name]
        after_value = after_metrics[name]
        delta = after_value - before_value
        percent_delta = (delta / before_value * 100.0) if before_value else None
        improved = None
        if direction == "lower":
            improved = delta < 0
        elif direction == "higher":
            improved = delta > 0
        deltas[name] = {
            "before": before_value,
            "after": after_value,
            "delta": round(delta, 6),
            "percent_delta": round(percent_delta, 6) if percent_delta is not None else None,
            "direction": direction,
            "improved": improved,
        }

    return {
        "artifact_type": "measured_baseline_comparison",
        "before": {
            "source_path": before["source_path"],
            "evidence_type": before["evidence_type"],
            "status": before["status"],
            "benchmark_target": before["benchmark_target"],
        },
        "after": {
            "source_path": after["source_path"],
            "evidence_type": after["evidence_type"],
            "status": after["status"],
            "benchmark_target": after["benchmark_target"],
        },
        "warnings": warnings,
        "compared_metrics": deltas,
        "missing_metrics": {
            "before_only": sorted(set(before_metrics) - set(after_metrics)),
            "after_only": sorted(set(after_metrics) - set(before_metrics)),
        },
    }


def comparison_to_markdown(summary: dict) -> str:
    lines = [
        "# Measured Baseline Comparison",
        "",
        f"- Before: `{summary['before']['source_path']}`",
        f"- After: `{summary['after']['source_path']}`",
        f"- Before evidence: `{summary['before']['evidence_type']}`",
        f"- After evidence: `{summary['after']['evidence_type']}`",
    ]
    if summary["warnings"]:
        lines.append("")
        lines.append("## Warnings")
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Before | After | Delta | Direction | Improved |")
    lines.append("|---|---:|---:|---:|---|---|")
    for name, row in summary["compared_metrics"].items():
        lines.append(
            f"| `{name}` | {row['before']} | {row['after']} | "
            f"{row['delta']} | {row['direction']} | {row['improved']} |"
        )
    if not summary["compared_metrics"]:
        lines.append("| No common comparable metrics |  |  |  |  |  |")
    return "\n".join(lines) + "\n"


def _extract_recursive(value: Any, *, prefix: str, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        if _is_percentile_dict(value):
            for percentile in ("p50", "p95", "p99"):
                if percentile in value and _is_number(value[percentile]):
                    output[f"{prefix}.{percentile}".strip(".")] = float(value[percentile])
            return
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _extract_recursive(child, prefix=next_prefix, output=output)
        return
    if _is_number(value) and _is_comparable_metric_name(prefix):
        output[prefix] = float(value)


def _is_percentile_dict(value: dict) -> bool:
    return any(key in value for key in ("p50", "p95", "p99"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_comparable_metric_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in LOWER_IS_BETTER_TOKENS + HIGHER_IS_BETTER_TOKENS)


def _metric_direction(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in HIGHER_IS_BETTER_TOKENS):
        return "higher"
    if any(token in lowered for token in LOWER_IS_BETTER_TOKENS):
        return "lower"
    return "unknown"
