from __future__ import annotations

import json
import re
from typing import Any


def extract_backend_metrics(summary_text: str, report_text: str = "") -> list[dict[str, Any]]:
    data = json.loads(summary_text)
    if not isinstance(data, list):
        raise ValueError("backend summary must be a list")

    consistency = _extract_consistency(report_text)
    metrics: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        key = _consistency_key(row)
        metric = {
            "backend": row.get("backend"),
            "precision": row.get("precision"),
            "device": row.get("device"),
            "avg_latency_ms": row.get("avg_latency_ms"),
            "p95_latency_ms": row.get("p95_latency_ms"),
            "p99_latency_ms": row.get("p99_latency_ms"),
            "throughput_qps": row.get("throughput_qps"),
            "correctness": consistency.get(key, "not_reported"),
            "source": "results/backend_validation_summary.json",
        }
        metrics.append(metric)

    return metrics


def filter_candidates(
    metrics: list[dict[str, Any]],
    metric: str,
    operator: str,
    threshold: float,
) -> list[dict[str, Any]]:
    if operator != "<":
        raise ValueError("only '<' is supported for this eval task")

    candidates = []
    for row in metrics:
        value = row.get(metric)
        if value is None:
            continue
        if float(value) < threshold:
            candidates.append(row)
    return candidates


def compare_candidates(
    candidates: list[dict[str, Any]],
    primary: str,
    tie_breakers: list[str] | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("no candidates to compare")

    tie_breakers = tie_breakers or []
    fields = [primary, *tie_breakers]

    def key(row: dict[str, Any]) -> tuple:
        values = []
        for field in fields:
            value = row.get(field)
            if value is None:
                values.append(float("-inf"))
            elif "latency" in field or field.startswith("p"):
                values.append(-float(value))
            else:
                values.append(float(value))
        return tuple(values)

    return max(candidates, key=key)


def final_answer(recommendation: dict[str, Any], evidence: list[dict[str, Any]], caveats: list[str]) -> dict:
    return {
        "recommended_backend": recommendation.get("backend"),
        "recommended_precision": recommendation.get("precision"),
        "recommended_device": recommendation.get("device"),
        "p95_latency_ms": recommendation.get("p95_latency_ms"),
        "throughput_qps": recommendation.get("throughput_qps"),
        "correctness": recommendation.get("correctness"),
        "evidence": evidence,
        "caveats": caveats,
        "explanation": (
            f"Select {recommendation.get('backend')} / {recommendation.get('precision')} "
            f"/ {recommendation.get('device')} because it satisfies p95 < 5 ms "
            f"with p95={recommendation.get('p95_latency_ms')} ms and has the highest "
            f"throughput among valid candidates at {recommendation.get('throughput_qps')} QPS."
        ),
    }


def _extract_consistency(report_text: str) -> dict[str, str]:
    consistency: dict[str, str] = {}
    for line in report_text.splitlines():
        match = re.match(r"\|\s*(ONNX FP32|Optimized ONNX FP32|INT8 ONNX)\s*\|\s*([^|]+)\|", line)
        if not match:
            continue
        label, value = match.groups()
        normalized = value.strip()
        if label == "ONNX FP32":
            consistency["ONNXRuntime:FP32 CUDA"] = normalized
        elif label == "Optimized ONNX FP32":
            consistency["ONNXRuntime:Optimized FP32 CUDA"] = normalized
        elif label == "INT8 ONNX":
            consistency["ONNXRuntime:INT8 CPU"] = normalized
    return consistency


def _consistency_key(row: dict[str, Any]) -> str:
    return f"{row.get('backend')}:{row.get('precision')}"
