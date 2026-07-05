"""Loader and validator for compiler ExecutionPlan v2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deployment.execution_plan_v2.schema import ExecutionPlanV2


class ExecutionPlanV2Error(ValueError):
    """Raised when an ExecutionPlan v2 artifact is malformed or contaminated."""


_MEASURED_FIELDS = {
    "measured_latency_ms",
    "actual_latency_ms",
    "measured_memory_mb",
    "measured_speedup",
    "speedup",
    "performance_claim",
    "runtime_result",
    "metrics",
}


def load_execution_plan_v2(path: str | Path) -> ExecutionPlanV2:
    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExecutionPlanV2Error(f"execution plan not found: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise ExecutionPlanV2Error(f"invalid execution plan JSON: {exc}") from exc
    return parse_execution_plan_v2(payload)


def parse_execution_plan_v2(payload: dict[str, Any]) -> ExecutionPlanV2:
    validate_execution_plan_v2(payload)
    return ExecutionPlanV2.from_dict(payload)


def validate_execution_plan_v2(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        raise ExecutionPlanV2Error("ExecutionPlan v2 must be a JSON object")
    if payload.get("schema") != "execution_plan":
        errors.append("schema must be execution_plan")
    if payload.get("schema_version") != "2.0.0":
        errors.append("schema_version must be 2.0.0")
    for key in ("plan_id", "provenance", "model_identity", "global_decisions", "function_plans"):
        if key not in payload:
            errors.append(f"missing required field: {key}")
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict) or "capability_bundle" not in provenance:
        errors.append("missing required field: provenance.capability_bundle")
    if not isinstance(payload.get("function_plans", []), list):
        errors.append("function_plans must be a list")
    if _contains_measured_field(payload):
        errors.append("compiler execution plan must not contain measured runtime fields")
    if errors:
        raise ExecutionPlanV2Error("; ".join(errors))
    return []


def _contains_measured_field(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _MEASURED_FIELDS:
                return True
            if lowered.startswith("measured_"):
                return True
            if _contains_measured_field(child):
                return True
    elif isinstance(value, list):
        return any(_contains_measured_field(child) for child in value)
    return False
