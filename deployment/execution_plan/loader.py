"""Loader and validator for compiler ExecutionPlan v2 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deployment.execution_plan.schema import ExecutionPlan


class ExecutionPlanError(ValueError):
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


def load_execution_plan(path: str | Path) -> ExecutionPlan:
    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExecutionPlanError(f"execution plan not found: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise ExecutionPlanError(f"invalid execution plan JSON: {exc}") from exc
    return parse_execution_plan(payload)


def parse_execution_plan(payload: dict[str, Any]) -> ExecutionPlan:
    validate_execution_plan(payload)
    return ExecutionPlan.from_dict(payload)


def validate_execution_plan(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        raise ExecutionPlanError("ExecutionPlan v2 must be a JSON object")
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
    errors.extend(_validate_exact_rmsnorm_decisions(payload))
    if errors:
        raise ExecutionPlanError("; ".join(errors))
    return []


def _validate_exact_rmsnorm_decisions(payload: dict[str, Any]) -> list[str]:
    errors = []
    for function in payload.get("function_plans", []):
        for op in function.get("per_op_decisions", []):
            kernel = op.get("kernel") or {}
            if kernel.get("decision_kind") != "rmsnorm_gpu_exact_config_selection":
                continue
            for key in ("candidate_id", "operator", "semantics", "backend", "kernel_family", "kernel_entry_point", "dtype", "tokens", "hidden", "epsilon", "launch_config", "artifact", "target"):
                if kernel.get(key) is None:
                    errors.append(f"missing exact RMSNorm field: {key}")
            if kernel.get("selected_kernel") != kernel.get("candidate_id"):
                errors.append("exact RMSNorm selected_kernel must equal candidate_id")
            if kernel.get("operator") != "rmsnorm" or kernel.get("semantics") != "weighted_rmsnorm":
                errors.append("exact RMSNorm requires weighted_rmsnorm semantics")
    return errors


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
