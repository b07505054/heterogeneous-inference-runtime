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
    errors.extend(_validate_aarch64_native_decisions(payload))
    sharding = (payload.get("global_decisions") or {}).get("cpu_sharding")
    if sharding is not None:
        try:
            from deployment.cpu_sharding import validate_sharding
            validate_sharding(sharding)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid global_decisions.cpu_sharding: {exc}")
    attention = (payload.get("global_decisions") or {}).get("attention_execution")
    if attention is not None:
        try:
            from deployment.attention_runtime import validate_attention_execution
            validate_attention_execution(attention)
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid global_decisions.attention_execution: {exc}")
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


def _validate_aarch64_native_decisions(payload: dict[str, Any]) -> list[str]:
    errors = []
    required = ("candidate_id", "operator", "kernel_family", "dtype", "shape",
                "target", "lowering", "microkernel_id", "entry_point",
                "abi_version", "object_ref", "object_sha256",
                "backend_evidence_ref", "selection_mode",
                "selection_trace_ref", "runtime_no_redecision")
    for function in payload.get("function_plans", []):
        for op in function.get("per_op_decisions", []):
            contract = op.get("native_execution") or {}
            if contract.get("decision_kind") != "aarch64_native_exact_candidate_selection":
                continue
            for key in required:
                if contract.get(key) is None:
                    errors.append(f"missing exact AArch64 native field: {key}")
            if contract.get("operator") != "hir.fused_matmul_bias_relu":
                errors.append("AArch64 native contract operator mismatch")
            if contract.get("dtype") != "f32" or contract.get("shape") != {"m": 32, "n": 32, "k": 32}:
                errors.append("AArch64 native contract supports only f32 32x32x32")
            target, lowering = contract.get("target") or {}, contract.get("lowering") or {}
            if target.get("triple") != "aarch64-linux-gnu" or target.get("cpu") != "cortex-a76":
                errors.append("AArch64 native target contract mismatch")
            expected = {"tile_m": 8, "tile_n": 8, "tile_k": 8,
                        "vector_width_bits": 128,
                        "pipeline_id": "aarch64_tiled_scheduled_v1",
                        "loop_order_id": "tiled_mnk_row_major_v1"}
            if any(lowering.get(k) != v for k, v in expected.items()) or lowering.get("schedule_unroll_k") not in (1, 2, 4):
                errors.append("AArch64 native lowering contract mismatch")
            if contract.get("abi_version") != "mlir_ciface_memref_f32_v1":
                errors.append("AArch64 native ABI contract mismatch")
            if contract.get("runtime_no_redecision") is not True:
                errors.append("AArch64 native runtime_no_redecision must be true")
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
