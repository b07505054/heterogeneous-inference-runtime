"""Validation helpers for compiler-produced vLLM execution plans.

The plan is planning intent only. Runtime measurements are stored separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_ARTIFACT_TYPE = "vllm_execution_plan"
EXPECTED_SCHEMA_VERSION = "1.0.0"
TRUTH_BOUNDARY = "Execution planning artifact only; not measured performance."

_REQUIRED_PATHS: tuple[tuple[str, ...], ...] = (
    ("artifact_type",),
    ("schema_version",),
    ("truth_boundary",),
    ("source_artifacts",),
    ("model", "model_id"),
    ("model", "tokenizer"),
    ("model", "dtype"),
    ("model", "quantization"),
    ("model", "trust_remote_code"),
    ("hardware_profile", "gpu_name"),
    ("hardware_profile", "vram_gb"),
    ("backend_profile", "backend"),
    ("batch_policy", "max_num_seqs"),
    ("batch_policy", "max_num_batched_tokens"),
    ("batch_policy", "enable_chunked_prefill"),
    ("prefix_policy", "enable_prefix_caching"),
    ("memory_policy", "gpu_memory_utilization"),
    ("memory_policy", "max_model_len"),
    ("memory_policy", "block_size"),
    ("memory_policy", "swap_space"),
    ("quantization_policy", "dtype"),
    ("quantization_policy", "quantization"),
    ("speculative_policy", "enabled"),
    ("runtime_config", "tensor_parallel_size"),
    ("runtime_config", "pipeline_parallel_size"),
    ("runtime_config", "served_model_name"),
)


class VLLMExecutionPlanError(ValueError):
    """Raised when a vLLM execution plan is missing required contract fields."""


def load_vllm_execution_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a compiler-produced vLLM execution plan JSON file."""
    plan_path = Path(path)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VLLMExecutionPlanError(f"vLLM execution plan not found: {plan_path}") from exc
    except json.JSONDecodeError as exc:
        raise VLLMExecutionPlanError(f"invalid vLLM execution plan JSON: {plan_path}: {exc}") from exc

    validate_vllm_execution_plan(plan)
    return plan


def validate_vllm_execution_plan(plan: dict[str, Any]) -> list[str]:
    """Validate a plan and return an empty list when it is usable."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        raise VLLMExecutionPlanError("vLLM execution plan must be a JSON object")

    for path in _REQUIRED_PATHS:
        if _lookup(plan, path) is None:
            errors.append(f"missing required field: {_format_path(path)}")

    if plan.get("artifact_type") != EXPECTED_ARTIFACT_TYPE:
        errors.append("artifact_type must be vllm_execution_plan")
    if plan.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        errors.append("schema_version must be 1.0.0")
    if not plan.get("truth_boundary"):
        errors.append("truth_boundary is required")
    if plan.get("truth_boundary") != TRUTH_BOUNDARY:
        errors.append("truth_boundary must state planning intent only")
    if plan.get("backend_profile", {}).get("backend") != "vllm":
        errors.append("backend_profile.backend must be vllm")

    runtime_config = plan.get("runtime_config", {})
    if runtime_config.get("tensor_parallel_size") != 1:
        errors.append("single-GPU Phase 1 requires tensor_parallel_size == 1")
    if runtime_config.get("pipeline_parallel_size") != 1:
        errors.append("single-GPU Phase 1 requires pipeline_parallel_size == 1")

    speculative = plan.get("speculative_policy", {})
    if speculative.get("enabled") is not False:
        if not speculative.get("draft_model"):
            errors.append("speculative_policy.enabled requires draft_model")
        errors.append("Phase 1 requires speculative_policy.enabled == false")

    if _contains_measured_claim(plan):
        errors.append("plan must not contain measured performance or speedup claims")

    if errors:
        raise VLLMExecutionPlanError("; ".join(errors))
    return []


def _lookup(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _contains_measured_claim(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered_key = str(key).lower()
            if lowered_key in {"speedup", "measured_speedup", "performance_claim"}:
                return True
            if lowered_key.startswith("measured_") and lowered_key not in {"measured_metrics_artifact"}:
                return True
            if _contains_measured_claim(item):
                return True
    elif isinstance(value, list):
        return any(_contains_measured_claim(item) for item in value)
    return False
