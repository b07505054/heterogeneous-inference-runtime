"""CoreML compiler metadata contract.

This module validates the runtime-facing metadata that sits beside a
compiler-produced CoreML package. It does not import coremltools, execute
CoreML, or consume compiler IR.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deployment.model_adapter.neutral_runtime_graph import (
    NeutralBackendTarget,
    NeutralConstraint,
    NeutralKVCacheRequirement,
    NeutralMemoryRequirement,
    NeutralStage,
    NeutralTensor,
)


class CoreMLMetadataError(ValueError):
    """Raised when a CoreML package metadata contract is invalid."""


@dataclass(frozen=True)
class CoreMLCompilerMetadata:
    model_family: str
    stages: tuple[NeutralStage, ...]
    input_tensors: tuple[NeutralTensor, ...]
    output_tensors: tuple[NeutralTensor, ...]
    memory_requirements: NeutralMemoryRequirement
    kv_cache_requirements: NeutralKVCacheRequirement
    preferred_backend: str
    constraints: tuple[NeutralConstraint, ...]
    compiler_version: str
    source_artifact_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages or ()))
        object.__setattr__(self, "input_tensors", tuple(self.input_tensors or ()))
        object.__setattr__(self, "output_tensors", tuple(self.output_tensors or ()))
        object.__setattr__(self, "constraints", tuple(self.constraints or ()))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def tensors(self) -> tuple[NeutralTensor, ...]:
        return (*self.input_tensors, *self.output_tensors)

    @property
    def backend_target(self) -> NeutralBackendTarget:
        return NeutralBackendTarget(
            preferred_backend=self.preferred_backend,
            allowed_backends=(self.preferred_backend,),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.model_family:
            errors.append("model_family_required")
        if not self.stages:
            errors.append("stages_required")
        if not self.input_tensors:
            errors.append("input_tensors_required")
        if not self.output_tensors:
            errors.append("output_tensors_required")
        if not self.preferred_backend:
            errors.append("preferred_backend_required")
        if not self.compiler_version:
            errors.append("compiler_version_required")
        if not self.source_artifact_kind:
            errors.append("source_artifact_kind_required")
        if self.kv_cache_requirements.required:
            if self.kv_cache_requirements.max_context_tokens is None:
                errors.append("kv_cache_max_context_tokens_required")
            if self.kv_cache_requirements.bytes_per_token is None:
                errors.append("kv_cache_bytes_per_token_required")
        return errors


def load_coreml_compiler_metadata(package_path: str | Path) -> CoreMLCompilerMetadata:
    """Load and validate metadata next to a CoreML package directory."""
    package = Path(package_path)
    if not package.exists():
        raise CoreMLMetadataError(f"CoreML package path does not exist: {package}")

    metadata_path = package.parent / "compiler_metadata.json"
    if not metadata_path.exists():
        raise CoreMLMetadataError(
            f"compiler_metadata.json not found next to CoreML package: {metadata_path}"
        )

    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CoreMLMetadataError(
            f"invalid compiler_metadata.json: {exc.msg}"
        ) from exc
    except OSError as exc:
        raise CoreMLMetadataError(f"failed to read compiler_metadata.json: {exc}") from exc

    metadata = parse_coreml_compiler_metadata(payload)
    errors = metadata.validate()
    if errors:
        raise CoreMLMetadataError(
            f"invalid CoreML compiler metadata: {', '.join(errors)}"
        )
    return metadata


def parse_coreml_compiler_metadata(payload: dict[str, Any]) -> CoreMLCompilerMetadata:
    if not isinstance(payload, dict):
        raise CoreMLMetadataError("compiler metadata must be a JSON object")

    _reject_compiler_ir_fields(payload)

    try:
        return CoreMLCompilerMetadata(
            model_family=_required_string(payload, "model_family"),
            stages=tuple(_stage(item) for item in _required_list(payload, "stages")),
            input_tensors=tuple(
                _tensor(item, role_default="input")
                for item in _required_list(payload, "input_tensors")
            ),
            output_tensors=tuple(
                _tensor(item, role_default="output")
                for item in _required_list(payload, "output_tensors")
            ),
            memory_requirements=_memory(
                _optional_dict(payload, "memory_requirements")
            ),
            kv_cache_requirements=_kv_cache(
                _optional_dict(payload, "kv_cache_requirements")
            ),
            preferred_backend=_required_string(payload, "preferred_backend"),
            constraints=tuple(
                _constraint(item) for item in _optional_list(payload, "constraints")
            ),
            compiler_version=_required_string(payload, "compiler_version"),
            source_artifact_kind=_required_string(payload, "source_artifact_kind"),
            metadata=_optional_dict(payload, "metadata"),
        )
    except KeyError as exc:
        raise CoreMLMetadataError(f"missing required metadata field: {exc.args[0]}") from exc
    except TypeError as exc:
        raise CoreMLMetadataError(f"invalid metadata field type: {exc}") from exc


def _reject_compiler_ir_fields(payload: dict[str, Any]) -> None:
    forbidden = {"compiler_ir", "mlir", "execution_plan", "function_plans"}
    present = sorted(forbidden.intersection(payload))
    if present:
        raise CoreMLMetadataError(
            f"compiler metadata exposes compiler IR fields: {', '.join(present)}"
        )


def _stage(payload: Any) -> NeutralStage:
    if not isinstance(payload, dict):
        raise TypeError("stage entries must be objects")
    return NeutralStage(
        stage_id=_required_string(payload, "stage_id"),
        stage_type=_required_string(payload, "stage_type"),
        inputs=tuple(_optional_list(payload, "inputs")),
        outputs=tuple(_optional_list(payload, "outputs")),
        required=bool(payload.get("required", True)),
        metadata=_optional_dict(payload, "metadata"),
    )


def _tensor(payload: Any, *, role_default: str) -> NeutralTensor:
    if not isinstance(payload, dict):
        raise TypeError("tensor entries must be objects")
    return NeutralTensor(
        name=_required_string(payload, "name"),
        role=str(payload.get("role", role_default)),
        shape=tuple(_optional_list(payload, "shape")),
        dtype=str(payload.get("dtype", "unknown")),
        layout=str(payload.get("layout", "unknown")),
        dynamic=bool(payload.get("dynamic", False)),
        metadata=_optional_dict(payload, "metadata"),
    )


def _memory(payload: dict[str, Any]) -> NeutralMemoryRequirement:
    return NeutralMemoryRequirement(
        estimated_static_mb=_optional_number(payload, "estimated_static_mb"),
        estimated_peak_mb=_optional_number(payload, "estimated_peak_mb"),
        requires_unified_memory=_optional_bool(payload, "requires_unified_memory"),
        metadata=_optional_dict(payload, "metadata"),
    )


def _kv_cache(payload: dict[str, Any]) -> NeutralKVCacheRequirement:
    return NeutralKVCacheRequirement(
        required=bool(payload.get("required", False)),
        max_context_tokens=_optional_int(payload, "max_context_tokens"),
        bytes_per_token=_optional_int(payload, "bytes_per_token"),
        cache_layout=str(payload.get("cache_layout", "not_applicable")),
        metadata=_optional_dict(payload, "metadata"),
    )


def _constraint(payload: Any) -> NeutralConstraint:
    if not isinstance(payload, dict):
        raise TypeError("constraint entries must be objects")
    if "value" not in payload:
        raise KeyError("constraints.value")
    return NeutralConstraint(
        name=_required_string(payload, "name"),
        value=payload["value"],
        operator=str(payload.get("operator", "max")),
        unit=str(payload.get("unit", "")),
        metadata=_optional_dict(payload, "metadata"),
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload[key]
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _optional_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list")
    return value


def _optional_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be an object")
    return value


def _optional_number(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    return float(value)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean")
    return value
