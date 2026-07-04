"""Neutral runtime graph schema.

The schema describes runtime concepts only. Source formats, compiler IR,
backend package internals, and exact model names belong in adapters or metadata,
not in the core dataclass fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ShapeDim = int | str | None


@dataclass(frozen=True)
class NeutralStage:
    stage_id: str
    stage_type: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    required: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set_tuple(self, "inputs", self.inputs)
        _set_tuple(self, "outputs", self.outputs)
        _set_dict(self, "metadata", self.metadata)


@dataclass(frozen=True)
class NeutralTensor:
    name: str
    role: str
    shape: tuple[ShapeDim, ...] = ()
    dtype: str = "unknown"
    layout: str = "unknown"
    dynamic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set_tuple(self, "shape", self.shape)
        _set_dict(self, "metadata", self.metadata)


@dataclass(frozen=True)
class NeutralMemoryRequirement:
    estimated_static_mb: float | None = None
    estimated_peak_mb: float | None = None
    requires_unified_memory: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set_dict(self, "metadata", self.metadata)


@dataclass(frozen=True)
class NeutralKVCacheRequirement:
    required: bool = False
    max_context_tokens: int | None = None
    bytes_per_token: int | None = None
    cache_layout: str = "not_applicable"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set_dict(self, "metadata", self.metadata)


@dataclass(frozen=True)
class NeutralBackendTarget:
    preferred_backend: str = "unspecified"
    allowed_backends: tuple[str, ...] = ()
    fallback_backends: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set_tuple(self, "allowed_backends", self.allowed_backends)
        _set_tuple(self, "fallback_backends", self.fallback_backends)
        _set_dict(self, "metadata", self.metadata)


@dataclass(frozen=True)
class NeutralConstraint:
    name: str
    value: int | float | str | bool
    operator: str = "max"
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set_dict(self, "metadata", self.metadata)


@dataclass(frozen=True)
class NeutralRuntimeGraph:
    graph_id: str
    model_family: str
    stages: tuple[NeutralStage, ...]
    tensors: tuple[NeutralTensor, ...]
    memory_requirements: NeutralMemoryRequirement = field(
        default_factory=NeutralMemoryRequirement
    )
    kv_cache_requirements: NeutralKVCacheRequirement = field(
        default_factory=NeutralKVCacheRequirement
    )
    backend_target: NeutralBackendTarget = field(default_factory=NeutralBackendTarget)
    constraints: tuple[NeutralConstraint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _set_tuple(self, "stages", self.stages)
        _set_tuple(self, "tensors", self.tensors)
        _set_tuple(self, "constraints", self.constraints)
        _set_dict(self, "metadata", self.metadata)

    def validate(self) -> list[str]:
        """Return validation errors without importing source-format packages."""
        errors: list[str] = []
        if not self.graph_id:
            errors.append("graph_id_required")
        if not self.model_family:
            errors.append("model_family_required")
        if not self.stages:
            errors.append("at_least_one_stage_required")
        if not self.tensors:
            errors.append("at_least_one_tensor_required")

        stage_ids = [stage.stage_id for stage in self.stages]
        if any(not stage_id for stage_id in stage_ids):
            errors.append("stage_id_required")
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("stage_ids_must_be_unique")

        tensor_names = [tensor.name for tensor in self.tensors]
        if any(not name for name in tensor_names):
            errors.append("tensor_name_required")
        if len(tensor_names) != len(set(tensor_names)):
            errors.append("tensor_names_must_be_unique")

        if self.kv_cache_requirements.required:
            if self.kv_cache_requirements.max_context_tokens is None:
                errors.append("kv_cache_max_context_tokens_required")
            if self.kv_cache_requirements.bytes_per_token is None:
                errors.append("kv_cache_bytes_per_token_required")

        return errors


def _set_tuple(instance: object, name: str, value: Any) -> None:
    object.__setattr__(instance, name, tuple(value or ()))


def _set_dict(instance: object, name: str, value: Any) -> None:
    object.__setattr__(instance, name, dict(value or {}))
