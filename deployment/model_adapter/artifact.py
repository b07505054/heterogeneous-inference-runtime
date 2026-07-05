"""Neutral model artifact descriptor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelArtifact:
    """Descriptor used before selecting a concrete model adapter.

    ``kind`` is an adapter selection key, not a source-format dependency.
    Future values can be added without requiring this descriptor to import
    backend libraries.
    """

    kind: str
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.kind:
            errors.append("artifact_kind_required")
        return errors
