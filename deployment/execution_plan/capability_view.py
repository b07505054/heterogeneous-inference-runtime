"""Thin runtime view over ml-platform-capabilities profile refs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deployment.execution_plan.schema import CapabilityBundleRef


DEFAULT_PROFILE_ROOT = Path(__file__).resolve().parents[3] / "ml-platform-capabilities" / "profiles"


class CapabilityValidationError(ValueError):
    """Raised when capability refs cannot be resolved."""


class CapabilityValidationView:
    """Validates refs against ml-platform-capabilities without duplicating truth."""

    def __init__(self, profile_root: str | Path = DEFAULT_PROFILE_ROOT) -> None:
        self.profile_root = Path(profile_root)

    def resolve_ref(self, ref: str) -> Path:
        path = self.profile_root / ref
        if not path.is_file():
            raise CapabilityValidationError(f"missing capability profile ref: {ref}")
        return path

    def load_ref(self, ref: str) -> dict[str, Any]:
        path = self.resolve_ref(ref)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CapabilityValidationError(f"invalid capability profile JSON: {ref}") from exc

    def has_profile_ref(self, ref: str) -> bool:
        return (self.profile_root / ref).is_file()

    def validate_bundle(self, bundle: CapabilityBundleRef) -> list[str]:
        errors: list[str] = []
        for ref in bundle.refs():
            if not self.has_profile_ref(ref):
                errors.append(f"missing_capability_ref:{ref}")
        if errors:
            raise CapabilityValidationError("; ".join(errors))
        return []

    def validate_refs(self, refs: tuple[str, ...]) -> list[str]:
        errors = [f"missing_capability_ref:{ref}" for ref in refs if not self.has_profile_ref(ref)]
        if errors:
            raise CapabilityValidationError("; ".join(errors))
        return []
