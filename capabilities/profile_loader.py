from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from capabilities.schema import (
    BackendCapability,
    CapabilityEvidence,
    HardwareCapability,
    KernelAvailability,
    KernelLibraryCapability,
)


KNOWN_PROFILE_TYPES = {"hardware", "backend", "kernels"}


def load_profile(path: str | Path) -> dict[str, Any]:
    """Load and validate one concrete capability profile."""

    profile_path = Path(path)
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed profile {profile_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("profile must be a JSON object")
    profile_type = payload.get("capability_type")
    if profile_type not in KNOWN_PROFILE_TYPES:
        raise ValueError(f"unknown capability type: {profile_type!r}")

    if profile_type == "hardware":
        return _normalize_hardware(payload, profile_path)
    if profile_type == "backend":
        return _normalize_backend(payload, profile_path)
    return _normalize_kernels(payload, profile_path)


def load_profiles(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [load_profile(path) for path in paths]


def _normalize_hardware(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    _require_keys(
        payload,
        "schema_version",
        "hardware_id",
        "vendor",
        "family",
        "model",
    )
    capability = HardwareCapability(
        hardware_id=_require_string(payload, "hardware_id"),
        vendor=_require_string(payload, "vendor"),
        family=_require_string(payload, "family"),
        model=_require_string(payload, "model"),
        components=tuple(_optional_string_list(payload, "components")),
        memory=_optional_dict(payload, "memory"),
        attributes=_optional_dict(payload, "attributes"),
        evidence=_evidence(payload.get("evidence", "declared")),
        source=payload.get("source"),
        notes=tuple(_optional_string_list(payload, "notes")),
    )
    return {
        "schema_version": _require_string(payload, "schema_version"),
        "capability_type": "hardware",
        "source_path": str(path),
        "capability": _jsonable_dataclass(capability),
    }


def _normalize_backend(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    _require_keys(payload, "schema_version", "backend_id", "backend", "backend_api", "supports")
    supports = _optional_dict(payload, "supports")
    capability = BackendCapability(
        backend_id=_require_string(payload, "backend_id"),
        backend_name=_require_string(payload, "backend"),
        backend_api=_require_string(payload, "backend_api"),
        supported_features=tuple(_supports_list(supports, "features")),
        supported_precisions=tuple(_supports_list(supports, "precisions")),
        supported_compute_units=tuple(_supports_list(supports, "compute_units")),
        unsupported_features=tuple(_optional_string_list(payload, "unsupported_features")),
        fallback_backends=tuple(_optional_string_list(payload, "fallback_backends")),
        evidence=_evidence(payload.get("evidence", "declared")),
        source=payload.get("source"),
        notes=tuple(_optional_string_list(payload, "notes")),
    )
    return {
        "schema_version": _require_string(payload, "schema_version"),
        "capability_type": "backend",
        "source_path": str(path),
        "backend": payload["backend"],
        "supports": supports,
        "does_not_claim": _optional_string_list(payload, "does_not_claim"),
        "future_capabilities": _optional_string_list(payload, "future_capabilities"),
        "capability": _jsonable_dataclass(capability),
    }


def _normalize_kernels(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    _require_keys(payload, "schema_version", "profile_id", "backend_id", "kernels")
    kernels = payload.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise ValueError("kernels must be a non-empty array")

    normalized_kernels = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise ValueError("kernel entry must be an object")
        _require_keys(kernel, "kernel_id", "operation", "availability")
        capability = KernelLibraryCapability(
            kernel_id=_require_string(kernel, "kernel_id"),
            operation=_require_string(kernel, "operation"),
            backend_id=_require_string(payload, "backend_id"),
            availability=_availability(kernel["availability"]),
            library=kernel.get("library"),
            supported_precisions=tuple(_optional_string_list(kernel, "supported_precisions")),
            supported_features=tuple(_optional_string_list(kernel, "supported_features")),
            evidence=_evidence(kernel.get("evidence", "declared")),
            source=kernel.get("source"),
            notes=tuple(_optional_string_list(kernel, "notes")),
        )
        normalized = {
            **_jsonable_dataclass(capability),
            "support_status": _optional_string(kernel, "support_status"),
            "measured": _optional_bool(kernel, "measured"),
        }
        normalized_kernels.append(normalized)

    return {
        "schema_version": _require_string(payload, "schema_version"),
        "capability_type": "kernels",
        "source_path": str(path),
        "profile_id": _require_string(payload, "profile_id"),
        "backend_id": _require_string(payload, "backend_id"),
        "kernels": normalized_kernels,
        "notes": _optional_string_list(payload, "notes"),
    }


def _require_keys(payload: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(f"profile missing required keys: {missing}")


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _optional_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _optional_string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return value


def _supports_list(supports: dict[str, Any], key: str) -> list[str]:
    value = supports.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"supports.{key} must be an array of strings")
    return value


def _evidence(value: Any) -> CapabilityEvidence:
    try:
        return CapabilityEvidence(value)
    except ValueError as exc:
        raise ValueError(f"unknown capability evidence: {value!r}") from exc


def _availability(value: Any) -> KernelAvailability:
    try:
        return KernelAvailability(value)
    except ValueError as exc:
        raise ValueError(f"unknown kernel availability: {value!r}") from exc


def _jsonable_dataclass(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError("expected dataclass")
    return _jsonable(asdict(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value
