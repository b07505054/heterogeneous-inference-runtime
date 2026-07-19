"""D3B Part J: version-aware CLI argument compatibility.

Every CLI field the materializer wants to emit is checked against the
*installed* vLLM argument registry (deployment.vllm_adapter.
distributed_capability_inventory.discover_argument_registry). If a requested
argument's dest is missing from the registry, or the resolved flag spelling
does not appear in the registry's option_strings for that dest, validation
fails closed -- the materializer must not silently drop or rename it.

Tests exercise this against both the real installed registry and a mocked
"alternate" registry (an older/incompatible vLLM release simulated by
deleting/renaming a dest) to prove incompatibility is actually detected, not
assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArgumentCompatibilityRecord:
    requested_argument: str
    dest: str
    installed_version_support_status: str  # "supported" | "unsupported"
    resolved_spelling: str | None
    resolved_type: str | None
    resolved_default: Any
    source: str  # provenance source of the *value*, not the flag's existence

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_argument": self.requested_argument,
            "dest": self.dest,
            "installed_version_support_status": self.installed_version_support_status,
            "resolved_spelling": self.resolved_spelling,
            "resolved_type": self.resolved_type,
            "resolved_default": self.resolved_default,
            "source": self.source,
        }


class UnsupportedArgumentError(ValueError):
    """Raised when a requested CLI argument does not exist in the installed registry."""


def check_argument(
    dest: str,
    *,
    registry: dict[str, Any],
    value_source: str,
    preferred_flag: str | None = None,
) -> ArgumentCompatibilityRecord:
    """Fail-closed check of a single dest against the installed argument registry."""
    arguments = registry.get("arguments", {})
    spec = arguments.get(dest)
    if spec is None:
        return ArgumentCompatibilityRecord(
            requested_argument=preferred_flag or dest,
            dest=dest,
            installed_version_support_status="unsupported",
            resolved_spelling=None,
            resolved_type=None,
            resolved_default=None,
            source=value_source,
        )
    option_strings: list[str] = spec.get("option_strings", [])
    resolved_spelling = None
    if preferred_flag and preferred_flag in option_strings:
        resolved_spelling = preferred_flag
    elif option_strings:
        # Prefer the long form (first non "-x" short alias), else whatever exists.
        long_forms = [o for o in option_strings if o.startswith("--")]
        resolved_spelling = long_forms[0] if long_forms else option_strings[0]
    if resolved_spelling is None:
        return ArgumentCompatibilityRecord(
            requested_argument=preferred_flag or dest,
            dest=dest,
            installed_version_support_status="unsupported",
            resolved_spelling=None,
            resolved_type=spec.get("type_name"),
            resolved_default=spec.get("default"),
            source=value_source,
        )
    return ArgumentCompatibilityRecord(
        requested_argument=preferred_flag or dest,
        dest=dest,
        installed_version_support_status="supported",
        resolved_spelling=resolved_spelling,
        resolved_type=spec.get("type_name"),
        resolved_default=spec.get("default"),
        source=value_source,
    )


def validate_all_supported(
    records: list[ArgumentCompatibilityRecord],
) -> tuple[bool, list[str]]:
    """Fail closed: every requested argument must be supported. Returns (ok, unsupported_dests)."""
    unsupported = [r.dest for r in records if r.installed_version_support_status == "unsupported"]
    return (len(unsupported) == 0, unsupported)


def build_mock_registry_without(dests: list[str], *, base_registry: dict[str, Any]) -> dict[str, Any]:
    """Build a mocked 'alternate vLLM version' registry missing the given dests.

    Used only by tests to prove version-incompatibility detection actually
    fires -- this function is never called by the real materialization path.
    """
    arguments = dict(base_registry.get("arguments", {}))
    for dest in dests:
        arguments.pop(dest, None)
    return {
        **base_registry,
        "vllm_version": "0.0.0-mocked-incompatible",
        "arguments": arguments,
        "discovery_method": "mocked_for_negative_test",
    }
