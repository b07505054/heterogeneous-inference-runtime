"""Capability layer schema definitions.

This package is intentionally schema-only. Runtime policies may consume these
types in the future, but importing this package must not change benchmark,
runtime, or simulator behavior.
"""

from capabilities.schema import (
    BackendCapability,
    CapabilityEvidence,
    CapabilityProfile,
    HardwareCapability,
    KernelAvailability,
    KernelLibraryCapability,
    MeasuredSupport,
)

__all__ = [
    "BackendCapability",
    "CapabilityEvidence",
    "CapabilityProfile",
    "HardwareCapability",
    "KernelAvailability",
    "KernelLibraryCapability",
    "MeasuredSupport",
]
