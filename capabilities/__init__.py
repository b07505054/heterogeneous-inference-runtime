"""Capability layer schema definitions and profile loading.

Importing this package must not change benchmark, runtime, or simulator
behavior. Profiles are explicit facts for policy consumers, not measured
performance results.
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
from capabilities.profile_loader import load_profile, load_profiles

__all__ = [
    "BackendCapability",
    "CapabilityEvidence",
    "CapabilityProfile",
    "HardwareCapability",
    "KernelAvailability",
    "KernelLibraryCapability",
    "MeasuredSupport",
    "load_profile",
    "load_profiles",
]
