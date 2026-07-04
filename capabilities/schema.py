"""First-class capability layer schemas.

These dataclasses describe what exists and what has been measured. They do not
select policies, execute benchmarks, import CoreML/vLLM, or infer performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class KernelAvailability(str, Enum):
    """Runtime/kernel availability category for an operation or feature."""

    BUILTIN = "builtin"
    OPAQUE = "opaque"
    CUSTOM = "custom"
    UNSUPPORTED = "unsupported"


class CapabilityEvidence(str, Enum):
    """Evidence level for a capability entry."""

    DECLARED = "declared"
    MEASURED = "measured"
    ARTIFACT = "artifact"
    SIMULATED = "simulated"
    FUTURE = "future"


@dataclass(frozen=True)
class HardwareCapability:
    """Physical hardware facts only.

    Examples include Apple M-series SoC, Apple GPU, Apple ANE, unified memory,
    NVIDIA GPU, CUDA compute capability, VRAM, or CPU details. Benchmark results
    do not belong here.
    """

    hardware_id: str
    vendor: str
    family: str
    model: str
    components: tuple[str, ...] = ()
    memory: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence: CapabilityEvidence = CapabilityEvidence.DECLARED
    source: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendCapability:
    """Runtime/backend feature support without measured performance."""

    backend_id: str
    backend_name: str
    backend_api: str
    supported_features: tuple[str, ...] = ()
    supported_precisions: tuple[str, ...] = ()
    supported_compute_units: tuple[str, ...] = ()
    unsupported_features: tuple[str, ...] = ()
    fallback_backends: tuple[str, ...] = ()
    evidence: CapabilityEvidence = CapabilityEvidence.DECLARED
    source: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelLibraryCapability:
    """Runtime or kernel implementation availability.

    This is not compiler lowering. It records whether a runtime exposes an
    operation or feature as builtin, opaque, custom, or unsupported.
    """

    kernel_id: str
    operation: str
    backend_id: str
    availability: KernelAvailability
    library: str | None = None
    supported_precisions: tuple[str, ...] = ()
    supported_features: tuple[str, ...] = ()
    evidence: CapabilityEvidence = CapabilityEvidence.DECLARED
    source: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasuredSupport:
    """Experimentally verified support facts only.

    Examples include FP16 benchmark completed, palettization benchmark
    completed, input size 224 measured, CoreML ComputeUnit ALL measured, vLLM
    TTFT measured, or concurrency benchmark completed. Predictions do not
    belong here.
    """

    measurement_id: str
    benchmark_target: dict[str, Any]
    measured_artifact_path: str
    measured_features: tuple[str, ...]
    status: str
    metrics_available: tuple[str, ...] = ()
    evidence: CapabilityEvidence = CapabilityEvidence.MEASURED
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityProfile:
    """Container joining capability declarations and measured support."""

    profile_id: str
    hardware: tuple[HardwareCapability, ...] = ()
    backends: tuple[BackendCapability, ...] = ()
    kernels: tuple[KernelLibraryCapability, ...] = ()
    measured_support: tuple[MeasuredSupport, ...] = ()
    schema_version: str = "0.1"
    notes: tuple[str, ...] = ()
