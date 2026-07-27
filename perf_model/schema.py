"""Typed performance-model input/prediction schema.

Every numeric estimate in this package must carry an explicit source/method
tag drawn from the enums below. Nothing may silently mix methods, and
nothing may claim a stronger truth boundary than it actually has.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Classification enums (plain strings, validated by membership check)
# ---------------------------------------------------------------------------

ESTIMATE_SOURCES = (
    "measured_microbenchmark",
    "analytical_flop_bandwidth",
    "derived_from_phase_measurement",
    "unavailable",
)

PREDICTION_METHODS = (
    "analytical",
    "measured_lookup",
    "calibrated_constant",
    "unsupported",
)

HARDWARE_FEATURE_CLASSES = ("measured", "device_reported", "vendor_spec", "unknown")

TRUTH_BOUNDARIES = (
    "analytical_no_measurement",
    "analytical_with_phase_derived_constant",
    "measured_lookup_same_model_hardware_workload",
    "unsupported_no_estimate",
)


def _check(value: str, allowed: tuple[str, ...], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field_name}={value!r} not in {allowed}")
    return value


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Level 0 inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelFeatures:
    model_id: str
    architecture: str
    parameter_count: int
    layer_count: int
    hidden_size: int
    intermediate_size: int
    attention_head_count: int
    kv_head_count: int
    head_dimension: int
    vocabulary_size: int
    dtype: str
    quantization: str
    maximum_model_length: int
    estimated_weight_bytes: int
    estimated_weight_bytes_source: str
    model_revision: str | None = None
    tie_word_embeddings: bool = False

    def __post_init__(self):
        _check(self.estimated_weight_bytes_source, ESTIMATE_SOURCES, "estimated_weight_bytes_source")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def features_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class HardwareFeature:
    """A single hardware quantity with its provenance class."""
    value: float | int | str | None
    source_class: str

    def __post_init__(self):
        _check(self.source_class, HARDWARE_FEATURE_CLASSES, "source_class")

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source_class": self.source_class}


@dataclass(frozen=True)
class HardwareFeatures:
    gpu_name: str
    gpu_memory_bytes: HardwareFeature
    gpu_count: int
    cuda_version: str
    compute_capability: HardwareFeature
    memory_bandwidth_bytes_per_s: HardwareFeature
    compute_throughput_flops: HardwareFeature
    pcie_topology: HardwareFeature = field(
        default_factory=lambda: HardwareFeature(None, "unknown")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory_bytes": self.gpu_memory_bytes.to_dict(),
            "gpu_count": self.gpu_count,
            "cuda_version": self.cuda_version,
            "compute_capability": self.compute_capability.to_dict(),
            "memory_bandwidth_bytes_per_s": self.memory_bandwidth_bytes_per_s.to_dict(),
            "compute_throughput_flops": self.compute_throughput_flops.to_dict(),
            "pcie_topology": self.pcie_topology.to_dict(),
        }

    def features_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class WorkloadFeatures:
    workload_id: str
    request_count: int
    concurrency: int
    prompt_token_distribution: dict[str, float]
    output_token_distribution: dict[str, float]
    arrival_process: str
    prefix_sharing: bool
    warmup_count: int
    repetition_count: int
    streaming_mode: bool
    sampling_settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def features_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeConfiguration:
    """v1 compiler-owned surface plus recorded-but-not-owned fields.

    The recorded_not_owned fields are requested defaults only; the
    authoritative values for them come from ResolvedRuntimeFacts
    (server_info_client) after launch, never from this struct.
    """
    max_num_seqs: int | None
    max_num_batched_tokens: int
    max_model_len: int
    gpu_memory_utilization: float
    tensor_parallel_size: int
    dtype: str
    quantization: str
    recorded_not_owned: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def features_hash(self) -> str:
        return stable_hash(self.to_dict())


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------

@dataclass
class MetricEstimate:
    """One predicted metric with a mandatory method tag."""
    value: float | int | bool | None
    method: str
    truth_boundary: str
    note: str = ""

    def __post_init__(self):
        _check(self.method, PREDICTION_METHODS, "method")
        _check(self.truth_boundary, TRUTH_BOUNDARIES, "truth_boundary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "method": self.method,
            "truth_boundary": self.truth_boundary,
            "note": self.note,
        }


PREDICTION_SCHEMA_VERSION = "perf_model.prediction.v1"
PREDICTOR_IDENTITY = "analytical_phase_model"
PREDICTOR_VERSION = "0.1.0"


@dataclass
class PredictionResult:
    model_features_hash: str
    hardware_features_hash: str
    workload_features_hash: str
    runtime_configuration_hash: str

    predicted_weight_memory_bytes: MetricEstimate
    predicted_kv_memory_bytes: MetricEstimate
    predicted_total_memory_bytes: MetricEstimate
    predicted_oom: MetricEstimate

    predicted_prefill_ms: MetricEstimate
    predicted_decode_token_ms: MetricEstimate
    predicted_ttft_ms: MetricEstimate
    predicted_tpot_ms: MetricEstimate
    predicted_e2e_ms: MetricEstimate
    predicted_output_tokens_per_second: MetricEstimate
    predicted_total_tokens_per_second: MetricEstimate

    component_breakdown: dict[str, Any]
    assumptions: dict[str, Any]
    unsupported_terms: list[str]
    confidence_class: str

    prediction_schema_version: str = PREDICTION_SCHEMA_VERSION
    predictor_identity: str = PREDICTOR_IDENTITY
    predictor_version: str = PREDICTOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in (
            "predicted_weight_memory_bytes", "predicted_kv_memory_bytes",
            "predicted_total_memory_bytes", "predicted_oom",
            "predicted_prefill_ms", "predicted_decode_token_ms",
            "predicted_ttft_ms", "predicted_tpot_ms", "predicted_e2e_ms",
            "predicted_output_tokens_per_second", "predicted_total_tokens_per_second",
        ):
            d[key] = getattr(self, key).to_dict()
        return d
