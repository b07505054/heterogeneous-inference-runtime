"""E2E-9: unified operation-descriptor taxonomy spanning RMSNorm CUDA
launch-configuration selection and Linear operation-decomposition selection.

Relationship to the existing `deployment/execution_plan/schema.py`
`ExecutionPath`/`ExecutionPathKind` system (audited, not replaced): that
system is the mature, capability-validated, compiler-plan-consuming
abstraction used across the Hailo/CPU/multi-backend dispatch work in this
repository (see `deployment/custom_cuda_adapter.py`'s `CustomCudaBackendAdapter`,
which already materializes RMSNorm `rmsnorm_exact_config` plan entries with
their own `candidate_id` convention `cuda_rmsnorm_fp32_bs{N}_v1`). This
module is deliberately a SEPARATE, narrower schema scoped only to the two
decision categories this slice unifies (launch configuration, operation
decomposition) for RMSNorm + Linear. It does not replace, wrap, or require
`ExecutionPath` -- see perf_model/execution_policy.py's module docstring for
the explicit boundary.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DecisionKind(str, Enum):
    LAUNCH_CONFIGURATION = "launch_configuration"
    OPERATION_DECOMPOSITION = "operation_decomposition"
    KERNEL_IMPLEMENTATION = "kernel_implementation"  # added E2E-10: precision/kernel-implementation selection


class OperationFamily(str, Enum):
    RMS_NORM = "rms_norm"
    LINEAR = "linear"
    MATMUL_BIAS_RELU = "matmul_bias_relu"  # added E2E-10


class OperationSubtype(str, Enum):
    RMS_NORM_GENERIC = "rms_norm_generic"
    LM_HEAD = "lm_head"
    FUSED_MATMUL_BIAS_RELU = "fused_matmul_bias_relu"  # added E2E-10


_FAMILY_SUBTYPES = {
    OperationFamily.RMS_NORM: {OperationSubtype.RMS_NORM_GENERIC},
    OperationFamily.LINEAR: {OperationSubtype.LM_HEAD},
    OperationFamily.MATMUL_BIAS_RELU: {OperationSubtype.FUSED_MATMUL_BIAS_RELU},
}

# E2E-10: payload-type registry, replacing the E2E-9 two-entry inline dict
# that OperationDescriptor's validation/serialization used directly. This is
# a GENERIC_EXTENSION (adds an extensibility API with identical behavior for
# the two pre-existing families) made so a third family's payload dataclass
# can be registered from its own family-specific module
# (perf_model/matmul_bias_relu_descriptor.py) instead of requiring a new
# family branch inside this file.
_PAYLOAD_TYPES: dict["OperationFamily", type] = {}


def register_payload_type(family: "OperationFamily", payload_cls: type) -> None:
    _PAYLOAD_TYPES[family] = payload_cls


def _enum_from_value(enum_cls, value):
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise ValueError(f"unknown {enum_cls.__name__} value: {value!r}") from exc


@dataclass(frozen=True)
class OperationEnvelope:
    """Common fields present for every operation, regardless of family."""
    operation_family: OperationFamily
    operation_subtype: OperationSubtype
    dtype: str
    device_type: str
    target_arch: str
    phase: str  # "decode" | "prefill" | "n/a"
    logical_shape: tuple[int, ...]
    static_attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.operation_subtype not in _FAMILY_SUBTYPES[self.operation_family]:
            raise ValueError(
                f"operation_subtype {self.operation_subtype} not valid for family {self.operation_family}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_family": self.operation_family.value, "operation_subtype": self.operation_subtype.value,
            "dtype": self.dtype, "device_type": self.device_type, "target_arch": self.target_arch,
            "phase": self.phase, "logical_shape": list(self.logical_shape), "static_attributes": self.static_attributes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OperationEnvelope":
        return OperationEnvelope(
            operation_family=_enum_from_value(OperationFamily, d["operation_family"]),
            operation_subtype=_enum_from_value(OperationSubtype, d["operation_subtype"]),
            dtype=d["dtype"], device_type=d["device_type"], target_arch=d["target_arch"], phase=d["phase"],
            logical_shape=tuple(d["logical_shape"]), static_attributes=d.get("static_attributes", {}),
        )


@dataclass(frozen=True)
class RMSNormDescriptor:
    token_count: int
    hidden_size: int
    epsilon: float
    has_weight: bool
    input_contiguous: bool
    output_contiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RMSNormDescriptor":
        return RMSNormDescriptor(**d)


@dataclass(frozen=True)
class LinearDescriptor:
    M: int
    N: int
    K: int
    has_bias: bool
    decode_or_prefill: str
    graph_captured: bool
    eager_execution: bool
    tensor_parallel_size: int
    weight_layout: str
    input_contiguous: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "LinearDescriptor":
        return LinearDescriptor(**d)


@dataclass(frozen=True)
class OperationDescriptor:
    common: OperationEnvelope
    payload: Any

    def __post_init__(self):
        expected_type = _PAYLOAD_TYPES.get(self.common.operation_family)
        if expected_type is None:
            raise ValueError(f"no payload type registered for operation_family {self.common.operation_family}")
        if not isinstance(self.payload, expected_type):
            raise ValueError(
                f"payload type {type(self.payload).__name__} does not match operation_family "
                f"{self.common.operation_family} (expected {expected_type.__name__})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"common": self.common.to_dict(), "payload": self.payload.to_dict()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OperationDescriptor":
        common = OperationEnvelope.from_dict(d["common"])
        payload_cls = _PAYLOAD_TYPES.get(common.operation_family)
        if payload_cls is None:
            raise ValueError(f"no payload type registered for operation_family {common.operation_family}")
        payload = payload_cls.from_dict(d["payload"])
        return OperationDescriptor(common=common, payload=payload)

    @staticmethod
    def from_json(text: str) -> "OperationDescriptor":
        return OperationDescriptor.from_dict(json.loads(text))


register_payload_type(OperationFamily.RMS_NORM, RMSNormDescriptor)
register_payload_type(OperationFamily.LINEAR, LinearDescriptor)
