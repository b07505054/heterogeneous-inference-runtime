"""ExecutionPlan v2 runtime-facing schema.

These dataclasses represent compiler planning intent. They do not describe
measured runtime behavior and they do not duplicate capability truth from
ml-platform-capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


EXECUTION_PLAN_TRUTH_BOUNDARY = (
    "ExecutionPlan v2 is compiler planning intent, not measured runtime."
)
EXECUTION_STAGE_TRUTH_BOUNDARY = (
    "ExecutionStage is derived from compiler decisions, not measured runtime behavior."
)
EXECUTION_PATH_TRUTH_BOUNDARY = (
    "ExecutionPath is runtime routing intent, not measured performance."
)
COMPILER_GUIDED_VLLM_TRUTH_BOUNDARY = (
    "Compiler-guided vLLM materializes backend config from compiler decisions; "
    "it does not modify Qwen weights or vLLM internals."
)
RMSNORM_TRUTH_BOUNDARY = (
    "Custom CUDA RMSNorm is kernel-level evidence only, not end-to-end Qwen or "
    "vLLM speedup."
)


class ExecutionStageKind(str, Enum):
    FULL_MODEL_SERVING = "full_model_serving"
    PREFILL = "prefill"
    DECODE = "decode"
    OTHER = "other"
    RMSNORM = "rmsnorm"
    MATMUL = "matmul"
    ATTENTION = "attention"
    QUANTIZED_MODEL_SERVING = "quantized_model_serving"
    MICROBENCHMARK = "microbenchmark"


class ExecutionPathKind(str, Enum):
    BASELINE_VLLM = "baseline_vllm"
    COMPILER_GUIDED_VLLM = "compiler_guided_vllm"
    PYTORCH_REFERENCE = "pytorch_reference"
    CUSTOM_CUDA_MICROBENCHMARK = "custom_cuda_microbenchmark"
    QUANTIZED_MODEL = "quantized_model"
    UNSUPPORTED = "unsupported"


class ExecutionMethod(str, Enum):
    SERVING = "serving"
    OPENAI_COMPATIBLE_SERVER = "openai_compatible_server"
    COMPILER_MATERIALIZED_CONFIG = "compiler_materialized_config"
    EAGER_REFERENCE = "eager_reference"
    RMSNORM_KERNEL = "rmsnorm_kernel"
    RMSNORM_MICROBENCHMARK = "rmsnorm_microbenchmark"


@dataclass(frozen=True)
class CapabilityBundleRef:
    hardware_profile_ref: str
    backend_profile_refs: tuple[str, ...] = ()
    kernel_profile_refs: tuple[str, ...] = ()
    workload_ref: str | None = None
    deployment_profile_ref: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CapabilityBundleRef":
        return cls(
            hardware_profile_ref=str(payload.get("hardware_profile_ref", "")),
            backend_profile_refs=tuple(payload.get("backend_profile_refs", ())),
            kernel_profile_refs=tuple(payload.get("kernel_profile_refs", ())),
            workload_ref=payload.get("workload_ref"),
            deployment_profile_ref=payload.get("deployment_profile_ref"),
        )

    def refs(self) -> tuple[str, ...]:
        refs = [self.hardware_profile_ref]
        refs.extend(self.backend_profile_refs)
        refs.extend(self.kernel_profile_refs)
        if self.workload_ref:
            refs.append(self.workload_ref)
        if self.deployment_profile_ref:
            refs.append(self.deployment_profile_ref)
        return tuple(ref for ref in refs if ref)


@dataclass(frozen=True)
class PlanProvenance:
    compiler_tool: str
    model_spec_ref: str
    capability_bundle: CapabilityBundleRef
    truth_boundary: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlanProvenance":
        return cls(
            compiler_tool=str(payload.get("compiler_tool", "")),
            model_spec_ref=str(payload.get("model_spec_ref", "")),
            capability_bundle=CapabilityBundleRef.from_dict(
                _dict_at(payload, "capability_bundle")
            ),
            truth_boundary=str(payload.get("truth_boundary", "")),
        )


@dataclass(frozen=True)
class BackendDecision:
    decision_type: str
    scope: str
    selected_backend: str
    fallback_backends: tuple[str, ...] = ()
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BackendDecision":
        return cls(
            decision_type=str(payload.get("decision_type", "")),
            scope=str(payload.get("scope", "")),
            selected_backend=str(payload.get("selected_backend", "")),
            fallback_backends=tuple(payload.get("fallback_backends", ())),
            reason=payload.get("reason"),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class KernelDecision:
    decision_type: str
    scope: str
    selected_kernel: str
    kernel_library: str
    lowering_path: str
    kernel_exists: bool
    reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KernelDecision":
        return cls(
            decision_type=str(payload.get("decision_type", "")),
            scope=str(payload.get("scope", "")),
            selected_kernel=str(payload.get("selected_kernel", "")),
            kernel_library=str(payload.get("kernel_library", "")),
            lowering_path=str(payload.get("lowering_path", "")),
            kernel_exists=bool(payload.get("kernel_exists", False)),
            reason=payload.get("reason"),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class OpDecision:
    op_name: str
    op_type: str
    kernel: KernelDecision | None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpDecision":
        kernel_payload = _dict_at(payload, "kernel")
        return cls(
            op_name=str(payload.get("op_name", "")),
            op_type=str(payload.get("op_type", "")),
            kernel=KernelDecision.from_dict(kernel_payload) if kernel_payload else None,
            raw=dict(payload),
        )


@dataclass(frozen=True)
class FunctionPlan:
    function_name: str
    serving_phase: str
    backend: BackendDecision
    per_op_decisions: tuple[OpDecision, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FunctionPlan":
        return cls(
            function_name=str(payload.get("function_name", "")),
            serving_phase=str(payload.get("serving_phase", "other")),
            backend=BackendDecision.from_dict(_dict_at(payload, "backend")),
            per_op_decisions=tuple(
                OpDecision.from_dict(item)
                for item in payload.get("per_op_decisions", ())
                if isinstance(item, dict)
            ),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class GlobalDecisions:
    quantization: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    serving: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalDecisions":
        return cls(
            quantization=_dict_at(payload, "quantization"),
            memory=_dict_at(payload, "memory"),
            serving=_dict_at(payload, "serving"),
        )


@dataclass(frozen=True)
class ExecutionPlanV2:
    schema: str
    schema_version: str
    plan_id: str
    provenance: PlanProvenance
    model_identity: dict[str, Any]
    global_decisions: GlobalDecisions
    function_plans: tuple[FunctionPlan, ...]
    truth_boundary: str
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionPlanV2":
        return cls(
            schema=str(payload.get("schema", "")),
            schema_version=str(payload.get("schema_version", "")),
            plan_id=str(payload.get("plan_id", "")),
            provenance=PlanProvenance.from_dict(_dict_at(payload, "provenance")),
            model_identity=_dict_at(payload, "model_identity"),
            global_decisions=GlobalDecisions.from_dict(
                _dict_at(payload, "global_decisions")
            ),
            function_plans=tuple(
                FunctionPlan.from_dict(item)
                for item in payload.get("function_plans", ())
                if isinstance(item, dict)
            ),
            truth_boundary=str(
                _dict_at(payload, "provenance").get(
                    "truth_boundary", EXECUTION_PLAN_TRUTH_BOUNDARY
                )
            ),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class ExecutionStage:
    stage_id: str
    kind: ExecutionStageKind
    function_name: str | None
    serving_phase: str | None
    op_name: str | None
    op_type: str | None
    source_compiler_decision: dict[str, Any]
    truth_boundary: str = EXECUTION_STAGE_TRUTH_BOUNDARY


@dataclass(frozen=True)
class ExecutionPath:
    path_id: str
    path_kind: ExecutionPathKind
    stage_id: str
    function_name: str | None
    serving_phase: str | None
    selected_backend: str
    execution_method: ExecutionMethod
    selected_kernel: str | None
    kernel_library: str | None
    fallback_backends: tuple[str, ...]
    source_compiler_decision: dict[str, Any] | None
    required_capability_refs: tuple[str, ...]
    runtime_config: dict[str, Any]
    benchmark_config: dict[str, Any]
    output_artifact: str
    truth_boundary: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _dict_at(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}
