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
PORTABLE_CPU_KERNEL_TRUTH_BOUNDARY = (
    "Portable CPU fused MatMul+Bias+ReLU is a real compiler-selected, real "
    "compiled-kernel dispatch (kernel_selection_contract_v1) -- functional "
    "bring-up and correctness evidence only, not a performance or "
    "NEON/SIMD-optimization claim, and not a comparison against any other "
    "backend."
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
    # Future/non-goal for Phase 1. Active Qwen serving materialization is vLLM only.
    PYTORCH_REFERENCE = "pytorch_reference"
    CUSTOM_CUDA_MICROBENCHMARK = "custom_cuda_microbenchmark"
    # Future/non-goal for Phase 1. Quantized serving requires explicit support.
    QUANTIZED_MODEL = "quantized_model"
    # Phase P1B: one compiler-selected, dispatch-validated portable C++ CPU
    # kernel (kernel_selection_contract_v1), distinct from PYTORCH_REFERENCE
    # (which is an unimplemented adapter stub) and from CUSTOM_CUDA_MICROBENCHMARK
    # (GPU-only). See deployment/execution_plan/portable_cpu_kernel_adapter.py.
    PORTABLE_CPU_KERNEL = "portable_cpu_kernel"
    AARCH64_NATIVE_OBJECT = "aarch64_native_object"
    UNSUPPORTED = "unsupported"


class ExecutionMethod(str, Enum):
    SERVING = "serving"
    OPENAI_COMPATIBLE_SERVER = "openai_compatible_server"
    COMPILER_MATERIALIZED_CONFIG = "compiler_materialized_config"
    # Future/non-goal for Phase 1. Kept as schema vocabulary, not active routing.
    EAGER_REFERENCE = "eager_reference"
    RMSNORM_KERNEL = "rmsnorm_kernel"
    RMSNORM_MICROBENCHMARK = "rmsnorm_microbenchmark"
    # Phase P1B: dispatches native/cpu_kernels/portable_fused_matmul_bias_relu,
    # a real compiled executable (never PyTorch/ONNXRuntime/NumPy/mock).
    FUSED_MATMUL_BIAS_RELU_KERNEL = "fused_matmul_bias_relu_kernel"
    AARCH64_NATIVE_OBJECT = "aarch64_native_object"
    CPU_ATTENTION_KERNEL = "cpu_attention_kernel"


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


# ---------------------------------------------------------------------------
# DecisionCost — static compiler cost evidence from ServingCostModelPass.
#
# All components are relative static penalty scores, NOT measured latency ms.
# truth_boundary must be read and stored verbatim; it distinguishes static
# estimates from measured profiling data.
#
# Runtime may use total_cost as a scheduling weight but must label it as a
# static estimate, never as a measured timing value.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionCost:
    compute_cost:          int = 0
    memory_cost:           int = 0
    dequant_cost:          int = 0
    requant_cost:          int = 0
    layout_transform_cost: int = 0
    cast_cost:             int = 0
    backend_switch_cost:   int = 0
    launch_overhead_cost:  int = 0
    kv_cache_cost:         int = 0
    transfer_cost:         int = 0
    unsupported_penalty:   int = 0
    total_cost:            int = 0
    cost_model_id:         str = ""
    truth_boundary:        str = ""   # preserved verbatim from compiler

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionCost":
        return cls(
            compute_cost=int(payload.get("compute_cost", 0) or 0),
            memory_cost=int(payload.get("memory_cost", 0) or 0),
            dequant_cost=int(payload.get("dequant_cost", 0) or 0),
            requant_cost=int(payload.get("requant_cost", 0) or 0),
            layout_transform_cost=int(payload.get("layout_transform_cost", 0) or 0),
            cast_cost=int(payload.get("cast_cost", 0) or 0),
            backend_switch_cost=int(payload.get("backend_switch_cost", 0) or 0),
            launch_overhead_cost=int(payload.get("launch_overhead_cost", 0) or 0),
            kv_cache_cost=int(payload.get("kv_cache_cost", 0) or 0),
            transfer_cost=int(payload.get("transfer_cost", 0) or 0),
            unsupported_penalty=int(payload.get("unsupported_penalty", 0) or 0),
            total_cost=int(payload.get("total_cost", 0) or 0),
            cost_model_id=str(payload.get("cost_model_id", "")),
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
    # Parsed from meta.evidence.cost when ServingCostModelPass ran; None otherwise.
    # cost.total_cost is a static penalty score, NOT measured latency.
    # cost.truth_boundary is preserved verbatim.
    cost: DecisionCost | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KernelDecision":
        cost_dict = _dict_at(_dict_at(_dict_at(payload, "meta"), "evidence"), "cost")
        cost = DecisionCost.from_dict(cost_dict) if cost_dict else None
        return cls(
            decision_type=str(payload.get("decision_type", "")),
            scope=str(payload.get("scope", "")),
            selected_kernel=str(payload.get("selected_kernel", "")),
            kernel_library=str(payload.get("kernel_library", "")),
            lowering_path=str(payload.get("lowering_path", "")),
            kernel_exists=bool(payload.get("kernel_exists", False)),
            reason=payload.get("reason"),
            cost=cost,
            raw=dict(payload),
        )


# ---------------------------------------------------------------------------
# KernelSelectionDecision — kernel_selection_contract_v1 (KernelSelectionPass).
#
# Distinct from KernelDecision above: KernelDecision reflects
# LoweringDecisionPlanningPass / third-party-LIBRARY coverage (backendCapabilities
# / kernelLibraries) and is "unsupported" whenever no such library capability is
# declared. KernelSelectionDecision reflects the separate, concrete
# RuntimeKernelDescriptor registry (target.runtime_kernels) -- a real,
# dispatchable runtime kernel contract. The two can and do disagree (e.g.
# KernelDecision.kernel_exists == False while KernelSelectionDecision.status ==
# "selected") when a target declares a handwritten runtime kernel without also
# declaring third-party library coverage. Both are genuine, non-contradictory
# compiler facts about two different layers -- see ml-graph-compiler-runtime
# CLAUDE.md "Kernel Selection Framework (kernel_selection_contract_v1)".
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KernelSelectionDecision:
    contract_version: str
    status: str
    selected_kernel: str | None = None
    source: str | None = None
    rejection_reasons: tuple[str, ...] = ()
    truth_boundary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> bool:
        return self.status == "selected"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KernelSelectionDecision":
        return cls(
            contract_version=str(payload.get("contract_version", "")),
            status=str(payload.get("status", "")),
            selected_kernel=payload.get("selected_kernel"),
            source=payload.get("source"),
            rejection_reasons=tuple(payload.get("rejection_reasons", ())),
            truth_boundary=str(payload.get("truth_boundary", "")),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class OpDecision:
    op_name: str
    op_type: str
    kernel: KernelDecision | None
    kernel_selection: KernelSelectionDecision | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpDecision":
        kernel_payload = _dict_at(payload, "kernel")
        kernel_selection_payload = _dict_at(payload, "kernel_selection")
        return cls(
            op_name=str(payload.get("op_name", "")),
            op_type=str(payload.get("op_type", "")),
            kernel=KernelDecision.from_dict(kernel_payload) if kernel_payload else None,
            kernel_selection=(
                KernelSelectionDecision.from_dict(kernel_selection_payload)
                if kernel_selection_payload else None
            ),
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


# ---------------------------------------------------------------------------
# Typed global decision sub-objects.
#
# MemoryPlanDecision and ServingPlanDecision replace the previous untyped
# dict[str, Any] fields in GlobalDecisions. Fields default to 0/False/"" when
# absent — the compiler builder may not emit all fields in every compilation
# mode.
#
# Key-name notes:
#   MemoryPlanDecision.kv_layout      ← compiler emits "kv_cache_layout"
#   MemoryPlanDecision.kv_byte_estimate_mb  ← compiler emits "estimated_kv_peak_mb"
#   Both aliases are accepted for forward compatibility.
#
#   ServingPlanDecision.colocated_cost_estimate_ms is a static formula estimate,
#   NOT measured runtime latency. It may be used for simulation clock advance
#   only; it must never appear alongside measured timing fields in RuntimeResult.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryPlanDecision:
    memory_budget_fraction: float = 0.0
    kv_layout: str = ""          # "paged" | "contiguous" | "" if builder hasn't emitted it
    kv_block_size_tokens: int = 0
    kv_byte_estimate_mb: float = 0.0   # static formula estimate, not measured allocation
    truth_boundary: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryPlanDecision":
        kv_layout = str(
            payload.get("kv_cache_layout") or payload.get("kv_layout", "")
        )
        kv_byte = float(
            payload.get("estimated_kv_peak_mb")
            or payload.get("kv_byte_estimate_mb", 0.0)
            or 0.0
        )
        return cls(
            memory_budget_fraction=float(payload.get("memory_budget_fraction", 0.0) or 0.0),
            kv_layout=kv_layout,
            kv_block_size_tokens=int(payload.get("kv_block_size_tokens", 0) or 0),
            kv_byte_estimate_mb=kv_byte,
            truth_boundary=str(payload.get("truth_boundary", "")),
        )


@dataclass(frozen=True)
class ServingPlanDecision:
    topology: str = ""             # "colocated" | "prefill_decode_split"
    replay_eligible: bool = False  # from ServingDecision.replay_eligible
    colocated_cost_estimate_ms: float = 0.0  # static formula, NOT measured runtime latency
    prefix_reuse_eligible: bool = False
    chunked_prefill_eligible: bool = False
    token_budget_per_step: int = 0
    parallelism_kind: str = ""     # "none" | "tensor_parallel" | "pipeline_parallel"
    parallelism_degree: int = 1
    truth_boundary: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ServingPlanDecision":
        return cls(
            topology=str(payload.get("topology", "")),
            replay_eligible=bool(payload.get("replay_eligible", False)),
            colocated_cost_estimate_ms=float(
                payload.get("colocated_cost_estimate_ms", 0.0) or 0.0
            ),
            prefix_reuse_eligible=bool(payload.get("prefix_reuse_eligible", False)),
            chunked_prefill_eligible=bool(payload.get("chunked_prefill_eligible", False)),
            token_budget_per_step=int(payload.get("token_budget_per_step", 0) or 0),
            parallelism_kind=str(payload.get("parallelism_kind", "")),
            parallelism_degree=int(payload.get("parallelism_degree", 1) or 1),
            truth_boundary=str(payload.get("truth_boundary", "")),
        )


@dataclass(frozen=True)
class GlobalDecisions:
    # quantization stays dict[str, Any] — only the vLLM materializer reads it via dict access.
    quantization: dict[str, Any] = field(default_factory=dict)
    memory: MemoryPlanDecision = field(default_factory=MemoryPlanDecision)
    serving: ServingPlanDecision = field(default_factory=ServingPlanDecision)
    # Optional Architecture-C single-node CPU sharding intent. Empty preserves
    # compatibility with all existing ExecutionPlan v2 artifacts.
    cpu_sharding: dict[str, Any] = field(default_factory=dict)
    # Optional exact attention execution decision. Empty preserves all legacy
    # ExecutionPlan v2 artifacts.
    attention_execution: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GlobalDecisions":
        return cls(
            quantization=_dict_at(payload, "quantization"),
            memory=MemoryPlanDecision.from_dict(_dict_at(payload, "memory")),
            serving=ServingPlanDecision.from_dict(_dict_at(payload, "serving")),
            cpu_sharding=_dict_at(payload, "cpu_sharding"),
            attention_execution=_dict_at(payload, "attention_execution"),
        )


# ---------------------------------------------------------------------------
# D1: distributed execution plan (compiler-planned TP=2 multi-process
# simulation). Mirrors ml-graph-compiler-runtime's
# mlir_passes/include/serving/ExecutionPlan.h DistributedPlan struct and
# ExecutionPlanExporter's "distributed" JSON block field-for-field. Absent
# entirely on legacy/TP1 plans -- from_dict returns None for those, and every
# existing ExecutionPlan.from_dict caller is unaffected.
#
# Truth boundary: this describes a compiler PLAN for a localhost CPU
# multi-process simulation. It is not a real GPU/NCCL/vLLM distributed
# execution claim -- see DISTRIBUTED_TRUTH_BOUNDARY below and
# docs/DISTRIBUTED_D1_COMPILER_PLANNED_TP2_MULTIPROCESS_REPORT.md.
# ---------------------------------------------------------------------------

DISTRIBUTED_TRUTH_BOUNDARY = (
    "D1 distributed plan: single-host CPU multi-process simulation over "
    "localhost IPC. Not NCCL, not GPU-to-GPU communication, not real vLLM "
    "tensor parallelism, not representative of multi-GPU scaling."
)

# D1 implements exactly one collective kind. Any other declared kind must be
# rejected explicitly, never silently ignored.
KNOWN_COLLECTIVE_KINDS = frozenset({"all_reduce"})


@dataclass(frozen=True)
class DistributedRankPlacement:
    rank_id: int
    logical_device: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistributedRankPlacement":
        return cls(
            rank_id=int(payload.get("rank_id", -1)),
            logical_device=str(payload.get("logical_device", "")),
        )


@dataclass(frozen=True)
class DistributedTensorShard:
    tensor_id: str
    partition_axis: int
    partition_count: int
    shard_index: int
    range_start: int
    range_end: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistributedTensorShard":
        return cls(
            tensor_id=str(payload.get("tensor_id", "")),
            partition_axis=int(payload.get("partition_axis", 0)),
            partition_count=int(payload.get("partition_count", 0)),
            shard_index=int(payload.get("shard_index", -1)),
            range_start=int(payload.get("range_start", 0)),
            range_end=int(payload.get("range_end", 0)),
        )


@dataclass(frozen=True)
class DistributedCollectiveStep:
    collective_id: str
    sequence_id: int
    kind: str
    participants: tuple[int, ...]
    tensor_id: str
    reduction: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistributedCollectiveStep":
        return cls(
            collective_id=str(payload.get("collective_id", "")),
            sequence_id=int(payload.get("sequence_id", -1)),
            kind=str(payload.get("kind", "")),
            participants=tuple(int(p) for p in payload.get("participants", ())),
            tensor_id=str(payload.get("tensor_id", "")),
            reduction=str(payload.get("reduction", "")),
        )


@dataclass(frozen=True)
class DistributedPlan:
    strategy: str
    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    ranks: tuple[DistributedRankPlacement, ...]
    tensor_shards: tuple[DistributedTensorShard, ...]
    collectives: tuple[DistributedCollectiveStep, ...]
    truth_boundary: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DistributedPlan":
        return cls(
            strategy=str(payload.get("strategy", "")),
            world_size=int(payload.get("world_size", 0)),
            tensor_parallel_size=int(payload.get("tensor_parallel_size", 0)),
            pipeline_parallel_size=int(payload.get("pipeline_parallel_size", 0)),
            ranks=tuple(
                DistributedRankPlacement.from_dict(r)
                for r in payload.get("ranks", ())
                if isinstance(r, dict)
            ),
            tensor_shards=tuple(
                DistributedTensorShard.from_dict(s)
                for s in payload.get("tensor_shards", ())
                if isinstance(s, dict)
            ),
            collectives=tuple(
                DistributedCollectiveStep.from_dict(c)
                for c in payload.get("collectives", ())
                if isinstance(c, dict)
            ),
            truth_boundary=str(payload.get("truth_boundary", "")),
        )


@dataclass(frozen=True)
class ExecutionPlan:
    schema: str
    schema_version: str
    plan_id: str
    provenance: PlanProvenance
    model_identity: dict[str, Any]
    global_decisions: GlobalDecisions
    function_plans: tuple[FunctionPlan, ...]
    truth_boundary: str
    # D1: None for legacy/TP1 plans -- every plan predating this field, and
    # every explicit TP1 candidate export, parses identically to before.
    distributed: DistributedPlan | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExecutionPlan":
        distributed_payload = payload.get("distributed")
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
            distributed=(
                DistributedPlan.from_dict(distributed_payload)
                if isinstance(distributed_payload, dict)
                else None
            ),
            raw=dict(payload),
        )


# ExecutionStage is backend-independent. It describes routing intent only:
# what to dispatch (function, phase, op) — not how or to which backend.
# Backend selection belongs to ExecutionPath.
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
