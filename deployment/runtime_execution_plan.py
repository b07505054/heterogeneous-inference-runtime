"""Runtime IR for compiler-derived execution plans.

RuntimeExecutionPlan is the runtime-side intermediate representation (IR).
It consumes compiler ServingExecutionPlan-like dicts produced by the
ml-graph-compiler-runtime serving-optimization-pipeline and makes their
decisions available to runtime consumers (Admission, Scheduler, MemoryPlanner,
ReplayManager, BackendDispatcher).

Design boundaries:
- RuntimeExecutionPlan is a compiler-plan reader, not an execution record.
- It preserves compiler provenance and truth boundaries verbatim.
- selected_backend is initialized from compiler primary_backend only.
  BackendDispatcher may later overwrite selected_backend and MUST set
  runtime_override_reason when it does so.
- capture_claimed is always False here. RuntimeResult (not yet implemented)
  owns actual CUDA graph capture claims.
- measured_latency_ms, measured_memory_mb, and cuda_graph_captured belong
  to RuntimeResult, not to this type.

Future schema mirrors:
- Swift (PocketChef): same field names, same truth-boundary strings
- C++ (lower-level runtime): same struct layout
- FlatBuffer/protobuf: same field vocabulary if serialization is needed
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchedulingPolicy:
    compiler_cost_ms: float = 0.0
    confidence: str = "unknown"
    policy: str = "unknown"
    priority: str = "normal"


@dataclass
class MemoryPolicy:
    kv_layout: str = "unknown"
    kv_byte_estimate_mb: float = 0.0
    kv_layout_reason: str = ""
    truth_boundary: str = ""


@dataclass
class ReplayPolicy:
    eligible: bool = False
    cuda_graph_bucket: str = ""
    override_reason: str = ""
    truth_boundary: str = ""
    capture_claimed: bool = False


@dataclass
class BackendPolicy:
    # Compiler-produced (set by adapter, immutable semantically after construction)
    primary_backend: str = "cpu"
    fallback_chain: list[str] = field(default_factory=list)
    compiler_decision_source: str = "unknown"
    required_precision: str = "fp16"
    required_kv_layout: str = "unknown"
    requires_replay: bool = False
    # Runtime-dispatch fields (BackendDispatcher fills these at dispatch time)
    selected_backend: str = ""
    backend_state: str = "planned"
    runtime_override_reason: str = ""

    def has_runtime_override(self) -> bool:
        return self.selected_backend != self.primary_backend


@dataclass
class CompilerProvenance:
    cost_source: str = ""
    truth_boundary: str = ""
    compiler_pipeline: str = "serving-optimization-pipeline"
    source_passes: list[str] = field(default_factory=list)


@dataclass
class RuntimeExecutionPlan:
    function_name: str = ""
    target_profile_id: str = ""
    execution_policy: str = "unknown"
    scheduling_policy: SchedulingPolicy = field(default_factory=SchedulingPolicy)
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)
    replay_policy: ReplayPolicy = field(default_factory=ReplayPolicy)
    backend_policy: BackendPolicy = field(default_factory=BackendPolicy)
    compiler_provenance: CompilerProvenance = field(default_factory=CompilerProvenance)

    def validate_runtime_override(self) -> bool:
        """True iff backend dispatch is in a consistent state.

        Consistent means either:
          (a) selected_backend == primary_backend (no override), or
          (b) selected_backend != primary_backend AND runtime_override_reason is non-empty.
        """
        bp = self.backend_policy
        if bp.selected_backend == bp.primary_backend:
            return True
        return bool(bp.runtime_override_reason)


class RuntimeExecutionPlanAdapter:
    @staticmethod
    def from_dict(d: dict, *, target_profile_id: str = "") -> RuntimeExecutionPlan:
        cost = d.get("cost_summary", {})
        execution_mode = d.get("execution_mode", "unknown")
        if execution_mode == "pd_split":
            compiler_cost_ms = float(cost.get("pd_split_total_ms", 0.0))
        else:
            compiler_cost_ms = float(cost.get("colocated_total_ms", 0.0))

        confidence = cost.get("confidence", "unknown")
        priority = "conservative" if confidence == "low" else "normal"

        scheduling = SchedulingPolicy(
            compiler_cost_ms=compiler_cost_ms,
            confidence=confidence,
            policy=cost.get("policy", "unknown"),
            priority=priority,
        )

        kv = d.get("kv_plan", {})
        memory = MemoryPolicy(
            kv_layout=kv.get("layout", "unknown"),
            kv_byte_estimate_mb=float(kv.get("kv_byte_estimate_mb", 0.0)),
            kv_layout_reason=kv.get("layout_reason", ""),
            truth_boundary=kv.get("truth_boundary", ""),
        )

        rp = d.get("replay_plan", {})
        replay = ReplayPolicy(
            eligible=bool(rp.get("replay_eligible", False)),
            cuda_graph_bucket=rp.get("cuda_graph_bucket", ""),
            override_reason=rp.get("override_reason", ""),
            truth_boundary=rp.get("truth_boundary", ""),
            capture_claimed=False,
        )

        bep = d.get("backend_execution_plan", {})
        primary = bep.get("primary_backend", "cpu")
        backend = BackendPolicy(
            primary_backend=primary,
            fallback_chain=list(bep.get("fallback_chain", [])),
            compiler_decision_source=bep.get("decision_source", "unknown"),
            required_precision=bep.get("required_precision", "fp16"),
            required_kv_layout=bep.get("required_kv_layout", "unknown"),
            requires_replay=bool(bep.get("requires_replay", False)),
            selected_backend=primary,
            backend_state="planned",
            runtime_override_reason="",
        )

        prov = d.get("provenance", {})
        provenance = CompilerProvenance(
            cost_source=prov.get("cost_source", cost.get("cost_source", "")),
            truth_boundary=prov.get("truth_boundary", ""),
            compiler_pipeline="serving-optimization-pipeline",
            source_passes=list(d.get("source_passes", [])),
        )

        if target_profile_id:
            profile_id = target_profile_id
        else:
            profile_id = d.get("target_profile_id", "")

        return RuntimeExecutionPlan(
            function_name=d.get("function_name", ""),
            target_profile_id=profile_id,
            execution_policy=execution_mode,
            scheduling_policy=scheduling,
            memory_policy=memory,
            replay_policy=replay,
            backend_policy=backend,
            compiler_provenance=provenance,
        )
