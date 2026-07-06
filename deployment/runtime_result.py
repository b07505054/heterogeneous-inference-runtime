"""RuntimeResult: execution record produced by ExecutionEngine.

RuntimeResult is NOT a compiler plan. It records all four runtime decisions
alongside a summary of the compiler's plan so they can be compared.

Truth boundary: "runtime_result_not_compiler_plan" — always set on every
RuntimeResult. Fields that belong to future work and must NOT appear here:
  - actual_latency_ms   → ExecutionStatistics (future)
  - measured_memory_mb  → MemoryDecision future extension
  - cuda_graph_captured → ReplayDecision.captured (always False until implemented)

decision_trace records the ordered stage labels of every component that
contributed to this result, in execution order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deployment.backend_dispatcher import BackendDecision
from deployment.runtime_decisions import MemoryDecision, ReplayDecision, SchedulingDecision


@dataclass(frozen=True)
class CompilerSummary:
    """Compiler-produced fields echoed into RuntimeResult for side-by-side comparison.

    These are copied verbatim from ExecutionPlan / FunctionPlan at execute() time.
    They are not runtime decisions — they are the compiler's static plan.
    """

    function_name: str
    compiler_primary_backend: str
    compiler_decision_source: str
    compiler_cost_ms: float
    compiler_kv_layout: str
    compiler_truth_boundary: str


@dataclass(frozen=True)
class RuntimeResult:
    """Frozen execution record for one FunctionPlan processed by ExecutionEngine.

    All four decisions are required and typed. execution_statistics remains
    None until measured throughput/timing sub-components are wired in.
    """

    function_name: str
    compiler_summary: CompilerSummary
    scheduling_decision: SchedulingDecision
    memory_decision: MemoryDecision
    replay_decision: ReplayDecision
    backend_decision: BackendDecision
    compiler_vs_runtime_backend: str       # "match" | "override"
    runtime_truth_boundary: str = "runtime_result_not_compiler_plan"
    decision_trace: list[str] = field(default_factory=lambda: [
        "compiler_runtime_adapter",
        "scheduling_decision_evaluator",
        "memory_decision_evaluator",
        "replay_decision_evaluator",
        "backend_dispatcher",
        "execution_engine",
    ])
    execution_statistics: object | None = None
