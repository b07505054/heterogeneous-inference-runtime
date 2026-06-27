"""RuntimeResult: execution record produced by ExecutionEngine.

RuntimeResult is NOT a compiler plan. It records runtime decisions alongside
a summary of the compiler's plan so the two can be compared.

Truth boundary: "runtime_result_not_compiler_plan" — always set on every
RuntimeResult. Actual latency, measured memory, and CUDA graph capture belong
to stub fields (replay_result, memory_result, execution_statistics) that are
None until the corresponding sub-components are wired into ExecutionEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deployment.backend_dispatcher import BackendDecision


@dataclass(frozen=True)
class CompilerSummary:
    """Compiler-produced fields echoed into RuntimeResult for side-by-side comparison.

    These are copied verbatim from RuntimeExecutionPlan at execute() time.
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
    """Execution record for one RuntimeExecutionPlan processed by ExecutionEngine.

    Stub fields (replay_result, memory_result, execution_statistics) are None
    until Scheduler, MemoryPlanner, and ReplayManager are wired in. They are
    typed ``object | None`` to remain schema-visible without over-constraining
    the future design.

    Fields that belong to future stubs and must NOT appear here:
    - actual_latency_ms   → ExecutionStatistics
    - measured_memory_mb  → MemoryResult
    - cuda_graph_captured → ReplayResult
    """

    function_name: str
    compiler_summary: CompilerSummary
    backend_decision: BackendDecision
    compiler_vs_runtime_backend: str       # "match" | "override"
    runtime_truth_boundary: str = "runtime_result_not_compiler_plan"
    replay_result: object | None = None
    memory_result: object | None = None
    execution_statistics: object | None = None
