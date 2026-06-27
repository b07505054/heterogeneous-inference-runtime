"""ExecutionEngine: runtime orchestrator for compiler-derived execution plans.

ExecutionEngine is the root of the runtime execution framework. It owns
sub-components (BackendDispatcher; future: Scheduler, MemoryPlanner,
ReplayManager). It does not mutate RuntimeExecutionPlan. It creates an
ExecutionContext, passes the plan to each sub-component, and assembles
RuntimeResult from their decisions.

Pipeline:
  RuntimeExecutionPlan
    → ExecutionContext (mutable scratchpad)
    → BackendDispatcher.dispatch()  → BackendDecision
    → (future) Scheduler.plan()     → SchedulingDecision
    → (future) MemoryPlanner.allocate() → MemoryDecision
    → (future) ReplayManager.attempt()  → ReplayDecision
    → RuntimeResult
"""

from __future__ import annotations

from deployment.backend_dispatcher import BackendDecision, BackendDispatcher
from deployment.execution_context import ExecutionContext
from deployment.runtime_execution_plan import RuntimeExecutionPlan
from deployment.runtime_result import CompilerSummary, RuntimeResult


class ExecutionEngine:
    def __init__(self, backend_dispatcher: BackendDispatcher | None = None) -> None:
        self._dispatcher = backend_dispatcher or BackendDispatcher()

    def execute(self, plan: RuntimeExecutionPlan) -> RuntimeResult:
        ctx = ExecutionContext(plan=plan)

        ctx.backend_decision = self._dispatcher.dispatch(plan)
        assert isinstance(ctx.backend_decision, BackendDecision)

        compiler_summary = CompilerSummary(
            function_name=plan.function_name,
            compiler_primary_backend=plan.backend_policy.primary_backend,
            compiler_decision_source=plan.backend_policy.compiler_decision_source,
            compiler_cost_ms=plan.scheduling_policy.compiler_cost_ms,
            compiler_kv_layout=plan.memory_policy.kv_layout,
            compiler_truth_boundary=plan.compiler_provenance.truth_boundary,
        )

        match_flag = (
            "match"
            if ctx.backend_decision.selected_backend == plan.backend_policy.primary_backend
            else "override"
        )

        return RuntimeResult(
            function_name=plan.function_name,
            compiler_summary=compiler_summary,
            backend_decision=ctx.backend_decision,
            compiler_vs_runtime_backend=match_flag,
        )
