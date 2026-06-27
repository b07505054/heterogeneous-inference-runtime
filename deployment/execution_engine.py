"""ExecutionEngine: runtime orchestrator for compiler-derived execution plans.

ExecutionEngine owns the decision pipeline. It does not mutate
RuntimeExecutionPlan. It creates an ExecutionContext, calls each evaluator
and dispatcher in order, and assembles a frozen RuntimeResult.

Pipeline (in execution order):
  RuntimeExecutionPlan
    → ExecutionContext (mutable scratchpad)
    → SchedulingDecisionEvaluator.evaluate()  → SchedulingDecision
    → MemoryDecisionEvaluator.evaluate()      → MemoryDecision
    → ReplayDecisionEvaluator.evaluate()      → ReplayDecision
    → BackendDispatcher.dispatch()            → BackendDecision
    → RuntimeResult (frozen)
"""

from __future__ import annotations

from deployment.backend_dispatcher import BackendDecision, BackendDispatcher
from deployment.execution_context import ExecutionContext
from deployment.runtime_decisions import (
    MemoryDecisionEvaluator,
    ReplayDecisionEvaluator,
    SchedulingDecisionEvaluator,
)
from deployment.runtime_execution_plan import RuntimeExecutionPlan
from deployment.runtime_result import CompilerSummary, RuntimeResult

_DECISION_TRACE: list[str] = [
    "compiler_runtime_adapter",
    "scheduling_decision_evaluator",
    "memory_decision_evaluator",
    "replay_decision_evaluator",
    "backend_dispatcher",
    "execution_engine",
]


class ExecutionEngine:
    def __init__(self, backend_dispatcher: BackendDispatcher | None = None) -> None:
        self._dispatcher = backend_dispatcher or BackendDispatcher()

    def execute(self, plan: RuntimeExecutionPlan) -> RuntimeResult:
        ctx = ExecutionContext(plan=plan)

        ctx.scheduling_decision = SchedulingDecisionEvaluator.evaluate(plan)
        ctx.memory_decision = MemoryDecisionEvaluator.evaluate(plan)
        ctx.replay_decision = ReplayDecisionEvaluator.evaluate(plan)
        ctx.backend_decision = self._dispatcher.dispatch(plan)

        assert isinstance(ctx.backend_decision, BackendDecision)
        assert ctx.scheduling_decision is not None
        assert ctx.memory_decision is not None
        assert ctx.replay_decision is not None

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
            scheduling_decision=ctx.scheduling_decision,
            memory_decision=ctx.memory_decision,
            replay_decision=ctx.replay_decision,
            backend_decision=ctx.backend_decision,
            compiler_vs_runtime_backend=match_flag,
            decision_trace=list(_DECISION_TRACE),
        )
