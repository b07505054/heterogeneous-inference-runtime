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

Optional instrumentation:
  When recorder is not None, ExecutionEngine calls recorder.begin_stage(),
  recorder.end_stage(), recorder.instant_event(), and recorder.advance_clock()
  around each decision stage. The recorder accumulates trace events as a side
  effect. RuntimeResult is identical whether recorder is None or not.
"""

from __future__ import annotations

from deployment.backend_dispatcher import BackendDecision, BackendDispatcher
from deployment.execution_context import ExecutionContext
from deployment.execution_trace_recorder import (
    MEMORY_PHASE_GAP_MS,
    SCHEDULING_PHASE_GAP_MS,
    ExecutionTraceRecorder,
)
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

_GPU_BACKENDS: frozenset[str] = frozenset({"coreml", "metal"})


class ExecutionEngine:
    def __init__(self, backend_dispatcher: BackendDispatcher | None = None) -> None:
        self._dispatcher = backend_dispatcher or BackendDispatcher()

    def execute(
        self,
        plan: RuntimeExecutionPlan,
        recorder: ExecutionTraceRecorder | None = None,
    ) -> RuntimeResult:
        ctx = ExecutionContext(plan=plan)

        # ── Stage 1: Scheduling ──────────────────────────────────────────
        if recorder is not None:
            recorder.begin_stage(
                "scheduler", "scheduling_decision", "scheduler",
                request_id=plan.function_name,
                truth_boundary="compiler_cost_estimate_not_measured_latency",
            )
        ctx.scheduling_decision = SchedulingDecisionEvaluator.evaluate(plan)
        if recorder is not None:
            recorder.end_stage(metadata_update={
                "priority": ctx.scheduling_decision.priority,
                "admitted": str(ctx.scheduling_decision.admitted_to_batch),
                "confidence": ctx.scheduling_decision.confidence,
                "batch_policy": ctx.scheduling_decision.batch_policy,
            })
            recorder.advance_clock(SCHEDULING_PHASE_GAP_MS)

        # ── Stage 2: Memory ──────────────────────────────────────────────
        if recorder is not None:
            recorder.begin_stage(
                "memory", "memory_decision", "kv_cache",
                request_id=plan.function_name,
                truth_boundary=plan.memory_policy.truth_boundary,
            )
        ctx.memory_decision = MemoryDecisionEvaluator.evaluate(plan)
        if recorder is not None:
            recorder.end_stage(metadata_update={
                "kv_layout_used": ctx.memory_decision.kv_layout_used,
                "allocator_kind": ctx.memory_decision.allocator_kind,
                "admitted": str(ctx.memory_decision.admitted),
                "page_budget_estimate": str(ctx.memory_decision.page_budget_estimate),
            })
            recorder.advance_clock(MEMORY_PHASE_GAP_MS)

        # ── Stage 3: Replay (instant — no allocator or hardware involved) ─
        ctx.replay_decision = ReplayDecisionEvaluator.evaluate(plan)
        if recorder is not None:
            recorder.instant_event(
                "replay", "replay_decision", "runtime",
                request_id=plan.function_name,
                metadata={
                    "replay_requested": str(ctx.replay_decision.replay_requested),
                    "eligible": str(ctx.replay_decision.replay_eligible_from_compiler),
                    "bucket": ctx.replay_decision.bucket,
                    "skipped_reason": ctx.replay_decision.skipped_reason,
                },
                truth_boundary=ctx.replay_decision.truth_boundary,
            )

        # ── Stage 4: Backend dispatch (instant — no compute) ─────────────
        ctx.backend_decision = self._dispatcher.dispatch(plan)
        if recorder is not None:
            recorder.instant_event(
                "backend", "backend_dispatch", "runtime",
                request_id=plan.function_name,
                metadata={
                    "selected_backend": ctx.backend_decision.selected_backend,
                    "override_reason": ctx.backend_decision.override_reason,
                    "attempted": ",".join(ctx.backend_decision.attempted_backends),
                },
                truth_boundary="runtime_backend_dispatch_not_compiler_plan",
            )

        # ── Stage 5: Compute (duration = compiler cost estimate) ──────────
        if recorder is not None:
            _lane = "gpu" if ctx.backend_decision.selected_backend in _GPU_BACKENDS else "cpu"
            recorder.begin_stage(
                "compute", plan.function_name, _lane,
                request_id=plan.function_name,
                metadata={"backend": ctx.backend_decision.selected_backend},
                truth_boundary="compiler_cost_estimate_not_measured_latency",
            )
            recorder.advance_clock(plan.scheduling_policy.compiler_cost_ms)

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

        result = RuntimeResult(
            function_name=plan.function_name,
            compiler_summary=compiler_summary,
            scheduling_decision=ctx.scheduling_decision,
            memory_decision=ctx.memory_decision,
            replay_decision=ctx.replay_decision,
            backend_decision=ctx.backend_decision,
            compiler_vs_runtime_backend=match_flag,
            decision_trace=list(_DECISION_TRACE),
        )

        if recorder is not None:
            recorder.end_stage(metadata_update={
                "kv_layout": ctx.memory_decision.kv_layout_used,
                "replay_requested": str(ctx.replay_decision.replay_requested),
                "vs_compiler_backend": match_flag,
            })

        return result
