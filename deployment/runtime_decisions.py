"""Typed runtime decision records and their evaluators.

Each evaluator reads a RuntimeExecutionPlan and produces a frozen decision
record. Evaluators are not planners — they do not allocate memory, dispatch
to hardware, or capture CUDA graphs. They translate compiler-derived plan
fields into typed, immutable runtime records for ExecutionEngine to assemble
into RuntimeResult.

Decision pipeline order in ExecutionEngine:
  1. SchedulingDecisionEvaluator  → SchedulingDecision
  2. MemoryDecisionEvaluator      → MemoryDecision
  3. ReplayDecisionEvaluator      → ReplayDecision
  4. BackendDispatcher            → BackendDecision (in backend_dispatcher.py)

Truth boundaries:
  SchedulingDecision: "compiler_cost_estimate_not_measured_latency"
  MemoryDecision:     verbatim from memory_policy.truth_boundary
  ReplayDecision:     verbatim from replay_policy.truth_boundary
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deployment.runtime_execution_plan import RuntimeExecutionPlan

_DEFAULT_KV_MB_PER_PAGE: float = 1.0


# ---------------------------------------------------------------------------
# SchedulingDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulingDecision:
    """Typed scheduling record derived from RuntimeExecutionPlan.scheduling_policy.

    All cost and confidence values are compiler estimates, not measured latency.
    truth_boundary = "compiler_cost_estimate_not_measured_latency"
    """

    execution_policy: str
    priority: str
    compiler_cost_ms: float
    confidence: str
    batch_policy: str
    admitted_to_batch: bool
    reason: str
    truth_boundary: str


class SchedulingDecisionEvaluator:
    @staticmethod
    def evaluate(plan: RuntimeExecutionPlan) -> SchedulingDecision:
        return SchedulingDecision(
            execution_policy=plan.execution_policy,
            priority=plan.scheduling_policy.priority,
            compiler_cost_ms=plan.scheduling_policy.compiler_cost_ms,
            confidence=plan.scheduling_policy.confidence,
            batch_policy="single_request",
            admitted_to_batch=True,
            reason="compiler_plan_admitted",
            truth_boundary="compiler_cost_estimate_not_measured_latency",
        )


# ---------------------------------------------------------------------------
# MemoryDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryDecision:
    """Typed memory record derived from RuntimeExecutionPlan.memory_policy.

    admitted is always True in this implementation — no real block pool is
    connected. rejection_reason is "" when admitted.
    page_budget_estimate is a formula from the compiler's kv_byte_estimate_mb
    using DEFAULT_KV_MB_PER_PAGE; it is not a measured allocation.
    truth_boundary is copied verbatim from memory_policy.truth_boundary.
    """

    kv_layout_used: str
    estimated_mb_from_compiler: float
    admitted: bool
    rejection_reason: str
    allocator_kind: str           # "paged" | "contiguous" | "unknown"
    page_budget_estimate: int
    truth_boundary: str


class MemoryDecisionEvaluator:
    @staticmethod
    def evaluate(plan: RuntimeExecutionPlan) -> MemoryDecision:
        kv_layout = plan.memory_policy.kv_layout
        estimated_mb = plan.memory_policy.kv_byte_estimate_mb

        if kv_layout == "paged":
            allocator_kind = "paged"
        elif kv_layout == "contiguous":
            allocator_kind = "contiguous"
        else:
            allocator_kind = "unknown"

        page_budget = math.ceil(max(estimated_mb, 0.0) / _DEFAULT_KV_MB_PER_PAGE)

        return MemoryDecision(
            kv_layout_used=kv_layout,
            estimated_mb_from_compiler=estimated_mb,
            admitted=True,
            rejection_reason="",
            allocator_kind=allocator_kind,
            page_budget_estimate=page_budget,
            truth_boundary=plan.memory_policy.truth_boundary,
        )


# ---------------------------------------------------------------------------
# ReplayDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayDecision:
    """Typed replay record derived from RuntimeExecutionPlan.replay_policy.

    captured and capture_attempted are always False — no CUDA graph
    infrastructure exists. truth_boundary is copied verbatim from
    replay_policy.truth_boundary:
    "static_shape_replay_eligibility_not_cuda_graph_capture"
    """

    replay_requested: bool
    replay_eligible_from_compiler: bool
    bucket: str
    capture_attempted: bool       # always False
    captured: bool                # always False
    skipped_reason: str
    truth_boundary: str


class ReplayDecisionEvaluator:
    @staticmethod
    def evaluate(plan: RuntimeExecutionPlan) -> ReplayDecision:
        eligible = plan.replay_policy.eligible
        requested = eligible and plan.backend_policy.requires_replay

        if requested:
            skipped_reason = "capture_not_implemented"
        elif not eligible:
            skipped_reason = "not_eligible_per_compiler_plan"
        else:
            skipped_reason = ""

        return ReplayDecision(
            replay_requested=requested,
            replay_eligible_from_compiler=eligible,
            bucket=plan.replay_policy.cuda_graph_bucket,
            capture_attempted=False,
            captured=False,
            skipped_reason=skipped_reason,
            truth_boundary=plan.replay_policy.truth_boundary,
        )
