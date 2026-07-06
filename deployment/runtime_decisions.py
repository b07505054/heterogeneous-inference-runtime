"""Typed runtime decision records and their evaluators.

Each evaluator accepts typed inputs from ExecutionPlanV2 and produces a frozen
decision record. Evaluators are not planners — they do not allocate memory,
dispatch to hardware, or capture CUDA graphs. They translate compiler-derived
plan fields into typed, immutable runtime records for ExecutionEngine to
assemble into RuntimeResult.

Decision pipeline order in ExecutionEngine:
  1. SchedulingDecisionEvaluator  → SchedulingDecision
  2. MemoryDecisionEvaluator      → MemoryDecision
  3. ReplayDecisionEvaluator      → ReplayDecision
  4. BackendDispatcher            → BackendDecision (in backend_dispatcher.py)

Truth boundaries:
  SchedulingDecision: "compiler_cost_estimate_not_measured_latency"
  MemoryDecision:     verbatim from MemoryPlanDecision.truth_boundary
  ReplayDecision:     "static_shape_replay_eligibility_not_cuda_graph_capture"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from deployment.execution_plan_v2.schema import FunctionPlan, MemoryPlanDecision, ServingPlanDecision

_DEFAULT_KV_MB_PER_PAGE: float = 1.0


# ---------------------------------------------------------------------------
# SchedulingDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulingDecision:
    """Typed scheduling record derived from FunctionPlan and ServingPlanDecision.

    All cost values are compiler estimates, not measured latency.
    truth_boundary = "compiler_cost_estimate_not_measured_latency"
    priority and confidence are fixed: V2 carries no confidence field.
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
    def evaluate(function_plan: FunctionPlan, serving: ServingPlanDecision) -> SchedulingDecision:
        execution_policy = serving.topology if serving.topology else function_plan.serving_phase
        return SchedulingDecision(
            execution_policy=execution_policy,
            priority="normal",
            compiler_cost_ms=serving.colocated_cost_estimate_ms,
            confidence="unknown",
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
    """Typed memory record derived from MemoryPlanDecision.

    admitted is always True — no real block pool is connected.
    page_budget_estimate is a formula from kv_byte_estimate_mb; not measured.
    truth_boundary is copied verbatim from MemoryPlanDecision.truth_boundary.
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
    def evaluate(memory: MemoryPlanDecision) -> MemoryDecision:
        kv_layout = memory.kv_layout
        estimated_mb = memory.kv_byte_estimate_mb

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
            truth_boundary=(
                memory.truth_boundary
                or "static_formula_estimate_not_measured_memory"
            ),
        )


# ---------------------------------------------------------------------------
# ReplayDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayDecision:
    """Typed replay record derived from FunctionPlan and ServingPlanDecision.

    captured and capture_attempted are always False — no CUDA graph
    infrastructure exists. truth_boundary is hardcoded:
    "static_shape_replay_eligibility_not_cuda_graph_capture"
    bucket is derived from serving_phase: "decode_static" for decode, "" otherwise.
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
    def evaluate(function_plan: FunctionPlan, serving: ServingPlanDecision) -> ReplayDecision:
        eligible = serving.replay_eligible
        requested = eligible  # V2 has no separate requires_replay; request iff eligible
        bucket = "decode_static" if function_plan.serving_phase == "decode" else ""

        if requested:
            skipped_reason = "capture_not_implemented"
        elif not eligible:
            skipped_reason = "not_eligible_per_compiler_plan"
        else:
            skipped_reason = ""

        return ReplayDecision(
            replay_requested=requested,
            replay_eligible_from_compiler=eligible,
            bucket=bucket,
            capture_attempted=False,
            captured=False,
            skipped_reason=skipped_reason,
            truth_boundary="static_shape_replay_eligibility_not_cuda_graph_capture",
        )
