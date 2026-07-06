"""Mutable runtime-local state accumulated during a single execution request.

ExecutionContext is not compiler IR and not the final result. It is the
scratchpad that ExecutionEngine populates as it calls each sub-component
in order:
  1. SchedulingDecisionEvaluator
  2. MemoryDecisionEvaluator
  3. ReplayDecisionEvaluator
  4. BackendDispatcher

Once all sub-components have run, ExecutionEngine assembles RuntimeResult
from this context and discards it.

ExecutionContext must never be returned to callers or stored beyond a single
execute() call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from deployment.execution_plan_v2.schema import ExecutionPlanV2, FunctionPlan

if TYPE_CHECKING:
    from deployment.backend_dispatcher import BackendDecision
    from deployment.runtime_decisions import (
        MemoryDecision,
        ReplayDecision,
        SchedulingDecision,
    )


@dataclass
class ExecutionContext:
    plan: ExecutionPlanV2
    function_plan: FunctionPlan
    scheduling_decision: SchedulingDecision | None = None
    memory_decision: MemoryDecision | None = None
    replay_decision: ReplayDecision | None = None
    backend_decision: BackendDecision | None = None
