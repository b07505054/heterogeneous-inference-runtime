"""Mutable runtime-local state accumulated during a single execution request.

ExecutionContext is not compiler IR and not the final result. It is the
scratchpad that ExecutionEngine populates as it calls each sub-component
in order: BackendDispatcher, then (future) Scheduler, MemoryPlanner, and
ReplayManager. Once all sub-components have run, ExecutionEngine assembles
RuntimeResult from this context and discards it.

ExecutionContext must never be returned to callers or stored beyond a single
execute() call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deployment.runtime_execution_plan import RuntimeExecutionPlan


@dataclass
class ExecutionContext:
    plan: RuntimeExecutionPlan
    backend_decision: object | None = None
    scheduling_decision: object | None = None
    memory_decision: object | None = None
    replay_decision: object | None = None
