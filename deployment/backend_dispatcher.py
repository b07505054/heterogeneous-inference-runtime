"""BackendDispatcher: selects a runtime backend from a compiler-derived plan.

BackendDispatcher reads RuntimeExecutionPlan.backend_policy and walks the
primary backend then fallback chain until it finds an available backend.
It does NOT mutate RuntimeExecutionPlan. It returns BackendDecision.

Availability is determined by the injectable ``unavailable`` set, not by
real hardware probing. This makes dispatch deterministic and fully testable.

Decision logic (strict order):
  1. Try primary_backend; if available → return with override_reason="".
  2. Walk fallback_chain in order; first available → return with
     override_reason="primary_backend_unavailable".
  3. All exhausted → emergency cpu fallback, always state="available".
     override_reason depends on compiler_decision_source:
       "constraint_conflict"  → "constraint_conflict_emergency_cpu"
       anything else          → "all_backends_unavailable_cpu_emergency"
"""

from __future__ import annotations

from dataclasses import dataclass

from deployment.runtime_execution_plan import RuntimeExecutionPlan


@dataclass(frozen=True)
class BackendDecision:
    """Immutable result of a single BackendDispatcher.dispatch() call."""

    selected_backend: str
    backend_state: str       # "available" | "unavailable" | "thermal_limited" |
                             # "unsupported_precision" | "out_of_memory"
    override_reason: str     # "" iff selected_backend == primary_backend via step 1
    attempted_backends: list[str]   # ordered audit trail of every backend tried


class BackendDispatcher:
    def __init__(self, *, unavailable: set[str] | None = None) -> None:
        self._unavailable: frozenset[str] = frozenset(unavailable or set())

    def dispatch(self, plan: RuntimeExecutionPlan) -> BackendDecision:
        primary = plan.backend_policy.primary_backend
        fallback = list(plan.backend_policy.fallback_chain)
        decision_source = plan.backend_policy.compiler_decision_source

        candidates = [primary] + fallback
        attempted: list[str] = []

        for backend in candidates:
            attempted.append(backend)
            if backend not in self._unavailable:
                override = "" if backend == primary else "primary_backend_unavailable"
                return BackendDecision(
                    selected_backend=backend,
                    backend_state="available",
                    override_reason=override,
                    attempted_backends=list(attempted),
                )

        # All candidates unavailable: emergency cpu fallback.
        if "cpu" not in attempted:
            attempted.append("cpu")
        if decision_source == "constraint_conflict":
            override_reason = "constraint_conflict_emergency_cpu"
        else:
            override_reason = "all_backends_unavailable_cpu_emergency"
        return BackendDecision(
            selected_backend="cpu",
            backend_state="available",
            override_reason=override_reason,
            attempted_backends=list(attempted),
        )
