"""E2E-9: the unified selector. select_implementation() is the single entry
point spanning both decision categories (LaunchConfiguration for RMSNorm,
OperationDecomposition for Linear/LM-head): generate candidates -> filter to
legal candidates -> cost-predict each via the per-family model from
CostModelRegistry -> pick the minimum predicted latency -> ImplementationDecision.

This module does NOT know anything CUDA- or vLLM-specific -- it only
orchestrates the generic candidate/legality/cost-model interfaces. Family-
specific knowledge lives in implementation_candidate.py (what candidates
exist), legality.py (what's legal), and the two cost model adapters (how
fast each legal candidate is expected to be).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perf_model.cost_model_registry import CostEstimate, CostModelRegistry
from perf_model.implementation_candidate import ImplementationCandidate, generate_candidates
from perf_model.legality import LegalityResult, filter_legal_candidates
from perf_model.operation_descriptor import OperationDescriptor


@dataclass(frozen=True)
class ImplementationDecision:
    operation_family: str
    operation_subtype: str
    decision_kind: str
    selected_candidate: ImplementationCandidate
    predicted_cost: CostEstimate
    fallback_candidate: ImplementationCandidate | None
    all_legal_candidates: tuple[ImplementationCandidate, ...]
    all_cost_estimates: tuple[CostEstimate, ...]
    legality_results: tuple[LegalityResult, ...]
    selector_version: str = "unified_selector_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_family": self.operation_family, "operation_subtype": self.operation_subtype,
            "decision_kind": self.decision_kind, "selected_candidate": self.selected_candidate.to_dict(),
            "predicted_cost": self.predicted_cost.to_dict(),
            "fallback_candidate": self.fallback_candidate.to_dict() if self.fallback_candidate else None,
            "all_legal_candidates": [c.to_dict() for c in self.all_legal_candidates],
            "all_cost_estimates": [e.to_dict() for e in self.all_cost_estimates],
            "legality_results": [r.to_dict() for r in self.legality_results],
            "selector_version": self.selector_version,
        }


class NoLegalCandidateError(RuntimeError):
    pass


def select_implementation(operation: OperationDescriptor, registry: CostModelRegistry,
                           target: dict[str, Any] | None = None,
                           runtime_context: dict[str, Any] | None = None) -> ImplementationDecision:
    target = target or {}
    runtime_context = runtime_context or {}
    all_candidates = generate_candidates(operation, target, runtime_context)
    legal_candidates, legality_results = filter_legal_candidates(operation, all_candidates, target, runtime_context)
    if not legal_candidates:
        raise NoLegalCandidateError(
            f"no legal candidate for {operation.common.operation_family}/{operation.common.operation_subtype}: "
            f"{[r.to_dict() for r in legality_results]}"
        )

    estimates = [registry.predict(operation, c) for c in legal_candidates]
    best_idx = min(range(len(estimates)), key=lambda i: estimates[i].predicted_latency_ms)
    selected = legal_candidates[best_idx]
    fallback = next((c for c in legal_candidates if c.candidate_id == selected.fallback_candidate_id), None)

    return ImplementationDecision(
        operation_family=operation.common.operation_family.value,
        operation_subtype=operation.common.operation_subtype.value,
        decision_kind=selected.decision_kind.value,
        selected_candidate=selected, predicted_cost=estimates[best_idx], fallback_candidate=fallback,
        all_legal_candidates=tuple(legal_candidates), all_cost_estimates=tuple(estimates),
        legality_results=tuple(legality_results),
    )
