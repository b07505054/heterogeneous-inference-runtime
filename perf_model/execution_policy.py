"""E2E-9: ExecutionPolicy -- a small, standalone, JSON-round-trippable plan
schema produced by the unified selector, consumed by the unified runtime
dispatcher (perf_model/runtime_dispatcher.py).

Relationship to the existing `deployment/execution_plan/schema.py`
(`ExecutionPath`, `ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK`) and
`deployment/custom_cuda_adapter.py` (`CustomCudaBackendAdapter`): that is a
larger, pre-existing, differently-scoped compiler-plan abstraction covering
multi-backend (Hailo/CPU/custom_cuda) dispatch and capability validation
across this whole repository, already tested and depended on elsewhere. This
slice's "extend the execution-plan schema with an optional, backward-
compatible field" requirement is satisfied WITHOUT touching that file: an
`ExecutionPolicy` here is designed to be embeddable as an additional,
optional key (suggested name: `operation_policies`) on any existing plan
object -- including, if desired, a future `ExecutionPath` -- without that
object needing to know this schema's internals, and without this schema
needing to know `ExecutionPath`'s internals. Nothing in `deployment/` is
imported, modified, or required by this module. `attach_to_plan_dict` /
`extract_from_plan_dict` below are the only two functions that touch a host
plan dict, and they operate purely by additive key insertion/lookup -- an
existing plan dict with no `operation_policies` key round-trips through
`extract_from_plan_dict` as an empty policy set, so this is non-breaking by
construction.

Two selection modes:
  - "static": one fixed candidate, correct for a single anticipated
    (tokens, hidden) shape (used for RMSNorm, whose optimal block size is
    shape-dependent per the real measured sweep -- see
    perf_model/rmsnorm_cost_model_adapter.py).
  - "runtime_piecewise": a small explicit table mapping a dynamic operand
    (currently only "M") to a candidate_id, plus a default for any M not in
    the table, evaluated by the runtime dispatcher with O(1) dict lookup.
    Used for LM-head, whose optimal decomposition depends on the per-request
    batch size M and must be re-decided every call without re-running the
    full selector (see build_runtime_piecewise_policy, which derives the
    table by calling select_implementation() once per M up front, at plan-
    build time, not per-token).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from perf_model.cost_model_registry import CostModelRegistry
from perf_model.implementation_candidate import ImplementationCandidate
from perf_model.implementation_decision import ImplementationDecision, select_implementation
from perf_model.operation_descriptor import LinearDescriptor, OperationDescriptor

SCHEMA_VERSION = "e2e9_execution_policy_v1"


@dataclass(frozen=True)
class PolicyRule:
    attr: str
    value: int
    candidate_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"attr": self.attr, "value": self.value, "candidate_id": self.candidate_id}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PolicyRule":
        return PolicyRule(attr=d["attr"], value=d["value"], candidate_id=d["candidate_id"])


@dataclass(frozen=True)
class ExecutionPolicy:
    schema_version: str
    operation_family: str
    operation_subtype: str
    decision_kind: str
    selection_mode: str  # "static" | "runtime_piecewise"
    candidates_by_id: dict[str, ImplementationCandidate]
    static_candidate_id: str | None = None
    rules: tuple[PolicyRule, ...] = ()
    default_candidate_id: str | None = None
    fallback_candidate_id: str | None = None
    expected_latency_ms: float | None = None
    confidence: float | None = None
    provenance: str = ""

    def resolve_candidate_id(self, dynamic_operands: dict[str, int] | None = None) -> str:
        if self.selection_mode == "static":
            if self.static_candidate_id is None:
                raise ValueError("static policy missing static_candidate_id")
            return self.static_candidate_id
        if self.selection_mode == "runtime_piecewise":
            operands = dynamic_operands or {}
            for rule in self.rules:
                if operands.get(rule.attr) == rule.value:
                    return rule.candidate_id
            if self.default_candidate_id is None:
                raise ValueError("runtime_piecewise policy missing default_candidate_id")
            return self.default_candidate_id
        raise ValueError(f"unknown selection_mode: {self.selection_mode}")

    def resolve_candidate(self, dynamic_operands: dict[str, int] | None = None) -> ImplementationCandidate:
        return self.candidates_by_id[self.resolve_candidate_id(dynamic_operands)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "operation_family": self.operation_family,
            "operation_subtype": self.operation_subtype, "decision_kind": self.decision_kind,
            "selection_mode": self.selection_mode,
            "candidates_by_id": {k: v.to_dict() for k, v in self.candidates_by_id.items()},
            "static_candidate_id": self.static_candidate_id,
            "rules": [r.to_dict() for r in self.rules], "default_candidate_id": self.default_candidate_id,
            "fallback_candidate_id": self.fallback_candidate_id, "expected_latency_ms": self.expected_latency_ms,
            "confidence": self.confidence, "provenance": self.provenance,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExecutionPolicy":
        from perf_model.implementation_candidate import DecisionKind, get_implementation_kind_type
        from perf_model.operation_descriptor import OperationFamily

        candidates_by_id = {}
        for cid, cd in d["candidates_by_id"].items():
            family = OperationFamily(cd["operation_family"])
            kind_cls = get_implementation_kind_type(family)
            candidates_by_id[cid] = ImplementationCandidate(
                candidate_id=cd["candidate_id"], operation_family=family,
                operation_subtype=cd["operation_subtype"], decision_kind=DecisionKind(cd["decision_kind"]),
                implementation_kind=kind_cls(cd["implementation_kind"]), parameters=cd["parameters"],
                supported_dtypes=tuple(cd["supported_dtypes"]), supported_devices=tuple(cd["supported_devices"]),
                required_runtime_features=tuple(cd["required_runtime_features"]),
                fallback_candidate_id=cd["fallback_candidate_id"], provenance=cd["provenance"],
            )
        return ExecutionPolicy(
            schema_version=d["schema_version"], operation_family=d["operation_family"],
            operation_subtype=d["operation_subtype"], decision_kind=d["decision_kind"],
            selection_mode=d["selection_mode"], candidates_by_id=candidates_by_id,
            static_candidate_id=d.get("static_candidate_id"),
            rules=tuple(PolicyRule.from_dict(r) for r in d.get("rules", [])),
            default_candidate_id=d.get("default_candidate_id"), fallback_candidate_id=d.get("fallback_candidate_id"),
            expected_latency_ms=d.get("expected_latency_ms"), confidence=d.get("confidence"),
            provenance=d.get("provenance", ""),
        )

    @staticmethod
    def from_json(text: str) -> "ExecutionPolicy":
        return ExecutionPolicy.from_dict(json.loads(text))


def build_static_policy(operation: OperationDescriptor, decision: ImplementationDecision) -> ExecutionPolicy:
    candidates_by_id = {c.candidate_id: c for c in decision.all_legal_candidates}
    return ExecutionPolicy(
        schema_version=SCHEMA_VERSION, operation_family=decision.operation_family,
        operation_subtype=decision.operation_subtype, decision_kind=decision.decision_kind,
        selection_mode="static", candidates_by_id=candidates_by_id,
        static_candidate_id=decision.selected_candidate.candidate_id,
        fallback_candidate_id=decision.selected_candidate.fallback_candidate_id,
        expected_latency_ms=decision.predicted_cost.predicted_latency_ms,
        confidence=decision.predicted_cost.confidence,
        provenance=f"static selection for shape={operation.common.logical_shape}; "
                   f"source={decision.predicted_cost.evidence_note}",
    )


def build_runtime_piecewise_policy(operation_template: OperationDescriptor, registry: CostModelRegistry,
                                    m_values: list[int], target: dict[str, Any] | None = None,
                                    runtime_context: dict[str, Any] | None = None) -> ExecutionPolicy:
    """Derives a per-M candidate table by calling select_implementation() once
    per M value up front (at plan-build time), not per-token/per-request.
    This is what makes the resulting policy safe to cache and reuse in the
    runtime dispatcher's hot path (O(1) dict lookup on M, no re-selection)."""
    assert isinstance(operation_template.payload, LinearDescriptor)
    rules: list[PolicyRule] = []
    candidates_by_id: dict[str, ImplementationCandidate] = {}
    last_decision: ImplementationDecision | None = None
    for m in m_values:
        payload = LinearDescriptor(
            M=m, N=operation_template.payload.N, K=operation_template.payload.K,
            has_bias=operation_template.payload.has_bias,
            decode_or_prefill=operation_template.payload.decode_or_prefill,
            graph_captured=operation_template.payload.graph_captured,
            eager_execution=operation_template.payload.eager_execution,
            tensor_parallel_size=operation_template.payload.tensor_parallel_size,
            weight_layout=operation_template.payload.weight_layout,
            input_contiguous=operation_template.payload.input_contiguous,
        )
        op = OperationDescriptor(common=operation_template.common, payload=payload)
        decision = select_implementation(op, registry, target, runtime_context)
        last_decision = decision
        for c in decision.all_legal_candidates:
            candidates_by_id[c.candidate_id] = c
        rules.append(PolicyRule(attr="M", value=m, candidate_id=decision.selected_candidate.candidate_id))

    default_id = next((c.candidate_id for c in candidates_by_id.values()
                        if c.fallback_candidate_id is None and c.decision_kind.value == "operation_decomposition"),
                       None)
    return ExecutionPolicy(
        schema_version=SCHEMA_VERSION, operation_family=last_decision.operation_family,
        operation_subtype=last_decision.operation_subtype, decision_kind=last_decision.decision_kind,
        selection_mode="runtime_piecewise", candidates_by_id=candidates_by_id, rules=tuple(rules),
        default_candidate_id=default_id, fallback_candidate_id=default_id,
        provenance=f"derived from select_implementation() at plan-build time for M in {m_values}; "
                   f"cost model={registry.get_model(op.common.operation_family).model_id}",
    )


def attach_to_plan_dict(plan: dict[str, Any], policies: dict[str, ExecutionPolicy]) -> dict[str, Any]:
    """Additive-only: copies `plan` and inserts/overwrites the
    `operation_policies` key. Never touches any other key, so this is safe to
    call on any existing plan dict (including deployment/execution_plan
    plans) without affecting their own schema/validation."""
    out = dict(plan)
    out["operation_policies"] = {k: v.to_dict() for k, v in policies.items()}
    return out


def extract_from_plan_dict(plan: dict[str, Any]) -> dict[str, ExecutionPolicy]:
    raw = plan.get("operation_policies", {})
    return {k: ExecutionPolicy.from_dict(v) for k, v in raw.items()}
