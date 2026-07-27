"""E2E-12 Phase 1: typed evidence taxonomy separating declared capabilities,
planning estimates, simulator outputs, and hardware measurements.

This is a NEW, standalone schema (perf_model/evidence_schema.py), not an
extension of CostEstimate (perf_model/cost_model_registry.py) -- deliberately
kept separate per the task's explicit instruction not to overload
CostEstimate with provenance data it was never designed to carry. CostEstimate
answers "what does the selector predict"; Evidence answers "what do we
actually know, from where, and how strongly can we trust it."
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceKind(str, Enum):
    DECLARED_CAPABILITY = "declared_capability"
    PLANNING_ESTIMATE = "planning_estimate"
    SIMULATOR_OUTPUT = "simulator_output"
    HARDWARE_BENCHMARK = "hardware_benchmark"
    HARDWARE_PROFILER = "hardware_profiler"
    DERIVED_ANALYSIS = "derived_analysis"


class ValidationState(str, Enum):
    UNVALIDATED = "unvalidated"
    SOURCE_VALIDATED = "source_validated"
    SIMULATED = "simulated"
    HARDWARE_CORRELATED = "hardware_correlated"
    HARDWARE_VALIDATED = "hardware_validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ClaimLevel(str, Enum):
    ARCHITECTURAL_CLAIM = "architectural_claim"
    DIRECTIONAL_PERFORMANCE_CLAIM = "directional_performance_claim"
    TARGET_SPECIFIC_PERFORMANCE_CLAIM = "target_specific_performance_claim"
    CROSS_TARGET_GENERALIZATION_CLAIM = "cross_target_generalization_claim"


# Evidence kinds that are ALLOWED to claim they observed real execution.
# DECLARED_CAPABILITY and PLANNING_ESTIMATE never satisfy this -- enforced
# both here (construction-time) and again in the claim gate (decision-time),
# so the constraint cannot be bypassed by only checking one layer.
_MEASURED_EXECUTION_KINDS = frozenset({EvidenceKind.HARDWARE_BENCHMARK, EvidenceKind.HARDWARE_PROFILER})
_SIMULATED_KINDS = frozenset({EvidenceKind.SIMULATOR_OUTPUT})


class EvidenceValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    evidence_kind: EvidenceKind
    workload_id: str
    operation_family: str
    candidate_id: str
    workload_shape: dict[str, int]
    target_profile_id: str
    validation_state: ValidationState
    timestamp: str
    raw_artifact_path: str
    source_commit: str = ""
    source_hashes: dict[str, str] = field(default_factory=dict)
    binary_hash: str = ""
    trace_hash: str = ""
    execution_plan_hash: str = ""
    simulator_name: str = ""
    simulator_version: str = ""
    simulator_commit: str = ""
    simulator_config_hash: str = ""
    benchmark_config_hash: str = ""
    hardware_identity: str = ""
    parent_evidence_ids: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise EvidenceValidationError("evidence_id is required")
        if not self.workload_id or not self.operation_family or not self.candidate_id:
            raise EvidenceValidationError("workload_id/operation_family/candidate_id are required")
        if not self.target_profile_id:
            raise EvidenceValidationError("target_profile_id is required")
        if not self.timestamp:
            raise EvidenceValidationError("timestamp is required")

        # Phase 2 rules 2/3, enforced here at construction time as well as
        # in the claim gate: a DECLARED_CAPABILITY or PLANNING_ESTIMATE
        # record can never carry a HARDWARE_VALIDATED / HARDWARE_CORRELATED
        # / SIMULATED validation_state -- those states assert something was
        # actually run.
        if self.evidence_kind in (EvidenceKind.DECLARED_CAPABILITY, EvidenceKind.PLANNING_ESTIMATE):
            if self.validation_state in (ValidationState.HARDWARE_VALIDATED, ValidationState.HARDWARE_CORRELATED,
                                          ValidationState.SIMULATED):
                raise EvidenceValidationError(
                    f"evidence_kind={self.evidence_kind.value} cannot carry validation_state="
                    f"{self.validation_state.value} -- declared capabilities and planning estimates "
                    f"were never executed, measured, or simulated"
                )
        if self.evidence_kind in _MEASURED_EXECUTION_KINDS and not (self.binary_hash and self.hardware_identity):
            raise EvidenceValidationError(
                f"evidence_kind={self.evidence_kind.value} requires binary_hash and hardware_identity "
                f"-- a hardware measurement with no binary/hardware identity cannot be trusted or reproduced"
            )
        if self.evidence_kind in _SIMULATED_KINDS and not (self.simulator_config_hash and self.binary_hash):
            raise EvidenceValidationError(
                f"evidence_kind={self.evidence_kind.value} requires simulator_config_hash and binary_hash "
                f"-- a simulator result with no config/binary identity cannot be trusted or reproduced"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence_kind"] = self.evidence_kind.value
        d["validation_state"] = self.validation_state.value
        d["parent_evidence_ids"] = list(self.parent_evidence_ids)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Evidence":
        d = dict(d)
        d["evidence_kind"] = EvidenceKind(d["evidence_kind"])
        d["validation_state"] = ValidationState(d["validation_state"])
        d["parent_evidence_ids"] = tuple(d.get("parent_evidence_ids", ()))
        return Evidence(**d)

    @staticmethod
    def from_json(text: str) -> "Evidence":
        return Evidence.from_dict(json.loads(text))


def validate_parent_evidence(evidence: Evidence, known_evidence_ids: set[str]) -> list[str]:
    """Returns a list of parent_evidence_ids that do NOT resolve to a known
    evidence record -- dangling parent references, analogous to E2E-9/10's
    dangling-fallback-candidate check."""
    return [pid for pid in evidence.parent_evidence_ids if pid not in known_evidence_ids]
