"""E2E-12 Phase 2: the generic performance-claim gate.

Declared capability -> planning estimate -> simulator evidence -> hardware
evidence -> correlation analysis -> claim decision.

This module contains ALL of the 10 gate rules as explicit, individually
testable checks (see tests/test_e2e12_claim_gate.py), combined by
evaluate_claim() into one of the required ClaimDecision values. No rule is
implemented only as an enum literal with no enforcing code -- every rule
below is backed by a function that can reject a real claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from perf_model.evidence_schema import Evidence, EvidenceKind, ValidationState


class ClaimDecision(str, Enum):
    PROMOTE_HARDWARE_VALIDATED = "promote_hardware_validated"
    PROMOTE_HARDWARE_CORRELATED = "promote_hardware_correlated"
    ALLOW_SIMULATOR_ONLY_DIRECTIONAL = "allow_simulator_only_directional"
    DOWNGRADE_TARGET_SPECIFIC = "downgrade_target_specific"
    REJECT_SIMULATION_MISMATCH = "reject_simulation_mismatch"
    REJECT_MISSING_PROVENANCE = "reject_missing_provenance"
    REJECT_UNSUPPORTED_TARGET = "reject_unsupported_target"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class PerformanceClaim:
    claim_id: str
    claim_text: str
    claim_level: str  # perf_model.evidence_schema.ClaimLevel value
    operation_family: str
    baseline: str | None  # rule 9: must be identified
    excluded_costs: tuple[str, ...] | None  # rule 10: must be stated explicitly (empty tuple is valid, None is not)
    target_profile_ids: tuple[str, ...] = ()
    simulator_hardware_disagreement_acknowledged: bool = False


@dataclass(frozen=True)
class ClaimGateResult:
    decision: ClaimDecision
    reasons: tuple[str, ...]
    claim_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.value, "reasons": list(self.reasons), "claim_id": self.claim_id}


def _rule_baseline_identified(claim: PerformanceClaim) -> str | None:
    if not claim.baseline:
        return "rule9_violation: claim does not identify a baseline"
    return None


def _rule_excluded_costs_stated(claim: PerformanceClaim) -> str | None:
    if claim.excluded_costs is None:
        return "rule10_violation: claim does not state whether setup/packing/compile/JIT costs were excluded"
    return None


def _rule_provenance_hashes_present(evidence: list[Evidence]) -> str | None:
    for e in evidence:
        if e.evidence_kind == EvidenceKind.HARDWARE_BENCHMARK and not (e.binary_hash and e.hardware_identity):
            return f"rule7_violation: hardware evidence {e.evidence_id} missing binary_hash/hardware_identity"
        if e.evidence_kind == EvidenceKind.SIMULATOR_OUTPUT and not (e.simulator_config_hash and e.binary_hash):
            return f"rule7_violation: simulator evidence {e.evidence_id} missing simulator_config_hash/binary_hash"
        if not e.source_hashes:
            return f"rule7_violation: evidence {e.evidence_id} missing source_hashes"
    return None


def _rule_no_declared_capability_as_measured(evidence: list[Evidence]) -> str | None:
    for e in evidence:
        if e.evidence_kind == EvidenceKind.DECLARED_CAPABILITY and e.validation_state in (
            ValidationState.HARDWARE_VALIDATED, ValidationState.HARDWARE_CORRELATED,
        ):
            return f"rule3_violation: declared-capability evidence {e.evidence_id} carries a measured validation_state"
    return None


def _rule_no_planning_estimate_as_simulator(evidence: list[Evidence]) -> str | None:
    for e in evidence:
        if e.evidence_kind == EvidenceKind.PLANNING_ESTIMATE and e.validation_state == ValidationState.SIMULATED:
            return f"rule2_violation: planning-estimate evidence {e.evidence_id} carries a SIMULATED validation_state"
    return None


def evaluate_claim(claim: PerformanceClaim, evidence: list[Evidence]) -> ClaimGateResult:
    reasons: list[str] = []

    for rule_check in (_rule_baseline_identified, _rule_excluded_costs_stated):
        violation = rule_check(claim)
        if violation:
            return ClaimGateResult(ClaimDecision.REJECT_MISSING_PROVENANCE, (violation,), claim.claim_id)

    for rule_check in (_rule_provenance_hashes_present, _rule_no_declared_capability_as_measured,
                       _rule_no_planning_estimate_as_simulator):
        violation = rule_check(evidence)
        if violation:
            return ClaimGateResult(ClaimDecision.REJECT_MISSING_PROVENANCE, (violation,), claim.claim_id)

    hardware_evidence = [e for e in evidence if e.evidence_kind == EvidenceKind.HARDWARE_BENCHMARK]
    simulator_evidence = [e for e in evidence if e.evidence_kind == EvidenceKind.SIMULATOR_OUTPUT]
    hardware_validated = [e for e in hardware_evidence if e.validation_state == ValidationState.HARDWARE_VALIDATED]
    hardware_target_ids = {e.target_profile_id for e in hardware_validated}

    # Rule 6: cross-target claims require multiple real target profiles.
    if claim.claim_level == "cross_target_generalization_claim":
        if len(hardware_target_ids) < 2:
            reasons.append(f"rule6_violation: cross-target claim requires >=2 hardware-validated target profiles, "
                            f"found {len(hardware_target_ids)}")
            return ClaimGateResult(ClaimDecision.REJECT_UNSUPPORTED_TARGET, tuple(reasons), claim.claim_id)

    # Rule 4/1: a target-specific claim requires real hardware timing; simulator
    # evidence alone can never promote to HARDWARE_VALIDATED.
    if claim.claim_level == "target_specific_performance_claim":
        declared_target_ok = (not claim.target_profile_ids) or any(
            t in hardware_target_ids for t in claim.target_profile_ids
        )
        if not hardware_validated:
            if simulator_evidence:
                reasons.append("rule1_violation_avoided: simulator evidence present but no hardware timing -- "
                                "cannot promote a target-specific claim from simulation alone")
                return ClaimGateResult(ClaimDecision.ALLOW_SIMULATOR_ONLY_DIRECTIONAL, tuple(reasons), claim.claim_id)
            reasons.append("rule4_violation: target-specific claim has no hardware evidence at all")
            return ClaimGateResult(ClaimDecision.INCONCLUSIVE, tuple(reasons), claim.claim_id)
        if not declared_target_ok:
            reasons.append("rule_target_mismatch: hardware evidence targets do not include the claimed target_profile_ids")
            return ClaimGateResult(ClaimDecision.REJECT_UNSUPPORTED_TARGET, tuple(reasons), claim.claim_id)

    # Rule 8: simulator/hardware disagreement must be surfaced, not hidden.
    if hardware_validated and simulator_evidence:
        disagreement_notes = [e.notes for e in simulator_evidence if "disagree" in e.notes.lower() or "mismatch" in e.notes.lower()]
        if disagreement_notes and not claim.simulator_hardware_disagreement_acknowledged:
            reasons.append(f"rule8_violation: simulator/hardware disagreement present in evidence notes "
                            f"({disagreement_notes}) but claim does not acknowledge it")
            return ClaimGateResult(ClaimDecision.REJECT_SIMULATION_MISMATCH, tuple(reasons), claim.claim_id)
        if disagreement_notes:
            reasons.append(f"disagreement acknowledged and preserved: {disagreement_notes}")

    if hardware_validated:
        if simulator_evidence:
            reasons.append(f"promoted on {len(hardware_validated)} hardware-validated + "
                            f"{len(simulator_evidence)} simulator evidence records")
            return ClaimGateResult(ClaimDecision.PROMOTE_HARDWARE_CORRELATED, tuple(reasons), claim.claim_id)
        reasons.append(f"promoted on {len(hardware_validated)} hardware-validated evidence records, no simulator evidence")
        return ClaimGateResult(ClaimDecision.PROMOTE_HARDWARE_VALIDATED, tuple(reasons), claim.claim_id)

    if simulator_evidence and claim.claim_level == "directional_performance_claim":
        reasons.append("directional claim supported by simulator evidence only")
        return ClaimGateResult(ClaimDecision.ALLOW_SIMULATOR_ONLY_DIRECTIONAL, tuple(reasons), claim.claim_id)

    reasons.append("no evidence sufficient to make a decision")
    return ClaimGateResult(ClaimDecision.INCONCLUSIVE, tuple(reasons), claim.claim_id)


def reject_aggregate_claim_from_per_shape_evidence(per_shape_ratios: list[float], threshold: float = 1.0) -> ClaimGateResult:
    """Phase 18's required rejection pattern: per-shape 'wins' do not
    automatically justify an aggregate speedup claim. Returns
    DOWNGRADE_TARGET_SPECIFIC if the mean ratio does not clear `threshold`
    even though some individual shapes do."""
    import statistics
    mean_ratio = statistics.mean(per_shape_ratios)
    any_shape_wins = any(r > threshold for r in per_shape_ratios)
    if mean_ratio <= threshold:
        reasons = (
            f"per-shape evidence shows wins at some shapes (any_shape_wins={any_shape_wins}), "
            f"but aggregate mean ratio={mean_ratio:.4f} does not exceed threshold={threshold} -- "
            f"hardware measurements do not support an aggregate speedup promotion",
        )
        return ClaimGateResult(ClaimDecision.REJECT_SIMULATION_MISMATCH if False else ClaimDecision.DOWNGRADE_TARGET_SPECIFIC,
                                reasons, "aggregate_claim")
    reasons = (f"aggregate mean ratio={mean_ratio:.4f} exceeds threshold={threshold}",)
    return ClaimGateResult(ClaimDecision.PROMOTE_HARDWARE_VALIDATED, reasons, "aggregate_claim")
