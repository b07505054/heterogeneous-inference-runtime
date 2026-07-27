import pytest

from perf_model.evidence_schema import (
    Evidence, EvidenceKind, EvidenceValidationError, ValidationState, validate_parent_evidence,
)
from perf_model.claim_gate import (
    ClaimDecision, PerformanceClaim, evaluate_claim, reject_aggregate_claim_from_per_shape_evidence,
)


def _hw_evidence(evidence_id="hw1", validation_state=ValidationState.HARDWARE_VALIDATED, **overrides):
    base = dict(
        evidence_id=evidence_id, evidence_kind=EvidenceKind.HARDWARE_BENCHMARK, workload_id="rmsnorm_16x4096",
        operation_family="rms_norm", candidate_id="cuda_rmsnorm_fp32_bs512_v1", workload_shape={"tokens": 16, "hidden": 4096},
        target_profile_id="gtx1650_maxq", validation_state=validation_state, timestamp="2026-07-27T00:00:00Z",
        raw_artifact_path="artifacts/e2e12/rmsnorm/raw.json", source_hashes={"runtime_dispatcher.py": "abc123"},
        binary_hash="deadbeef", hardware_identity="gtx1650_maxq_cc75",
    )
    base.update(overrides)
    return Evidence(**base)


def _sim_evidence(evidence_id="sim1", notes="", **overrides):
    base = dict(
        evidence_id=evidence_id, evidence_kind=EvidenceKind.SIMULATOR_OUTPUT, workload_id="rmsnorm_16x4096",
        operation_family="rms_norm", candidate_id="cuda_rmsnorm_fp32_bs512_v1", workload_shape={"tokens": 16, "hidden": 4096},
        target_profile_id="gtx1650_maxq_accelsim_approx", validation_state=ValidationState.SIMULATED,
        timestamp="2026-07-27T00:00:00Z", raw_artifact_path="artifacts/e2e12/accel_sim/raw.json",
        source_hashes={"runtime_dispatcher.py": "abc123"}, binary_hash="deadbeef",
        simulator_config_hash="cfg123", simulator_name="accel-sim", simulator_version="c5296df", notes=notes,
    )
    base.update(overrides)
    return Evidence(**base)


def _cap_evidence(evidence_id="cap1", validation_state=ValidationState.SOURCE_VALIDATED):
    return Evidence(
        evidence_id=evidence_id, evidence_kind=EvidenceKind.DECLARED_CAPABILITY, workload_id="rmsnorm_16x4096",
        operation_family="rms_norm", candidate_id="cuda_rmsnorm_fp32_bs512_v1", workload_shape={"tokens": 16, "hidden": 4096},
        target_profile_id="gtx1650_maxq", validation_state=validation_state, timestamp="2026-07-27T00:00:00Z",
        raw_artifact_path="configs/target_profile.json", source_hashes={"target_profile.json": "xyz"},
    )


def _plan_evidence(evidence_id="plan1", validation_state=ValidationState.UNVALIDATED):
    return Evidence(
        evidence_id=evidence_id, evidence_kind=EvidenceKind.PLANNING_ESTIMATE, workload_id="rmsnorm_16x4096",
        operation_family="rms_norm", candidate_id="cuda_rmsnorm_fp32_bs512_v1", workload_shape={"tokens": 16, "hidden": 4096},
        target_profile_id="gtx1650_maxq", validation_state=validation_state, timestamp="2026-07-27T00:00:00Z",
        raw_artifact_path="perf_model/rmsnorm_cost_model_adapter.py", source_hashes={"rmsnorm_cost_model_adapter.py": "def"},
    )


# ---------- Evidence schema ----------

def test_evidence_json_round_trip():
    e = _hw_evidence()
    e2 = Evidence.from_json(e.to_json())
    assert e2 == e


def test_evidence_requires_evidence_id():
    with pytest.raises(EvidenceValidationError):
        _hw_evidence(evidence_id="")


def test_evidence_kind_validation_hardware_requires_binary_and_identity():
    with pytest.raises(EvidenceValidationError):
        _hw_evidence(binary_hash="")


def test_evidence_kind_validation_simulator_requires_config_and_binary_hash():
    with pytest.raises(EvidenceValidationError):
        _sim_evidence(simulator_config_hash="")


def test_declared_capability_cannot_carry_hardware_validated_state():
    with pytest.raises(EvidenceValidationError):
        _cap_evidence(validation_state=ValidationState.HARDWARE_VALIDATED)


def test_planning_estimate_cannot_carry_simulated_state():
    with pytest.raises(EvidenceValidationError):
        _plan_evidence(validation_state=ValidationState.SIMULATED)


def test_parent_evidence_validation_detects_dangling_reference():
    e = _hw_evidence(evidence_id="child1")
    e = Evidence(**{**e.to_dict(), "evidence_kind": EvidenceKind(e.to_dict()["evidence_kind"]),
                     "validation_state": ValidationState(e.to_dict()["validation_state"]),
                     "parent_evidence_ids": ("nonexistent_parent",)})
    dangling = validate_parent_evidence(e, known_evidence_ids={"hw1"})
    assert dangling == ["nonexistent_parent"]


def test_hardware_identity_required_for_hardware_benchmark():
    with pytest.raises(EvidenceValidationError):
        _hw_evidence(hardware_identity="")


# ---------- Claim gate ----------

def _claim(**overrides):
    base = dict(claim_id="c1", claim_text="test claim", claim_level="target_specific_performance_claim",
                operation_family="rms_norm", baseline="torch.compile", excluded_costs=())
    base.update(overrides)
    return PerformanceClaim(**base)


def test_simulator_only_cannot_become_hardware_validated():
    result = evaluate_claim(_claim(), [_sim_evidence()])
    assert result.decision != ClaimDecision.PROMOTE_HARDWARE_VALIDATED
    assert result.decision == ClaimDecision.ALLOW_SIMULATOR_ONLY_DIRECTIONAL


def test_declared_capability_alone_does_not_promote():
    result = evaluate_claim(_claim(), [_cap_evidence()])
    assert result.decision != ClaimDecision.PROMOTE_HARDWARE_VALIDATED


def test_missing_source_hash_rejects_promotion():
    e = _hw_evidence(source_hashes={})
    result = evaluate_claim(_claim(), [e])
    assert result.decision == ClaimDecision.REJECT_MISSING_PROVENANCE


def test_missing_config_hash_rejects_simulator_promotion():
    # construct evidence dict manually bypassing Evidence.__post_init__'s own check,
    # to test the GATE's independent enforcement of rule 7 (defense in depth)
    e = _hw_evidence()
    d = e.to_dict()
    d["evidence_kind"] = EvidenceKind.SIMULATOR_OUTPUT
    d["simulator_config_hash"] = "present"
    d["source_hashes"] = {}
    e2 = Evidence(evidence_id=d["evidence_id"], evidence_kind=EvidenceKind.SIMULATOR_OUTPUT, workload_id=d["workload_id"],
                  operation_family=d["operation_family"], candidate_id=d["candidate_id"], workload_shape=d["workload_shape"],
                  target_profile_id=d["target_profile_id"], validation_state=ValidationState.SIMULATED,
                  timestamp=d["timestamp"], raw_artifact_path=d["raw_artifact_path"], binary_hash=d["binary_hash"],
                  simulator_config_hash="present", source_hashes={})
    result = evaluate_claim(_claim(), [e2])
    assert result.decision == ClaimDecision.REJECT_MISSING_PROVENANCE


def test_baseline_missing_rejects():
    result = evaluate_claim(_claim(baseline=None), [_hw_evidence()])
    assert result.decision == ClaimDecision.REJECT_MISSING_PROVENANCE


def test_excluded_costs_none_rejects():
    result = evaluate_claim(_claim(excluded_costs=None), [_hw_evidence()])
    assert result.decision == ClaimDecision.REJECT_MISSING_PROVENANCE


def test_target_mismatch_rejects_cross_target_claim():
    result = evaluate_claim(_claim(claim_level="cross_target_generalization_claim"), [_hw_evidence()])
    assert result.decision == ClaimDecision.REJECT_UNSUPPORTED_TARGET


def test_cross_target_claim_promotes_with_two_targets():
    e1 = _hw_evidence(evidence_id="hw_a", target_profile_id="gtx1650_maxq")
    e2 = _hw_evidence(evidence_id="hw_b", target_profile_id="raspberry_pi5_cortex_a76")
    result = evaluate_claim(_claim(claim_level="cross_target_generalization_claim"), [e1, e2])
    assert result.decision in (ClaimDecision.PROMOTE_HARDWARE_VALIDATED, ClaimDecision.PROMOTE_HARDWARE_CORRELATED)


def test_valid_hardware_only_evidence_promotes_hardware_validated():
    result = evaluate_claim(_claim(), [_hw_evidence()])
    assert result.decision == ClaimDecision.PROMOTE_HARDWARE_VALIDATED


def test_valid_hardware_plus_simulator_promotes_hardware_correlated():
    result = evaluate_claim(_claim(), [_hw_evidence(), _sim_evidence()])
    assert result.decision == ClaimDecision.PROMOTE_HARDWARE_CORRELATED


def test_simulator_hardware_disagreement_rejected_if_not_acknowledged():
    sim = _sim_evidence(notes="simulator winner disagrees with hardware winner at this shape")
    result = evaluate_claim(_claim(), [_hw_evidence(), sim])
    assert result.decision == ClaimDecision.REJECT_SIMULATION_MISMATCH


def test_simulator_hardware_disagreement_preserved_when_acknowledged():
    sim = _sim_evidence(notes="simulator winner disagrees with hardware winner at this shape")
    result = evaluate_claim(_claim(simulator_hardware_disagreement_acknowledged=True), [_hw_evidence(), sim])
    assert result.decision == ClaimDecision.PROMOTE_HARDWARE_CORRELATED
    assert any("disagree" in r for r in result.reasons)


def test_rejected_rmsnorm_selector_aggregate_claim_remains_rejected():
    # real E2E-11 per-shape selector-vs-fixed-block ratios (fixed512/selected)
    per_shape_ratios = [0.984, 0.934, 0.978, 1.000, 0.968, 0.994, 0.966, 0.950, 1.000, 1.071, 0.993, 0.993]
    result = reject_aggregate_claim_from_per_shape_evidence(per_shape_ratios, threshold=1.0)
    assert result.decision == ClaimDecision.DOWNGRADE_TARGET_SPECIFIC
    assert "does not exceed threshold" in result.reasons[0]
