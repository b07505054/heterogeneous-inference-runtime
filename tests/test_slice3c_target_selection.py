from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.execution_plan.int8_quantization import (
    PACKED_B_TRANSPOSE_LAYOUT,
    PACKED_B_TRANSPOSE_SCHEME,
    PACKED_INT8_KERNEL_ID,
    create_calibration_artifact,
)
from deployment.execution_plan.slice3c_target_selection import (
    CandidateGenerator,
    GENERATION_REJECTION_REASONS,
    EXECUTORCH_XNNPACK_INT8_T1_CANDIDATE_ID,
    EXECUTORCH_XNNPACK_INT8_T4_CANDIDATE_ID,
    FP32_CANDIDATE_ID,
    INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,
    INT8_PACKED_GENERIC_CANDIDATE_ID,
    candidate_legality,
    create_build_manifest,
    enumerate_complete_candidates,
    generate_candidate_graph,
    load_codegen_capabilities,
    select_candidate,
    validate_measurement,
)


def test_candidate_generator_cartesian_expansion_and_no_ranking():
    graph = generate_candidate_graph()
    expected = 1
    for dimension in graph.dimensions:
        expected *= len(dimension.values)
    assert len(graph.nodes) == expected
    assert len(graph.accepted_candidates) == 7
    assert not hasattr(CandidateGenerator, "rank")
    assert not hasattr(CandidateGenerator, "select")


def test_candidate_graph_records_pruned_combinations_and_stable_reasons():
    graph = generate_candidate_graph()
    rejected = [node for node in graph.nodes if node.candidate is None]
    assert rejected
    assert all(node.rejection_reasons for node in rejected)
    assert {reason for node in rejected for reason in node.rejection_reasons} <= GENERATION_REJECTION_REASONS
    view = graph.to_visualization_dict()
    assert view["summary"] == {
        "combination_count": len(graph.nodes),
        "accepted_count": 7,
        "rejected_count": len(graph.nodes) - 7,
    }


def test_generation_identity_is_stable_and_dimension_sensitive():
    first = generate_candidate_graph()
    second = CandidateGenerator().generate()
    assert [c.candidate_id for c in first.accepted_candidates] == [c.candidate_id for c in second.accepted_candidates]
    ids = {c.candidate_id for c in first.accepted_candidates}
    assert len(ids) == 7
    assert EXECUTORCH_XNNPACK_INT8_T1_CANDIDATE_ID in ids
    assert EXECUTORCH_XNNPACK_INT8_T4_CANDIDATE_ID in ids


def test_dimensions_have_identity_constraints_and_dependency_rules():
    dimensions = CandidateGenerator().discover_dimensions()
    assert {d.name for d in dimensions} == {
        "backend", "runtime", "delegate", "precision", "quantization_scheme",
        "kernel", "layout", "packing", "tile", "thread_count",
        "memory_placement", "target_isa", "artifact_kind",
        "execution_strategy", "measurement_evidence",
    }
    assert all(d.identity for d in dimensions)
    assert all(d.values for d in dimensions)
    assert any(d.legality_constraints for d in dimensions)
    assert any(d.dependency_rules for d in dimensions)


def test_generation_and_target_legality_are_independent():
    candidates = enumerate_complete_candidates()
    a76 = next(c for c in candidates if c.candidate_id == INT8_PACKED_A76_DOTPROD_CANDIDATE_ID)
    generic = load_codegen_capabilities(_profile(isaFeatures=["asimd"], supportsInt8DotProduct=False, microarchitecture="generic"))
    assert a76 in candidates
    assert "missing_microarchitecture_capability" in candidate_legality(
        a76, generic, has_calibration_artifact=True, has_packed_artifact=True,
        build_tool_flags=generic.supported_compiler_target_flags,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json"


def _profile(**overrides):
    p = json.loads(PROFILE.read_text())
    p["cpuCodegenCapabilities"].update(overrides)
    return p


def _artifact():
    return create_calibration_artifact(
        workload_id="w", operator_kind="fused_matmul_bias_relu", m=2, n=3, k=4,
        activation_values=[0.1] * 8, weight_values=[0.2] * 12,
        calibration_dataset={"dataset_id": "synthetic", "sample_count": 1},
    )


def _packed():
    return {"artifact_id": "p", "artifact_sha256": "packedhash"}


def test_target_capabilities_parse_and_round_trip():
    caps = load_codegen_capabilities(json.loads(PROFILE.read_text()))
    assert caps.architecture == "aarch64"
    assert caps.microarchitecture == "cortex-a76"
    assert "asimd" in caps.isa_features
    assert "asimddp" in caps.isa_features
    assert caps.supports_int8_dot_product is True


def test_complete_candidate_identities_differ():
    ids = [c.candidate_id for c in enumerate_complete_candidates()]
    assert len(ids) == len(set(ids))
    assert INT8_PACKED_A76_DOTPROD_CANDIDATE_ID in ids
    assert INT8_PACKED_GENERIC_CANDIDATE_ID in ids


@pytest.mark.parametrize("override,reason", [
    ({"architecture": ""}, "missing_target_architecture"),
    ({"microarchitecture": ""}, "missing_microarchitecture_capability"),
    ({"isaFeatures": ["asimddp"]}, "missing_asimd"),
    ({"isaFeatures": ["asimd"], "supportsInt8DotProduct": False}, "missing_asimddp"),
    ({"supportedCompilerTargetFlags": ["-O2"]}, "unsupported_codegen_flags"),
])
def test_a76_candidate_legality_rejection_reasons(override, reason):
    caps = load_codegen_capabilities(_profile(**override))
    cand = {c.candidate_id: c for c in enumerate_complete_candidates()}[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]
    reasons = candidate_legality(cand, caps, has_calibration_artifact=True, has_packed_artifact=True,
                                 build_tool_flags=("-O2", "-O3", "-mcpu=cortex-a76"))
    assert reason in reasons


def test_build_manifest_contains_compiler_owned_flags(tmp_path):
    cand = {c.candidate_id: c for c in enumerate_complete_candidates()}[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]
    manifest = create_build_manifest(cand, source_root=REPO_ROOT, output_dir=tmp_path)
    assert manifest["compiler_flags"] == ["-O3", "-mcpu=cortex-a76", "-std=c++17"]
    assert manifest["codegen_target_id"] == "cortex_a76_dotprod"


def _measurement(candidate_id, *, binary="bin", packed="packedhash", latency=1.0, cosine=0.999, rel=0.01, target="raspberry-pi5-cortex-a76-cpu"):
    c = {c.candidate_id: c for c in enumerate_complete_candidates()}[candidate_id]
    return {
        "candidate_id": candidate_id, "kernel_id": c.kernel_id, "target_id": target,
        "target_architecture": "aarch64", "shape": {"M": 2, "N": 3, "K": 4},
        "binary_sha256": binary, "packed_artifact_sha256": packed,
        "calibration_artifact_sha256": _artifact()["artifact_sha256"],
        "correctness_metrics": {"cosine_similarity": cosine, "relative_l2_error": rel},
        "latency_median_ms": latency, "latency_p95_ms": latency * 1.05,
        "latency_stddev_ms": 0.01, "sample_count": 7,
    }


def test_measurement_identity_validation_rejects_wrong_binary_and_artifact():
    caps = load_codegen_capabilities(json.loads(PROFILE.read_text()))
    cand = {c.candidate_id: c for c in enumerate_complete_candidates()}[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]
    good = _measurement(cand.candidate_id)
    assert validate_measurement(good, cand, caps, shape={"M": 2, "N": 3, "K": 4}, binary_sha256="bin", packed_artifact=_packed(), calibration_artifact=_artifact()) == []
    assert "measurement_binary_sha256_mismatch" in validate_measurement(good, cand, caps, shape={"M": 2, "N": 3, "K": 4}, binary_sha256="other", packed_artifact=_packed(), calibration_artifact=_artifact())
    assert "wrong_packed_artifact" in validate_measurement(good, cand, caps, shape={"M": 2, "N": 3, "K": 4}, binary_sha256="bin", packed_artifact={"artifact_sha256": "other"}, calibration_artifact=_artifact())


def test_correctness_gate_and_lowest_latency_selection():
    caps = load_codegen_capabilities(json.loads(PROFILE.read_text()))
    candidates = enumerate_complete_candidates()
    meas = {
        FP32_CANDIDATE_ID: _measurement(FP32_CANDIDATE_ID, binary="fp", packed="", latency=2.0),
        INT8_PACKED_A76_DOTPROD_CANDIDATE_ID: _measurement(INT8_PACKED_A76_DOTPROD_CANDIDATE_ID, binary="a76", latency=0.5),
    }
    bins = {FP32_CANDIDATE_ID: "fp", INT8_PACKED_A76_DOTPROD_CANDIDATE_ID: "a76"}
    sel = select_candidate(candidates, target=caps, shape={"M": 2, "N": 3, "K": 4}, measurements=meas,
                           binary_sha256_by_candidate=bins, packed_artifact=_packed(), calibration_artifact=_artifact(),
                           has_packed_artifact=True, has_calibration_artifact=True,
                           build_tool_flags=("-O2", "-O3", "-mcpu=cortex-a76"))
    assert sel["selected_candidate_id"] == INT8_PACKED_A76_DOTPROD_CANDIDATE_ID
    meas[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID] = _measurement(INT8_PACKED_A76_DOTPROD_CANDIDATE_ID, binary="a76", latency=0.5, cosine=0.5)
    sel = select_candidate(candidates, target=caps, shape={"M": 2, "N": 3, "K": 4}, measurements=meas,
                           binary_sha256_by_candidate=bins, packed_artifact=_packed(), calibration_artifact=_artifact(),
                           has_packed_artifact=True, has_calibration_artifact=True,
                           build_tool_flags=("-O2", "-O3", "-mcpu=cortex-a76"))
    assert sel["selected_candidate_id"] == FP32_CANDIDATE_ID


def test_generic_target_rejects_a76_candidate():
    caps = load_codegen_capabilities(_profile(isaFeatures=["asimd"], supportsInt8DotProduct=False, microarchitecture="generic"))
    cand = {c.candidate_id: c for c in enumerate_complete_candidates()}[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]
    reasons = candidate_legality(cand, caps, has_calibration_artifact=True, has_packed_artifact=True,
                                 build_tool_flags=("-O2", "-O3", "-mcpu=cortex-a76"))
    assert "missing_asimddp" in reasons
    assert "missing_microarchitecture_capability" in reasons
