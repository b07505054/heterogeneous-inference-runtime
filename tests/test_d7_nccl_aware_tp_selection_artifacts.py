import json
from pathlib import Path

import pytest

from deployment.vllm_adapter.tp_cost_model import (
    MODEL_IDENTITY_FEATURES,
    estimated_communication_bytes,
    load_communication_predictor,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_BY_LABEL = {
    "qwen05b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}


def _load_predictor():
    profile = json.loads(
        (REPO_ROOT / "results/runtime_paths/nccl_calibration/communication_cost_profile.json").read_text()
    )
    fit = json.loads((REPO_ROOT / "results/runtime_paths/nccl_calibration/fit_report.json").read_text())
    return load_communication_predictor(profile, fit)


def test_python_predictor_matches_d7_artifact_communication_estimates():
    predictor = _load_predictor()
    evidence = json.loads(
        (REPO_ROOT / "results/runtime_paths/distributed_d7_nccl_aware_tp_selection/candidate_evidence.json")
        .read_text()
    )
    assert len(evidence) == 21
    for report in evidence:
        model_features = MODEL_IDENTITY_FEATURES[MODEL_BY_LABEL[report["model_label"]]]
        for candidate in report["candidates"]:
            tp = candidate["tensor_parallel_size"]
            expected_bytes = estimated_communication_bytes(model_features, tp)
            assert candidate["estimated_communication_bytes"] == expected_bytes
            assert candidate["estimated_nccl_comm_time_us"] == pytest.approx(
                predictor.predict_time_us(expected_bytes), abs=1e-9
            )
            assert candidate["communication_profile_id"] == predictor.profile_id
            assert candidate["communication_predictor_kind"] == predictor.predictor_kind
            assert candidate["nccl_transport"] == "SHM/direct/direct"
            assert candidate["p2p_available"] is False


def test_d7_artifact_reports_flip_outcomes_against_measured_winner():
    flips = json.loads(
        (REPO_ROOT / "results/runtime_paths/distributed_d7_nccl_aware_tp_selection"
         "/decision_flips_caused_by_nccl_aware_cost.json").read_text()
    )
    assert "count" in flips
    assert "flips" in flips
    assert flips["count"] == len(flips["flips"])
    for row in flips["flips"]:
        assert "measured_winner" in row
        assert "d7_matches_measured_winner" in row
