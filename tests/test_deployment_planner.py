import json
from pathlib import Path

from deployment.planner.deployment_plan_schema import TRUTH_BOUNDARY
from deployment.planner.planner import plan_deployment


def test_planner_selects_coreml_latency_candidate_from_mixed_artifacts(tmp_path: Path):
    coreml = _write_json(tmp_path / "coreml.json", _coreml_artifact(p95=3.0))
    server = _write_json(tmp_path / "server.json", _server_artifact(e2e=900.0, throughput=120.0))

    plan = plan_deployment(
        profile_paths=[Path("capabilities/profiles/backend/coreml.json")],
        artifact_paths=[coreml, server],
        runtime=None,
        constraints={"max_p95_ms": 1000.0},
        objective="latency",
    )

    assert plan["status"] == "selected"
    assert plan["selected_runtime"] == "coreml"
    assert plan["selected_policy"] == "coreml_edge_policy"
    assert plan["selected_candidate"] == {
        "compute_unit": "all",
        "compression": "none",
        "input_size": 224,
    }


def test_planner_selects_server_throughput_candidate(tmp_path: Path):
    c1 = _write_json(tmp_path / "server_c1.json", _server_artifact(concurrency=1, throughput=40.0))
    c4 = _write_json(tmp_path / "server_c4.json", _server_artifact(concurrency=4, throughput=140.0))

    plan = plan_deployment(
        artifact_paths=[c1, c4],
        runtime="server",
        constraints={"min_tokens_per_second": 50.0},
        objective="throughput",
    )

    assert plan["status"] == "selected"
    assert plan["selected_runtime"] == "server"
    assert plan["selected_candidate"]["concurrency"] == 4
    assert str(c4) in plan["source_artifacts"]


def test_planner_no_eligible_candidate_path(tmp_path: Path):
    coreml = _write_json(tmp_path / "coreml.json", _coreml_artifact(p95=10.0))

    plan = plan_deployment(
        artifact_paths=[coreml],
        runtime="coreml",
        constraints={"max_p95_ms": 5.0},
        objective="latency",
    )

    assert plan["status"] == "no_eligible_candidate"
    assert plan["selected_candidate"] is None
    assert "latency_exceeds_max" in plan["decision_reason"][-1]


def test_planner_preserves_truth_boundary(tmp_path: Path):
    coreml = _write_json(tmp_path / "coreml.json", _coreml_artifact(p95=3.0))

    plan = plan_deployment(
        artifact_paths=[coreml],
        runtime="coreml",
        constraints={},
        objective="latency",
    )

    assert plan["artifact_type"] == "deployment_plan"
    assert plan["planner_version"] == "v1"
    assert plan["truth_boundary"] == TRUTH_BOUNDARY


def test_planner_consumes_policy_artifact_candidates(tmp_path: Path):
    policy = _write_json(
        tmp_path / "policy.json",
        {
            "artifact_type": "optimization_policy",
            "policy_name": "coreml_edge_policy",
            "candidates": [
                {
                    "artifact": "slow.json",
                    "input_size": 224,
                    "compression": "none",
                    "compute_unit": "cpu",
                    "p95_ms": 5.0,
                    "package_mb": 6.0,
                    "rss_mb": 20.0,
                    "drift": 0.0,
                    "reasons": [],
                },
                {
                    "artifact": "fast.json",
                    "input_size": 224,
                    "compression": "none",
                    "compute_unit": "all",
                    "p95_ms": 2.0,
                    "package_mb": 6.0,
                    "rss_mb": 20.0,
                    "drift": 0.0,
                    "reasons": [],
                },
            ],
        },
    )

    plan = plan_deployment(
        artifact_paths=[policy],
        runtime="coreml",
        constraints={"max_package_mb": 10.0},
        objective="latency",
    )

    assert plan["status"] == "selected"
    assert plan["selected_candidate"]["compute_unit"] == "all"
    assert plan["source_artifacts"] == ["fast.json", "slow.json"]


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _coreml_artifact(p95: float) -> dict:
    return {
        "artifact_type": "measured_baseline",
        "evidence_type": "measured",
        "status": "ok",
        "benchmark_target": {
            "kind": "native_coreml_cv",
            "backend": "coreml",
            "model_compression": "none",
            "input_size": 224,
        },
        "metrics": {
            "model": {
                "compression": "none",
                "input_size": 224,
                "package_size_mb": 6.7,
            },
            "coreml": {
                "status": "ok",
                "metrics": {
                    "steady_state_latency_ms": {"p95": p95},
                    "package_size_mb": 6.7,
                    "rss_delta_mb": 18.0,
                    "numerical_drift": {"max_abs": 0.0},
                },
            },
        },
        "execution": {"compute_unit": "all"},
    }


def _server_artifact(concurrency: int = 1, e2e: float = 900.0, throughput: float = 80.0) -> dict:
    return {
        "artifact_type": "measured_baseline",
        "evidence_type": "measured",
        "status": "ok",
        "benchmark_target": {
            "kind": "openai_compatible_server",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "concurrency": concurrency,
        },
        "metrics": {
            "ttft_ms": {"p95": 150.0},
            "tpot_ms": {"p95": 12.0},
            "e2e_latency_ms": {"p95": e2e},
            "tokens_per_second": throughput,
            "success_count": 28,
            "error_count": 0,
        },
    }
