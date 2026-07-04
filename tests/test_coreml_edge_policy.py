import json
from pathlib import Path

from deployment.coreml_edge_policy import generate_coreml_edge_policy
from scripts.generate_coreml_edge_policy import build_parser


def test_selects_lowest_p95_when_prefer_latency(tmp_path: Path):
    slow = _write_json(tmp_path / "slow.json", _coreml_artifact(p95=4.0, package=3.0, rss=10.0))
    fast = _write_json(tmp_path / "fast.json", _coreml_artifact(p95=2.0, package=6.0, rss=12.0))

    policy = _policy([slow, fast], prefer="latency")

    assert policy["status"] == "selected"
    assert policy["selected"]["source_artifact"] == str(fast)
    assert policy["selected"]["compute_unit"] == "all"


def test_selects_smallest_package_when_prefer_size(tmp_path: Path):
    large = _write_json(tmp_path / "large.json", _coreml_artifact(p95=2.0, package=8.0, rss=10.0))
    small = _write_json(tmp_path / "small.json", _coreml_artifact(p95=4.0, package=3.0, rss=12.0))

    policy = _policy([large, small], prefer="size")

    assert policy["status"] == "selected"
    assert policy["selected"]["source_artifact"] == str(small)
    assert "package size" in policy["decision_reason"]


def test_rejects_candidate_exceeding_latency(tmp_path: Path):
    path = _write_json(tmp_path / "too_slow.json", _coreml_artifact(p95=6.0, package=3.0, rss=10.0))

    policy = _policy([path], prefer="latency", max_p95_ms=5.0)

    assert policy["status"] == "no_eligible_candidate"
    assert policy["selected"] is None
    assert policy["candidates"][0]["eligible"] is False
    assert "p95_exceeds_max" in policy["candidates"][0]["reasons"]


def test_rejects_simulated_artifact(tmp_path: Path):
    payload = _coreml_artifact(p95=2.0, package=3.0, rss=10.0)
    payload["artifact_type"] = "scheduler_decision_report"
    payload["evidence_type"] = "simulated"
    path = _write_json(tmp_path / "simulated.json", payload)

    policy = _policy([path], prefer="latency")

    assert policy["status"] == "no_eligible_candidate"
    assert "artifact_type_not_measured_baseline" in policy["candidates"][0]["reasons"]
    assert "evidence_type_not_measured" in policy["candidates"][0]["reasons"]


def test_handles_missing_metric_safely(tmp_path: Path):
    payload = _coreml_artifact(p95=2.0, package=3.0, rss=10.0)
    del payload["metrics"]["coreml"]["metrics"]["rss_delta_mb"]
    path = _write_json(tmp_path / "missing_rss.json", payload)

    policy = _policy([path], prefer="memory")

    assert policy["status"] == "no_eligible_candidate"
    assert policy["candidates"][0]["rss_mb"] is None
    assert "missing_rss_mb" in policy["candidates"][0]["reasons"]


def test_no_eligible_candidate_path(tmp_path: Path):
    too_large = _write_json(tmp_path / "too_large.json", _coreml_artifact(p95=2.0, package=20.0, rss=10.0))
    too_drifty = _write_json(tmp_path / "too_drifty.json", _coreml_artifact(p95=2.0, package=3.0, rss=10.0, drift=0.2))

    policy = _policy([too_large, too_drifty], prefer="latency", max_package_mb=10.0, max_drift=0.01)

    assert policy["status"] == "no_eligible_candidate"
    assert policy["selected"] is None
    assert "package_exceeds_max" in policy["decision_reason"]
    assert "drift_exceeds_max" in policy["decision_reason"]


def test_capability_profile_filters_to_measured_support(tmp_path: Path):
    supported = _write_json(tmp_path / "supported.json", _coreml_artifact(p95=4.0, package=3.0, rss=10.0))
    unsupported = _write_json(tmp_path / "unsupported.json", _coreml_artifact(p95=2.0, package=3.0, rss=10.0))
    profile = {
        "measured_support": [
            {
                "measured_artifact_path": str(supported),
                "status": "ok",
                "evidence": "measured",
            }
        ]
    }

    policy = _policy([supported, unsupported], prefer="latency", capability_profile=profile)

    assert policy["selected"]["source_artifact"] == str(supported)
    assert "not_in_capability_measured_support" in policy["candidates"][1]["reasons"]


def test_cli_parser_accepts_policy_options():
    args = build_parser().parse_args(
        [
            "--baselines",
            "a.json",
            "b.json",
            "--max-p95-ms",
            "5.0",
            "--max-package-mb",
            "10.0",
            "--max-drift",
            "0.01",
            "--prefer",
            "latency",
            "--output",
            "results/policies/coreml_edge_policy.json",
        ]
    )

    assert args.baselines == ["a.json", "b.json"]
    assert args.prefer == "latency"


def _policy(paths: list[Path], **overrides):
    options = {
        "max_p95_ms": 5.0,
        "max_package_mb": 10.0,
        "max_drift": 0.01,
        "prefer": "latency",
    }
    options.update(overrides)
    return generate_coreml_edge_policy(paths, **options)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _coreml_artifact(
    *,
    p95: float,
    package: float,
    rss: float,
    drift: float = 0.0,
    input_size: int = 224,
    compression: str = "none",
    compute_unit: str = "all",
) -> dict:
    return {
        "artifact_type": "measured_baseline",
        "evidence_type": "measured",
        "status": "ok",
        "benchmark_target": {
            "kind": "native_coreml_cv",
            "backend": "coreml",
            "model": "MobileNetV2",
            "model_compression": compression,
            "input_size": input_size,
        },
        "hardware": {},
        "software_versions": {},
        "command": [],
        "git_commit": "abc123",
        "metrics": {
            "model": {
                "name": "MobileNetV2",
                "precision": "fp16",
                "compression": compression,
                "input_size": input_size,
                "package_size_mb": package,
            },
            "coreml": {
                "status": "ok",
                "backend": "coreml_mlpackage",
                "metrics": {
                    "steady_state_latency_ms": {"p50": p95 - 0.5, "p95": p95, "p99": p95 + 0.5},
                    "package_size_mb": package,
                    "rss_delta_mb": rss,
                    "numerical_drift": {"max_abs": drift, "mean_abs": drift / 2.0},
                },
            },
        },
        "notes": [],
        "execution": {"compute_unit": compute_unit},
    }
