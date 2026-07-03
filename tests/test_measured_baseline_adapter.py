import json
from pathlib import Path

import pytest

from benchmark.measured_baseline_adapter import (
    compare_measured_baselines,
    comparison_to_markdown,
    load_measured_baseline,
)
from scripts.compare_measured_baselines import build_parser


def test_adapter_loads_valid_measured_baseline(tmp_path: Path):
    path = _write_json(tmp_path / "baseline.json", _openai_artifact(tokens_per_second=10.0))
    loaded = load_measured_baseline(path)
    assert loaded["evidence_type"] == "measured"
    assert loaded["metrics"]["ttft_ms.p95"] == 20.0
    assert loaded["metrics"]["tokens_per_second"] == 10.0


def test_adapter_rejects_simulated_artifact(tmp_path: Path):
    payload = _openai_artifact(tokens_per_second=10.0)
    payload["artifact_type"] = "scheduler_decision_report"
    payload["evidence_type"] = "simulated"
    path = _write_json(tmp_path / "simulated.json", payload)
    with pytest.raises(ValueError, match="artifact_type"):
        load_measured_baseline(path)


def test_compare_openai_compatible_server_metrics(tmp_path: Path):
    before = _write_json(tmp_path / "before.json", _openai_artifact(tokens_per_second=10.0, p95=30.0))
    after = _write_json(tmp_path / "after.json", _openai_artifact(tokens_per_second=12.0, p95=25.0))
    summary = compare_measured_baselines(before, after)
    assert summary["compared_metrics"]["ttft_ms.p95"]["improved"] is True
    assert summary["compared_metrics"]["tokens_per_second"]["improved"] is True
    assert summary["warnings"] == []


def test_compare_coreml_metrics_package_size_and_drift(tmp_path: Path):
    before = _write_json(
        tmp_path / "before_coreml.json",
        _coreml_artifact(package_size=20.0, p95=5.0, mean_abs=0.02),
    )
    after = _write_json(
        tmp_path / "after_coreml.json",
        _coreml_artifact(package_size=10.0, p95=6.0, mean_abs=0.03),
    )
    summary = compare_measured_baselines(before, after)
    assert summary["compared_metrics"]["model.package_size_mb"]["improved"] is True
    assert summary["compared_metrics"]["coreml.metrics.steady_state_latency_ms.p95"]["improved"] is False
    assert summary["compared_metrics"]["coreml.metrics.numerical_drift.mean_abs"]["improved"] is False


def test_compare_warns_on_mismatched_benchmark_target(tmp_path: Path):
    before_payload = _openai_artifact(tokens_per_second=10.0)
    after_payload = _openai_artifact(tokens_per_second=11.0)
    after_payload["benchmark_target"]["model"] = "different"
    before = _write_json(tmp_path / "before.json", before_payload)
    after = _write_json(tmp_path / "after.json", after_payload)
    summary = compare_measured_baselines(before, after)
    assert summary["warnings"] == ["benchmark_target differs between before and after artifacts"]


def test_compare_cli_parser_and_markdown_render(tmp_path: Path):
    args = build_parser().parse_args(["--before", "a.json", "--after", "b.json", "--output", "out.md"])
    assert args.output == "out.md"
    before = _write_json(tmp_path / "before.json", _openai_artifact(tokens_per_second=10.0))
    after = _write_json(tmp_path / "after.json", _openai_artifact(tokens_per_second=11.0))
    markdown = comparison_to_markdown(compare_measured_baselines(before, after))
    assert "# Measured Baseline Comparison" in markdown
    assert "tokens_per_second" in markdown


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_artifact(metrics: dict, target: dict) -> dict:
    return {
        "artifact_type": "measured_baseline",
        "evidence_type": "measured",
        "status": "ok",
        "benchmark_target": target,
        "hardware": {},
        "software_versions": {},
        "command": [],
        "git_commit": "abc123",
        "metrics": metrics,
        "notes": [],
    }


def _openai_artifact(tokens_per_second: float, p95: float = 20.0) -> dict:
    return _base_artifact(
        {
            "ttft_ms": {"p50": 10.0, "p95": p95, "p99": p95 + 5.0},
            "tpot_ms": {"p50": 2.0, "p95": 3.0, "p99": 4.0},
            "e2e_latency_ms": {"p50": 100.0, "p95": 120.0, "p99": 150.0},
            "tokens_per_second": tokens_per_second,
        },
        {"kind": "openai_compatible_server", "model": "test"},
    )


def _coreml_artifact(package_size: float, p95: float, mean_abs: float) -> dict:
    return _base_artifact(
        {
            "model": {
                "name": "MobileNetV2",
                "precision": "fp16",
                "compression": "palettize",
                "package_size_mb": package_size,
            },
            "coreml": {
                "status": "ok",
                "metrics": {
                    "steady_state_latency_ms": {"p50": 4.0, "p95": p95, "p99": 8.0},
                    "package_size_mb": package_size,
                    "numerical_drift": {"max_abs": mean_abs * 2.0, "mean_abs": mean_abs},
                },
            },
        },
        {"kind": "native_coreml_cv", "backend": "coreml", "model": "MobileNetV2"},
    )
