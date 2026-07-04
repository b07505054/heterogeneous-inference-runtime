import json
from pathlib import Path

from deployment.server_runtime_policy import generate_server_runtime_policy
from scripts.generate_server_runtime_policy import build_parser


def test_selects_c1_when_c4_violates_tpot_constraint(tmp_path: Path):
    c1 = _write_json(tmp_path / "c1.json", _server_artifact(concurrency=1, tpot=20.0, throughput=40.0))
    c4 = _write_json(tmp_path / "c4.json", _server_artifact(concurrency=4, tpot=80.0, throughput=100.0))

    policy = _policy([c1, c4], prefer="latency", max_tpot_p95_ms=50.0)

    assert policy["status"] == "selected"
    assert policy["selected"]["concurrency"] == 1
    assert policy["selected"]["source_artifact"] == str(c1)
    assert "tpot_p95_exceeds_max" in policy["candidates"][1]["reasons"]


def test_selects_higher_throughput_if_eligible_and_prefer_throughput(tmp_path: Path):
    c1 = _write_json(tmp_path / "c1.json", _server_artifact(concurrency=1, tpot=20.0, e2e=900.0, throughput=40.0))
    c4 = _write_json(tmp_path / "c4.json", _server_artifact(concurrency=4, tpot=30.0, e2e=1000.0, throughput=100.0))

    policy = _policy([c1, c4], prefer="throughput")

    assert policy["status"] == "selected"
    assert policy["selected"]["concurrency"] == 4
    assert policy["selected"]["source_artifact"] == str(c4)
    assert "tokens/sec" in policy["decision_reason"]


def test_rejects_candidate_with_errors(tmp_path: Path):
    path = _write_json(tmp_path / "errors.json", _server_artifact(error_count=1))

    policy = _policy([path], prefer="latency")

    assert policy["status"] == "no_eligible_candidate"
    assert policy["candidates"][0]["eligible"] is False
    assert "error_count_nonzero" in policy["candidates"][0]["reasons"]


def test_allow_errors_keeps_erroring_candidate_eligible(tmp_path: Path):
    path = _write_json(tmp_path / "errors_allowed.json", _server_artifact(error_count=1))

    policy = _policy([path], prefer="latency", allow_errors=True)

    assert policy["status"] == "selected"
    assert policy["constraints"]["allow_errors"] is True
    assert policy["candidates"][0]["reasons"] == []


def test_rejects_non_openai_compatible_artifact(tmp_path: Path):
    payload = _server_artifact()
    payload["benchmark_target"]["kind"] = "native_coreml_cv"
    path = _write_json(tmp_path / "wrong_kind.json", payload)

    policy = _policy([path], prefer="latency")

    assert policy["status"] == "no_eligible_candidate"
    assert "benchmark_target_not_openai_compatible_server" in policy["candidates"][0]["reasons"]


def test_rejects_simulated_artifact(tmp_path: Path):
    payload = _server_artifact()
    payload["artifact_type"] = "scheduler_decision_report"
    payload["evidence_type"] = "simulated"
    path = _write_json(tmp_path / "simulated.json", payload)

    policy = _policy([path], prefer="latency")

    assert policy["status"] == "no_eligible_candidate"
    assert "artifact_type_not_measured_baseline" in policy["candidates"][0]["reasons"]
    assert "evidence_type_not_measured" in policy["candidates"][0]["reasons"]


def test_handles_missing_metrics_safely(tmp_path: Path):
    payload = _server_artifact()
    del payload["metrics"]["tokens_per_second"]
    path = _write_json(tmp_path / "missing_throughput.json", payload)

    policy = _policy([path], prefer="throughput")

    assert policy["status"] == "no_eligible_candidate"
    assert policy["candidates"][0]["tokens_per_second"] is None
    assert "missing_tokens_per_second" in policy["candidates"][0]["reasons"]


def test_no_eligible_candidate_path(tmp_path: Path):
    slow_ttft = _write_json(tmp_path / "slow_ttft.json", _server_artifact(ttft=500.0))
    slow_e2e = _write_json(tmp_path / "slow_e2e.json", _server_artifact(e2e=5000.0))

    policy = _policy([slow_ttft, slow_e2e], prefer="latency")

    assert policy["status"] == "no_eligible_candidate"
    assert policy["selected"] is None
    assert "ttft_p95_exceeds_max" in policy["decision_reason"]
    assert "e2e_p95_exceeds_max" in policy["decision_reason"]


def test_selected_includes_optional_model_limits(tmp_path: Path):
    path = _write_json(
        tmp_path / "limits.json",
        _server_artifact(max_model_len=4096, max_tokens=256),
    )

    policy = _policy([path], prefer="latency")

    assert policy["selected"]["max_model_len"] == 4096
    assert policy["selected"]["max_tokens"] == 256
    assert policy["candidates"][0]["max_model_len"] == 4096
    assert policy["candidates"][0]["max_tokens"] == 256


def test_cli_parser_accepts_policy_options():
    args = build_parser().parse_args(
        [
            "--baselines",
            "c1.json",
            "c4.json",
            "--max-ttft-p95-ms",
            "200",
            "--max-tpot-p95-ms",
            "50",
            "--max-e2e-p95-ms",
            "2000",
            "--prefer",
            "latency",
            "--allow-errors",
            "--output",
            "results/policies/server_runtime_policy.json",
        ]
    )

    assert args.baselines == ["c1.json", "c4.json"]
    assert args.prefer == "latency"
    assert args.allow_errors is True


def _policy(paths: list[Path], **overrides):
    options = {
        "max_ttft_p95_ms": 200.0,
        "max_tpot_p95_ms": 50.0,
        "max_e2e_p95_ms": 2000.0,
        "prefer": "latency",
    }
    options.update(overrides)
    return generate_server_runtime_policy(paths, **options)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _server_artifact(
    *,
    concurrency: int = 1,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ttft: float = 150.0,
    tpot: float = 12.0,
    e2e: float = 900.0,
    throughput: float = 71.47,
    success_count: int = 28,
    error_count: int = 0,
    max_model_len: int | None = None,
    max_tokens: int | None = None,
) -> dict:
    target = {
        "kind": "openai_compatible_server",
        "base_url": "http://127.0.0.1:8000",
        "endpoint": "/v1/chat/completions",
        "model": model,
        "concurrency": concurrency,
        "warmup": 0,
        "stream": True,
    }
    if max_model_len is not None:
        target["max_model_len"] = max_model_len
    if max_tokens is not None:
        target["max_tokens"] = max_tokens
    return {
        "artifact_type": "measured_baseline",
        "evidence_type": "measured",
        "status": "ok",
        "benchmark_target": target,
        "hardware": {},
        "software_versions": {},
        "command": [],
        "git_commit": "abc123",
        "metrics": {
            "ttft_ms": {"p50": ttft - 10.0, "p95": ttft, "p99": ttft + 10.0},
            "tpot_ms": {"p50": tpot - 1.0, "p95": tpot, "p99": tpot + 1.0},
            "e2e_latency_ms": {"p50": e2e - 100.0, "p95": e2e, "p99": e2e + 100.0},
            "tokens_per_second": throughput,
            "success_count": success_count,
            "error_count": error_count,
            "total_output_tokens": 1024,
        },
        "notes": [
            "Generic OpenAI-compatible benchmark client only; it does not install, start, stop, or manage the server.",
        ],
    }
