from pathlib import Path

import pytest

from agentic_eval.artifact_tools import Artifact, ArtifactAccessError, ArtifactStore
from agentic_eval.benchmark_agent import BenchmarkAgent, DEFAULT_TASK
from agentic_eval.metric_tools import compare_candidates, extract_backend_metrics, filter_candidates


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def build_fixture_agent() -> BenchmarkAgent:
    store = ArtifactStore(
        root=FIXTURE_DIR,
        allowed_artifacts=[
            Artifact(
                path="backend_validation_summary.json",
                kind="json",
                description="MobileNetV2 backend summary with p95 latency and throughput",
            ),
            Artifact(
                path="report.md",
                kind="markdown",
                description="MobileNetV2 benchmark report with correctness consistency",
            ),
        ],
    )
    return BenchmarkAgent(store)


def test_agent_uses_tools_and_recommends_expected_backend():
    run = build_fixture_agent().run(DEFAULT_TASK)

    assert run.answer["recommended_backend"] == "ONNXRuntime"
    assert run.answer["recommended_precision"] == "Optimized FP32 CUDA"
    assert run.answer["recommended_device"] == "CUDAExecutionProvider"
    assert run.answer["p95_latency_ms"] == 3.3046
    assert run.answer["throughput_qps"] == 339.1333
    assert run.eval["task_success"] is True
    assert run.eval["tool_selection_accuracy"] == 1.0
    assert run.eval["wrong_file_access"] == 0
    assert run.eval["p95_constraint_checked"] is True
    assert run.eval["throughput_tiebreak_used"] is True
    assert run.eval["correctness_considered"] is True

    tools = [call.tool for call in run.trace]
    for expected_tool in [
        "list_artifacts",
        "read_artifact",
        "extract_backend_metrics",
        "filter_candidates",
        "compare_candidates",
        "final_answer",
    ]:
        assert expected_tool in tools
    assert tools.count("read_artifact") == 2


def test_candidate_filter_excludes_missing_or_slow_p95_results():
    summary = (FIXTURE_DIR / "backend_validation_summary.json").read_text(encoding="utf-8")
    report = (FIXTURE_DIR / "report.md").read_text(encoding="utf-8")
    metrics = extract_backend_metrics(summary, report)

    candidates = filter_candidates(metrics, "p95_latency_ms", "<", 5.0)
    labels = {(row["backend"], row["precision"], row["device"]) for row in candidates}

    assert ("ONNXRuntime", "FP32 CUDA", "CUDAExecutionProvider") in labels
    assert ("ONNXRuntime", "Optimized FP32 CUDA", "CUDAExecutionProvider") in labels
    assert ("ThreadScaling", "Optimized FP32", "8 threads") not in labels
    assert ("CppInference", "FP32", "CPU C++") not in labels
    assert ("ONNXRuntime", "INT8 CPU", "CPUExecutionProvider") not in labels
    assert ("PyTorch", "FP32", "cpu") not in labels

    best = compare_candidates(candidates, primary="throughput_qps", tie_breakers=["p95_latency_ms"])
    assert best["precision"] == "Optimized FP32 CUDA"


def test_artifact_store_rejects_wrong_file_access():
    store = ArtifactStore(
        root=FIXTURE_DIR,
        allowed_artifacts=[
            Artifact(
                path="backend_validation_summary.json",
                kind="json",
                description="MobileNetV2 backend summary with p95 latency and throughput",
            )
        ],
    )

    with pytest.raises(ArtifactAccessError):
        store.read_artifact("secrets.env")

