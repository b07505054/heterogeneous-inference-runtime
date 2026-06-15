from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from .artifact_tools import Artifact, ArtifactAccessError, ArtifactStore
from .metric_tools import (
    compare_candidates,
    extract_backend_metrics,
    filter_candidates,
    final_answer,
)
from .trace_judge import judge_trace


DEFAULT_TASK = "Find the best backend for MobileNetV2 under p95 < 5ms and explain why."


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None


@dataclass
class AgentRun:
    task: str
    answer: dict[str, Any]
    trace: list[ToolCall]
    eval: dict[str, Any]


class BenchmarkAgent:
    """Deterministic policy for CI, with the same tool surface an LLM policy would use."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.trace: list[ToolCall] = []

    def run(self, task: str = DEFAULT_TASK) -> AgentRun:
        start = perf_counter()
        retry_count = 0
        safety_violation = 0

        artifacts = self._call("list_artifacts", self.artifact_store.list_artifacts)
        summary_path = self._choose_artifact(artifacts, "backend summary")
        report_path = self._choose_artifact(artifacts, "benchmark report")

        try:
            summary_text = self._call(
                "read_artifact",
                self.artifact_store.read_artifact,
                path=summary_path,
            )
            report_text = self._call(
                "read_artifact",
                self.artifact_store.read_artifact,
                path=report_path,
            )
        except ArtifactAccessError:
            safety_violation = 1
            raise

        metrics = self._call(
            "extract_backend_metrics",
            extract_backend_metrics,
            summary_text=summary_text,
            report_text=report_text,
        )
        candidates = self._call(
            "filter_candidates",
            filter_candidates,
            metrics=metrics,
            metric="p95_latency_ms",
            operator="<",
            threshold=5.0,
        )
        recommendation = self._call(
            "compare_candidates",
            compare_candidates,
            candidates=candidates,
            primary="throughput_qps",
            tie_breakers=["p95_latency_ms"],
        )
        answer = self._call(
            "final_answer",
            final_answer,
            recommendation=recommendation,
            evidence=[recommendation],
            caveats=[
                "Candidates without p95 latency were excluded because they cannot prove p95 < 5 ms.",
                "Correctness is taken from the benchmark report when available.",
            ],
        )

        elapsed_ms = (perf_counter() - start) * 1000.0
        eval_result = judge_trace(
            task=task,
            answer=answer,
            trace=self.trace,
            elapsed_ms=elapsed_ms,
            retry_count=retry_count,
            safety_violation=safety_violation,
        )
        return AgentRun(task=task, answer=answer, trace=self.trace, eval=eval_result)

    def _choose_artifact(self, artifacts: list[dict], description_hint: str) -> str:
        for artifact in artifacts:
            description = artifact.get("description", "").lower()
            if description_hint in description:
                return artifact["path"]
        raise ValueError(f"no artifact matches: {description_hint}")

    def _call(self, tool: str, fn, **kwargs):
        call = ToolCall(tool=tool, args=kwargs)
        self.trace.append(call)
        try:
            return fn(**kwargs)
        except Exception as exc:
            call.ok = False
            call.error = str(exc)
            raise


def build_default_agent(root: Path | None = None) -> BenchmarkAgent:
    repo_root = root or Path(__file__).resolve().parents[1]
    store = ArtifactStore(
        root=repo_root,
        allowed_artifacts=[
            Artifact(
                path="results/backend_validation_summary.json",
                kind="json",
                description="MobileNetV2 backend summary with p95 latency and throughput",
            ),
            Artifact(
                path="results/report.md",
                kind="markdown",
                description="MobileNetV2 benchmark report with correctness consistency",
            ),
        ],
    )
    return BenchmarkAgent(store)

