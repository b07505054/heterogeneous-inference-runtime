from __future__ import annotations

from typing import Any


REQUIRED_TOOLS = [
    "list_artifacts",
    "read_artifact",
    "extract_backend_metrics",
    "filter_candidates",
    "compare_candidates",
    "final_answer",
]


def judge_trace(
    task: str,
    answer: dict[str, Any],
    trace: list[Any],
    elapsed_ms: float,
    retry_count: int,
    safety_violation: int,
) -> dict[str, Any]:
    tool_names = [_tool_name(call) for call in trace]
    wrong_file_access = _wrong_file_access(trace)
    p95_checked = _has_filter(trace, metric="p95_latency_ms", threshold=5.0)
    throughput_tiebreak = _has_compare(trace, primary="throughput_qps")
    evidence = answer.get("evidence") or []
    explanation = answer.get("explanation") or ""
    correctness_considered = any(
        (item.get("correctness") not in (None, "not_reported"))
        for item in evidence
        if isinstance(item, dict)
    )
    expected_answer = (
        answer.get("recommended_backend") == "ONNXRuntime"
        and answer.get("recommended_precision") == "Optimized FP32 CUDA"
        and answer.get("recommended_device") == "CUDAExecutionProvider"
        and answer.get("p95_latency_ms") == 3.3046
        and answer.get("throughput_qps") == 339.1333
    )

    required_tool_hits = sum(1 for tool in REQUIRED_TOOLS if tool in tool_names)
    tool_selection_accuracy = required_tool_hits / len(REQUIRED_TOOLS)
    has_read_artifact = tool_names.count("read_artifact") >= 2
    has_evidence = bool(evidence) and "p95" in explanation and "throughput" in explanation
    task_success = all(
        [
            expected_answer,
            tool_selection_accuracy == 1.0,
            has_read_artifact,
            wrong_file_access == 0,
            p95_checked,
            throughput_tiebreak,
            correctness_considered,
            has_evidence,
            retry_count == 0,
            safety_violation == 0,
        ]
    )

    return {
        "task_success": task_success,
        "task": task,
        "tool_selection_accuracy": tool_selection_accuracy,
        "wrong_file_access": wrong_file_access,
        "required_evidence_used": has_evidence,
        "constraint_followed": p95_checked and expected_answer,
        "p95_constraint_checked": p95_checked,
        "throughput_tiebreak_used": throughput_tiebreak,
        "correctness_considered": correctness_considered,
        "p95_agent_latency_ms": round(elapsed_ms, 4),
        "retry_count": retry_count,
        "safety_violation": safety_violation,
    }


def _tool_name(call: Any) -> str:
    return getattr(call, "tool", call.get("tool") if isinstance(call, dict) else "")


def _tool_args(call: Any) -> dict[str, Any]:
    return getattr(call, "args", call.get("args", {}) if isinstance(call, dict) else {})


def _wrong_file_access(trace: list[Any]) -> int:
    allowed = {
        "results/backend_validation_summary.json",
        "results/report.md",
        "backend_validation_summary.json",
        "report.md",
    }
    count = 0
    for call in trace:
        if _tool_name(call) != "read_artifact":
            continue
        path = _tool_args(call).get("path")
        if path not in allowed:
            count += 1
    return count


def _has_filter(trace: list[Any], metric: str, threshold: float) -> bool:
    for call in trace:
        if _tool_name(call) != "filter_candidates":
            continue
        args = _tool_args(call)
        if (
            args.get("metric") == metric
            and args.get("operator") == "<"
            and float(args.get("threshold")) == threshold
        ):
            return True
    return False


def _has_compare(trace: list[Any], primary: str) -> bool:
    for call in trace:
        if _tool_name(call) != "compare_candidates":
            continue
        if _tool_args(call).get("primary") == primary:
            return True
    return False
