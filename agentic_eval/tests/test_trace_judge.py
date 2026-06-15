from agentic_eval.benchmark_agent import ToolCall
from agentic_eval.trace_judge import judge_trace


VALID_ANSWER = {
    "recommended_backend": "ONNXRuntime",
    "recommended_precision": "Optimized FP32 CUDA",
    "recommended_device": "CUDAExecutionProvider",
    "p95_latency_ms": 3.3046,
    "throughput_qps": 339.1333,
    "evidence": [
        {
            "backend": "ONNXRuntime",
            "precision": "Optimized FP32 CUDA",
            "device": "CUDAExecutionProvider",
            "p95_latency_ms": 3.3046,
            "throughput_qps": 339.1333,
            "correctness": "100%",
        }
    ],
    "explanation": "p95 is below 5 ms and throughput is highest among valid candidates.",
}


def valid_trace():
    return [
        ToolCall("list_artifacts", {}),
        ToolCall("read_artifact", {"path": "results/backend_validation_summary.json"}),
        ToolCall("read_artifact", {"path": "results/report.md"}),
        ToolCall("extract_backend_metrics", {}),
        ToolCall("filter_candidates", {"metric": "p95_latency_ms", "operator": "<", "threshold": 5.0}),
        ToolCall("compare_candidates", {"primary": "throughput_qps"}),
        ToolCall("final_answer", {}),
    ]


def test_trace_judge_accepts_valid_agentic_trace():
    result = judge_trace("task", VALID_ANSWER, valid_trace(), 10.0, 0, 0)
    assert result["task_success"] is True


def test_trace_judge_fails_without_p95_filter():
    trace = [call for call in valid_trace() if call.tool != "filter_candidates"]
    result = judge_trace("task", VALID_ANSWER, trace, 10.0, 0, 0)
    assert result["task_success"] is False
    assert result["p95_constraint_checked"] is False


def test_trace_judge_fails_without_throughput_tiebreak():
    trace = valid_trace()
    trace[-2] = ToolCall("compare_candidates", {"primary": "avg_latency_ms"})
    result = judge_trace("task", VALID_ANSWER, trace, 10.0, 0, 0)
    assert result["task_success"] is False
    assert result["throughput_tiebreak_used"] is False


def test_trace_judge_fails_without_evidence_explanation():
    answer = dict(VALID_ANSWER)
    answer["evidence"] = []
    answer["explanation"] = "Optimized FP32 CUDA is best."
    result = judge_trace("task", answer, valid_trace(), 10.0, 0, 0)
    assert result["task_success"] is False
    assert result["required_evidence_used"] is False


def test_trace_judge_fails_hard_coded_answer_without_tool_trace():
    result = judge_trace("task", VALID_ANSWER, [], 10.0, 0, 0)
    assert result["task_success"] is False
    assert result["tool_selection_accuracy"] == 0.0

