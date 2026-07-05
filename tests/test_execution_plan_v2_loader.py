import json
from pathlib import Path

import pytest

from deployment.execution_plan_v2.loader import (
    ExecutionPlanV2Error,
    load_execution_plan_v2,
    parse_execution_plan_v2,
)


def test_loader_accepts_execution_plan_v2(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan()), encoding="utf-8")

    plan = load_execution_plan_v2(path)

    assert plan.schema == "execution_plan"
    assert plan.schema_version == "2.0.0"
    assert plan.plan_id == "qwen-plan"
    assert plan.provenance.capability_bundle.hardware_profile_ref == "hardware/nvidia_gtx1650_maxq.json"
    assert plan.function_plans[0].backend.selected_backend == "vllm"


def test_loader_rejects_wrong_schema():
    payload = _plan()
    payload["schema"] = "vllm_execution_plan"

    with pytest.raises(ExecutionPlanV2Error, match="schema must be execution_plan"):
        parse_execution_plan_v2(payload)


def test_loader_rejects_wrong_schema_version():
    payload = _plan()
    payload["schema_version"] = "1.0.0"

    with pytest.raises(ExecutionPlanV2Error, match="schema_version"):
        parse_execution_plan_v2(payload)


def test_loader_rejects_measured_runtime_fields():
    payload = _plan()
    payload["function_plans"][0]["measured_latency_ms"] = 1.0

    with pytest.raises(ExecutionPlanV2Error, match="measured runtime"):
        parse_execution_plan_v2(payload)


def _plan() -> dict:
    return {
        "schema": "execution_plan",
        "schema_version": "2.0.0",
        "plan_id": "qwen-plan",
        "provenance": {
            "compiler_tool": "test-exporter",
            "model_spec_ref": "profiles/models/qwen2_5_0_5b_instruct.json",
            "capability_bundle": {
                "hardware_profile_ref": "hardware/nvidia_gtx1650_maxq.json",
                "backend_profile_refs": ["backend/vllm.json"],
                "kernel_profile_refs": ["kernels/triton.json"],
                "workload_ref": "workloads/qwen_short_to_medium_32.json",
            },
            "truth_boundary": "execution_planning_declared_profiles_not_measured_runtime",
        },
        "model_identity": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
            "trust_remote_code": False,
        },
        "global_decisions": {
            "quantization": {"strategy": "none", "dtype": "float16"},
            "memory": {"memory_budget_fraction": 0.75, "kv_block_size_tokens": 16},
            "serving": {
                "token_budget_per_step": 2048,
                "prefix_reuse_eligible": True,
                "chunked_prefill_eligible": True,
                "parallelism_kind": "none",
                "parallelism_degree": 1,
            },
        },
        "function_plans": [
            {
                "function_name": "qwen_decode",
                "serving_phase": "decode",
                "backend": {
                    "decision_type": "BackendDecision",
                    "scope": "Function",
                    "selected_backend": "vllm",
                    "fallback_backends": ["custom_runtime"],
                },
                "per_op_decisions": [],
            }
        ],
    }
