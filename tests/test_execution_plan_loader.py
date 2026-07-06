import json
from pathlib import Path

import pytest

from deployment.execution_plan.loader import (
    ExecutionPlanError,
    load_execution_plan,
    parse_execution_plan,
)
from deployment.execution_plan.schema import (
    DecisionCost,
    MemoryPlanDecision,
    ServingPlanDecision,
)


def test_loader_accepts_execution_plan(tmp_path: Path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan()), encoding="utf-8")

    plan = load_execution_plan(path)

    assert plan.schema == "execution_plan"
    assert plan.schema_version == "2.0.0"
    assert plan.plan_id == "qwen-plan"
    assert plan.provenance.capability_bundle.hardware_profile_ref == "hardware/nvidia_gtx1650_maxq.json"
    assert plan.function_plans[0].backend.selected_backend == "cuda_triton"


def test_loader_rejects_wrong_schema():
    payload = _plan()
    payload["schema"] = "vllm_execution_plan"

    with pytest.raises(ExecutionPlanError, match="schema must be execution_plan"):
        parse_execution_plan(payload)


def test_loader_rejects_wrong_schema_version():
    payload = _plan()
    payload["schema_version"] = "1.0.0"

    with pytest.raises(ExecutionPlanError, match="schema_version"):
        parse_execution_plan(payload)


def test_loader_rejects_measured_runtime_fields():
    payload = _plan()
    payload["function_plans"][0]["measured_latency_ms"] = 1.0

    with pytest.raises(ExecutionPlanError, match="measured runtime"):
        parse_execution_plan(payload)


# ---------------------------------------------------------------------------
# Step A — typed schema tests
# ---------------------------------------------------------------------------

def test_memory_plan_decision_parsed_from_global_decisions():
    plan = parse_execution_plan(_plan())

    mem = plan.global_decisions.memory
    assert isinstance(mem, MemoryPlanDecision)
    assert mem.memory_budget_fraction == 0.75
    assert mem.kv_block_size_tokens == 16


def test_serving_plan_decision_parsed_from_global_decisions():
    plan = parse_execution_plan(_plan())

    srv = plan.global_decisions.serving
    assert isinstance(srv, ServingPlanDecision)
    assert srv.token_budget_per_step == 2048
    assert srv.prefix_reuse_eligible is True
    assert srv.chunked_prefill_eligible is True
    assert srv.parallelism_kind == "none"
    assert srv.parallelism_degree == 1


def test_memory_plan_decision_kv_layout_from_kv_cache_layout_key():
    payload = _plan()
    payload["global_decisions"]["memory"]["kv_cache_layout"] = "paged"
    plan = parse_execution_plan(payload)

    assert plan.global_decisions.memory.kv_layout == "paged"


def test_memory_plan_decision_kv_byte_estimate_from_estimated_kv_peak_mb_key():
    payload = _plan()
    payload["global_decisions"]["memory"]["estimated_kv_peak_mb"] = 6.75
    plan = parse_execution_plan(payload)

    assert plan.global_decisions.memory.kv_byte_estimate_mb == 6.75


def test_serving_plan_decision_replay_eligible_parsed():
    payload = _plan()
    payload["global_decisions"]["serving"]["replay_eligible"] = True
    plan = parse_execution_plan(payload)

    assert plan.global_decisions.serving.replay_eligible is True


def test_serving_plan_decision_colocated_cost_estimate_parsed():
    payload = _plan()
    payload["global_decisions"]["serving"]["colocated_cost_estimate_ms"] = 31.2
    plan = parse_execution_plan(payload)

    assert plan.global_decisions.serving.colocated_cost_estimate_ms == 31.2


def test_global_decisions_empty_memory_and_serving_use_defaults():
    payload = _plan()
    payload["global_decisions"] = {}
    plan = parse_execution_plan(payload)

    assert isinstance(plan.global_decisions.memory, MemoryPlanDecision)
    assert isinstance(plan.global_decisions.serving, ServingPlanDecision)
    assert plan.global_decisions.memory.memory_budget_fraction == 0.0
    assert plan.global_decisions.serving.replay_eligible is False


def test_kernel_decision_cost_parsed_from_meta_evidence():
    payload = _plan_with_kernel_cost()
    plan = parse_execution_plan(payload)

    op = plan.function_plans[0].per_op_decisions[0]
    assert op.kernel is not None
    cost = op.kernel.cost
    assert isinstance(cost, DecisionCost)
    assert cost.cost_model_id == "serving_static_cost_model_v1"
    assert cost.total_cost == 0
    assert cost.compute_cost == 0
    assert cost.backend_switch_cost == 0


def test_kernel_decision_cost_truth_boundary_preserved_verbatim():
    payload = _plan_with_kernel_cost()
    plan = parse_execution_plan(payload)

    cost = plan.function_plans[0].per_op_decisions[0].kernel.cost
    assert cost.truth_boundary == "serving_static_cost_model_v1_not_measured_latency"


def test_kernel_decision_cost_none_when_meta_evidence_absent():
    plan = parse_execution_plan(_plan_with_kernel_no_cost())

    op = plan.function_plans[0].per_op_decisions[0]
    assert op.kernel is not None
    assert op.kernel.cost is None


def test_kernel_decision_cost_backend_fallback_values():
    payload = _plan_with_kernel_cost(
        candidate_type="backend_fallback",
        backend_switch_cost=20,
        launch_overhead_cost=2,
        transfer_cost=5,
        total_cost=27,
    )
    plan = parse_execution_plan(payload)

    cost = plan.function_plans[0].per_op_decisions[0].kernel.cost
    assert cost.backend_switch_cost == 20
    assert cost.launch_overhead_cost == 2
    assert cost.transfer_cost == 5
    assert cost.total_cost == 27


def test_contamination_check_still_rejects_measured_fields():
    payload = _plan()
    payload["function_plans"][0]["measured_latency_ms"] = 1.0

    with pytest.raises(ExecutionPlanError, match="measured runtime"):
        parse_execution_plan(payload)


def _plan_with_kernel_cost(
    candidate_type: str = "direct_lower",
    backend_switch_cost: int = 0,
    launch_overhead_cost: int = 0,
    transfer_cost: int = 0,
    total_cost: int = 0,
) -> dict:
    p = _plan()
    p["function_plans"][0]["per_op_decisions"] = [
        {
            "op_name": "rmsnorm_0",
            "op_type": "RMSNorm",
            "kernel": {
                "decision_type": "KernelDecision",
                "scope": "PerOp",
                "selected_kernel": "fused_rmsnorm_forward",
                "kernel_library": "local_cuda_extension",
                "lowering_path": "direct_lower",
                "kernel_exists": True,
                "meta": {
                    "evidence": {
                        "cost": {
                            "compute_cost": 0,
                            "memory_cost": 0,
                            "dequant_cost": 0,
                            "requant_cost": 0,
                            "layout_transform_cost": 0,
                            "cast_cost": 0,
                            "backend_switch_cost": backend_switch_cost,
                            "launch_overhead_cost": launch_overhead_cost,
                            "kv_cache_cost": 0,
                            "transfer_cost": transfer_cost,
                            "unsupported_penalty": 0,
                            "total_cost": total_cost,
                            "cost_model_id": "serving_static_cost_model_v1",
                            "truth_boundary": (
                                "serving_static_cost_model_v1_not_measured_latency"
                            ),
                        }
                    }
                },
            },
        }
    ]
    return p


def _plan_with_kernel_no_cost() -> dict:
    p = _plan()
    p["function_plans"][0]["per_op_decisions"] = [
        {
            "op_name": "rmsnorm_0",
            "op_type": "RMSNorm",
            "kernel": {
                "decision_type": "KernelDecision",
                "scope": "PerOp",
                "selected_kernel": "fused_rmsnorm_forward",
                "kernel_library": "local_cuda_extension",
                "lowering_path": "direct_lower",
                "kernel_exists": True,
            },
        }
    ]
    return p


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
                    "selected_backend": "cuda_triton",
                    "fallback_backends": ["custom_runtime"],
                },
                "per_op_decisions": [],
            }
        ],
    }
