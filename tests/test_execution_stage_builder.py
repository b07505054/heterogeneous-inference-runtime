from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.schema import ExecutionStageKind
from deployment.execution_plan.stage_builder import build_execution_stages


def test_stage_builder_creates_function_and_op_stages():
    plan = parse_execution_plan(_plan_with_rmsnorm())

    stages = build_execution_stages(plan)

    assert [stage.stage_id for stage in stages] == ["qwen_prefill", "rmsnorm_0"]
    assert stages[0].kind == ExecutionStageKind.PREFILL
    assert stages[1].kind == ExecutionStageKind.RMSNORM
    assert stages[1].op_name == "rmsnorm_0"
    assert "not measured runtime behavior" in stages[1].truth_boundary


def _plan_with_rmsnorm() -> dict:
    return {
        "schema": "execution_plan",
        "schema_version": "2.0.0",
        "plan_id": "qwen-plan",
        "provenance": {
            "compiler_tool": "test",
            "model_spec_ref": "profiles/models/qwen2_5_0_5b_instruct.json",
            "capability_bundle": {
                "hardware_profile_ref": "hardware/nvidia_gtx1650_maxq.json",
                "backend_profile_refs": ["backend/vllm.json"],
                "kernel_profile_refs": ["kernels/triton.json"],
            },
            "truth_boundary": "execution_planning_declared_profiles_not_measured_runtime",
        },
        "model_identity": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
        "global_decisions": {},
        "function_plans": [
            {
                "function_name": "qwen_prefill",
                "serving_phase": "prefill",
                "backend": {
                    "decision_type": "BackendDecision",
                    "scope": "Function",
                    "selected_backend": "vllm",
                },
                "per_op_decisions": [
                    {
                        "op_name": "rmsnorm_0",
                        "op_type": "RMSNorm",
                        "kernel": {
                            "decision_type": "KernelDecision",
                            "scope": "PerOp",
                            "selected_kernel": "fused_rmsnorm_forward",
                            "kernel_library": "local_cuda_extension",
                            "lowering_path": "custom_cuda",
                            "kernel_exists": True,
                        },
                    }
                ],
            }
        ],
    }
