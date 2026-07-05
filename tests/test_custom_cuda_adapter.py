from deployment.custom_cuda_adapter import CustomCudaBackendAdapter
from deployment.execution_plan_v2.capability_view import CapabilityValidationView
from deployment.execution_plan_v2.loader import parse_execution_plan_v2
from deployment.execution_plan_v2.path_builder import build_execution_paths
from deployment.execution_plan_v2.schema import ExecutionPathKind
from deployment.execution_plan_v2.stage_builder import build_execution_stages


def test_custom_cuda_adapter_materializes_rmsnorm_commands():
    plan = parse_execution_plan_v2(_plan())
    paths = build_execution_paths(plan, build_execution_stages(plan))
    path = next(item for item in paths if item.path_kind == ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK)
    adapter = CustomCudaBackendAdapter()

    assert adapter.validate(path, CapabilityValidationView()) == []
    materialized = adapter.materialize(path)

    assert materialized.backend == "custom_cuda"
    assert materialized.command == (
        ".venv-rmsnorm/bin/python",
        "scripts/test_rmsnorm_cuda_correctness.py",
    )
    assert materialized.benchmark_command == (
        ".venv-rmsnorm/bin/python",
        "scripts/benchmark_rmsnorm_cuda.py",
        "--output",
        "results/runtime_paths/rmsnorm_custom_cuda_microbenchmark.json",
    )
    assert "kernel-level evidence only" in materialized.truth_boundary


def _plan() -> dict:
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
