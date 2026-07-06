from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.path_builder import (
    build_baseline_vllm_path,
    build_execution_paths,
)
from deployment.execution_plan.schema import ExecutionPathKind
from deployment.execution_plan.stage_builder import build_execution_stages


def test_baseline_request_builds_baseline_vllm_path():
    path = build_baseline_vllm_path(model_id="Qwen/Qwen2.5-0.5B-Instruct")

    assert path.path_kind == ExecutionPathKind.BASELINE_VLLM
    assert path.selected_backend == "vllm"
    assert path.runtime_config["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert path.source_compiler_decision is None


def test_compiler_plan_builds_vllm_and_rmsnorm_paths():
    # Main fixture uses cuda_triton — execution-unit vocabulary from the compiler.
    plan = parse_execution_plan(_plan())
    stages = build_execution_stages(plan)

    paths = build_execution_paths(plan, stages)

    kinds = {path.path_kind for path in paths}
    assert ExecutionPathKind.COMPILER_GUIDED_VLLM in kinds
    assert ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK in kinds
    vllm_path = next(path for path in paths if path.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)
    assert vllm_path.runtime_config["max_num_batched_tokens"] == 2048
    assert vllm_path.runtime_config["enable_prefix_caching"] is True
    rmsnorm = next(path for path in paths if path.path_kind == ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK)
    assert rmsnorm.selected_kernel == "fused_rmsnorm_forward"


def test_cuda_triton_backend_builds_compiler_guided_vllm_path():
    plan = parse_execution_plan(_plan())
    stages = build_execution_stages(plan)

    paths = build_execution_paths(plan, stages)

    vllm_paths = [p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM]
    assert len(vllm_paths) >= 1
    assert vllm_paths[0].selected_backend == "cuda_triton"
    assert vllm_paths[0].metadata["compiler_execution_unit"] == "cuda_triton"


def test_cuda_cublas_backend_builds_compiler_guided_vllm_path():
    payload = _plan()
    payload["function_plans"][0]["backend"]["selected_backend"] = "cuda_cublas"
    plan = parse_execution_plan(payload)

    paths = build_execution_paths(plan, build_execution_stages(plan))

    vllm_paths = [p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM]
    assert len(vllm_paths) >= 1
    assert vllm_paths[0].selected_backend == "cuda_cublas"


def test_cpu_backend_builds_unsupported_path_adapter_not_implemented():
    payload = _plan()
    payload["function_plans"][0]["backend"]["selected_backend"] = "cpu"
    plan = parse_execution_plan(payload)

    paths = build_execution_paths(plan, build_execution_stages(plan))

    fn_paths = [p for p in paths if p.path_kind != ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK]
    assert len(fn_paths) == 1
    assert fn_paths[0].path_kind == ExecutionPathKind.UNSUPPORTED
    assert fn_paths[0].selected_backend == "cpu"
    assert fn_paths[0].metadata["reason"] == "adapter_not_implemented"


def test_coreml_ane_backend_builds_unsupported_path_adapter_not_implemented():
    payload = _plan()
    payload["function_plans"][0]["backend"]["selected_backend"] = "coreml_ane"
    plan = parse_execution_plan(payload)

    paths = build_execution_paths(plan, build_execution_stages(plan))

    fn_paths = [p for p in paths if p.path_kind != ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK]
    assert len(fn_paths) == 1
    assert fn_paths[0].path_kind == ExecutionPathKind.UNSUPPORTED
    assert fn_paths[0].metadata["reason"] == "adapter_not_implemented"


def test_unknown_execution_unit_builds_unsupported_path():
    payload = _plan()
    payload["function_plans"][0]["backend"]["selected_backend"] = "unknown_unit"
    plan = parse_execution_plan(payload)

    paths = build_execution_paths(plan, build_execution_stages(plan))

    fn_paths = [p for p in paths if p.path_kind != ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK]
    assert len(fn_paths) == 1
    assert fn_paths[0].path_kind == ExecutionPathKind.UNSUPPORTED
    assert fn_paths[0].selected_backend == "unknown_unit"
    assert fn_paths[0].metadata["reason"] == "unknown_execution_unit"


def test_vllm_product_name_as_execution_unit_is_unknown():
    # "vllm" is a runtime product name, not a compiler execution unit.
    # It must never silently route to the vLLM adapter.
    payload = _plan()
    payload["function_plans"][0]["backend"]["selected_backend"] = "vllm"
    plan = parse_execution_plan(payload)

    paths = build_execution_paths(plan, build_execution_stages(plan))

    fn_paths = [p for p in paths if p.path_kind != ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK]
    assert len(fn_paths) == 1
    assert fn_paths[0].path_kind == ExecutionPathKind.UNSUPPORTED
    assert fn_paths[0].metadata["reason"] == "unknown_execution_unit"


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
        "model_identity": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
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
