from deployment.execution_plan.capability_view import CapabilityValidationView
from deployment.execution_plan.path_builder import (
    build_baseline_vllm_path,
    build_execution_paths,
)
from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.schema import ExecutionPathKind
from deployment.execution_plan.stage_builder import build_execution_stages
from deployment.vllm_adapter.backend_adapter import VLLMBackendAdapter


def test_vllm_adapter_materializes_baseline_config_and_commands():
    adapter = VLLMBackendAdapter()
    path = build_baseline_vllm_path(model_id="Qwen/Qwen2.5-0.5B-Instruct")

    assert adapter.validate(path, CapabilityValidationView()) == []
    materialized = adapter.materialize(path)

    assert materialized.backend == "vllm"
    assert materialized.config["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert materialized.command[:3] == (
        ".venv/bin/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    )
    assert "scripts/benchmark_openai_compatible_server.py" in materialized.benchmark_command


def test_vllm_adapter_accepts_compiler_guided_path_with_execution_unit_vocab():
    # COMPILER_GUIDED_VLLM path carries the compiler's execution-unit as selected_backend
    # (e.g. "cuda_triton"), not the runtime product name "vllm". The adapter must
    # accept it based on path_kind alone.
    adapter = VLLMBackendAdapter()
    plan = parse_execution_plan(_compiler_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    compiler_path = next(p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)

    assert compiler_path.selected_backend == "cuda_triton"
    assert adapter.validate(compiler_path, CapabilityValidationView()) == []


def test_vllm_adapter_rejects_parallelism_above_one():
    adapter = VLLMBackendAdapter()
    path = build_baseline_vllm_path(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    bad_path = path.__class__(
        **{**path.__dict__, "runtime_config": {**path.runtime_config, "tensor_parallel_size": 2}}
    )

    assert "tensor_parallel_size_must_be_1" in adapter.validate(bad_path, CapabilityValidationView())


def _compiler_plan() -> dict:
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
                    "fallback_backends": [],
                },
                "per_op_decisions": [],
            }
        ],
    }
