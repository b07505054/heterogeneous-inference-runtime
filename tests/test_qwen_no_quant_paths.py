"""Dry-run integration tests for the no-quant Qwen compiler plan.

These tests verify that:
- A plan with absent global_decisions.quantization loads without error.
- The no-quant path materializes as a COMPILER_GUIDED_VLLM path.
- The vLLM materializer omits --quantization from the server command.
- The benchmark runner command references the correct script.

No vLLM installation is required. No server is started.
"""
from __future__ import annotations

from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.path_builder import build_baseline_vllm_path, build_execution_paths
from deployment.execution_plan.schema import ExecutionPathKind
from deployment.execution_plan.stage_builder import build_execution_stages
from deployment.vllm_adapter.backend_adapter import VLLMBackendAdapter


# ---------------------------------------------------------------------------
# Fixture: minimal no-quant plan matching the GTX 1650 compiler output.
# global_decisions has no quantization key — reflects actual compiler output
# when profile declares no required_weight_quant_mode.
# ---------------------------------------------------------------------------

def _no_quant_plan() -> dict:
    return {
        "schema": "execution_plan",
        "schema_version": "2.0.0",
        "plan_id": "nvidia-gtx1650-maxq_serving_plan",
        "provenance": {
            "compiler_tool": "compile-for-target",
            "model_spec_ref": "",
            "capability_bundle": {
                "hardware_profile_ref": "nvidia-gtx1650-maxq",
                "backend_profile_refs": ["cuda_triton", "cuda_cublas"],
                "kernel_profile_refs": ["cublas", "triton"],
            },
            "truth_boundary": "execution_planning_declared_profiles_not_measured_runtime",
        },
        "model_identity": {
            "model_id": "qwen2.5-0.5b",
            "hidden_size": 896,
            "num_attention_heads": 14,
            "num_kv_heads": 2,
            "num_layers": 24,
            "truth_boundary": "declared_model_config_not_full_graph_import",
        },
        # No "quantization" key — this is the no-quant compiler output.
        "global_decisions": {
            "memory": {
                "memory_budget_fraction": 0.75,
                "kv_cache_layout": "paged",
                "kv_block_size_tokens": 0,
                "estimated_kv_peak_mb": 42,
                "truth_boundary": "decision_collected_from_v1_mlir_attrs_evidence_not_tracked",
            },
            "serving": {
                "topology": "pd_split",
                "replay_eligible": False,
                "prefix_reuse_eligible": False,
                "chunked_prefill_eligible": False,
                "token_budget_per_step": 0,
                "parallelism_degree": 1,
                "truth_boundary": "estimated_cost_not_measured_latency",
            },
        },
        "function_plans": [
            {
                "function_name": "qwen_prefill",
                "serving_phase": "prefill",
                "backend": {
                    "decision_type": "BackendDecision",
                    "scope": "Function",
                    "selected_backend": "cuda",
                    "source_pass": "target_preferred",
                    "fallback_backends": [],
                    "truth_boundary": "decision_collected_from_v1_mlir_attrs_evidence_not_tracked",
                },
                "per_op_decisions": [
                    {
                        "op_name": "op_1",
                        "op_type": "llm.rmsnorm",
                        "quantization": {
                            "strategy": "fp16_fallback",
                            "weight_dtype": "fp16",
                            "activation_dtype": "fp16",
                            "accumulation_dtype": "fp16",
                        },
                    }
                ],
            },
            {
                "function_name": "qwen_decode",
                "serving_phase": "decode",
                "backend": {
                    "decision_type": "BackendDecision",
                    "scope": "Function",
                    "selected_backend": "cuda",
                    "source_pass": "target_preferred",
                    "fallback_backends": ["cpu"],
                    "truth_boundary": "decision_collected_from_v1_mlir_attrs_evidence_not_tracked",
                },
                "per_op_decisions": [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests: plan loading tolerates absent quantization
# ---------------------------------------------------------------------------

def test_no_quant_plan_loads_without_error():
    plan = parse_execution_plan(_no_quant_plan())
    assert plan.schema == "execution_plan"
    assert plan.schema_version == "2.0.0"


def test_no_quant_plan_global_decisions_quantization_is_empty_dict():
    plan = parse_execution_plan(_no_quant_plan())
    assert plan.global_decisions.quantization == {}


def test_no_quant_plan_memory_decision_parsed():
    plan = parse_execution_plan(_no_quant_plan())
    assert plan.global_decisions.memory.memory_budget_fraction == 0.75
    assert plan.global_decisions.memory.kv_layout == "paged"


# ---------------------------------------------------------------------------
# Tests: execution stages from no-quant plan
# ---------------------------------------------------------------------------

def test_no_quant_plan_builds_prefill_and_decode_stages():
    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    phase_names = {s.function_name for s in stages}
    assert "qwen_prefill" in phase_names
    assert "qwen_decode" in phase_names


# ---------------------------------------------------------------------------
# Tests: path routing — "cuda" backend routes to vLLM
# ---------------------------------------------------------------------------

def test_no_quant_plan_cuda_backend_routes_to_compiler_guided_vllm():
    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    vllm_paths = [p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM]
    assert len(vllm_paths) >= 1, "expected at least one COMPILER_GUIDED_VLLM path"


def test_no_quant_plan_no_unsupported_paths_for_function_stages():
    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    function_paths = [
        p for p in paths
        if p.path_kind == ExecutionPathKind.UNSUPPORTED
        and p.stage_id in ("qwen_prefill", "qwen_decode")
    ]
    assert function_paths == [], f"unexpected unsupported paths: {[p.stage_id for p in function_paths]}"


# ---------------------------------------------------------------------------
# Tests: vLLM materialization — no-quant omits --quantization flag
# ---------------------------------------------------------------------------

def test_no_quant_materialization_omits_quantization_flag():
    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    vllm_path = next(p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)

    adapter = VLLMBackendAdapter()
    mat = adapter.materialize(vllm_path)

    assert "--quantization" not in mat.command, (
        f"--quantization should not appear for no-quant plan; command: {mat.command}"
    )


def test_no_quant_materialization_uses_float16():
    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    vllm_path = next(p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)

    mat = VLLMBackendAdapter().materialize(vllm_path)

    assert "--dtype" in mat.command
    dtype_idx = mat.command.index("--dtype")
    assert mat.command[dtype_idx + 1] == "float16"


def test_no_quant_materialization_includes_gpu_memory_utilization():
    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    vllm_path = next(p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)

    mat = VLLMBackendAdapter().materialize(vllm_path)

    assert "--gpu-memory-utilization" in mat.command
    idx = mat.command.index("--gpu-memory-utilization")
    assert float(mat.command[idx + 1]) == 0.75


def test_no_quant_benchmark_command_references_benchmark_script():
    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    vllm_path = next(p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)

    mat = VLLMBackendAdapter().materialize(vllm_path)

    assert mat.benchmark_command is not None
    assert any("benchmark_openai_compatible_server.py" in part for part in mat.benchmark_command)


# ---------------------------------------------------------------------------
# Tests: baseline path for no-quant comparison
# ---------------------------------------------------------------------------

def test_baseline_vllm_path_for_no_quant_comparison():
    path = build_baseline_vllm_path(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    assert path.path_kind == ExecutionPathKind.BASELINE_VLLM
    assert path.runtime_config["dtype"] == "float16"
    assert path.runtime_config.get("quantization") in ("none", None, "")
