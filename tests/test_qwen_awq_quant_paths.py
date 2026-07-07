"""Dry-run integration tests for the AWQ (Phase C, minimal) Qwen compiler plan.

These tests verify that:
- A plan with global_decisions.quantization = {strategy: weight_only_int4,
  algorithm: awq, quantized_model_artifact_ref: ...} loads without error.
- The AWQ path materializes as a COMPILER_GUIDED_VLLM path.
- The vLLM materializer emits --quantization awq and points --model /
  --tokenizer at the quantized artifact ref, not the original HF repo id.
- The no-quant (B) fixture and materialization are unaffected by this change
  (see tests/test_qwen_no_quant_paths.py, unmodified by this file).

No vLLM installation is required. No server is started. No AWQ artifact is
required to exist on disk -- these tests only exercise plan parsing and
command materialization.
"""
from __future__ import annotations

from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.path_builder import build_execution_paths
from deployment.execution_plan.schema import ExecutionPathKind
from deployment.execution_plan.stage_builder import build_execution_stages
from deployment.vllm_adapter.backend_adapter import VLLMBackendAdapter

QUANTIZED_MODEL_ARTIFACT_REF = "artifacts/qwen_awq"
FORCED_QUANT_TRUTH_BOUNDARY = "experimental_forced_quant_not_native_int4_support_on_gtx1650"


# ---------------------------------------------------------------------------
# Fixture: minimal AWQ-forced plan matching the real compiler output from
# nvidia_gtx1650_maxq_awq_forced.json (see
# ml-graph-compiler-runtime/artifacts/qwen_awq_plan/execution_plan.json).
# ---------------------------------------------------------------------------

def _awq_forced_plan() -> dict:
    return {
        "schema": "execution_plan",
        "schema_version": "2.0.0",
        "plan_id": "nvidia-gtx1650-maxq-awq-forced-experimental_serving_plan",
        "provenance": {
            "compiler_tool": "compile-for-target",
            "model_spec_ref": "",
            "capability_bundle": {
                "hardware_profile_ref": "nvidia-gtx1650-maxq-awq-forced-experimental",
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
        "global_decisions": {
            "quantization": {
                "decision_id": "qd_global_collected",
                "decision_type": "QuantizationDecision",
                "scope": "Global",
                "source_pass": "forced_quant_profile",
                "strategy": "weight_only_int4",
                "algorithm": "awq",
                "weight_dtype": "int4",
                "activation_dtype": "int4",
                "quantized_model_artifact_ref": QUANTIZED_MODEL_ARTIFACT_REF,
                "truth_boundary": FORCED_QUANT_TRUTH_BOUNDARY,
            },
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
                "per_op_decisions": [],
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


def _awq_vllm_path():
    plan = parse_execution_plan(_awq_forced_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    return next(p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)


# ---------------------------------------------------------------------------
# Tests: plan loading carries the forced AWQ quantization decision
# ---------------------------------------------------------------------------

def test_awq_plan_loads_without_error():
    plan = parse_execution_plan(_awq_forced_plan())
    assert plan.schema == "execution_plan"
    assert plan.schema_version == "2.0.0"


def test_awq_plan_global_decisions_quantization_populated():
    plan = parse_execution_plan(_awq_forced_plan())
    quant = plan.global_decisions.quantization
    assert quant["strategy"] == "weight_only_int4"
    assert quant["algorithm"] == "awq"
    assert quant["quantized_model_artifact_ref"] == QUANTIZED_MODEL_ARTIFACT_REF
    assert quant["truth_boundary"] == FORCED_QUANT_TRUTH_BOUNDARY


# ---------------------------------------------------------------------------
# Tests: routing — AWQ plan still routes to COMPILER_GUIDED_VLLM
# ---------------------------------------------------------------------------

def test_awq_plan_cuda_backend_routes_to_compiler_guided_vllm():
    vllm_path = _awq_vllm_path()
    assert vllm_path.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM


def test_awq_runtime_config_carries_algorithm_and_artifact_ref():
    vllm_path = _awq_vllm_path()
    assert vllm_path.runtime_config["quantization"] == "awq"
    assert vllm_path.runtime_config["quantized_model_artifact_ref"] == QUANTIZED_MODEL_ARTIFACT_REF


# ---------------------------------------------------------------------------
# Tests: vLLM materialization — AWQ emits --quantization awq and routes
# --model / --tokenizer to the quantized artifact, not the HF repo id.
# ---------------------------------------------------------------------------

def test_awq_materialization_emits_quantization_flag():
    vllm_path = _awq_vllm_path()
    mat = VLLMBackendAdapter().materialize(vllm_path)

    assert "--quantization" in mat.command
    idx = mat.command.index("--quantization")
    assert mat.command[idx + 1] == "awq"


def test_awq_materialization_points_model_at_quantized_artifact():
    vllm_path = _awq_vllm_path()
    mat = VLLMBackendAdapter().materialize(vllm_path)

    assert "--model" in mat.command
    idx = mat.command.index("--model")
    assert mat.command[idx + 1] == QUANTIZED_MODEL_ARTIFACT_REF


def test_awq_materialization_points_tokenizer_at_quantized_artifact():
    vllm_path = _awq_vllm_path()
    mat = VLLMBackendAdapter().materialize(vllm_path)

    assert "--tokenizer" in mat.command
    idx = mat.command.index("--tokenizer")
    assert mat.command[idx + 1] == QUANTIZED_MODEL_ARTIFACT_REF


def test_awq_materialization_config_dict_matches_command():
    vllm_path = _awq_vllm_path()
    mat = VLLMBackendAdapter().materialize(vllm_path)

    assert mat.config["model"] == QUANTIZED_MODEL_ARTIFACT_REF
    assert mat.config["tokenizer"] == QUANTIZED_MODEL_ARTIFACT_REF
    assert mat.config["quantization"] == "awq"


# ---------------------------------------------------------------------------
# Regression guard: the no-quant (B) fixture's quantization dict stays empty
# and unaffected by the algorithm-aware branch added to path_builder.py.
# The full no-quant assertions live in tests/test_qwen_no_quant_paths.py;
# this only re-checks the specific branch this change touched.
# ---------------------------------------------------------------------------

def test_no_quant_plan_still_resolves_to_none_quantization():
    from tests.test_qwen_no_quant_paths import _no_quant_plan

    plan = parse_execution_plan(_no_quant_plan())
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    vllm_path = next(p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM)

    assert vllm_path.runtime_config["quantization"] == "none"
    assert vllm_path.runtime_config.get("quantized_model_artifact_ref") is None
