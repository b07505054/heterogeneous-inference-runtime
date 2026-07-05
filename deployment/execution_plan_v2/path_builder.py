"""Build runtime ExecutionPath objects from baselines and compiler decisions."""

from __future__ import annotations

from typing import Any

from deployment.execution_plan_v2.schema import (
    COMPILER_GUIDED_VLLM_TRUTH_BOUNDARY,
    EXECUTION_PATH_TRUTH_BOUNDARY,
    RMSNORM_TRUTH_BOUNDARY,
    ExecutionMethod,
    ExecutionPath,
    ExecutionPathKind,
    ExecutionPlanV2,
    ExecutionStage,
    ExecutionStageKind,
)


def build_baseline_vllm_path(
    *,
    model_id: str,
    trace: str = "traces/llm_request_trace_32.jsonl",
    output_artifact: str = "results/runtime_paths/qwen_0_5b_baseline_vllm.json",
    concurrency: int = 1,
    warmup: int = 4,
) -> ExecutionPath:
    return ExecutionPath(
        path_id="qwen_0_5b_baseline_vllm",
        path_kind=ExecutionPathKind.BASELINE_VLLM,
        stage_id="full_model_serving",
        function_name=None,
        serving_phase=None,
        selected_backend="vllm",
        execution_method=ExecutionMethod.OPENAI_COMPATIBLE_SERVER,
        selected_kernel=None,
        kernel_library=None,
        fallback_backends=(),
        source_compiler_decision=None,
        required_capability_refs=(
            "hardware/nvidia_gtx1650_maxq.json",
            "backend/vllm.json",
        ),
        runtime_config={
            "model": model_id,
            "tokenizer": model_id,
            "dtype": "float16",
            "quantization": "none",
            "trust_remote_code": False,
        },
        benchmark_config={
            "trace": trace,
            "concurrency": concurrency,
            "warmup": warmup,
            "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/chat/completions",
        },
        output_artifact=output_artifact,
        truth_boundary=(
            "Baseline vLLM path measures the original HuggingFace Qwen model "
            "through an OpenAI-compatible vLLM server."
        ),
        metadata={},
    )


def build_execution_paths(plan: ExecutionPlanV2, stages: list[ExecutionStage]) -> list[ExecutionPath]:
    paths: list[ExecutionPath] = []
    function_by_name = {fn.function_name: fn for fn in plan.function_plans}
    for stage in stages:
        if stage.kind == ExecutionStageKind.RMSNORM:
            paths.append(_rmsnorm_path(plan, stage))
            continue
        function_plan = function_by_name.get(stage.function_name or "")
        backend = function_plan.backend.selected_backend if function_plan else ""
        if backend == "vllm":
            paths.append(_compiler_guided_vllm_path(plan, stage, backend))
        else:
            paths.append(_unsupported_path(plan, stage, backend))
    return paths


def _compiler_guided_vllm_path(plan: ExecutionPlanV2, stage: ExecutionStage, backend: str) -> ExecutionPath:
    refs = plan.provenance.capability_bundle.refs()
    global_config = _runtime_config_from_decisions(plan)
    model_id = str(plan.model_identity.get("model_id") or plan.model_identity.get("model") or "")
    global_config.setdefault("model", model_id)
    global_config.setdefault("tokenizer", plan.model_identity.get("tokenizer", model_id))
    return ExecutionPath(
        path_id=f"{plan.plan_id}:{stage.stage_id}:vllm",
        path_kind=ExecutionPathKind.COMPILER_GUIDED_VLLM,
        stage_id=stage.stage_id,
        function_name=stage.function_name,
        serving_phase=stage.serving_phase,
        selected_backend="vllm",
        execution_method=ExecutionMethod.COMPILER_MATERIALIZED_CONFIG,
        selected_kernel=None,
        kernel_library=None,
        fallback_backends=tuple(_fallbacks(stage.source_compiler_decision)),
        source_compiler_decision=stage.source_compiler_decision,
        required_capability_refs=refs,
        runtime_config=global_config,
        benchmark_config={
            "trace": "traces/llm_request_trace_32.jsonl",
            "concurrency": 1,
            "warmup": 4,
            "base_url": "http://127.0.0.1:8000",
            "endpoint": "/v1/chat/completions",
        },
        output_artifact="results/runtime_paths/qwen_0_5b_compiler_guided_vllm.json",
        truth_boundary=COMPILER_GUIDED_VLLM_TRUTH_BOUNDARY,
        metadata={"compiler_plan_id": plan.plan_id, "compiler_backend": backend},
    )


def _rmsnorm_path(plan: ExecutionPlanV2, stage: ExecutionStage) -> ExecutionPath:
    kernel = _dict_at(stage.source_compiler_decision, "kernel")
    return ExecutionPath(
        path_id=f"{plan.plan_id}:{stage.stage_id}:custom_cuda",
        path_kind=ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK,
        stage_id=stage.stage_id,
        function_name=stage.function_name,
        serving_phase=stage.serving_phase,
        selected_backend="custom_cuda",
        execution_method=ExecutionMethod.RMSNORM_MICROBENCHMARK,
        selected_kernel=kernel.get("selected_kernel") or "fused_rmsnorm_forward",
        kernel_library=kernel.get("kernel_library") or "local_cuda_extension",
        fallback_backends=(),
        source_compiler_decision=stage.source_compiler_decision,
        required_capability_refs=(
            "hardware/nvidia_gtx1650_maxq.json",
            "platform/cuda.json",
        ),
        runtime_config={},
        benchmark_config={
            "correctness_script": "scripts/test_rmsnorm_cuda_correctness.py",
            "benchmark_script": "scripts/benchmark_rmsnorm_cuda.py",
        },
        output_artifact="results/runtime_paths/rmsnorm_custom_cuda_microbenchmark.json",
        truth_boundary=RMSNORM_TRUTH_BOUNDARY,
        metadata={"compiler_plan_id": plan.plan_id},
    )


def _unsupported_path(plan: ExecutionPlanV2, stage: ExecutionStage, backend: str) -> ExecutionPath:
    return ExecutionPath(
        path_id=f"{plan.plan_id}:{stage.stage_id}:unsupported",
        path_kind=ExecutionPathKind.UNSUPPORTED,
        stage_id=stage.stage_id,
        function_name=stage.function_name,
        serving_phase=stage.serving_phase,
        selected_backend=backend,
        execution_method=ExecutionMethod.SERVING,
        selected_kernel=None,
        kernel_library=None,
        fallback_backends=tuple(_fallbacks(stage.source_compiler_decision)),
        source_compiler_decision=stage.source_compiler_decision,
        required_capability_refs=plan.provenance.capability_bundle.refs(),
        runtime_config={},
        benchmark_config={},
        output_artifact="",
        truth_boundary=EXECUTION_PATH_TRUTH_BOUNDARY,
        metadata={"reason": "unsupported_backend_for_phase1"},
    )


def _runtime_config_from_decisions(plan: ExecutionPlanV2) -> dict[str, Any]:
    quantization = plan.global_decisions.quantization
    memory = plan.global_decisions.memory
    serving = plan.global_decisions.serving
    parallelism_kind = serving.get("parallelism_kind")
    parallelism_degree = int(serving.get("parallelism_degree", 1) or 1)
    return {
        "dtype": quantization.get("dtype", "float16"),
        "quantization": quantization.get("strategy", quantization.get("quantization", "none")),
        "gpu_memory_utilization": memory.get("memory_budget_fraction"),
        "block_size": memory.get("kv_block_size_tokens"),
        "max_num_batched_tokens": serving.get("token_budget_per_step"),
        "enable_prefix_caching": serving.get("prefix_reuse_eligible"),
        "enable_chunked_prefill": serving.get("chunked_prefill_eligible"),
        "tensor_parallel_size": parallelism_degree if parallelism_kind == "tensor_parallel" else 1,
        "pipeline_parallel_size": 1,
        "served_model_name": serving.get("served_model_name", "qwen-0.5b-compiler-plan"),
        "trust_remote_code": bool(plan.model_identity.get("trust_remote_code", False)),
    }


def _fallbacks(decision: dict[str, Any]) -> list[str]:
    backend = _dict_at(decision, "backend")
    value = backend.get("fallback_backends", ())
    return list(value) if isinstance(value, list) else []


def _dict_at(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}
