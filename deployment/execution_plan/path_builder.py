"""Build runtime ExecutionPath objects from baselines and compiler decisions."""

from __future__ import annotations

from typing import Any

from deployment.execution_plan.schema import (
    COMPILER_GUIDED_VLLM_TRUTH_BOUNDARY,
    EXECUTION_PATH_TRUTH_BOUNDARY,
    PORTABLE_CPU_KERNEL_TRUTH_BOUNDARY,
    RMSNORM_TRUTH_BOUNDARY,
    ExecutionMethod,
    ExecutionPath,
    ExecutionPathKind,
    ExecutionPlan,
    ExecutionStage,
    ExecutionStageKind,
)
from deployment.execution_plan.portable_cpu_kernel_adapter import ALL_KNOWN_KERNEL_IDS, KNOWN_KERNEL_IDS
from deployment.execution_unit_router import ExecutionUnitRouter


def build_baseline_vllm_path(
    *,
    model_id: str,
    trace: str = "traces/llm_request_trace_32.jsonl",
    output_artifact: str = "results/runtime_paths/qwen_0_5b_baseline_vllm.json",
    concurrency: int = 1,
    warmup: int = 4,
    served_model_name: str | None = None,
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
            "served_model_name": served_model_name,
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


def build_execution_paths(plan: ExecutionPlan, stages: list[ExecutionStage]) -> list[ExecutionPath]:
    paths: list[ExecutionPath] = []
    function_by_name = {fn.function_name: fn for fn in plan.function_plans}
    for stage in stages:
        if stage.kind == ExecutionStageKind.RMSNORM:
            paths.append(_rmsnorm_path(plan, stage))
            continue
        if stage.kind == ExecutionStageKind.MATMUL:
            paths.append(_portable_cpu_fused_matmul_bias_relu_path(plan, stage))
            continue
        if stage.kind == ExecutionStageKind.ATTENTION and isinstance(stage.source_compiler_decision.get("attention_execution"), dict):
            paths.append(_portable_cpu_attention_path(plan, stage))
            continue
        function_plan = function_by_name.get(stage.function_name or "")
        execution_unit = function_plan.backend.selected_backend if function_plan else ""
        adapter_id = ExecutionUnitRouter.serving_adapter_id(execution_unit)
        if adapter_id == "vllm":
            paths.append(_compiler_guided_vllm_path(plan, stage, execution_unit))
        elif adapter_id in ("coreml", "pytorch_reference"):
            paths.append(_unsupported_path(plan, stage, execution_unit, reason="adapter_not_implemented"))
        else:
            paths.append(_unsupported_path(plan, stage, execution_unit, reason="unknown_execution_unit"))
    return paths


def _compiler_guided_vllm_path(plan: ExecutionPlan, stage: ExecutionStage, execution_unit: str) -> ExecutionPath:
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
        selected_backend=execution_unit,
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
        metadata={"compiler_plan_id": plan.plan_id, "compiler_execution_unit": execution_unit},
    )


def _rmsnorm_path(plan: ExecutionPlan, stage: ExecutionStage) -> ExecutionPath:
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


def _portable_cpu_fused_matmul_bias_relu_path(plan: ExecutionPlan, stage: ExecutionStage) -> ExecutionPath:
    """Phase P1B: the one real, compiler-selected, dispatch-validated CPU
    kernel path. Mirrors _rmsnorm_path's per-op bypass of the generic
    function-level vLLM/coreml/pytorch_reference routing above -- this stage
    kind is a single fused op, not a full-model serving unit.

    Honestly reports UNSUPPORTED (never a fabricated selection) when the
    compiler's kernel_selection_contract_v1 decision for this op is anything
    other than status == "selected" with selected_kernel exactly matching
    the one kernel this repo implements -- e.g. a target profile that does
    not declare this runtime kernel, or declares a different one.
    """
    kernel_selection = _dict_at(stage.source_compiler_decision, "kernel_selection")
    status = kernel_selection.get("status")
    selected_kernel = kernel_selection.get("selected_kernel")
    if status != "selected" or selected_kernel not in ALL_KNOWN_KERNEL_IDS:
        return _unsupported_path(
            plan, stage, "cpu",
            reason=f"kernel_selection_status_{status or 'absent'}",
        )
    quantization = _dict_at(stage.source_compiler_decision, "quantization")
    memory_placement = _dict_at(stage.source_compiler_decision, "memory_placement")
    runtime_config = {}
    if quantization:
        runtime_config["quantization_contract"] = quantization
    if memory_placement:
        runtime_config["memory_placement_contract"] = memory_placement
    return ExecutionPath(
        path_id=f"{plan.plan_id}:{stage.stage_id}:portable_cpu_kernel",
        path_kind=ExecutionPathKind.PORTABLE_CPU_KERNEL,
        stage_id=stage.stage_id,
        function_name=stage.function_name,
        serving_phase=stage.serving_phase,
        selected_backend="cpu",
        execution_method=ExecutionMethod.FUSED_MATMUL_BIAS_RELU_KERNEL,
        selected_kernel=selected_kernel,
        kernel_library=None,
        fallback_backends=(),
        source_compiler_decision=stage.source_compiler_decision,
        required_capability_refs=plan.provenance.capability_bundle.refs(),
        runtime_config=runtime_config,
        benchmark_config={
            "correctness_script": "scripts/test_portable_cpu_matmul_bias_relu_correctness.py",
            "benchmark_script": "scripts/benchmark_portable_cpu_matmul_bias_relu.py",
        },
        output_artifact="results/runtime_paths/portable_cpu_fused_matmul_bias_relu.json",
        truth_boundary=PORTABLE_CPU_KERNEL_TRUTH_BOUNDARY,
        metadata={
            "compiler_plan_id": plan.plan_id,
            "kernel_selection_source": kernel_selection.get("source"),
            "quantization_contract": quantization,
            "memory_placement_contract": memory_placement,
        },
    )

def _portable_cpu_attention_path(plan: ExecutionPlan, stage: ExecutionStage) -> ExecutionPath:
    contract = _dict_at(stage.source_compiler_decision, "attention_execution")
    kv_contract = (stage.source_compiler_decision.get("paged_kv_execution")
                   or stage.source_compiler_decision.get("kv_cache_execution"))
    kv_config_key = ("paged_kv_execution"
                     if stage.source_compiler_decision.get("paged_kv_execution")
                     else "kv_cache_execution")
    return ExecutionPath(
        path_id=f"{plan.plan_id}:{stage.stage_id}:portable_cpu_attention",
        path_kind=ExecutionPathKind.PORTABLE_CPU_KERNEL,
        stage_id=stage.stage_id, function_name=stage.function_name,
        serving_phase=stage.serving_phase, selected_backend="portable_cpu",
        execution_method=ExecutionMethod.CPU_ATTENTION_KERNEL,
        selected_kernel=contract.get("kernel_id"), kernel_library=contract.get("artifact_ref"),
        fallback_backends=(), source_compiler_decision=stage.source_compiler_decision,
        required_capability_refs=plan.provenance.capability_bundle.refs(),
        runtime_config={"attention_execution": contract,
                        **({kv_config_key: kv_contract} if isinstance(kv_contract, dict) else {})},
        benchmark_config={}, output_artifact="",
        truth_boundary=str(contract.get("truth_boundary", "")),
        metadata={"runtime_no_redecision": True, "compiler_plan_id": plan.plan_id},
    )


def _unsupported_path(
    plan: ExecutionPlan,
    stage: ExecutionStage,
    execution_unit: str,
    reason: str = "unsupported_backend_for_phase1",
) -> ExecutionPath:
    return ExecutionPath(
        path_id=f"{plan.plan_id}:{stage.stage_id}:unsupported",
        path_kind=ExecutionPathKind.UNSUPPORTED,
        stage_id=stage.stage_id,
        function_name=stage.function_name,
        serving_phase=stage.serving_phase,
        selected_backend=execution_unit,
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
        metadata={"reason": reason},
    )


def _runtime_config_from_decisions(plan: ExecutionPlan) -> dict[str, Any]:
    quantization = plan.global_decisions.quantization  # dict — vLLM materializer reads via .get()
    memory = plan.global_decisions.memory              # MemoryPlanDecision — typed
    serving = plan.global_decisions.serving            # ServingPlanDecision — typed
    model_id = str(
        plan.model_identity.get("model_id") or plan.model_identity.get("model") or ""
    )
    conservative_limits = _conservative_serving_limits(plan)
    # Vocabulary note: quantization.strategy is a schema-level descriptor
    # ("weight_only_int4", "none", ...); quantization.algorithm names the
    # method used to produce it ("awq", "gptq", "none"). vLLM's --quantization
    # flag takes the algorithm name (e.g. "awq"), not the strategy string, so
    # algorithm takes priority here when present. Absent for every plan
    # without a forced quantization override, so this is additive: existing
    # no-quant plans (empty quantization dict) still resolve to "none".
    algorithm = quantization.get("algorithm")
    if algorithm and algorithm != "none":
        quant_flag = algorithm
    else:
        quant_flag = quantization.get("strategy", quantization.get("quantization", "none"))
    return {
        "dtype": quantization.get("dtype", "float16"),
        "quantization": quant_flag,
        # Set only when the compiler plan carries a quantized model artifact
        # reference (forced-quant profiles today). None for every no-quant
        # plan -- the vLLM materializer falls back to the HF model id.
        "quantized_model_artifact_ref": quantization.get("quantized_model_artifact_ref") or None,
        "memory_budget_fraction": memory.memory_budget_fraction or None,
        "kv_block_size_tokens": (
            memory.kv_block_size_tokens
            or conservative_limits.get("kv_block_size_tokens")
            or None
        ),
        "token_budget_per_step": (
            serving.token_budget_per_step
            or conservative_limits.get("token_budget_per_step")
            or None
        ),
        "request_concurrency_budget": conservative_limits.get("request_concurrency_budget"),
        "sequence_length_budget": conservative_limits.get("sequence_length_budget"),
        "enable_prefix_caching": serving.prefix_reuse_eligible,
        "enable_chunked_prefill": serving.chunked_prefill_eligible,
        "tensor_parallel_size": (
            serving.parallelism_degree if serving.parallelism_kind == "tensor_parallel" else 1
        ),
        "pipeline_parallel_size": 1,
        "served_model_name": model_id or "qwen-0.5b-compiler-plan",
        "trust_remote_code": bool(plan.model_identity.get("trust_remote_code", False)),
    }


def _conservative_serving_limits(plan: ExecutionPlan) -> dict[str, int]:
    """Backend-neutral serving limits for small-memory Qwen targets.

    Some compiler artifacts currently carry the neutral budget fields with a
    zero value when the target profile has a fixed conservative serving policy.
    Keep that policy in ExecutionPath terms; backend adapters translate it to
    product-specific flags.
    """
    hardware_ref = plan.provenance.capability_bundle.hardware_profile_ref
    model_id = str(
        plan.model_identity.get("model_id") or plan.model_identity.get("model") or ""
    ).lower()
    if "nvidia-gtx1650-maxq" not in hardware_ref:
        return {}
    if "qwen2.5-0.5b" not in model_id and "qwen/qwen2.5-0.5b" not in model_id:
        return {}
    return {
        "request_concurrency_budget": 4,
        "token_budget_per_step": 2048,
        "kv_block_size_tokens": 16,
        "sequence_length_budget": 2048,
    }


def _fallbacks(decision: dict[str, Any]) -> list[str]:
    backend = _dict_at(decision, "backend")
    value = backend.get("fallback_backends", ())
    return list(value) if isinstance(value, list) else []


def _dict_at(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}
