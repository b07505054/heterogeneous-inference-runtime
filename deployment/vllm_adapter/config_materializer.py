"""Materialize vLLM CLI arguments from compiler decisions or legacy plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deployment.execution_plan.schema import ExecutionPath
from deployment.vllm_adapter.plan_schema import load_vllm_execution_plan, validate_vllm_execution_plan


def materialize_vllm_cli_args(plan_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Convert a validated vLLM execution plan into vLLM server CLI arguments."""
    if isinstance(plan_or_path, dict):
        plan = plan_or_path
        validate_vllm_execution_plan(plan)
    else:
        plan = load_vllm_execution_plan(plan_or_path)

    return {
        "model": plan["model"]["model_id"],
        "tokenizer": plan["model"]["tokenizer"],
        "dtype": plan["model"]["dtype"],
        "quantization": plan["quantization_policy"]["quantization"],
        "max_model_len": plan["memory_policy"]["max_model_len"],
        "gpu_memory_utilization": plan["memory_policy"]["gpu_memory_utilization"],
        "block_size": plan["memory_policy"]["block_size"],
        "swap_space": plan["memory_policy"]["swap_space"],
        "max_num_seqs": plan["batch_policy"]["max_num_seqs"],
        "max_num_batched_tokens": plan["batch_policy"]["max_num_batched_tokens"],
        "enable_chunked_prefill": plan["batch_policy"]["enable_chunked_prefill"],
        "enable_prefix_caching": plan["prefix_policy"]["enable_prefix_caching"],
        "tensor_parallel_size": plan["runtime_config"]["tensor_parallel_size"],
        "pipeline_parallel_size": plan["runtime_config"]["pipeline_parallel_size"],
        "served_model_name": plan["runtime_config"]["served_model_name"],
        "trust_remote_code": plan["model"]["trust_remote_code"],
    }


def materialize_vllm_cli_args_from_path(path: ExecutionPath) -> dict[str, Any]:
    """Convert an ExecutionPath into vLLM server CLI arguments.

    The input is runtime routing intent derived from compiler decisions or a
    baseline request. vLLM flags are produced only at this materializer boundary.
    """
    config = path.runtime_config
    model = config.get("model") or config.get("model_id")
    if not model:
        raise ValueError("ExecutionPath runtime_config must include model")
    tokenizer = config.get("tokenizer") or model
    return {
        "model": model,
        "tokenizer": tokenizer,
        "dtype": config.get("dtype", "float16"),
        "quantization": config.get("quantization", "none"),
        "max_model_len": config.get("max_model_len"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization"),
        "block_size": config.get("block_size"),
        "swap_space": config.get("swap_space"),
        "max_num_seqs": config.get("max_num_seqs"),
        "max_num_batched_tokens": config.get("max_num_batched_tokens"),
        "enable_chunked_prefill": config.get("enable_chunked_prefill"),
        "enable_prefix_caching": config.get("enable_prefix_caching"),
        "tensor_parallel_size": config.get("tensor_parallel_size", 1),
        "pipeline_parallel_size": config.get("pipeline_parallel_size", 1),
        "served_model_name": config.get("served_model_name"),
        "trust_remote_code": bool(config.get("trust_remote_code", False)),
    }
