"""vLLM backend adapter for ExecutionPath materialization."""

from __future__ import annotations

from deployment.backend_adapter import BackendMaterialization
from deployment.execution_plan_v2.capability_view import CapabilityValidationView
from deployment.execution_plan_v2.schema import (
    COMPILER_GUIDED_VLLM_TRUTH_BOUNDARY,
    ExecutionPath,
    ExecutionPathKind,
)
from deployment.vllm_adapter.config_materializer import materialize_vllm_cli_args_from_path


class VLLMBackendAdapter:
    backend_id = "vllm"

    def supports(self, path: ExecutionPath, capabilities: CapabilityValidationView) -> bool:
        return not self.validate(path, capabilities)

    def validate(self, path: ExecutionPath, capabilities: CapabilityValidationView) -> list[str]:
        errors: list[str] = []
        if path.path_kind not in {
            ExecutionPathKind.BASELINE_VLLM,
            ExecutionPathKind.COMPILER_GUIDED_VLLM,
        }:
            errors.append("path_kind_not_vllm")
        if path.selected_backend != "vllm":
            errors.append("selected_backend_not_vllm")
        try:
            capabilities.validate_refs(path.required_capability_refs)
        except ValueError as exc:
            errors.append(str(exc))
        if int(path.runtime_config.get("tensor_parallel_size", 1) or 1) != 1:
            errors.append("tensor_parallel_size_must_be_1")
        if int(path.runtime_config.get("pipeline_parallel_size", 1) or 1) != 1:
            errors.append("pipeline_parallel_size_must_be_1")
        return errors

    def materialize(self, path: ExecutionPath) -> BackendMaterialization:
        config = materialize_vllm_cli_args_from_path(path)
        command = _server_command(config)
        benchmark_command = (
            ".venv/bin/python",
            "scripts/benchmark_openai_compatible_server.py",
            "--base-url",
            str(path.benchmark_config.get("base_url", "http://127.0.0.1:8000")),
            "--model",
            str(config["served_model_name"] or config["model"]),
            "--trace",
            str(path.benchmark_config.get("trace", "traces/llm_request_trace_32.jsonl")),
            "--concurrency",
            str(path.benchmark_config.get("concurrency", 1)),
            "--warmup",
            str(path.benchmark_config.get("warmup", 4)),
            "--claimed-server",
            "vllm",
            "--output",
            path.output_artifact,
        )
        return BackendMaterialization(
            backend="vllm",
            method=path.execution_method.value,
            config=config,
            command=command,
            benchmark_command=benchmark_command,
            expected_output_artifact=path.output_artifact,
            truth_boundary=path.truth_boundary or COMPILER_GUIDED_VLLM_TRUTH_BOUNDARY,
        )


def _server_command(config: dict) -> tuple[str, ...]:
    command = [
        ".venv/bin/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(config["model"]),
        "--tokenizer",
        str(config["tokenizer"]),
        "--dtype",
        str(config["dtype"]),
    ]
    quantization = config.get("quantization")
    if quantization == "none":
        quantization = None
    optional_flags = {
        "--quantization": quantization,
        "--max-model-len": config.get("max_model_len"),
        "--gpu-memory-utilization": config.get("gpu_memory_utilization"),
        "--block-size": config.get("block_size"),
        "--swap-space": config.get("swap_space"),
        "--max-num-seqs": config.get("max_num_seqs"),
        "--max-num-batched-tokens": config.get("max_num_batched_tokens"),
        "--tensor-parallel-size": config.get("tensor_parallel_size"),
        "--pipeline-parallel-size": config.get("pipeline_parallel_size"),
        "--served-model-name": config.get("served_model_name"),
    }
    for flag, value in optional_flags.items():
        if value is not None:
            command.extend([flag, str(value)])
    if config.get("enable_chunked_prefill"):
        command.append("--enable-chunked-prefill")
    if config.get("enable_prefix_caching"):
        command.append("--enable-prefix-caching")
    if config.get("trust_remote_code"):
        command.append("--trust-remote-code")
    return tuple(command)
