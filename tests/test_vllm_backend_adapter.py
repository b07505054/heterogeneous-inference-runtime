from deployment.execution_plan_v2.capability_view import CapabilityValidationView
from deployment.execution_plan_v2.path_builder import build_baseline_vllm_path
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


def test_vllm_adapter_rejects_parallelism_above_one():
    adapter = VLLMBackendAdapter()
    path = build_baseline_vllm_path(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    bad_path = path.__class__(
        **{**path.__dict__, "runtime_config": {**path.runtime_config, "tensor_parallel_size": 2}}
    )

    assert "tensor_parallel_size_must_be_1" in adapter.validate(bad_path, CapabilityValidationView())
