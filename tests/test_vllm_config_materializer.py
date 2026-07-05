import json
import tempfile
import unittest
from pathlib import Path

from deployment.vllm_adapter.config_materializer import materialize_vllm_cli_args
from deployment.vllm_adapter.plan_schema import TRUTH_BOUNDARY


class TestVLLMConfigMaterializer(unittest.TestCase):
    def test_materialized_config_contains_expected_vllm_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_plan(Path(tmp), _valid_plan())
            config = materialize_vllm_cli_args(path)

        self.assertEqual(
            config,
            {
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
                "dtype": "float16",
                "quantization": "none",
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.75,
                "block_size": 16,
                "swap_space": 2,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 2048,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "served_model_name": "qwen-0.5b-compiler-plan",
                "trust_remote_code": False,
            },
        )

    def test_materializer_accepts_loaded_plan_dict(self) -> None:
        config = materialize_vllm_cli_args(_valid_plan())

        self.assertEqual(config["model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(config["tensor_parallel_size"], 1)
        self.assertEqual(config["pipeline_parallel_size"], 1)


def _write_plan(tmp_path: Path, plan: dict) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _valid_plan() -> dict:
    return {
        "artifact_type": "vllm_execution_plan",
        "schema_version": "1.0.0",
        "truth_boundary": TRUTH_BOUNDARY,
        "source_artifacts": ["compiler_artifact.json"],
        "model": {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
            "dtype": "float16",
            "quantization": "none",
            "trust_remote_code": False,
        },
        "hardware_profile": {
            "gpu_name": "NVIDIA GeForce GTX 1650 Max-Q",
            "vram_gb": 4,
        },
        "backend_profile": {"backend": "vllm"},
        "batch_policy": {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "enable_chunked_prefill": True,
        },
        "prefix_policy": {"enable_prefix_caching": True},
        "memory_policy": {
            "gpu_memory_utilization": 0.75,
            "max_model_len": 2048,
            "block_size": 16,
            "swap_space": 2,
        },
        "quantization_policy": {"dtype": "float16", "quantization": "none"},
        "speculative_policy": {
            "enabled": False,
            "draft_model": None,
            "num_speculative_tokens": None,
        },
        "runtime_config": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "served_model_name": "qwen-0.5b-compiler-plan",
        },
    }


if __name__ == "__main__":
    unittest.main()
