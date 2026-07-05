import json
import tempfile
import unittest
from pathlib import Path

from deployment.vllm_adapter.plan_schema import (
    TRUTH_BOUNDARY,
    VLLMExecutionPlanError,
    load_vllm_execution_plan,
    validate_vllm_execution_plan,
)


class TestVLLMPlanSchema(unittest.TestCase):
    def test_valid_fixture_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_plan(Path(tmp), _valid_plan())
            plan = load_vllm_execution_plan(path)

        self.assertEqual(plan["artifact_type"], "vllm_execution_plan")
        self.assertEqual(validate_vllm_execution_plan(plan), [])

    def test_required_fields_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            del plan["memory_policy"]["max_model_len"]
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "memory_policy.max_model_len"):
                load_vllm_execution_plan(path)

    def test_single_gpu_constraints_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            plan["runtime_config"]["tensor_parallel_size"] = 2
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "tensor_parallel_size == 1"):
                load_vllm_execution_plan(path)


    def test_pipeline_parallel_size_above_one_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            plan["runtime_config"]["pipeline_parallel_size"] = 2
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "pipeline_parallel_size == 1"):
                load_vllm_execution_plan(path)

    def test_speculative_enabled_without_draft_model_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            plan["speculative_policy"]["enabled"] = True
            plan["speculative_policy"]["draft_model"] = None
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "requires draft_model"):
                load_vllm_execution_plan(path)

    def test_invalid_artifact_type_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            plan["artifact_type"] = "serving_execution_plan"
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "artifact_type"):
                load_vllm_execution_plan(path)

    def test_invalid_schema_version_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            plan["schema_version"] = "0.1"
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "schema_version"):
                load_vllm_execution_plan(path)

    def test_speculative_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            self.assertFalse(plan["speculative_policy"]["enabled"])
            plan["speculative_policy"]["enabled"] = True
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "speculative_policy.enabled == false"):
                load_vllm_execution_plan(path)

    def test_rejects_measured_speedup_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _valid_plan()
            plan["measured_speedup"] = 1.2
            path = _write_plan(Path(tmp), plan)

            with self.assertRaisesRegex(VLLMExecutionPlanError, "measured performance or speedup"):
                load_vllm_execution_plan(path)


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
