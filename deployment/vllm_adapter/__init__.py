"""vLLM execution-plan adapter skeleton.

This package validates compiler planning artifacts and materializes vLLM config
arguments. It does not start vLLM or run inference.
"""

from deployment.vllm_adapter.backend_adapter import VLLMBackendAdapter
from deployment.vllm_adapter.config_materializer import (
    materialize_vllm_cli_args,
    materialize_vllm_cli_args_from_path,
)
from deployment.vllm_adapter.plan_schema import (
    EXPECTED_ARTIFACT_TYPE,
    EXPECTED_SCHEMA_VERSION,
    TRUTH_BOUNDARY,
    VLLMExecutionPlanError,
    load_vllm_execution_plan,
    validate_vllm_execution_plan,
)

__all__ = [
    "EXPECTED_ARTIFACT_TYPE",
    "EXPECTED_SCHEMA_VERSION",
    "TRUTH_BOUNDARY",
    "VLLMExecutionPlanError",
    "load_vllm_execution_plan",
    "materialize_vllm_cli_args",
    "materialize_vllm_cli_args_from_path",
    "validate_vllm_execution_plan",
    "VLLMBackendAdapter",
]
