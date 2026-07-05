"""Custom CUDA adapter for op-level microbenchmark paths."""

from __future__ import annotations

from deployment.backend_adapter import BackendMaterialization
from deployment.execution_plan_v2.capability_view import CapabilityValidationView
from deployment.execution_plan_v2.schema import (
    RMSNORM_TRUTH_BOUNDARY,
    ExecutionPath,
    ExecutionPathKind,
)


class CustomCudaBackendAdapter:
    backend_id = "custom_cuda"

    def supports(self, path: ExecutionPath, capabilities: CapabilityValidationView) -> bool:
        return not self.validate(path, capabilities)

    def validate(self, path: ExecutionPath, capabilities: CapabilityValidationView) -> list[str]:
        errors: list[str] = []
        if path.path_kind != ExecutionPathKind.CUSTOM_CUDA_MICROBENCHMARK:
            errors.append("path_kind_not_custom_cuda_microbenchmark")
        if path.selected_backend != "custom_cuda":
            errors.append("selected_backend_not_custom_cuda")
        if path.selected_kernel not in {"fused_rmsnorm_forward", "rmsnorm"}:
            errors.append("unsupported_custom_cuda_kernel")
        try:
            capabilities.validate_refs(path.required_capability_refs)
        except ValueError as exc:
            errors.append(str(exc))
        return errors

    def materialize(self, path: ExecutionPath) -> BackendMaterialization:
        correctness_script = path.benchmark_config.get(
            "correctness_script", "scripts/test_rmsnorm_cuda_correctness.py"
        )
        benchmark_script = path.benchmark_config.get(
            "benchmark_script", "scripts/benchmark_rmsnorm_cuda.py"
        )
        correctness_command = (
            ".venv-rmsnorm/bin/python",
            str(correctness_script),
        )
        benchmark_command = (
            ".venv-rmsnorm/bin/python",
            str(benchmark_script),
            "--output",
            path.output_artifact,
        )
        return BackendMaterialization(
            backend="custom_cuda",
            method=path.execution_method.value,
            config={
                "op": "RMSNorm",
                "selected_kernel": path.selected_kernel,
                "kernel_library": path.kernel_library,
                "correctness_command": correctness_command,
            },
            command=correctness_command,
            benchmark_command=benchmark_command,
            expected_output_artifact=path.output_artifact,
            truth_boundary=path.truth_boundary or RMSNORM_TRUTH_BOUNDARY,
        )
