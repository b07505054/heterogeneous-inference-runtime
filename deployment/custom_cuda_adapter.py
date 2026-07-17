"""Custom CUDA adapter for op-level microbenchmark paths."""

from __future__ import annotations

from deployment.backend_adapter import BackendMaterialization
from deployment.execution_plan.capability_view import CapabilityValidationView
from deployment.execution_plan.schema import (
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
        exact = path.benchmark_config.get("rmsnorm_exact_config")
        if exact:
            errors.extend(self._validate_exact(exact, path))
        else:
            if path.selected_backend != "custom_cuda":
                errors.append("selected_backend_not_custom_cuda")
            if path.selected_kernel not in {"fused_rmsnorm_forward", "rmsnorm"}:
                errors.append("unsupported_custom_cuda_kernel")
        try:
            capabilities.validate_refs(path.required_capability_refs)
        except ValueError as exc:
            errors.append(str(exc))
        return errors

    @staticmethod
    def _validate_exact(exact, path):
        errors = []
        for key in ("candidate_id", "operator", "semantics", "backend", "dtype", "tokens", "hidden", "launch_config", "target", "artifact"):
            if exact.get(key) is None:
                errors.append(f"missing_exact_rmsnorm_{key}")
        if exact.get("operator") != "rmsnorm" or exact.get("semantics") != "weighted_rmsnorm":
            errors.append("rmsnorm_semantic_contract_mismatch")
        if exact.get("dtype") != "fp32":
            errors.append("unsupported_rmsnorm_dtype")
        if exact.get("candidate_id") != path.selected_kernel:
            errors.append("selected_candidate_id_mismatch")
        launch = exact.get("launch_config") or {}
        if exact.get("backend") == "cuda":
            if launch.get("block_size") not in {64, 128, 256, 512}:
                errors.append("unsupported_cuda_rmsnorm_block_size")
            if not str(exact.get("candidate_id", "")).startswith("cuda_rmsnorm_fp32_bs"):
                errors.append("candidate_backend_mismatch")
        elif exact.get("backend") == "triton":
            if launch.get("block_size", 0) < exact.get("hidden", 0) or launch.get("num_warps") not in {4, 8} or launch.get("num_stages") != "default":
                errors.append("unsupported_triton_rmsnorm_launch_config")
            if not str(exact.get("candidate_id", "")).startswith("triton_rmsnorm_fp32_block"):
                errors.append("candidate_backend_mismatch")
        else:
            errors.append("unsupported_exact_rmsnorm_backend")
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
        config = {
            "op": "RMSNorm",
            "selected_kernel": path.selected_kernel,
            "kernel_library": path.kernel_library,
            "correctness_command": correctness_command,
        }
        # Optional kernel launch policy (block-size policy lab, Phase 1).
        # Absent -> commands and config are byte-identical to the previous
        # behavior; the kernel then uses its default block size (256, the
        # original fixed launch configuration).
        exact = path.benchmark_config.get("rmsnorm_exact_config")
        if exact:
            launch = exact["launch_config"]
            benchmark_command = (
                ".venv-rmsnorm/bin/python" if exact["backend"] == "cuda" else ".venv/bin/python",
                str(benchmark_script), "--output", path.output_artifact,
                "--tokens", str(exact["tokens"]), "--hidden", str(exact["hidden"]),
                "--eps", str(exact.get("epsilon", 1e-6)),
            )
            if exact["backend"] == "cuda":
                benchmark_command += ("--block-sizes", str(launch["block_size"]),)
            else:
                benchmark_command += ("--block-size", str(launch["block_size"]), "--num-warps", str(launch["num_warps"]),)
            benchmark_command += ("--selected-candidate-id", exact["candidate_id"], "--proof-output", path.output_artifact + ".proof.json")
            config["exact_candidate"] = exact
            config["selected_candidate_id"] = exact["candidate_id"]
            config["redecision_count"] = 0
            return BackendMaterialization(backend=exact["backend"], method=path.execution_method.value,
                config=config, command=benchmark_command, benchmark_command=benchmark_command,
                expected_output_artifact=path.output_artifact,
                truth_boundary=path.truth_boundary or RMSNORM_TRUTH_BOUNDARY)
        block_size = path.benchmark_config.get("rmsnorm_block_size")
        if block_size is not None:
            benchmark_command = benchmark_command + (
                "--block-sizes",
                str(block_size),
            )
            config["kernel_policy"] = {"block_size": int(block_size)}
        return BackendMaterialization(
            backend="custom_cuda",
            method=path.execution_method.value,
            config=config,
            command=correctness_command,
            benchmark_command=benchmark_command,
            expected_output_artifact=path.output_artifact,
            truth_boundary=path.truth_boundary or RMSNORM_TRUTH_BOUNDARY,
        )
