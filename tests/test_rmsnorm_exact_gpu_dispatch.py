import pytest

from deployment.custom_cuda_adapter import CustomCudaBackendAdapter
from deployment.execution_plan.capability_view import CapabilityValidationView
from deployment.execution_plan.loader import ExecutionPlanError, parse_execution_plan
from deployment.execution_plan.path_builder import build_execution_paths
from deployment.execution_plan.stage_builder import build_execution_stages


def plan(backend="cuda", candidate="cuda_rmsnorm_fp32_bs256_v1", launch=None):
    launch = launch or {"block_size": 256, "num_warps": None, "num_stages": None}
    kernel = {"decision_type": "KernelDecision", "scope": "PerOp", "selected_kernel": candidate,
              "kernel_library": "custom_cuda_rmsnorm", "lowering_path": "rmsnorm_gpu_exact_config", "kernel_exists": True,
              "decision_kind": "rmsnorm_gpu_exact_config_selection", "operator": "rmsnorm", "semantics": "weighted_rmsnorm",
              "backend": backend, "candidate_id": candidate, "kernel_family": "custom_cuda_rmsnorm" if backend == "cuda" else "triton_rmsnorm",
              "kernel_entry_point": "fused_rmsnorm_forward" if backend == "cuda" else "rmsnorm_kernel", "dtype": "fp32",
              "tokens": 16, "hidden": 4096, "epsilon": 1e-6, "launch_config": launch,
              "artifact": {"source_hash": "a" * 64, "measurement_artifact_hash": "b" * 64, "compiled_artifact_hash": None},
              "target": {"gpu_name": "NVIDIA GeForce GTX 1650 Max-Q", "compute_capability": "7.5"}}
    return {"schema": "execution_plan", "schema_version": "2.0.0", "plan_id": "rmsnorm",
            "provenance": {"compiler_tool": "test", "model_spec_ref": "operator", "capability_bundle": {"hardware_profile_ref": "hardware/nvidia_gtx1650_maxq.json"}, "truth_boundary": "operator-only"},
            "model_identity": {"model_id": "operator"}, "global_decisions": {},
            "function_plans": [{"function_name": "rmsnorm", "serving_phase": "other", "backend": {"decision_type": "BackendDecision", "scope": "Function", "selected_backend": backend},
                                "per_op_decisions": [{"op_name": "rmsnorm_0", "op_type": "RMSNorm", "kernel": kernel}]}]}


def materialize(payload):
    parsed = parse_execution_plan(payload)
    path = next(path for path in build_execution_paths(parsed, build_execution_stages(parsed)) if path.stage_id == "rmsnorm_0")
    adapter = CustomCudaBackendAdapter()
    assert adapter.validate(path, CapabilityValidationView()) == []
    return adapter.materialize(path)


def test_cuda_exact_dispatch_has_no_redecision():
    result = materialize(plan())
    assert result.config["selected_candidate_id"] == "cuda_rmsnorm_fp32_bs256_v1"
    assert result.config["redecision_count"] == 0
    index = result.benchmark_command.index("--block-sizes")
    assert result.benchmark_command[index:index + 2] == ("--block-sizes", "256")


def test_triton_exact_dispatch_preserves_fields():
    candidate = "triton_rmsnorm_fp32_block4096_warps8_stages_default_v1"
    result = materialize(plan("triton", candidate, {"block_size": 4096, "num_warps": 8, "num_stages": "default"}))
    assert result.backend == "triton"
    assert "--block-size" in result.benchmark_command and "--num-warps" in result.benchmark_command
    assert result.config["exact_candidate"]["launch_config"]["num_warps"] == 8


def test_missing_launch_config_rejected():
    payload = plan()
    del payload["function_plans"][0]["per_op_decisions"][0]["kernel"]["launch_config"]
    with pytest.raises(ExecutionPlanError, match="launch_config"):
        parse_execution_plan(payload)


def test_runtime_rejects_target_hash_and_candidate_config_mismatch():
    for mutate, expected in (
        (lambda k: k["target"].update(gpu_name="Other GPU"), "target_gpu_name_mismatch"),
        (lambda k: k["artifact"].update(source_hash="not-a-hash"), "invalid_rmsnorm_source_hash"),
        (lambda k: k["launch_config"].update(block_size=128), "candidate_launch_config_mismatch"),
    ):
        payload = plan()
        kernel = payload["function_plans"][0]["per_op_decisions"][0]["kernel"]
        mutate(kernel)
        parsed = parse_execution_plan(payload)
        path = next(path for path in build_execution_paths(parsed, build_execution_stages(parsed)) if path.stage_id == "rmsnorm_0")
        assert expected in CustomCudaBackendAdapter().validate(path, CapabilityValidationView())
