import copy
import hashlib
from pathlib import Path
import pytest

from deployment.execution_plan.aarch64_native_object_adapter import (
    AArch64NativeObjectAdapter, AArch64NativeObjectError)
from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.path_builder import build_execution_paths
from deployment.execution_plan.stage_builder import build_execution_stages


def contract(tmp_path):
    obj = tmp_path / "kernel.o"; obj.write_bytes(b"object")
    return {"decision_kind": "aarch64_native_exact_candidate_selection",
            "candidate_id": "tile8x8x8_uk1",
            "operator": "hir.fused_matmul_bias_relu",
            "kernel_family": "aarch64_generated_fused_matmul_bias_relu",
            "dtype": "f32", "shape": {"m": 32, "n": 32, "k": 32},
            "target": {"triple": "aarch64-linux-gnu", "cpu": "cortex-a76",
                       "features": [], "target_profile_id": "raspberry-pi5-cortex-a76-cpu"},
            "lowering": {"pipeline_id": "aarch64_tiled_scheduled_v1",
                         "tile_m": 8, "tile_n": 8, "tile_k": 8,
                         "schedule_unroll_k": 1, "vector_width_bits": 128,
                         "loop_order_id": "tiled_mnk_row_major_v1"},
            "microkernel_id": "hir_fused_matmul_bias_relu_tiled_scheduled_v1",
            "entry_point": "_mlir_ciface_matmul_bias_relu_tiled_32x32x32",
            "abi_version": "mlir_ciface_memref_f32_v1",
            "object_ref": obj.name,
            "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
            "backend_evidence_ref": "evidence.json",
            "correctness_evidence_ref": None, "measurement_evidence_ref": None,
            "selection_mode": "deterministic_static_lexicographic_estimate",
            "selection_trace_ref": "selection.json", "runtime_no_redecision": True}


def plan(c):
    return {"schema": "execution_plan", "schema_version": "2.0.0", "plan_id": "p",
            "provenance": {"compiler_tool": "test", "model_spec_ref": "x",
                           "capability_bundle": {"hardware_profile_ref": "pi"},
                           "truth_boundary": "test"},
            "model_identity": {"model_id": "op"}, "global_decisions": {},
            "function_plans": [{"function_name": "matmul_bias_relu_tiled_32x32x32",
                "serving_phase": "other",
                "backend": {"selected_backend": "aarch64_native_object"},
                "per_op_decisions": [{"op_name": "fused", "op_type": "hir.fused_matmul_bias_relu",
                                      "native_execution": c}]}]}


def test_valid_contract_and_path(tmp_path):
    c = contract(tmp_path)
    AArch64NativeObjectAdapter(c, plan_root=tmp_path).validate(require_running_target=False)
    parsed = parse_execution_plan(plan(c))
    path = next(x for x in build_execution_paths(parsed, build_execution_stages(parsed))
                if x.stage_id == "fused")
    assert path.selected_kernel == c["candidate_id"]
    assert path.runtime_config["native_execution"]["object_sha256"] == c["object_sha256"]


@pytest.mark.parametrize("field,value,error", [
    ("candidate_id", "unknown", "unknown_candidate"),
    ("dtype", "f16", "dtype_mismatch"),
    ("shape", {"m": 1, "n": 1, "k": 1}, "shape_mismatch"),
    ("abi_version", "bad", "abi_version_mismatch"),
    ("entry_point", "bad", "entry_point_mismatch"),
])
def test_fail_closed_identity(tmp_path, field, value, error):
    c = contract(tmp_path); c[field] = value
    with pytest.raises(AArch64NativeObjectError, match=error):
        AArch64NativeObjectAdapter(c, plan_root=tmp_path).validate(require_running_target=False)


def test_wrong_hash_and_no_substitution(tmp_path):
    c = contract(tmp_path); c["object_sha256"] = "0" * 64
    with pytest.raises(AArch64NativeObjectError, match="object_sha256_mismatch"):
        AArch64NativeObjectAdapter(c, plan_root=tmp_path).validate(require_running_target=False)


def test_selected_executed_proof(tmp_path):
    c = contract(tmp_path); a = AArch64NativeObjectAdapter(c, plan_root=tmp_path)
    result = {"executed": {"candidate_id": c["candidate_id"],
              "object_sha256": c["object_sha256"], "entry_point": c["entry_point"],
              "runtime_redecision_count": 0}}
    assert a.proof(result)["runtime_redecision_count"] == 0
    result["executed"]["candidate_id"] = "other"
    with pytest.raises(AArch64NativeObjectError, match="selected_executed"):
        a.proof(result)
