from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from deployment.execution_plan.int8_quantization import (
    INT8_KERNEL_ID,
    KERNEL_CAPABILITY,
    PACKED_B_TRANSPOSE_LAYOUT,
    PACKED_B_TRANSPOSE_SCHEME,
    PACKED_INT8_KERNEL_CAPABILITY,
    PACKED_INT8_KERNEL_ID,
    SCHEME,
    build_evaluation_artifact,
    create_calibration_artifact,
    create_packed_weight_artifact,
    latency_stats,
    load_and_validate_calibration_artifact,
    load_and_validate_packed_weight_artifact,
    numerical_metrics,
    pack_b_transposed_int8,
    quantize_symmetric,
    reference_fused_matmul_bias_relu,
    select_with_evidence,
    symmetric_scale,
    tensor_f32_sha256,
    theoretical_memory,
    write_packed_weight_artifact,
    write_json_deterministic,
)
from deployment.execution_plan.portable_cpu_kernel_adapter import (
    PortableCpuKernelError,
    Tensor,
    dispatch_fused_matmul_bias_relu,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
INT8_KERNEL = REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu_int8"
PACKED_INT8_KERNEL = REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu_int8_packed_b"
FP32_KERNEL = REPO_ROOT / "native" / "cpu_kernels" / "portable_fused_matmul_bias_relu"


def _memory_placement(m: int, n: int, k: int) -> dict:
    return {
        "status": "selected", "compute_unit": "cpu", "selected_memory_space": "cpu_visible_host_memory",
        "input_tile_bytes": m*k*4, "weight_tile_bytes": k*n*4, "output_tile_bytes": m*n*4,
        "scratch_bytes": 0, "padding_bytes": 0, "single_buffer_bytes": m*k*4 + k*n*4 + m*n*4,
        "additional_double_buffer_bytes": 0, "total_required_local_memory_bytes": m*k*4 + k*n*4 + m*n*4,
        "buffer_placements": [
            {"buffer_id": "input_tile", "role": "input", "memory_space": "cpu_visible_host_memory", "byte_count": m*k*4, "alignment": 64},
            {"buffer_id": "weight_tile", "role": "weight", "memory_space": "cpu_visible_host_memory", "byte_count": k*n*4, "alignment": 64},
            {"buffer_id": "output_tile", "role": "output", "memory_space": "cpu_visible_host_memory", "byte_count": m*n*4, "alignment": 64},
            {"buffer_id": "scratch", "role": "scratch", "memory_space": "cpu_visible_host_memory", "byte_count": 0, "alignment": 64},
        ],
        "transfer_operations": [], "compute_dependency_ids": [],
    }


def _tensors(m: int, n: int, k: int, seed: int) -> tuple[Tensor, Tensor, Tensor]:
    rng = random.Random(seed)
    return (
        Tensor((m, k), [rng.uniform(-1, 1) for _ in range(m*k)]),
        Tensor((k, n), [rng.uniform(-1, 1) for _ in range(k*n)]),
        Tensor((n,), [rng.uniform(-0.2, 0.2) for _ in range(n)]),
    )


def _artifact(tmp_path: Path, a: Tensor, b: Tensor, m: int, n: int, k: int) -> dict:
    artifact = create_calibration_artifact(
        workload_id=f"slice3a_fused_matmul_bias_relu_{m}x{n}x{k}",
        operator_kind="fused_matmul_bias_relu", m=m, n=n, k=k,
        activation_values=a.data, weight_values=b.data,
        calibration_dataset={"dataset_id": "synthetic_slice3a_seeded_uniform", "seed": 123, "distribution": "uniform[-1,1]", "sample_count": 1, "shape": {"M": m, "N": n, "K": k}},
    )
    path = tmp_path / "calibration.json"
    write_json_deterministic(path, artifact)
    artifact["path"] = str(path)
    return artifact


def _int8_decision(artifact: dict, m: int, n: int, k: int) -> dict:
    decision = {
        "op_name": "op_2", "op_type": "hir.fused_matmul_bias_relu",
        "kernel_selection": {"contract_version": "kernel_selection_contract_v1", "status": "selected", "selected_kernel": INT8_KERNEL_ID, "source": "slice3a_test"},
        "quantization": {
            "selected_candidate_id": f"fused_matmul_bias_relu:quant={SCHEME}:shape={m}x{n}x{k}:kernel={INT8_KERNEL_ID}",
            "scheme": SCHEME, "activation_dtype": "int8", "weight_dtype": "int8", "accumulation_dtype": "int32", "output_dtype": "fp32",
            "activation_granularity": "per_tensor", "weight_granularity": "per_tensor", "granularity": "per_tensor",
            "activation_scale": artifact["activation_scale"], "weight_scale": artifact["weight_scale"],
            "activation_zero_point": 0, "weight_zero_point": 0,
            "required_kernel_capability": KERNEL_CAPABILITY, "requires_calibration": True, "calibration_available": True,
            "calibration_artifact_ref": artifact["path"], "calibration_artifact_id": artifact["artifact_id"], "calibration_artifact_sha256": artifact["artifact_sha256"],
            "workload_id": artifact["workload_id"], "selection_reason": "slice3a_evidence_gate_passed",
        },
        "thread_schedule": {"status": "selected", "thread_count": 1, "partition_axis": "none", "partition_strategy": "serial"},
        "memory_placement": _memory_placement(m, n, k),
    }
    decision["quantization"]["execution_stages"] = [
        {
            "stage_id": "quantize_activation",
            "op": "hir.quantize",
            "dependency_ids": [],
            "produces": "quantized_activation_ready",
            "scale": artifact["activation_scale"],
            "zero_point": 0,
            "rounding_mode": "round_nearest_even",
            "clamp_min": -127,
            "clamp_max": 127,
            "source_dtype": "fp32",
            "destination_dtype": "int8",
        },
        {
            "stage_id": "execute_int8_kernel",
            "op": "hir.portable_cpu_int8_fused_matmul_bias_relu",
            "dependency_ids": ["quantized_activation_ready"],
            "produces": "fp32_output_ready",
            "kernel_id": INT8_KERNEL_ID,
        },
        {
            "stage_id": "return_fp32_output",
            "op": "runtime.return",
            "dependency_ids": ["fp32_output_ready"],
            "produces": "return_ready",
        },
    ]
    return decision


def _packed_artifact(tmp_path: Path, calibration: dict, b: Tensor, m: int, n: int, k: int) -> dict:
    manifest, packed_bytes = create_packed_weight_artifact(
        workload_id=calibration["workload_id"],
        operator_kind="fused_matmul_bias_relu",
        m=m, n=n, k=k,
        source_weight_values=b.data,
        weight_scale=calibration["weight_scale"],
    )
    manifest = write_packed_weight_artifact(
        tmp_path / "packed_weight.json",
        tmp_path / "packed_weight.bin",
        manifest,
        packed_bytes,
    )
    manifest["path"] = str(tmp_path / "packed_weight.json")
    return manifest


def _packed_int8_decision(calibration: dict, packed: dict, m: int, n: int, k: int) -> dict:
    decision = _int8_decision(calibration, m, n, k)
    decision["kernel_selection"]["selected_kernel"] = PACKED_INT8_KERNEL_ID
    decision["quantization"]["selected_candidate_id"] = (
        f"fused_matmul_bias_relu:quant={SCHEME}:shape={m}x{n}x{k}:"
        f"kernel={PACKED_INT8_KERNEL_ID}:packed={packed['artifact_id']}"
    )
    decision["quantization"]["required_kernel_capability"] = PACKED_INT8_KERNEL_CAPABILITY
    decision["quantization"]["kernel_requires_packed_weight"] = True
    decision["quantization"]["packed_weight_artifact_ref"] = packed["path"]
    decision["quantization"]["packed_weight_artifact_id"] = packed["artifact_id"]
    decision["quantization"]["packed_weight_sha256"] = packed["artifact_sha256"]
    decision["quantization"]["packed_layout"] = PACKED_B_TRANSPOSE_LAYOUT
    decision["quantization"]["packing_scheme"] = PACKED_B_TRANSPOSE_SCHEME
    decision["quantization"]["selection_reason"] = "slice3b_compiler_owned_packed_weight_artifact_selected"
    decision["quantization"]["execution_stages"] = [
        decision["quantization"]["execution_stages"][0],
        {
            "stage_id": "load_packed_weight",
            "op": "hir.load_quantized_weight",
            "dependency_ids": [],
            "produces": "packed_weight_ready",
            "artifact_ref": packed["path"],
            "artifact_sha256": packed["artifact_sha256"],
            "packed_layout": PACKED_B_TRANSPOSE_LAYOUT,
        },
        {
            "stage_id": "execute_int8_kernel",
            "op": "hir.portable_cpu_int8_fused_matmul_bias_relu",
            "dependency_ids": ["quantized_activation_ready", "packed_weight_ready"],
            "produces": "fp32_output_ready",
            "kernel_id": PACKED_INT8_KERNEL_ID,
            "fused_postprocess": "dequantize_bias_relu",
        },
        {
            "stage_id": "return_fp32_output",
            "op": "runtime.return",
            "dependency_ids": ["fp32_output_ready"],
            "produces": "return_ready",
        },
    ]
    return decision


def test_symmetric_scale_generation_and_zero_tensor():
    assert symmetric_scale([-2.0, 1.0]) == 2.0 / 127.0
    assert symmetric_scale([0.0, 0.0]) == 1.0


def test_quantization_clamp_round_is_deterministic():
    assert quantize_symmetric([-2.0, -0.51, 0.49, 2.0], 1.0) == [-2, -1, 0, 2]
    assert quantize_symmetric([-999.0, 999.0], 1.0) == [-127, 127]


def test_pack_b_transposed_layout_is_deterministic():
    # Original B is KxN = [[1, 2, 3], [4, 5, 6]].
    assert pack_b_transposed_int8([1, 2, 3, 4, 5, 6], n=3, k=2) == [1, 4, 2, 5, 3, 6]


def test_calibration_artifact_deterministic_and_hash_changes(tmp_path):
    a, b, _ = _tensors(2, 3, 4, seed=1)
    art1 = _artifact(tmp_path, a, b, 2, 3, 4)
    art2 = _artifact(tmp_path, a, b, 2, 3, 4)
    assert art1["artifact_sha256"] == art2["artifact_sha256"]
    b2 = Tensor(b.shape, list(b.data)); b2.data[0] += 0.01
    art3 = _artifact(tmp_path, a, b2, 2, 3, 4)
    assert art1["artifact_sha256"] != art3["artifact_sha256"]


def test_artifact_shape_and_hash_validation(tmp_path):
    a, b, _ = _tensors(2, 3, 4, seed=2)
    art = _artifact(tmp_path, a, b, 2, 3, 4)
    load_and_validate_calibration_artifact(Path(art["path"]), expected_artifact_id=art["artifact_id"], expected_artifact_sha256=art["artifact_sha256"], workload_id=art["workload_id"], operator_kind="fused_matmul_bias_relu", m=2, n=3, k=4, activation_scale=art["activation_scale"], weight_scale=art["weight_scale"], activation_zero_point=0, weight_zero_point=0, activation_values=a.data, weight_values=b.data)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_and_validate_calibration_artifact(Path(art["path"]), expected_artifact_id=art["artifact_id"], expected_artifact_sha256=art["artifact_sha256"], workload_id=art["workload_id"], operator_kind="fused_matmul_bias_relu", m=9, n=3, k=4, activation_scale=art["activation_scale"], weight_scale=art["weight_scale"], activation_zero_point=0, weight_zero_point=0)
    with pytest.raises(ValueError, match="sha256"):
        load_and_validate_calibration_artifact(Path(art["path"]), expected_artifact_id=art["artifact_id"], expected_artifact_sha256="bad", workload_id=art["workload_id"], operator_kind="fused_matmul_bias_relu", m=2, n=3, k=4, activation_scale=art["activation_scale"], weight_scale=art["weight_scale"], activation_zero_point=0, weight_zero_point=0)


def test_packed_weight_artifact_serialization_hashing_and_validation(tmp_path):
    a, b, _ = _tensors(2, 3, 4, seed=20)
    calibration = _artifact(tmp_path, a, b, 2, 3, 4)
    packed1 = _packed_artifact(tmp_path, calibration, b, 2, 3, 4)
    packed2 = _packed_artifact(tmp_path, calibration, b, 2, 3, 4)
    assert packed1["artifact_sha256"] == packed2["artifact_sha256"]
    assert packed1["packed_layout"] == PACKED_B_TRANSPOSE_LAYOUT
    assert packed1["packing_scheme"] == PACKED_B_TRANSPOSE_SCHEME
    payload, data_path = load_and_validate_packed_weight_artifact(
        Path(packed1["path"]),
        expected_artifact_id=packed1["artifact_id"],
        expected_artifact_sha256=packed1["artifact_sha256"],
        workload_id=calibration["workload_id"],
        operator_kind="fused_matmul_bias_relu",
        m=2, n=3, k=4,
        source_weight_values=b.data,
    )
    assert payload["artifact_id"] == packed1["artifact_id"]
    assert data_path.read_bytes()
    b2 = Tensor(b.shape, list(b.data)); b2.data[0] += 0.01
    manifest3, _ = create_packed_weight_artifact(
        workload_id=calibration["workload_id"], operator_kind="fused_matmul_bias_relu",
        m=2, n=3, k=4, source_weight_values=b2.data, weight_scale=calibration["weight_scale"],
    )
    assert manifest3["artifact_sha256"] != packed1["artifact_sha256"]


def test_packed_weight_artifact_rejects_hash_layout_and_shape_mismatch(tmp_path):
    a, b, _ = _tensors(2, 3, 4, seed=21)
    calibration = _artifact(tmp_path, a, b, 2, 3, 4)
    packed = _packed_artifact(tmp_path, calibration, b, 2, 3, 4)
    with pytest.raises(ValueError, match="sha256"):
        load_and_validate_packed_weight_artifact(
            Path(packed["path"]), expected_artifact_id=packed["artifact_id"], expected_artifact_sha256="bad",
            workload_id=calibration["workload_id"], operator_kind="fused_matmul_bias_relu", m=2, n=3, k=4)
    with pytest.raises(ValueError, match="packed_layout mismatch"):
        load_and_validate_packed_weight_artifact(
            Path(packed["path"]), expected_artifact_id=packed["artifact_id"], expected_artifact_sha256=packed["artifact_sha256"],
            workload_id=calibration["workload_id"], operator_kind="fused_matmul_bias_relu", m=2, n=3, k=4,
            expected_layout="row_major_kx_n")
    with pytest.raises(ValueError, match="shape mismatch"):
        load_and_validate_packed_weight_artifact(
            Path(packed["path"]), expected_artifact_id=packed["artifact_id"], expected_artifact_sha256=packed["artifact_sha256"],
            workload_id=calibration["workload_id"], operator_kind="fused_matmul_bias_relu", m=9, n=3, k=4)


@pytest.mark.skipif(not INT8_KERNEL.exists(), reason="INT8 kernel executable not built")
def test_runtime_executes_int8_and_does_not_fallback(tmp_path):
    m, n, k = 8, 7, 9
    a, b, bias = _tensors(m, n, k, seed=3)
    art = _artifact(tmp_path, a, b, m, n, k)
    result = dispatch_fused_matmul_bias_relu(op_decision=_int8_decision(art, m, n, k), backend="cpu", a=a, b=b, bias=bias, repeats=2)
    assert result.kernel_id == INT8_KERNEL_ID
    assert result.dtype == SCHEME
    assert result.raw_kernel_stdout["arithmetic"] == "int8_times_int8_accumulate_int32_dequantize_fp32"
    ref = reference_fused_matmul_bias_relu(a.data, b.data, bias.data, m, n, k)
    metrics = numerical_metrics(ref, result.output.data)
    assert metrics["cosine_similarity"] >= 0.99
    assert metrics["relative_l2_error"] <= 0.05


@pytest.mark.skipif(not PACKED_INT8_KERNEL.exists(), reason="packed INT8 kernel executable not built")
def test_runtime_executes_packed_int8_and_never_repacks(tmp_path):
    m, n, k = 8, 7, 9
    a, b, bias = _tensors(m, n, k, seed=22)
    calibration = _artifact(tmp_path, a, b, m, n, k)
    packed = _packed_artifact(tmp_path, calibration, b, m, n, k)
    result = dispatch_fused_matmul_bias_relu(
        op_decision=_packed_int8_decision(calibration, packed, m, n, k),
        backend="cpu", a=a, b=b, bias=bias, repeats=2,
    )
    assert result.kernel_id == PACKED_INT8_KERNEL_ID
    assert result.raw_kernel_stdout["runtime_packed_weight_transform"] is False
    assert result.raw_kernel_stdout["packed_layout"] == PACKED_B_TRANSPOSE_LAYOUT
    assert result.raw_kernel_stdout["arithmetic"] == "int8_times_int8_accumulate_int32_dequantize_fp32_packed_b"
    ref = reference_fused_matmul_bias_relu(a.data, b.data, bias.data, m, n, k)
    metrics = numerical_metrics(ref, result.output.data)
    assert metrics["cosine_similarity"] >= 0.99
    assert metrics["relative_l2_error"] <= 0.05


def test_runtime_rejects_packed_contract_mismatches(tmp_path):
    m, n, k = 4, 5, 6
    a, b, bias = _tensors(m, n, k, seed=23)
    calibration = _artifact(tmp_path, a, b, m, n, k)
    packed = _packed_artifact(tmp_path, calibration, b, m, n, k)
    decision = _packed_int8_decision(calibration, packed, m, n, k)
    decision["quantization"]["packed_weight_sha256"] = "bad"
    with pytest.raises((PortableCpuKernelError, ValueError), match="sha256"):
        dispatch_fused_matmul_bias_relu(op_decision=decision, backend="cpu", a=a, b=b, bias=bias)
    decision = _packed_int8_decision(calibration, packed, m, n, k)
    decision["quantization"]["packed_layout"] = "row_major_kx_n"
    with pytest.raises(PortableCpuKernelError, match="packed_layout"):
        dispatch_fused_matmul_bias_relu(op_decision=decision, backend="cpu", a=a, b=b, bias=bias)
    decision = _packed_int8_decision(calibration, packed, m, n, k)
    decision["quantization"]["kernel_requires_packed_weight"] = False
    with pytest.raises(PortableCpuKernelError, match="requires quantization.kernel_requires_packed_weight"):
        dispatch_fused_matmul_bias_relu(op_decision=decision, backend="cpu", a=a, b=b, bias=bias)


def test_runtime_rejects_missing_artifact_and_scale_mismatch(tmp_path):
    m, n, k = 4, 4, 4
    a, b, bias = _tensors(m, n, k, seed=4)
    art = _artifact(tmp_path, a, b, m, n, k)
    decision = _int8_decision(art, m, n, k)
    decision["quantization"]["calibration_artifact_ref"] = str(tmp_path / "missing.json")
    with pytest.raises((PortableCpuKernelError, FileNotFoundError)):
        dispatch_fused_matmul_bias_relu(op_decision=decision, backend="cpu", a=a, b=b, bias=bias)
    decision = _int8_decision(art, m, n, k)
    decision["quantization"]["activation_scale"] *= 2
    with pytest.raises((PortableCpuKernelError, ValueError), match="stage scale"):
        dispatch_fused_matmul_bias_relu(op_decision=decision, backend="cpu", a=a, b=b, bias=bias)


def test_runtime_rejects_int8_without_explicit_quantized_stages(tmp_path):
    m, n, k = 4, 4, 4
    a, b, bias = _tensors(m, n, k, seed=25)
    art = _artifact(tmp_path, a, b, m, n, k)
    decision = _int8_decision(art, m, n, k)
    del decision["quantization"]["execution_stages"]
    with pytest.raises(PortableCpuKernelError, match="execution_stages is required"):
        dispatch_fused_matmul_bias_relu(op_decision=decision, backend="cpu", a=a, b=b, bias=bias)


def test_runtime_uses_exact_plan_quantize_stage_scale(tmp_path):
    m, n, k = 4, 4, 4
    a, b, bias = _tensors(m, n, k, seed=26)
    art = _artifact(tmp_path, a, b, m, n, k)
    decision = _int8_decision(art, m, n, k)
    decision["quantization"]["execution_stages"][0]["scale"] *= 2
    with pytest.raises(PortableCpuKernelError, match="stage scale"):
        dispatch_fused_matmul_bias_relu(op_decision=decision, backend="cpu", a=a, b=b, bias=bias)


def test_evaluation_artifact_and_policy_round_trip(tmp_path):
    a, b, _ = _tensors(2, 3, 4, seed=5)
    art = _artifact(tmp_path, a, b, 2, 3, 4)
    metrics = {"max_absolute_error": 0.01, "mean_absolute_error": 0.001, "mean_squared_error": 0.0001, "relative_l2_error": 0.01, "cosine_similarity": 0.999, "relu_zero_state_mismatch_percent": 0.0}
    fp32 = latency_stats([10.0, 9.0, 11.0]); int8 = latency_stats([8.0, 8.1, 7.9])
    ev = build_evaluation_artifact(evaluation_id="ev", workload_id=art["workload_id"], shape=art["shape"], input_hash=tensor_f32_sha256(a.data), weight_hash=tensor_f32_sha256(b.data), fp32_candidate_id="fp32", int8_candidate_id="int8", calibration_artifact=art, thread_count=1, build_identity={"compiler_flags": "test"}, correctness_metrics=metrics, fp32_latency=fp32, int8_latency=int8, theoretical_tensor_bytes=theoretical_memory(2,3,4,123), observed_memory={"kind": "unavailable", "value": None}, thresholds={"min_cosine_similarity": 0.99, "max_relative_l2_error": 0.05}, timestamp="2026-07-14T00:00:00Z")
    path = tmp_path / "evaluation.json"; write_json_deterministic(path, ev)
    assert json.loads(path.read_text())["evaluation_sha256"] == ev["evaluation_sha256"]
    assert select_with_evidence(ev)["selected_scheme"] == SCHEME
    ev["correctness_metrics"]["value"]["cosine_similarity"] = 0.5
    assert select_with_evidence(ev)["rejection_reason"] == "accuracy_gate_failed"
    ev["correctness_metrics"]["value"]["cosine_similarity"] = 0.999
    ev["int8_latency_statistics"]["value"]["median_ms"] = 9.9
    assert select_with_evidence(ev, min_speedup_margin=0.02)["rejection_reason"] == "performance_gate_failed"
