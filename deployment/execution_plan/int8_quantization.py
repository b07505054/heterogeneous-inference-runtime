"""Deterministic Slice 3A INT8 calibration/evaluation helpers."""
from __future__ import annotations
import hashlib, json, math, platform, statistics, struct
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "slice3a.int8_static_symmetric.v1"
EVALUATION_SCHEMA_VERSION = "slice3a.int8_matmul_evaluation.v1"
SCHEME = "int8_static_symmetric"
KERNEL_CAPABILITY = "quant_kernel.int8_static_symmetric"
INT8_KERNEL_ID = "portable_fused_matmul_bias_relu_int8_symmetric"
PACKED_WEIGHT_SCHEMA_VERSION = "slice3b.int8_packed_weight.v1"
PACKED_B_TRANSPOSE_SCHEME = "b_transposed_nxk_contiguous"
PACKED_B_TRANSPOSE_LAYOUT = "packed_b_transposed_nxk"
PACKED_INT8_KERNEL_CAPABILITY = "quant_kernel.int8_static_symmetric.packed_b_transposed"
PACKED_INT8_KERNEL_ID = "portable_fused_matmul_bias_relu_int8_symmetric_packed_b"
PACKED_WEIGHT_PRODUCER_VERSION = "slice3b_packed_weight_v1"
TRUTH_BOUNDARY = "slice3a_operator_int8_static_symmetric_real_artifact_not_model_accuracy"

def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def tensor_f32_sha256(values: Iterable[float]) -> str:
    vals = list(values)
    return sha256_bytes(struct.pack(f"<{len(vals)}f", *vals))

def symmetric_scale(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("cannot calibrate an empty tensor")
    max_abs = max(abs(float(v)) for v in vals)
    return 1.0 if max_abs == 0.0 else max_abs / 127.0

def quantize_symmetric(values: Iterable[float], scale: float) -> list[int]:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}")
    return [max(-127, min(127, int(round(float(v) / scale)))) for v in values]

def i8_bytes(values: Iterable[int]) -> bytes:
    vals = list(values)
    return struct.pack(f"<{len(vals)}b", *vals)

def pack_b_transposed_int8(weight_int8: Iterable[int], *, n: int, k: int) -> list[int]:
    """Pack row-major B[K,N] into compiler-owned contiguous Bp[N,K].

    The packed kernel computes one output column j at a time and walks K as
    the innermost loop, so this layout makes Bp[j, kk] contiguous. This is a
    representation change owned by the compiler artifact, not a runtime
    transpose.
    """

    vals = list(weight_int8)
    if len(vals) != n * k:
        raise ValueError(f"weight_int8 has {len(vals)} elements, expected K*N={k*n}")
    packed = [0] * (n * k)
    for kk in range(k):
        for j in range(n):
            packed[j * k + kk] = vals[kk * n + j]
    return packed

def create_packed_weight_artifact(*, workload_id: str, operator_kind: str, m: int, n: int, k: int,
                                  source_weight_values: Iterable[float], weight_scale: float,
                                  producer_version: str = PACKED_WEIGHT_PRODUCER_VERSION,
                                  compiler_version: str = PACKED_WEIGHT_PRODUCER_VERSION,
                                  creation_time: str = "deterministic") -> tuple[dict[str, Any], bytes]:
    source_weight_values = list(source_weight_values)
    if len(source_weight_values) != k * n:
        raise ValueError(f"source weights have {len(source_weight_values)} elements, expected K*N={k*n}")
    quantized = quantize_symmetric(source_weight_values, weight_scale)
    packed = pack_b_transposed_int8(quantized, n=n, k=k)
    packed_bytes = i8_bytes(packed)
    packed_sha = sha256_bytes(packed_bytes)
    payload: dict[str, Any] = {
        "schema_version": PACKED_WEIGHT_SCHEMA_VERSION,
        "source_weight_sha256": tensor_f32_sha256(source_weight_values),
        "original_layout": "row_major_kx_n",
        "packed_layout": PACKED_B_TRANSPOSE_LAYOUT,
        "packing_scheme": PACKED_B_TRANSPOSE_SCHEME,
        "original_shape": {"K": k, "N": n},
        "packed_shape": {"N": n, "K": k},
        "dtype": "int8",
        "tile_shape": {"N": 1, "K": k},
        "alignment": 1,
        "producer_version": producer_version,
        "creation_time": creation_time,
        "kernel_capability": PACKED_INT8_KERNEL_CAPABILITY,
        "compiler_version": compiler_version,
        "workload_id": workload_id,
        "operator_kind": operator_kind,
        "shape": {"M": m, "N": n, "K": k},
        "weight_scale": weight_scale,
        "weight_zero_point": 0,
        "artifact_sha256": packed_sha,
        "truth_boundary": "compiler_owned_offline_int8_weight_packing_artifact_runtime_must_not_repack",
    }
    payload["artifact_id"] = "slice3b-packed-" + sha256_bytes(_canonical_json_bytes(payload))[:16]
    return payload, packed_bytes

def write_packed_weight_artifact(manifest_path: Path, data_path: Path, manifest: dict[str, Any], packed_bytes: bytes) -> dict[str, Any]:
    if sha256_bytes(packed_bytes) != manifest.get("artifact_sha256"):
        raise ValueError("packed weight bytes do not match manifest artifact_sha256")
    manifest = dict(manifest)
    manifest["packed_weight_data_ref"] = data_path.name
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(packed_bytes)
    write_json_deterministic(manifest_path, manifest)
    return manifest

def load_and_validate_packed_weight_artifact(path: Path, *, expected_artifact_id: str, expected_artifact_sha256: str,
                                             workload_id: str, operator_kind: str, m: int, n: int, k: int,
                                             source_weight_values: Iterable[float] | None = None,
                                             expected_layout: str = PACKED_B_TRANSPOSE_LAYOUT,
                                             expected_packing_scheme: str = PACKED_B_TRANSPOSE_SCHEME,
                                             expected_dtype: str = "int8") -> tuple[dict[str, Any], Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "artifact_id": expected_artifact_id,
        "schema_version": PACKED_WEIGHT_SCHEMA_VERSION,
        "workload_id": workload_id,
        "operator_kind": operator_kind,
        "packed_layout": expected_layout,
        "packing_scheme": expected_packing_scheme,
        "dtype": expected_dtype,
        "kernel_capability": PACKED_INT8_KERNEL_CAPABILITY,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(f"packed weight artifact {key} mismatch")
    if payload.get("shape") != {"M": m, "N": n, "K": k}:
        raise ValueError("packed weight artifact shape mismatch")
    if payload.get("original_shape") != {"K": k, "N": n}:
        raise ValueError("packed weight artifact original_shape mismatch")
    if payload.get("packed_shape") != {"N": n, "K": k}:
        raise ValueError("packed weight artifact packed_shape mismatch")
    if payload.get("artifact_sha256") != expected_artifact_sha256:
        raise ValueError("packed weight artifact sha256 does not match compiler contract")
    data_ref = payload.get("packed_weight_data_ref")
    if not data_ref:
        raise ValueError("packed weight artifact missing packed_weight_data_ref")
    data_path = Path(str(data_ref))
    if not data_path.is_absolute():
        data_path = path.parent / data_path
    data = data_path.read_bytes()
    if sha256_bytes(data) != expected_artifact_sha256:
        raise ValueError("packed weight data sha256 mismatch")
    if len(data) != n * k:
        raise ValueError(f"packed weight data has {len(data)} bytes, expected {n*k}")
    if source_weight_values is not None and payload.get("source_weight_sha256") != tensor_f32_sha256(source_weight_values):
        raise ValueError("packed weight source_weight_sha256 mismatch")
    return payload, data_path

def create_calibration_artifact(*, workload_id: str, operator_kind: str, m: int, n: int, k: int,
                                activation_values: Iterable[float], weight_values: Iterable[float],
                                calibration_dataset: dict[str, Any]) -> dict[str, Any]:
    activation_values = list(activation_values); weight_values = list(weight_values)
    manifest_hash = sha256_bytes(_canonical_json_bytes(calibration_dataset))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "workload_id": workload_id, "operator_kind": operator_kind,
        "shape": {"M": m, "N": n, "K": k}, "quantization_scheme": SCHEME,
        "activation_dtype": "int8", "weight_dtype": "int8", "accumulator_dtype": "int32", "output_dtype": "fp32",
        "activation_granularity": "per_tensor", "weight_granularity": "per_tensor",
        "activation_scale": symmetric_scale(activation_values), "weight_scale": symmetric_scale(weight_values),
        "activation_zero_point": 0, "weight_zero_point": 0,
        "output_dequantization_rule": "output_fp32 = accumulator_int32 * activation_scale * weight_scale; bias_fp32 and relu applied after dequantization",
        "calibration_method": "symmetric_minmax_absmax_div_127",
        "calibration_dataset_id": calibration_dataset.get("dataset_id", "synthetic_generated_input_set"),
        "calibration_dataset_manifest": calibration_dataset,
        "calibration_dataset_manifest_sha256": manifest_hash,
        "calibration_sample_count": int(calibration_dataset.get("sample_count", 1)),
        "input_data_sha256": tensor_f32_sha256(activation_values), "weight_data_sha256": tensor_f32_sha256(weight_values),
        "creation_tool": "deployment.execution_plan.int8_quantization:create_calibration_artifact",
        "truth_boundary": TRUTH_BOUNDARY,
    }
    payload["artifact_id"] = "slice3a-" + sha256_bytes(_canonical_json_bytes(payload))[:16]
    payload["artifact_sha256"] = sha256_bytes(_canonical_json_bytes(payload))
    return payload

def write_json_deterministic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def load_and_validate_calibration_artifact(path: Path, *, expected_artifact_id: str, expected_artifact_sha256: str,
                                           workload_id: str, operator_kind: str, m: int, n: int, k: int,
                                           activation_scale: float, weight_scale: float,
                                           activation_zero_point: int, weight_zero_point: int,
                                           activation_values: Iterable[float] | None = None,
                                           weight_values: Iterable[float] | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_hash = payload.get("artifact_sha256"); clone = dict(payload); clone.pop("artifact_sha256", None)
    if actual_hash != sha256_bytes(_canonical_json_bytes(clone)):
        raise ValueError("calibration artifact self-hash mismatch")
    if actual_hash != expected_artifact_sha256:
        raise ValueError("calibration artifact sha256 does not match compiler contract")
    checks = {"artifact_id": expected_artifact_id, "schema_version": SCHEMA_VERSION, "workload_id": workload_id,
              "operator_kind": operator_kind, "quantization_scheme": SCHEME, "activation_dtype": "int8",
              "weight_dtype": "int8", "accumulator_dtype": "int32", "output_dtype": "fp32",
              "activation_granularity": "per_tensor", "weight_granularity": "per_tensor"}
    for key, expected in checks.items():
        if payload.get(key) != expected: raise ValueError(f"calibration artifact {key} mismatch")
    if payload.get("shape") != {"M": m, "N": n, "K": k}: raise ValueError("calibration artifact shape mismatch")
    if int(payload.get("activation_zero_point")) != activation_zero_point: raise ValueError("activation zero point mismatch")
    if int(payload.get("weight_zero_point")) != weight_zero_point: raise ValueError("weight zero point mismatch")
    if not math.isclose(float(payload.get("activation_scale")), float(activation_scale), rel_tol=0.0, abs_tol=1e-12): raise ValueError("activation scale mismatch")
    if not math.isclose(float(payload.get("weight_scale")), float(weight_scale), rel_tol=0.0, abs_tol=1e-12): raise ValueError("weight scale mismatch")
    if activation_values is not None and payload.get("input_data_sha256") != tensor_f32_sha256(activation_values): raise ValueError("activation/input data hash mismatch")
    if weight_values is not None and payload.get("weight_data_sha256") != tensor_f32_sha256(weight_values): raise ValueError("weight data hash mismatch")
    return payload

def reference_fused_matmul_bias_relu(a: list[float], b: list[float], bias: list[float], m: int, n: int, k: int) -> list[float]:
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k): acc += a[i * k + kk] * b[kk * n + j]
            out[i * n + j] = max(0.0, acc + bias[j])
    return out

def numerical_metrics(reference: list[float], actual: list[float]) -> dict[str, float]:
    diffs = [float(a) - float(r) for r, a in zip(reference, actual)]; abs_diffs = [abs(d) for d in diffs]
    mse = sum(d*d for d in diffs) / len(diffs) if diffs else 0.0
    ref_norm = math.sqrt(sum(r*r for r in reference)); diff_norm = math.sqrt(sum(d*d for d in diffs)); actual_norm = math.sqrt(sum(a*a for a in actual))
    dot = sum(r*a for r, a in zip(reference, actual)); cosine = 1.0 if ref_norm == 0.0 and actual_norm == 0.0 else dot / (ref_norm * actual_norm)
    zero_mismatch = sum((r == 0.0) != (a == 0.0) for r, a in zip(reference, actual))
    return {"max_absolute_error": max(abs_diffs) if abs_diffs else 0.0, "mean_absolute_error": sum(abs_diffs)/len(abs_diffs) if abs_diffs else 0.0,
            "mean_squared_error": mse, "relative_l2_error": 0.0 if ref_norm == 0.0 else diff_norm/ref_norm,
            "cosine_similarity": cosine, "relu_zero_state_mismatch_percent": 100.0*zero_mismatch/len(reference) if reference else 0.0}

def latency_stats(samples_ms: Iterable[float]) -> dict[str, Any]:
    samples = sorted(float(s) for s in samples_ms)
    if not samples: raise ValueError("no latency samples")
    def pct(p: float) -> float:
        return samples[min(len(samples)-1, max(0, math.ceil((p/100.0)*len(samples))-1))]
    return {"samples_ms": samples, "minimum_ms": samples[0], "median_ms": statistics.median(samples), "p90_ms": pct(90), "p95_ms": pct(95), "mean_ms": statistics.mean(samples), "stddev_ms": statistics.pstdev(samples) if len(samples)>1 else 0.0}

def theoretical_memory(m: int, n: int, k: int, artifact_bytes: int, packed_weight_artifact_bytes: int = 0) -> dict[str, int]:
    return {"fp32_activation_bytes": m*k*4, "fp32_weight_bytes": k*n*4, "int8_activation_bytes": m*k, "int8_weight_bytes": k*n,
            "int32_accumulator_bytes": m*n*4, "output_bytes": m*n*4, "quantization_artifact_bytes": artifact_bytes,
            "packed_weight_artifact_bytes": packed_weight_artifact_bytes}

def rss_kb() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"): return int(line.split()[1])
    except OSError: return None
    return None

def build_evaluation_artifact(*, evaluation_id: str, workload_id: str, shape: dict[str, int], input_hash: str, weight_hash: str,
                              fp32_candidate_id: str, int8_candidate_id: str, calibration_artifact: dict[str, Any],
                              thread_count: int, build_identity: dict[str, Any], correctness_metrics: dict[str, float],
                              fp32_latency: dict[str, Any], int8_latency: dict[str, Any], theoretical_tensor_bytes: dict[str, int],
                              observed_memory: dict[str, Any], thresholds: dict[str, float], timestamp: str) -> dict[str, Any]:
    accuracy_gate = correctness_metrics["cosine_similarity"] >= thresholds["min_cosine_similarity"] and correctness_metrics["relative_l2_error"] <= thresholds["max_relative_l2_error"]
    speedup = fp32_latency["median_ms"] / int8_latency["median_ms"] if int8_latency["median_ms"] > 0 else None
    payload = {"schema_version": EVALUATION_SCHEMA_VERSION, "evaluation_id": evaluation_id, "workload_id": workload_id, "shape": shape,
               "input_hash": input_hash, "weight_hash": weight_hash, "fp32_candidate_id": fp32_candidate_id, "int8_candidate_id": int8_candidate_id,
               "calibration_artifact_id": calibration_artifact["artifact_id"], "calibration_artifact_sha256": calibration_artifact["artifact_sha256"],
               "hardware_identity": {"machine": platform.machine(), "processor": platform.processor(), "platform": platform.platform()}, "os_kernel": platform.release(),
               "cpu_architecture": platform.machine(), "thread_count": thread_count, "build_identity": build_identity,
               "correctness_metrics": {"kind": "measured", "value": correctness_metrics}, "fp32_latency_statistics": {"kind": "measured", "value": fp32_latency},
               "int8_latency_statistics": {"kind": "measured", "value": int8_latency}, "speedup": {"kind": "derived", "value": speedup},
               "theoretical_tensor_bytes": {"kind": "derived", "value": theoretical_tensor_bytes}, "observed_memory": observed_memory,
               "accuracy_gate_result": {"kind": "derived", "value": accuracy_gate, "thresholds": thresholds},
               "performance_comparison": {"kind": "derived", "value": "int8_faster" if speedup and speedup > 1.0 else "int8_not_faster"},
               "timestamp": timestamp, "truth_boundary": "operator_level_numerical_fidelity_and_kernel_timing_not_model_accuracy_or_perplexity"}
    payload["evaluation_sha256"] = sha256_bytes(_canonical_json_bytes(payload)); return payload

def select_with_evidence(evaluation: dict[str, Any], *, min_speedup_margin: float = 0.02) -> dict[str, Any]:
    metrics = evaluation["correctness_metrics"]["value"]; thresholds = evaluation["accuracy_gate_result"]["thresholds"]
    if metrics["cosine_similarity"] < thresholds["min_cosine_similarity"] or metrics["relative_l2_error"] > thresholds["max_relative_l2_error"]:
        return {"selected_scheme": "fp32_baseline", "rejection_reason": "accuracy_gate_failed"}
    fp32 = evaluation["fp32_latency_statistics"]["value"]["median_ms"]; int8 = evaluation["int8_latency_statistics"]["value"]["median_ms"]
    if int8 > fp32 * (1.0 - min_speedup_margin): return {"selected_scheme": "fp32_baseline", "rejection_reason": "performance_gate_failed"}
    return {"selected_scheme": SCHEME, "selection_reason": "accuracy_and_performance_gates_passed"}
