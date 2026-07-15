"""Slice 3C target-aware packed INT8 candidate selection.

This is compiler-owned operator materialization tooling for the current
portable fused MatMul+Bias+ReLU experiment. It is not LLVM code generation and
does not extract full-model weights.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import subprocess
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from deployment.execution_plan.int8_quantization import (
    INT8_KERNEL_ID,
    KERNEL_CAPABILITY,
    PACKED_B_TRANSPOSE_LAYOUT,
    PACKED_B_TRANSPOSE_SCHEME,
    PACKED_INT8_KERNEL_CAPABILITY,
    PACKED_INT8_KERNEL_ID,
    SCHEME,
    sha256_bytes,
    write_json_deterministic,
)

FP32_KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn128_bk32"
FP32_CANDIDATE_ID = "slice3c:portable_cpu:fp32:row_major_kx_n:generic_aarch64"
INT8_ROW_MAJOR_CANDIDATE_ID = "slice3c:portable_cpu:int8_static_symmetric:row_major_kx_n:generic_aarch64"
INT8_PACKED_GENERIC_CANDIDATE_ID = "slice3c:portable_cpu:int8_static_symmetric:packed_b_transposed_nxk:generic_aarch64"
INT8_PACKED_A76_DOTPROD_CANDIDATE_ID = "slice3c:portable_cpu:int8_static_symmetric:packed_b_transposed_nxk:cortex_a76_dotprod"
EXECUTORCH_XNNPACK_FP32_T1_CANDIDATE_ID = "slice3g:executorch_xnnpack:fp32:none:t1:aarch64"
EXECUTORCH_XNNPACK_INT8_T1_CANDIDATE_ID = "slice3g:executorch_xnnpack:int8:pt2e_per_tensor_affine_per_channel_symmetric_axis0:t1:aarch64"
EXECUTORCH_XNNPACK_INT8_T4_CANDIDATE_ID = "slice3g:executorch_xnnpack:int8:pt2e_per_tensor_affine_per_channel_symmetric_axis0:t4:aarch64"

MEASUREMENT_SCHEMA_VERSION = "slice3c.measurement_artifact.v1"
BUILD_MANIFEST_SCHEMA_VERSION = "slice3c.kernel_build_manifest.v1"
SELECTION_SCHEMA_VERSION = "slice3c.selection_result.v1"


@dataclass(frozen=True)
class CodegenCapabilities:
    target_id: str
    architecture: str
    microarchitecture: str
    isa_features: tuple[str, ...]
    vector_width_bits: int
    supports_int8_dot_product: bool
    supported_compiler_target_flags: tuple[str, ...]
    supported_kernel_capabilities: tuple[str, ...]
    supports_executorch_runtime: bool = False
    supports_xnnpack_delegate: bool = False
    available_runtime_artifacts: tuple[str, ...] = ()
    maximum_runtime_threads: int = 1
    physical_compute_units: int = 1
    supported_executorch_candidate_variants: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompleteCandidate:
    candidate_id: str
    backend: str
    precision: str
    kernel_id: str
    weight_layout: str
    packing_scheme: str
    codegen_target_id: str
    required_isa: tuple[str, ...]
    required_compiler_flags: tuple[str, ...]
    required_kernel_capability: str
    runtime: str = "native"
    delegate: str = "none"
    thread_count: int = 1
    quantization_scheme: str = "none"
    artifact_kind: str = "native_binary"
    tile_id: str = "untiled"
    memory_placement: str = "cpu_visible_host_memory"
    execution_strategy: str = "native_fused_call"
    measurement_evidence_kind: str = "steady_state_invoke"
    generation_provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateDimension:
    name: str
    values: tuple[str | int, ...]
    identity: str
    legality_constraints: tuple[str, ...] = ()
    dependency_rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateGraphNode:
    node_id: str
    dimensions: dict[str, str | int]
    candidate: CompleteCandidate | None
    generation_status: str
    rejection_reasons: tuple[str, ...]
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class CandidateGraph:
    schema_version: str
    dimensions: tuple[CandidateDimension, ...]
    nodes: tuple[CandidateGraphNode, ...]

    @property
    def accepted_candidates(self) -> tuple[CompleteCandidate, ...]:
        return tuple(n.candidate for n in self.nodes if n.candidate is not None)

    def to_visualization_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dimensions": [asdict(d) for d in self.dimensions],
            "summary": {
                "combination_count": len(self.nodes),
                "accepted_count": len(self.accepted_candidates),
                "rejected_count": len(self.nodes) - len(self.accepted_candidates),
            },
            "nodes": [
                {
                    "node_id": n.node_id,
                    "dimensions": n.dimensions,
                    "generation_status": n.generation_status,
                    "candidate_id": n.candidate.candidate_id if n.candidate else None,
                    "rejection_reasons": list(n.rejection_reasons),
                    "provenance": list(n.provenance),
                }
                for n in self.nodes
            ],
        }


GENERATION_REJECTION_REASONS = frozenset({
    "backend_runtime_mismatch", "backend_delegate_mismatch",
    "precision_quantization_mismatch", "kernel_precision_mismatch",
    "layout_kernel_mismatch", "packing_layout_mismatch",
    "target_kernel_mismatch", "artifact_runtime_mismatch",
    "thread_execution_mismatch", "memory_execution_mismatch",
})


class CandidateGenerator:
    """Generate complete candidates; never evaluate cost or select a winner."""

    def discover_dimensions(self) -> tuple[CandidateDimension, ...]:
        return (
            CandidateDimension("backend", ("portable_cpu", "executorch_xnnpack"), "backend.v1", ("backend_registered",), ("root_dimension",)),
            CandidateDimension("runtime", ("native", "executorch"), "runtime.v1", ("runtime_packaged",), ("backend_runtime",)),
            CandidateDimension("delegate", ("none", "xnnpack"), "delegate.v1", ("delegate_available",), ("backend_delegate",)),
            CandidateDimension("precision", ("fp32", SCHEME, "int8"), "precision.v1", ("precision_supported",), ("backend_precision",)),
            CandidateDimension("quantization_scheme", ("none", SCHEME, "pt2e_per_tensor_affine_per_channel_symmetric_axis0"), "quantization.v1", ("quantization_supported",), ("precision_quantization",)),
            CandidateDimension("kernel", (FP32_KERNEL_ID, INT8_KERNEL_ID, PACKED_INT8_KERNEL_ID, "xnnpack_delegate_linear_bias_relu", "xnnpack_delegate_quantized_linear_bias_relu"), "kernel.v1", ("kernel_available",), ("kernel_precision",)),
            CandidateDimension("layout", ("row_major_kx_n", PACKED_B_TRANSPOSE_LAYOUT, "embedded_pte"), "layout.v1", ("layout_supported",), ("layout_kernel",)),
            CandidateDimension("packing", ("none", PACKED_B_TRANSPOSE_SCHEME), "packing.v1", ("packed_artifact_available",), ("packing_layout",)),
            CandidateDimension("tile", ("untiled",), "tile.v1", ("tile_supported",), ("kernel_tile",)),
            CandidateDimension("thread_count", (1, 4), "thread.v1", ("target_thread_budget",), ("thread_execution",)),
            CandidateDimension("memory_placement", ("cpu_visible_host_memory", "runtime_managed"), "memory.v1", ("memory_space_available",), ("memory_execution",)),
            CandidateDimension("target_isa", ("generic_aarch64", "cortex_a76_dotprod", "aarch64"), "target_isa.v1", ("isa_supported",), ("target_kernel",)),
            CandidateDimension("artifact_kind", ("native_binary", "pte"), "artifact.v1", ("artifact_identity_valid",), ("artifact_runtime",)),
            CandidateDimension("execution_strategy", ("native_fused_call", "xnnpack_delegate_region"), "execution.v1", ("execution_strategy_supported",), ("thread_execution",)),
            CandidateDimension("measurement_evidence", ("steady_state_invoke",), "measurement.v1", ("measurement_identity_valid",), ("candidate_measurement",)),
        )

    @staticmethod
    def _node_id(values: dict[str, str | int]) -> str:
        encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        return "candidate-combination-" + hashlib.sha256(encoded).hexdigest()[:20]

    @staticmethod
    def _dependency_reasons(v: dict[str, str | int]) -> tuple[str, ...]:
        portable = v["backend"] == "portable_cpu"
        reasons: list[str] = []
        if (portable and v["runtime"] != "native") or (not portable and v["runtime"] != "executorch"): reasons.append("backend_runtime_mismatch")
        if (portable and v["delegate"] != "none") or (not portable and v["delegate"] != "xnnpack"): reasons.append("backend_delegate_mismatch")
        expected_q = {"fp32": "none", SCHEME: SCHEME, "int8": "pt2e_per_tensor_affine_per_channel_symmetric_axis0"}.get(v["precision"])
        if v["quantization_scheme"] != expected_q: reasons.append("precision_quantization_mismatch")
        expected_kernel = {
            ("portable_cpu", "fp32"): {FP32_KERNEL_ID},
            ("portable_cpu", SCHEME): {INT8_KERNEL_ID, PACKED_INT8_KERNEL_ID},
            ("executorch_xnnpack", "fp32"): {"xnnpack_delegate_linear_bias_relu"},
            ("executorch_xnnpack", "int8"): {"xnnpack_delegate_quantized_linear_bias_relu"},
        }.get((v["backend"], v["precision"]), set())
        if v["kernel"] not in expected_kernel: reasons.append("kernel_precision_mismatch")
        expected_layout = "embedded_pte" if not portable else (PACKED_B_TRANSPOSE_LAYOUT if v["kernel"] == PACKED_INT8_KERNEL_ID else "row_major_kx_n")
        if v["layout"] != expected_layout: reasons.append("layout_kernel_mismatch")
        expected_packing = PACKED_B_TRANSPOSE_SCHEME if v["layout"] == PACKED_B_TRANSPOSE_LAYOUT else "none"
        if v["packing"] != expected_packing: reasons.append("packing_layout_mismatch")
        expected_target = "aarch64" if not portable else ("cortex_a76_dotprod" if v["kernel"] == PACKED_INT8_KERNEL_ID and v["target_isa"] == "cortex_a76_dotprod" else "generic_aarch64")
        if v["target_isa"] != expected_target: reasons.append("target_kernel_mismatch")
        if v["artifact_kind"] != ("native_binary" if portable else "pte"): reasons.append("artifact_runtime_mismatch")
        if portable and v["thread_count"] != 1: reasons.append("thread_execution_mismatch")
        if not portable and v["precision"] == "fp32" and v["thread_count"] != 1: reasons.append("thread_execution_mismatch")
        if v["execution_strategy"] != ("native_fused_call" if portable else "xnnpack_delegate_region"): reasons.append("thread_execution_mismatch")
        if v["memory_placement"] != ("cpu_visible_host_memory" if portable else "runtime_managed"): reasons.append("memory_execution_mismatch")
        return tuple(sorted(set(reasons)))

    @staticmethod
    def _candidate_id(v: dict[str, str | int]) -> str:
        if v["backend"] == "portable_cpu":
            return f"slice3c:portable_cpu:{v['precision']}:{v['layout']}:{v['target_isa']}"
        return f"slice3g:executorch_xnnpack:{v['precision']}:{v['quantization_scheme']}:t{v['thread_count']}:aarch64"

    def generate(self) -> CandidateGraph:
        dimensions = self.discover_dimensions()
        nodes: list[CandidateGraphNode] = []
        provenance = ("candidate_generator.cartesian_product.v1", "dimension_catalog.slice3_complete_implementations.v1")
        for values in itertools.product(*(d.values for d in dimensions)):
            v = dict(zip((d.name for d in dimensions), values))
            reasons = self._dependency_reasons(v)
            candidate = None
            if not reasons:
                kernel = str(v["kernel"]);target = str(v["target_isa"])
                required_isa = ("asimd", "asimddp") if target == "cortex_a76_dotprod" else ()
                flags = ("-O3", "-mcpu=cortex-a76") if target == "cortex_a76_dotprod" else (("-O2",) if v["backend"] == "portable_cpu" else ())
                capability = {FP32_KERNEL_ID:"quant_kernel.none",INT8_KERNEL_ID:KERNEL_CAPABILITY,PACKED_INT8_KERNEL_ID:PACKED_INT8_KERNEL_CAPABILITY}.get(kernel,"runtime.executorch.xnnpack")
                candidate = CompleteCandidate(self._candidate_id(v),str(v["backend"]),str(v["precision"]),kernel,str(v["layout"]),"" if v["packing"] == "none" else str(v["packing"]),target,required_isa,flags,capability,str(v["runtime"]),str(v["delegate"]),int(v["thread_count"]),str(v["quantization_scheme"]),str(v["artifact_kind"]),str(v["tile"]),str(v["memory_placement"]),str(v["execution_strategy"]),str(v["measurement_evidence"]),provenance)
            nodes.append(CandidateGraphNode(self._node_id(v),v,candidate,"accepted" if candidate else "rejected_by_dependency_rules",reasons,provenance))
        return CandidateGraph("slice3.candidate_graph.v1",dimensions,tuple(nodes))


def load_codegen_capabilities(profile: dict[str, Any]) -> CodegenCapabilities:
    raw = profile.get("cpuCodegenCapabilities") or {}
    return CodegenCapabilities(
        target_id=str(profile.get("profileId", "")),
        architecture=str(raw.get("architecture", "")),
        microarchitecture=str(raw.get("microarchitecture", "")),
        isa_features=tuple(raw.get("isaFeatures") or ()),
        vector_width_bits=int(raw.get("vectorWidthBits", 0) or 0),
        supports_int8_dot_product=bool(raw.get("supportsInt8DotProduct", False)),
        supported_compiler_target_flags=tuple(raw.get("supportedCompilerTargetFlags") or ()),
        supported_kernel_capabilities=tuple(raw.get("supportedKernelCapabilities") or ()),
        supports_executorch_runtime=bool(profile.get("executorchRuntimeCapabilities", {}).get("supportsExecuTorchRuntime", False)),
        supports_xnnpack_delegate=bool(profile.get("executorchRuntimeCapabilities", {}).get("supportsXNNPACKDelegate", False)),
        available_runtime_artifacts=tuple(profile.get("executorchRuntimeCapabilities", {}).get("availableRuntimeArtifacts") or ()),
        maximum_runtime_threads=int(profile.get("executorchRuntimeCapabilities", {}).get("maximumRuntimeThreads", 1)),
        physical_compute_units=int(profile.get("hardwareExecutionProfile", {}).get("physicalComputeUnits", 1)),
        supported_executorch_candidate_variants=tuple(profile.get("executorchRuntimeCapabilities", {}).get("supportedExecuTorchCandidateVariants") or ()),
    )


@lru_cache(maxsize=1)
def generate_candidate_graph() -> CandidateGraph:
    return CandidateGenerator().generate()


def enumerate_complete_candidates() -> list[CompleteCandidate]:
    return list(generate_candidate_graph().accepted_candidates)


def candidate_legality(candidate: CompleteCandidate, caps: CodegenCapabilities, *, has_calibration_artifact: bool,
                       has_packed_artifact: bool, build_tool_flags: tuple[str, ...]) -> list[str]:
    reasons: list[str] = []
    if candidate.backend == "executorch_xnnpack":
        if not caps.supports_executorch_runtime:
            reasons.append("missing_executorch_runtime")
        if not caps.supports_xnnpack_delegate:
            reasons.append("missing_xnnpack_capability")
        if "executorch_xnnpack_runner" not in caps.available_runtime_artifacts:
            reasons.append("missing_runtime_package")
        if candidate.candidate_id not in caps.supported_executorch_candidate_variants:
            reasons.append("unsupported_thread_count")
        if candidate.thread_count > caps.maximum_runtime_threads or candidate.thread_count > caps.physical_compute_units:
            reasons.append("thread_budget_exceeded")
        if caps.architecture != "aarch64":
            reasons.append("target_mismatch")
    if candidate.codegen_target_id == "cortex_a76_dotprod":
        if caps.architecture != "aarch64":
            reasons.append("missing_target_architecture")
        if caps.microarchitecture != "cortex-a76":
            reasons.append("missing_microarchitecture_capability")
        if "asimd" not in caps.isa_features:
            reasons.append("missing_asimd")
        if "asimddp" not in caps.isa_features or not caps.supports_int8_dot_product:
            reasons.append("missing_asimddp")
    for flag in candidate.required_compiler_flags:
        if flag not in caps.supported_compiler_target_flags or flag not in build_tool_flags:
            reasons.append("unsupported_codegen_flags")
            break
    if candidate.required_kernel_capability not in caps.supported_kernel_capabilities:
        reasons.append("missing_kernel_capability")
    if candidate.precision == SCHEME and not has_calibration_artifact:
        reasons.append("missing_calibration_artifact")
    if candidate.weight_layout == PACKED_B_TRANSPOSE_LAYOUT and not has_packed_artifact:
        reasons.append("missing_packed_weight_artifact")
    return reasons


def create_build_manifest(candidate: CompleteCandidate, *, source_root: Path, output_dir: Path,
                          compiler_executable: str = "g++") -> dict[str, Any]:
    if candidate.kernel_id == FP32_KERNEL_ID:
        source = source_root / "native/cpu_kernels/portable_fused_matmul_bias_relu.cpp"
        binary = output_dir / "portable_fused_matmul_bias_relu"
        flags = ["-O2", "-std=c++17", "-pthread"]
    elif candidate.kernel_id == INT8_KERNEL_ID:
        source = source_root / "native/cpu_kernels/portable_fused_matmul_bias_relu_int8.cpp"
        binary = output_dir / "portable_fused_matmul_bias_relu_int8"
        flags = ["-O2", "-std=c++17"]
    else:
        source = source_root / "native/cpu_kernels/portable_fused_matmul_bias_relu_int8_packed_b.cpp"
        binary = output_dir / "portable_fused_matmul_bias_relu_int8_packed_b"
        flags = [*candidate.required_compiler_flags, "-std=c++17"]
    command = [compiler_executable, *flags, "-o", str(binary), str(source)]
    return {
        "schema_version": BUILD_MANIFEST_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "kernel_id": candidate.kernel_id,
        "codegen_target_id": candidate.codegen_target_id,
        "target_architecture": "aarch64" if "aarch64" in candidate.codegen_target_id or candidate.codegen_target_id != "host" else platform.machine(),
        "target_microarchitecture": "cortex-a76" if candidate.codegen_target_id == "cortex_a76_dotprod" else "",
        "required_isa_features": list(candidate.required_isa),
        "compiler_executable": compiler_executable,
        "compiler_flags": flags,
        "kernel_source": str(source),
        "output_binary": str(binary),
        "build_command": command,
        "truth_boundary": "compiler_owned_kernel_build_contract_external_cpp_compilation_not_llvm_codegen",
    }


def materialize_build_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    Path(manifest["output_binary"]).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(manifest["build_command"], check=True)
    binary = Path(manifest["output_binary"])
    out = dict(manifest)
    out["binary_sha256"] = sha256_bytes(binary.read_bytes())
    return out


def create_measurement_artifact(*, evaluation: dict[str, Any], candidate: CompleteCandidate,
                                target: CodegenCapabilities, binary_sha256: str,
                                build_manifest: dict[str, Any], packed_artifact: dict[str, Any] | None,
                                calibration_artifact: dict[str, Any], latency_key: str,
                                metrics_key: str) -> dict[str, Any]:
    latency = evaluation[latency_key]["value"]
    metrics = evaluation[metrics_key]["value"]
    payload = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "evaluation_id": evaluation["evaluation_id"],
        "workload_id": evaluation["workload_id"],
        "shape": evaluation["shape"],
        "candidate_id": candidate.candidate_id,
        "target_id": target.target_id,
        "target_architecture": target.architecture,
        "target_microarchitecture": target.microarchitecture,
        "required_isa": list(candidate.required_isa),
        "kernel_id": candidate.kernel_id,
        "binary_sha256": binary_sha256,
        "build_command": build_manifest["build_command"],
        "compiler_version": build_manifest.get("compiler_version", "external_g++"),
        "compiler_flags": build_manifest["compiler_flags"],
        "packed_artifact_id": (packed_artifact or {}).get("artifact_id", ""),
        "packed_artifact_sha256": (packed_artifact or {}).get("artifact_sha256", ""),
        "calibration_artifact_id": calibration_artifact.get("artifact_id", ""),
        "calibration_artifact_sha256": calibration_artifact.get("artifact_sha256", ""),
        "correctness_metrics": metrics,
        "latency_median_ms": latency["median_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "latency_stddev_ms": latency["stddev_ms"],
        "sample_count": len(latency["samples_ms"]),
        "thread_count": evaluation["thread_count"],
        "timestamp": evaluation["timestamp"],
        "measurement_provenance": "slice3c_pi_operator_measurement_real_native_kernel_json",
    }
    payload["measurement_artifact_id"] = "slice3c-measure-" + sha256_bytes(json.dumps(payload, sort_keys=True).encode())[:16]
    payload["measurement_sha256"] = sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return payload


def validate_measurement(measurement: dict[str, Any], candidate: CompleteCandidate, target: CodegenCapabilities,
                         *, shape: dict[str, int], binary_sha256: str,
                         packed_artifact: dict[str, Any] | None, calibration_artifact: dict[str, Any],
                         min_samples: int = 5, max_p95_over_median: float = 1.25,
                         min_cosine: float = 0.99, max_relative_l2: float = 0.05) -> list[str]:
    reasons: list[str] = []
    if not measurement:
        return ["missing_measurement_artifact"]
    if measurement.get("candidate_id") != candidate.candidate_id or measurement.get("kernel_id") != candidate.kernel_id:
        reasons.append("measurement_candidate_mismatch")
    if measurement.get("target_id") != target.target_id or measurement.get("target_architecture") != target.architecture:
        reasons.append("measurement_target_mismatch")
    if measurement.get("shape") != shape:
        reasons.append("measurement_shape_mismatch")
    if measurement.get("binary_sha256") != binary_sha256:
        reasons.append("measurement_binary_sha256_mismatch")
    if candidate.weight_layout == PACKED_B_TRANSPOSE_LAYOUT:
        if measurement.get("packed_artifact_sha256") != (packed_artifact or {}).get("artifact_sha256"):
            reasons.append("wrong_packed_artifact")
    if measurement.get("calibration_artifact_sha256") != calibration_artifact.get("artifact_sha256"):
        reasons.append("wrong_calibration_artifact")
    metrics = measurement.get("correctness_metrics") or {}
    if float(metrics.get("cosine_similarity", 0.0)) < min_cosine or float(metrics.get("relative_l2_error", 1.0)) > max_relative_l2:
        reasons.append("accuracy_gate_failed")
    if int(measurement.get("sample_count", 0)) < min_samples:
        reasons.append("missing_measurement_artifact")
    med = float(measurement.get("latency_median_ms", 0.0) or 0.0)
    p95 = float(measurement.get("latency_p95_ms", 0.0) or 0.0)
    if med <= 0.0 or p95 <= 0.0:
        reasons.append("missing_measurement_artifact")
    elif p95 / med > max_p95_over_median:
        reasons.append("performance_gate_failed")
    return reasons


def select_candidate(candidates: list[CompleteCandidate], *, target: CodegenCapabilities,
                     shape: dict[str, int], measurements: dict[str, dict[str, Any]],
                     binary_sha256_by_candidate: dict[str, str],
                     packed_artifact: dict[str, Any] | None, calibration_artifact: dict[str, Any],
                     has_packed_artifact: bool, has_calibration_artifact: bool,
                     build_tool_flags: tuple[str, ...]) -> dict[str, Any]:
    considered = []
    legal = []
    for c in candidates:
        reasons = candidate_legality(c, target, has_calibration_artifact=has_calibration_artifact,
                                     has_packed_artifact=has_packed_artifact, build_tool_flags=build_tool_flags)
        meas = measurements.get(c.candidate_id, {})
        reasons.extend(validate_measurement(meas, c, target, shape=shape,
                                            binary_sha256=binary_sha256_by_candidate.get(c.candidate_id, ""),
                                            packed_artifact=packed_artifact, calibration_artifact=calibration_artifact))
        item = {"candidate_id": c.candidate_id, "kernel_id": c.kernel_id, "rejection_reasons": sorted(set(reasons)),
                "latency_median_ms": meas.get("latency_median_ms")}
        considered.append(item)
        if not item["rejection_reasons"]:
            legal.append((float(meas["latency_median_ms"]), c, meas))
    if legal:
        legal.sort(key=lambda x: x[0])
        _, selected, measurement = legal[0]
        reason = "lowest_measured_median_latency_correctness_and_stability_gates_passed"
    else:
        selected = candidates[0]
        measurement = measurements.get(selected.candidate_id, {})
        reason = "fp32_fallback_no_non_fp32_candidate_legal"
    return {"schema_version": SELECTION_SCHEMA_VERSION, "selected_candidate_id": selected.candidate_id,
            "selected_kernel": selected.kernel_id, "selection_reason": reason,
            "selected_latency_median_ms": measurement.get("latency_median_ms"),
            "considered_candidates": considered}


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json_deterministic(path, payload)
