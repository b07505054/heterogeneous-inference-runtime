"""E2E-10: MatMulBiasReLUCostModel, registered directly into the existing
CostModelRegistry (perf_model.cost_model_registry.CostModelRegistry.register)
-- no changes needed to that registry's orchestration, since it was already
a generic dict-based registry in E2E-9.

Phase 0 audit finding: no GBDT, Ridge, or "Hybrid policy" cost model exists
anywhere in this repository for fused MatMul-Bias-ReLU (confirmed by
grepping for gbdt/xgboost/lightgbm/sklearn/ridge/hybrid across the whole
repo -- zero matches outside this project's own generic
AnalyticalFallbackCostModel naming). The strongest REAL evidence available
is deployment/execution_plan/slice3c_target_selection.select_candidate(),
which is itself a pure measured-lookup selector (lowest measured median
latency among legal candidates) over real per-shape measurement artifacts
-- not a trained/fit statistical model. This module adapts that same
measured-lookup approach against E2E-10's own evidence:
perf_model/evidence/matmul_bias_relu_e2e10.json, 9 shapes x 4 candidates,
DIRECTLY_MEASURED on a real Raspberry Pi 5 (cortex-a76) via
deployment.execution_plan.portable_cpu_kernel_adapter.dispatch_fused_matmul_bias_relu
(the real subprocess dispatcher for the real compiled kernels), not
simulated, not replayed from stale artifacts, not retrained.

Per the task's explicit instruction, MatMulBiasReLUGBDTCostModel and
MatMulBiasReLUHybridCostModel classes are NOT created here: building empty
or synthetic stand-ins for models that were never found would misrepresent
what evidence exists. Only MatMulBiasReLUCostModel (measured lookup) exists
for this slice; H3's evaluation reflects this honestly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perf_model.cost_model_registry import CostEstimate
from perf_model.implementation_candidate import ImplementationCandidate
from perf_model.operation_descriptor import OperationDescriptor, OperationFamily

DEFAULT_EVIDENCE_PATH = Path(__file__).resolve().parent / "evidence" / "matmul_bias_relu_e2e10.json"

_SLICE3C_ID_BY_KIND_TARGET = {
    ("fp32_tiled", None): "slice3c:portable_cpu:fp32:row_major_kx_n:generic_aarch64",
    ("int8_scalar", None): "slice3c:portable_cpu:int8_static_symmetric:row_major_kx_n:generic_aarch64",
    ("int8_packed_b", "generic_aarch64"): "slice3c:portable_cpu:int8_static_symmetric:packed_b_transposed_nxk:generic_aarch64",
    ("int8_packed_b", "cortex_a76_dotprod"): "slice3c:portable_cpu:int8_static_symmetric:packed_b_transposed_nxk:cortex_a76_dotprod",
}


class MatMulBiasReLUCostModel:
    model_id = "matmul_bias_relu_measured_lookup_v1"
    operation_family = OperationFamily.MATMUL_BIAS_RELU

    def __init__(self, evidence_path: Path = DEFAULT_EVIDENCE_PATH):
        self.evidence_path = evidence_path
        raw = json.loads(evidence_path.read_text())
        self.hardware_identity = raw.get("hardware_identity", {})
        self.target_profile_id = raw.get("target_profile_id")
        self._by_shape: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._shapes: set[tuple[int, int, int]] = set()
        for row in raw["results"]:
            s = row["shape"]
            key = (s["M"], s["N"], s["K"])
            self._by_shape[key] = row
            self._shapes.add(key)

    def _slice3c_id(self, candidate: ImplementationCandidate) -> str:
        kind = candidate.implementation_kind.value if hasattr(candidate.implementation_kind, "value") else str(candidate.implementation_kind)
        target_isa = candidate.parameters.get("target_isa") if kind != "fp32_tiled" and kind != "int8_scalar" else None
        key = (kind, target_isa) if kind == "int8_packed_b" else (kind, None)
        found = _SLICE3C_ID_BY_KIND_TARGET.get(key)
        if found is None:
            raise KeyError(f"no evidence mapping for candidate kind={kind} target_isa={target_isa}")
        return found

    def _nearest_shape(self, m: int, n: int, k: int) -> tuple[int, int, int]:
        return min(self._shapes, key=lambda s: (abs(s[0] - m) + abs(s[1] - n) + abs(s[2] - k)))

    def predict(self, operation: OperationDescriptor, candidate: ImplementationCandidate) -> CostEstimate:
        payload = operation.payload
        shape = (payload.M, payload.N, payload.K)
        slice3c_id = self._slice3c_id(candidate)

        exact = shape in self._shapes
        lookup_shape = shape if exact else self._nearest_shape(*shape)
        row = self._by_shape[lookup_shape]
        measurement = row["measurements"].get(slice3c_id)
        if measurement is None:
            raise KeyError(f"no measurement for {slice3c_id} at shape {lookup_shape}")

        latency = measurement["latency_median_ms"]
        note = (f"Pi5 cortex-a76 measured_median_ms at shape={lookup_shape} for {slice3c_id} "
                f"(target_profile={self.target_profile_id}, hardware={self.hardware_identity})")
        if not exact:
            note += f"; requested shape {shape} not measured, using nearest measured shape {lookup_shape}"
        return CostEstimate(
            candidate_id=candidate.candidate_id, predicted_latency_ms=latency,
            source="measured_lookup" if exact else "measured_interpolation",
            confidence=0.9 if exact else 0.35, is_extrapolation=not exact, evidence_note=note,
        )

    def measured_winner(self, m: int, n: int, k: int) -> tuple[str, float] | None:
        row = self._by_shape.get((m, n, k))
        if row is None:
            return None
        best_id = min(row["measurements"], key=lambda cid: row["measurements"][cid]["latency_median_ms"])
        return best_id, row["measurements"][best_id]["latency_median_ms"]

    def accuracy_gate_passed(self, m: int, n: int, k: int, slice3c_candidate_id: str) -> bool | None:
        row = self._by_shape.get((m, n, k))
        if row is None:
            return None
        met = row["measurements"].get(slice3c_candidate_id, {}).get("correctness_metrics")
        if met is None:
            return None
        return met["cosine_similarity"] >= 0.99 and met["relative_l2_error"] <= 0.05
