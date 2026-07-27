"""E2E-9: per-family cost model interface + registry.

Deliberately NOT one universal cost model. Each operation family gets its
own cost model class trained/adapted from that family's own real evidence
(RMSNormCostModel <- perf_model/evidence/rmsnorm_benchmark_e2e9.json measured
sweep; TinyMLinearCostModel <- E2E-7/E2E-8 measured LM-head evidence). A
generic AnalyticalFallbackCostModel exists only as a last-resort, always-
available estimator when a family-specific model can't answer (e.g. an
unmeasured shape with no measured neighbor) -- it is intentionally crude
(pure bytes/flops roofline arithmetic) and is never treated as more reliable
than a measured-lookup model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from perf_model.implementation_candidate import ImplementationCandidate
from perf_model.operation_descriptor import OperationDescriptor, OperationFamily


@dataclass(frozen=True)
class CostEstimate:
    candidate_id: str
    predicted_latency_ms: float
    source: str  # "measured_lookup" | "measured_interpolation" | "analytical_fallback"
    confidence: float  # 0..1, informal
    is_extrapolation: bool
    evidence_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "predicted_latency_ms": self.predicted_latency_ms,
            "source": self.source, "confidence": self.confidence, "is_extrapolation": self.is_extrapolation,
            "evidence_note": self.evidence_note,
        }


class OperationCostModel(Protocol):
    model_id: str
    operation_family: OperationFamily

    def predict(self, operation: OperationDescriptor, candidate: ImplementationCandidate) -> CostEstimate: ...


class AnalyticalFallbackCostModel:
    """Crude bytes/flops roofline estimate. Used only when a family-specific
    measured model has no applicable evidence at all. Deliberately generic
    across families -- this is the ONE place genuinely-universal logic is
    allowed, because it makes no claim of accuracy (confidence is always low)."""
    model_id = "analytical_fallback_v1"
    operation_family = None  # applies to any family

    ASSUMED_BANDWIDTH_GBPS = 150.0  # conservative GTX 1650 Max-Q memory-bound assumption
    ASSUMED_TFLOPS = 2.0

    def predict(self, operation: OperationDescriptor, candidate: ImplementationCandidate) -> CostEstimate:
        payload = operation.payload
        if operation.common.operation_family == OperationFamily.RMS_NORM:
            bytes_total = 2 * payload.token_count * payload.hidden_size * 4  # read+write, fp32
            flops_total = 4 * payload.token_count * payload.hidden_size
        else:
            bytes_total = (payload.M * payload.K + payload.K * payload.N + payload.M * payload.N) * 2
            flops_total = 2 * payload.M * payload.N * payload.K
        mem_ms = bytes_total / (self.ASSUMED_BANDWIDTH_GBPS * 1e9) * 1e3
        compute_ms = flops_total / (self.ASSUMED_TFLOPS * 1e12) * 1e3
        predicted = max(mem_ms, compute_ms)
        return CostEstimate(candidate.candidate_id, predicted, "analytical_fallback", confidence=0.15,
                             is_extrapolation=True,
                             evidence_note="no measured evidence available; roofline estimate only, low confidence")


class CostModelRegistry:
    def __init__(self):
        self._models: dict[OperationFamily, OperationCostModel] = {}
        self._fallback = AnalyticalFallbackCostModel()
        self._unavailable_reasons: dict[OperationFamily, str] = {}

    def register(self, family: OperationFamily, model: OperationCostModel) -> None:
        self._models[family] = model

    def mark_unavailable(self, family: OperationFamily, reason: str) -> None:
        self._unavailable_reasons[family] = reason

    def get_model(self, family: OperationFamily) -> OperationCostModel:
        model = self._models.get(family)
        if model is not None:
            return model
        return self._fallback

    def predict(self, operation: OperationDescriptor, candidate: ImplementationCandidate) -> CostEstimate:
        model = self.get_model(operation.common.operation_family)
        try:
            return model.predict(operation, candidate)
        except Exception:
            return self._fallback.predict(operation, candidate)

    def registered_families(self) -> list[OperationFamily]:
        return list(self._models.keys())
