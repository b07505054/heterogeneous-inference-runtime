"""E2E-9: TinyMLinearCostModel, wrapping the REAL E2E-7 isolated GEMM-vs-GEMV
benchmark evidence (scripts/isolated_gemm_benchmark.py, DIRECTLY_MEASURED on
the GTX 1650 Max-Q) for the LM-head operation subtype. This is the same
measured table already relied on by scripts/analyze_perf_model_results_e2e8.py
(E2E7_LM_HEAD_ISOLATED_MS) -- duplicated here (not imported) because
perf_model/ is the lower-level package that scripts/ depends on, not the
reverse; the numbers are identical and both cite the same E2E-7 artifact.

A measured-lookup table, not a learned/fit model: exact M values return the
exact measured median_ms; M values between measured points use linear
interpolation; M values outside [1,8] are extrapolated with low confidence.

Real end-to-end realization evidence (measured serving TPOT saved vs E2E-7's
isolated-operation prediction) is currently DIRECTLY_MEASURED only at M=2
from E2E-8 (baseline 168.94ms -> optimized 133.12ms TPOT, realization ratio
0.944); M=1,4,8 realization ratios are INFERENCE (not yet re-measured in this
slice) until Phase 12B's unified-selector re-run produces them.
"""
from __future__ import annotations

from typing import Any

from perf_model.cost_model_registry import CostEstimate
from perf_model.implementation_candidate import ImplementationCandidate, ImplementationKind
from perf_model.operation_descriptor import OperationDescriptor, OperationFamily

# Real E2E-7 isolated benchmark: scripts/isolated_gemm_benchmark.py, SHAPES["lm_head"],
# candidates "1_default_linear" vs "2_gemv_loop", median_ms, M = batch_size.
E2E7_LM_HEAD_ISOLATED_MS: dict[int, dict[str, float]] = {
    1: {"default": 2.7546, "gemv": 2.7766},
    2: {"default": 43.6263, "gemv": 5.6709},
    3: {"default": 43.6306, "gemv": 8.9406},
    4: {"default": 43.6288, "gemv": 12.1059},
    6: {"default": 43.9941, "gemv": 18.3890},
    8: {"default": 43.9992, "gemv": 25.1245},
}

# Real E2E-8 end-to-end realization ratios, DIRECTLY_MEASURED so far only at M=2.
E2E8_REALIZATION_RATIO: dict[int, float] = {2: 0.944}


class TinyMLinearCostModel:
    model_id = "tiny_m_linear_measured_lookup_v1"
    operation_family = OperationFamily.LINEAR

    def __init__(self, table: dict[int, dict[str, float]] = None):
        self.table = table or E2E7_LM_HEAD_ISOLATED_MS
        self._measured_ms = sorted(self.table.keys())

    def _interp(self, m: int, kind: str) -> tuple[float, bool]:
        if m in self.table:
            return self.table[m][kind], False
        lower = [x for x in self._measured_ms if x < m]
        upper = [x for x in self._measured_ms if x > m]
        if lower and upper:
            lo, hi = max(lower), min(upper)
            lo_v, hi_v = self.table[lo][kind], self.table[hi][kind]
            frac = (m - lo) / (hi - lo)
            return lo_v + frac * (hi_v - lo_v), True
        nearest = min(self._measured_ms, key=lambda x: abs(x - m))
        return self.table[nearest][kind], True

    def predict(self, operation: OperationDescriptor, candidate: ImplementationCandidate) -> CostEstimate:
        payload = operation.payload
        m = payload.M
        kind = "gemv" if candidate.implementation_kind == ImplementationKind.LINEAR_ROW_WISE_GEMV else "default"
        latency, is_extrap = self._interp(m, kind)
        exact = m in self.table
        note = f"E2E-7 isolated lm_head {kind} median_ms at M={m}"
        if not exact:
            note += " (interpolated/extrapolated from measured M values " + str(self._measured_ms) + ")"
        ratio = E2E8_REALIZATION_RATIO.get(m)
        if ratio is not None:
            note += f"; E2E-8 measured end-to-end realization_ratio={ratio}"
        return CostEstimate(candidate.candidate_id, latency,
                             "measured_lookup" if exact else "measured_interpolation",
                             confidence=0.9 if exact else 0.5, is_extrapolation=not exact, evidence_note=note)
