"""E2E-9: RMSNormCostModel, adapted from the REAL measured RMSNorm CUDA
benchmark sweep (perf_model/evidence/rmsnorm_benchmark_e2e9.json,
DIRECTLY_MEASURED on the GTX 1650 Max-Q via the existing, unmodified
scripts/benchmark_rmsnorm_cuda.py and cuda_transformer_kernels/rmsnorm_kernel.cu).

This is a measured-lookup adapter, NOT a trained/fit statistical model: for
shapes present in the sweep it returns the exact measured custom_median_ms
per (tokens, hidden, block_size); for shapes absent from the sweep it falls
back to the nearest measured shape (by token count, then hidden size) and
flags the estimate as an extrapolation with reduced confidence. This
intentionally reuses -- rather than replaces or retrains -- the evidence
already produced by the existing RMSNorm benchmark infrastructure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perf_model.cost_model_registry import CostEstimate
from perf_model.implementation_candidate import ImplementationCandidate
from perf_model.operation_descriptor import OperationDescriptor, OperationFamily

DEFAULT_EVIDENCE_PATH = Path(__file__).resolve().parent / "evidence" / "rmsnorm_benchmark_e2e9.json"


class RMSNormCostModel:
    model_id = "rmsnorm_measured_lookup_v1"
    operation_family = OperationFamily.RMS_NORM

    def __init__(self, evidence_path: Path = DEFAULT_EVIDENCE_PATH):
        self.evidence_path = evidence_path
        raw = json.loads(evidence_path.read_text())
        self.exact_candidates: list[dict[str, Any]] = raw["exact_candidates"]
        self.source_hash = raw["exact_candidates"][0]["source_hash"] if self.exact_candidates else None
        self._by_shape_block: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._by_shape_fallback: dict[tuple[int, int], float] = {}
        self._shapes: set[tuple[int, int]] = set()
        for row in self.exact_candidates:
            key = (row["tokens"], row["hidden"], row["block_size"])
            self._by_shape_block[key] = row
            self._shapes.add((row["tokens"], row["hidden"]))
            self._by_shape_fallback[(row["tokens"], row["hidden"])] = row["fallback_median_ms"]

    def _nearest_shape(self, tokens: int, hidden: int) -> tuple[int, int]:
        return min(self._shapes, key=lambda s: (abs(s[0] - tokens), abs(s[1] - hidden)))

    def predict(self, operation: OperationDescriptor, candidate: ImplementationCandidate) -> CostEstimate:
        payload = operation.payload
        tokens, hidden = payload.token_count, payload.hidden_size

        if candidate.parameters.get("is_eager_fallback"):
            shape = (tokens, hidden) if (tokens, hidden) in self._shapes else self._nearest_shape(tokens, hidden)
            is_extrap = shape != (tokens, hidden)
            latency = self._by_shape_fallback[shape]
            return CostEstimate(candidate.candidate_id, latency,
                                 "measured_interpolation" if is_extrap else "measured_lookup",
                                 confidence=0.5 if is_extrap else 0.9, is_extrapolation=is_extrap,
                                 evidence_note=f"eager torch_rmsnorm fallback_median_ms at shape={shape}, "
                                               f"source_hash={self.source_hash}")

        block_size = candidate.parameters.get("block_size")
        exact_key = (tokens, hidden, block_size)
        if exact_key in self._by_shape_block:
            row = self._by_shape_block[exact_key]
            return CostEstimate(candidate.candidate_id, row["custom_median_ms"], "measured_lookup",
                                 confidence=0.9, is_extrapolation=False,
                                 evidence_note=f"exact measured shape=({tokens},{hidden}) block={block_size}, "
                                               f"source_hash={row['source_hash']}")

        nearest = self._nearest_shape(tokens, hidden)
        nearest_key = (nearest[0], nearest[1], block_size)
        row = self._by_shape_block.get(nearest_key)
        if row is None:
            raise KeyError(f"no measured data for block_size={block_size} at any shape")
        return CostEstimate(candidate.candidate_id, row["custom_median_ms"], "measured_interpolation",
                             confidence=0.4, is_extrapolation=True,
                             evidence_note=f"requested shape=({tokens},{hidden}) not measured; "
                                           f"using nearest measured shape={nearest}, block={block_size}, "
                                           f"source_hash={row['source_hash']}")

    def measured_winner(self, tokens: int, hidden: int) -> tuple[int, float] | None:
        """Ground-truth best block size at a measured shape, for regret computation."""
        rows = [(bs, r["custom_median_ms"]) for (t, h, bs), r in self._by_shape_block.items()
                if t == tokens and h == hidden]
        if not rows:
            return None
        return min(rows, key=lambda x: x[1])
