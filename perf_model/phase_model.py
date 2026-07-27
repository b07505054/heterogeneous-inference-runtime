"""Level 2/3: phase equations converting Level-1 op/traffic counts into ms.

Exactly two free scalar constants exist in this whole module:
  effective_flops_per_second        -- calibrated from ONE real compute-bound
                                        measurement (prefill phase)
  effective_bandwidth_bytes_per_second -- calibrated from ONE real
                                        memory-bound measurement (decode phase)

Both are calibrated once (from real /metrics phase histograms on the dev
GPU) and then reused, unmodified, to predict every other workload/candidate
in this slice. This is a "calibrated constant" per perf_model.schema's
method taxonomy -- explicitly NOT a fitted/learned model (no regression, no
per-candidate refitting). Fixed per-phase overhead constants are held at
0 ms in v1 and any systematic residual is left for error attribution
(RUNTIME_OVERHEAD_ERROR) rather than absorbed into the constants.

Three distinct predictions are produced, per the required design, and must
never be used interchangeably:
  - single-request latency  (batch_size == 1, queue_ms == 0)
  - concurrent serving      (batch_size == admitted_concurrency, positional queue model)
  - steady-state throughput (same batch_size, reported as tokens/s, not ms)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perf_model import compute_model
from perf_model.schema import ModelFeatures, MetricEstimate


@dataclass(frozen=True)
class EffectiveThroughput:
    flops_per_second: float | None
    bandwidth_bytes_per_second: float | None
    fixed_prefill_overhead_ms: float
    fixed_decode_overhead_ms: float
    source: str  # "unavailable" | "derived_from_phase_measurement"
    calibrated_from: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flops_per_second": self.flops_per_second,
            "bandwidth_bytes_per_second": self.bandwidth_bytes_per_second,
            "fixed_prefill_overhead_ms": self.fixed_prefill_overhead_ms,
            "fixed_decode_overhead_ms": self.fixed_decode_overhead_ms,
            "source": self.source,
            "calibrated_from": self.calibrated_from,
        }


UNCALIBRATED = EffectiveThroughput(None, None, 0.0, 0.0, "unavailable", None)


def calibrate(
    *,
    prefill_flops: int,
    measured_prefill_ms: float,
    decode_memory_bytes_batch1: int,
    measured_decode_token_ms: float,
    calibrated_from: dict[str, Any],
) -> EffectiveThroughput:
    """Two-point roofline calibration: prefill measurement assumed
    compute-bound (solves flops/s), decode(batch=1) measurement assumed
    memory-bound (solves bytes/s). Both are approximations -- prefill also
    has some memory traffic and decode also has some compute -- the
    resulting constants absorb that as part of "effective", not "peak".
    """
    flops_per_second = prefill_flops / (measured_prefill_ms / 1000.0) if measured_prefill_ms > 0 else None
    bandwidth = (
        decode_memory_bytes_batch1 / (measured_decode_token_ms / 1000.0)
        if measured_decode_token_ms > 0 else None
    )
    return EffectiveThroughput(
        flops_per_second=flops_per_second,
        bandwidth_bytes_per_second=bandwidth,
        fixed_prefill_overhead_ms=0.0,
        fixed_decode_overhead_ms=0.0,
        source="derived_from_phase_measurement",
        calibrated_from=calibrated_from,
    )


def _truth_boundary(throughput: EffectiveThroughput) -> str:
    return (
        "analytical_with_phase_derived_constant"
        if throughput.source == "derived_from_phase_measurement"
        else "unsupported_no_estimate"
    )


def predict_prefill_ms(
    model: ModelFeatures, prompt_tokens: int, weight_bytes: int, throughput: EffectiveThroughput
) -> tuple[MetricEstimate, dict[str, Any]]:
    ops = compute_model.prefill_op_counts(model, prompt_tokens)
    traffic_bytes = compute_model.prefill_memory_traffic_bytes(weight_bytes=weight_bytes)
    breakdown = {"op_counts": ops.to_dict(), "memory_traffic_bytes": traffic_bytes}
    if throughput.source != "derived_from_phase_measurement":
        return MetricEstimate(None, "unsupported", "unsupported_no_estimate",
                               "effective_flops_per_second not yet calibrated"), breakdown
    compute_ms = (ops.total_flops / throughput.flops_per_second) * 1000.0
    memory_ms = (traffic_bytes / throughput.bandwidth_bytes_per_second) * 1000.0
    total = compute_ms + memory_ms + throughput.fixed_prefill_overhead_ms
    breakdown.update({"compute_ms": compute_ms, "memory_ms": memory_ms,
                       "fixed_overhead_ms": throughput.fixed_prefill_overhead_ms})
    return MetricEstimate(total, "analytical", _truth_boundary(throughput),
                           "compute_ms + memory_ms + fixed_overhead_ms"), breakdown


def predict_decode_token_ms(
    model: ModelFeatures, *, kv_context_tokens: int, weight_bytes: int, kv_bytes_per_token: int,
    batch_size: int, throughput: EffectiveThroughput,
) -> tuple[MetricEstimate, dict[str, Any]]:
    """batch_size == 1 -> single-request TPOT. batch_size > 1 -> the per-step
    time for a continuous-batching iteration advancing `batch_size` sequences
    together; this equals the per-request TPOT observed by every sequence in
    that step (vLLM advances all admitted sequences one token per iteration),
    NOT that time divided by batch_size.
    """
    single_ops = compute_model.decode_step_op_counts(model, kv_context_tokens)
    batch_flops = single_ops.total_flops * batch_size
    memory_bytes = weight_bytes + kv_bytes_per_token * kv_context_tokens * batch_size
    breakdown = {"op_counts_single_sequence": single_ops.to_dict(), "batch_size": batch_size,
                 "batch_flops": batch_flops, "memory_traffic_bytes": memory_bytes}
    if throughput.source != "derived_from_phase_measurement":
        return MetricEstimate(None, "unsupported", "unsupported_no_estimate",
                               "effective_bandwidth_bytes_per_second not yet calibrated"), breakdown
    compute_ms = (batch_flops / throughput.flops_per_second) * 1000.0
    memory_ms = (memory_bytes / throughput.bandwidth_bytes_per_second) * 1000.0
    total = compute_ms + memory_ms + throughput.fixed_decode_overhead_ms
    breakdown.update({"compute_ms": compute_ms, "memory_ms": memory_ms,
                       "fixed_overhead_ms": throughput.fixed_decode_overhead_ms})
    return MetricEstimate(total, "analytical", _truth_boundary(throughput),
                           "compute_ms + KV_and_weight_read_ms + fixed_overhead_ms"), breakdown


def predict_queue_ms_positional(request_index: int, admitted_concurrency: int, avg_service_ms: float) -> float:
    """First-order closed-loop queue approximation: requests beyond the
    admitted concurrency wait in round-robin batches for a slot. This is a
    deliberately simple analytical approximation, not a queueing-theory fit;
    it ignores max_num_batched_tokens interaction and heterogeneous request
    lengths. Flagged explicitly as QUEUE_MODEL_ERROR risk in calibration_row.
    """
    if admitted_concurrency <= 0:
        return 0.0
    return (request_index // admitted_concurrency) * avg_service_ms


def predict_ttft_ms(
    *, queue_ms: float, prefill_estimate: MetricEstimate, first_decode_step_estimate: MetricEstimate,
    request_overhead_ms: float,
) -> MetricEstimate:
    if prefill_estimate.value is None or first_decode_step_estimate.value is None:
        return MetricEstimate(None, "unsupported", "unsupported_no_estimate",
                               "prefill or first-decode-step estimate unavailable")
    total = queue_ms + prefill_estimate.value + first_decode_step_estimate.value + request_overhead_ms
    return MetricEstimate(total, "analytical", prefill_estimate.truth_boundary,
                           "queue_ms + prefill_ms + first_decode_step_ms + request_overhead_ms")


def predict_tpot_ms(decode_token_estimate: MetricEstimate) -> MetricEstimate:
    return MetricEstimate(decode_token_estimate.value, decode_token_estimate.method,
                           decode_token_estimate.truth_boundary, "identical to predicted_decode_token_ms")


def predict_e2e_ms(ttft_estimate: MetricEstimate, output_tokens: int, tpot_estimate: MetricEstimate) -> MetricEstimate:
    if ttft_estimate.value is None or tpot_estimate.value is None:
        return MetricEstimate(None, "unsupported", "unsupported_no_estimate", "TTFT or TPOT unavailable")
    total = ttft_estimate.value + max(output_tokens - 1, 0) * tpot_estimate.value
    return MetricEstimate(total, "analytical", ttft_estimate.truth_boundary,
                           "ttft_ms + max(output_tokens-1,0) * tpot_ms")


def predict_output_tokens_per_second_single(tpot_estimate: MetricEstimate) -> MetricEstimate:
    if not tpot_estimate.value:
        return MetricEstimate(None, "unsupported", "unsupported_no_estimate", "tpot unavailable or zero")
    return MetricEstimate(1000.0 / tpot_estimate.value, tpot_estimate.method, tpot_estimate.truth_boundary,
                           "single_request_only; not steady-state throughput")


def predict_output_tokens_per_second_concurrent(
    batch_size: int, decode_step_estimate: MetricEstimate
) -> MetricEstimate:
    """Steady-state aggregate throughput: `batch_size` sequences each advance one
    token per step of duration decode_step_estimate.value ms."""
    if not decode_step_estimate.value:
        return MetricEstimate(None, "unsupported", "unsupported_no_estimate", "decode step estimate unavailable or zero")
    tokens_per_sec = batch_size * 1000.0 / decode_step_estimate.value
    return MetricEstimate(tokens_per_sec, decode_step_estimate.method, decode_step_estimate.truth_boundary,
                           "steady_state_concurrent_only; not single-request rate")
