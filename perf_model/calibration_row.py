"""Normalized calibration row builder + predicted-vs-measured error attribution.

A calibration row is the unit of evidence this whole slice produces: one
(model, hardware, workload, candidate) experiment, with predictions,
measurements, adherence status, and an explainable error attribution --
never just "error was large".
"""
from __future__ import annotations

import statistics
from typing import Any

ERROR_ATTRIBUTION_CLASSES = (
    "MODEL_FEATURE_ERROR",
    "HARDWARE_FEATURE_ERROR",
    "OPERATION_COUNT_ERROR",
    "EFFECTIVE_THROUGHPUT_ERROR",
    "MEMORY_TRAFFIC_ERROR",
    "RUNTIME_OVERHEAD_ERROR",
    "QUEUE_MODEL_ERROR",
    "BATCHING_INTERACTION_ERROR",
    "VLLM_DERIVED_CONFIG_DIFFERENCE",
    "WARMUP_OR_COMPILATION_NOISE",
    "MEASUREMENT_INSTABILITY",
    "UNSUPPORTED_PREDICTION_TERM",
)

INSTABILITY_RELATIVE_MAD_THRESHOLD = 0.25


def median_abs_deviation(values: list[float]) -> float:
    if not values:
        return 0.0
    med = statistics.median(values)
    return statistics.median(abs(v - med) for v in values)


def distribution_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None, "mad": None,
                "relative_mad": None, "raw": []}
    ordered = sorted(values)
    med = statistics.median(ordered)
    mad = median_abs_deviation(ordered)
    idx95 = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(ordered), "median": med, "p95": ordered[idx95],
        "min": ordered[0], "max": ordered[-1], "mad": mad,
        "relative_mad": (mad / med) if med else None, "raw": ordered,
    }


def compute_error(predicted_value: float | int | bool | None, measured_value: float | None) -> dict[str, Any]:
    if predicted_value is None or measured_value is None or isinstance(predicted_value, bool):
        return {"absolute_error": None, "relative_error": None, "prediction_available": predicted_value is not None,
                "measurement_available": measured_value is not None}
    abs_err = abs(float(predicted_value) - float(measured_value))
    rel_err = abs_err / abs(measured_value) if measured_value else None
    return {"absolute_error": abs_err, "relative_error": rel_err,
            "prediction_available": True, "measurement_available": True}


def attribute_error(
    *, metric_name: str, error: dict[str, Any], measured_distribution: dict[str, Any] | None,
    adherence_mismatches: list[str], concurrency: int, resolved_max_num_seqs: int | None,
    warmup_count: int, relative_error_threshold: float = 0.30,
) -> list[dict[str, str]]:
    """Returns a ranked list of {category, explanation}. Multiple categories may
    apply; order is most-to-least likely given available evidence. This is a
    transparent rule-based attributor, not a classifier fit on data.
    """
    findings: list[dict[str, str]] = []

    if error.get("prediction_available") is False:
        findings.append({"category": "UNSUPPORTED_PREDICTION_TERM",
                          "explanation": f"{metric_name} has no calibrated prediction yet (effective throughput unavailable)."})
        return findings

    if error.get("measurement_available") is False:
        return findings  # nothing to attribute without a measurement

    rel = error.get("relative_error")
    if rel is None or rel < relative_error_threshold:
        return findings  # error small enough not to attribute

    if warmup_count <= 0:
        findings.append({"category": "WARMUP_OR_COMPILATION_NOISE",
                          "explanation": "No warmup requests were recorded before this measurement; "
                                         "first-call CUDA/compile/allocator effects may inflate the measurement."})

    if adherence_mismatches:
        findings.append({"category": "VLLM_DERIVED_CONFIG_DIFFERENCE",
                          "explanation": f"Resolved runtime config differs from requested for: {', '.join(adherence_mismatches)}; "
                                         "the prediction was computed against the requested value."})

    if measured_distribution and measured_distribution.get("relative_mad") is not None:
        if measured_distribution["relative_mad"] > INSTABILITY_RELATIVE_MAD_THRESHOLD:
            findings.append({"category": "MEASUREMENT_INSTABILITY",
                              "explanation": f"Measured {metric_name} relative MAD "
                                             f"{measured_distribution['relative_mad']:.2f} exceeds "
                                             f"{INSTABILITY_RELATIVE_MAD_THRESHOLD}; point comparison against the "
                                             "median may not be a fair test of the model."})

    if resolved_max_num_seqs is not None and concurrency > resolved_max_num_seqs:
        findings.append({"category": "BATCHING_INTERACTION_ERROR",
                          "explanation": f"concurrency={concurrency} exceeds resolved max_num_seqs="
                                         f"{resolved_max_num_seqs}; admission queueing occurs and the v1 compute/"
                                         "memory model does not capture batched-decode compute nonlinearity."})
        if metric_name in ("predicted_ttft_ms", "predicted_e2e_ms"):
            findings.append({"category": "QUEUE_MODEL_ERROR",
                              "explanation": "TTFT/E2E under admission-limited concurrency depend on the v1 "
                                             "positional round-robin queue approximation, which ignores "
                                             "max_num_batched_tokens interaction and request-length heterogeneity."})

    if metric_name in ("predicted_decode_token_ms", "predicted_tpot_ms"):
        findings.append({"category": "EFFECTIVE_THROUGHPUT_ERROR",
                          "explanation": "Decode-phase prediction depends entirely on the single calibrated "
                                         "effective_bandwidth_bytes_per_second constant; residual error here is "
                                         "most likely a miscalibrated or workload-dependent bandwidth constant, "
                                         "not a wrong operation count (decode FLOPs are small)."})
    if metric_name in ("predicted_prefill_ms", "predicted_ttft_ms"):
        findings.append({"category": "OPERATION_COUNT_ERROR",
                          "explanation": "Prefill prediction depends on both FLOP count and the compute-bound "
                                         "assumption; short prompts may be more memory- or overhead-bound than "
                                         "the model assumes, which would show up here as compute/memory split "
                                         "error rather than a wrong FLOP count."})
        findings.append({"category": "MEMORY_TRAFFIC_ERROR",
                          "explanation": "Prefill memory traffic is modeled as a single weight-bytes read; "
                                         "activation and workspace traffic are not counted."})

    if not findings:
        findings.append({"category": "RUNTIME_OVERHEAD_ERROR",
                          "explanation": "No specific cause identified from available evidence; v1 holds fixed "
                                         "per-phase overhead constants at 0 ms, so unexplained residual is "
                                         "attributed to unmodeled fixed overhead by default."})
    return findings


def build_calibration_row(
    *, identity: dict[str, Any], configuration: dict[str, Any], predictions: dict[str, Any],
    measurements: dict[str, Any], errors: dict[str, Any],
) -> dict[str, Any]:
    return {
        "identity": identity,
        "configuration": configuration,
        "predictions": predictions,
        "measurements": measurements,
        "errors": errors,
        "calibration_row_schema_version": "perf_model.calibration_row.v1",
    }
