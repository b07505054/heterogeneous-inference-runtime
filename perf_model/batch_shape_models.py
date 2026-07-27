"""E2E-6: batch-shape decode-latency models (N-S). Plain OLS/median fits
only (numpy.linalg.lstsq or statistics.median), consistent with every prior
slice's "no Ridge/GBDT/neural" constraint.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class FittedModel:
    name: str
    params: dict[str, Any]
    feature_names: list[str]
    n_calibration_rows: int
    calibration_batch_sizes: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "feature_names": self.feature_names,
                "n_calibration_rows": self.n_calibration_rows, "calibration_batch_sizes": self.calibration_batch_sizes}


def _target_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("batch_step_ms") is not None]


def _cal_sizes(rows: list[dict]) -> list[int]:
    return sorted({r["batch_size"] for r in rows})


def _lstsq(X, y) -> np.ndarray:
    coeffs, *_ = np.linalg.lstsq(np.array(X, dtype=float), np.array(y, dtype=float), rcond=None)
    return coeffs


# --- Model N: binary batch transition ---

def fit_model_n(rows: list[dict]) -> FittedModel:
    usable = _target_rows(rows)
    c1_rows = [r["batch_step_ms"] for r in usable if r["batch_size"] == 1]
    cm_rows = [r["batch_step_ms"] for r in usable if r["batch_size"] >= 2]
    c1 = statistics.median(c1_rows) if c1_rows else 0.0
    c_multi = statistics.median(cm_rows) if cm_rows else 0.0
    return FittedModel("N_binary_transition", {"C1": c1, "C_multi": c_multi}, ["batch_eq_1", "batch_ge_2"],
                        len(usable), _cal_sizes(usable))


def predict_model_n(params: dict, *, batch_size: int, **_) -> float:
    return params["C1"] if batch_size == 1 else params["C_multi"]


# --- Model O: linear batch scaling ---

def fit_model_o(rows: list[dict]) -> FittedModel:
    usable = _target_rows(rows)
    X = [[1.0, r["batch_size"]] for r in usable]
    y = [r["batch_step_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0, 0.0])
    return FittedModel("O_linear_scaling", {"C_fixed": float(coeffs[0]), "C_sequence": float(coeffs[1])},
                        ["intercept", "batch_size"], len(usable), _cal_sizes(usable))


def predict_model_o(params: dict, *, batch_size: int, **_) -> float:
    return params["C_fixed"] + batch_size * params["C_sequence"]


# --- Model P: piecewise graph-bucket ---

def fit_model_p(rows: list[dict], capture_sizes: list[int]) -> FittedModel:
    usable = _target_rows(rows)

    def bucket(bs: int) -> int:
        candidates = [c for c in capture_sizes if c >= bs]
        return min(candidates) if candidates else max(capture_sizes) if capture_sizes else bs

    buckets: dict[int, list[float]] = {}
    for r in usable:
        buckets.setdefault(bucket(r["batch_size"]), []).append(r["batch_step_ms"])
    c_bucket = {str(b): statistics.median(vals) for b, vals in buckets.items()}
    return FittedModel("P_graph_bucket", {"capture_sizes": capture_sizes, "C_bucket": c_bucket},
                        ["graph_bucket"], len(usable), _cal_sizes(usable))


def predict_model_p(params: dict, *, batch_size: int, **_) -> float:
    capture_sizes = params["capture_sizes"]
    candidates = [c for c in capture_sizes if c >= batch_size]
    bucket = min(candidates) if candidates else (max(capture_sizes) if capture_sizes else batch_size)
    return params["C_bucket"].get(str(bucket), 0.0)


# --- Model Q: roofline-inspired ---

def fit_model_q(rows: list[dict]) -> FittedModel:
    """Calibrates effective_compute_rate and effective_bandwidth from the
    batch=1 row of THIS experiment (compute-bound vs memory-bound split
    unknown a priori for decode, so both terms are fit jointly against the
    batch=1 measurement assuming it's the cleanest, least-contended sample)."""
    b1 = next((r for r in rows if r["batch_size"] == 1 and r.get("batch_step_ms")), None)
    if b1 is None or not b1.get("decode_flops") or not b1.get("decode_bytes"):
        return FittedModel("Q_roofline", {"effective_compute_flops_per_s": None, "effective_bandwidth_bytes_per_s": None,
                                           "fixed_overhead_ms": 0.0, "source": "unavailable"}, [], 0, [])
    # Attribute the batch=1 step time entirely to whichever term the FLOP/byte
    # ratio suggests is dominant (decode is traditionally memory-bound); rate
    # is solved so the memory term alone reproduces the measured step time.
    bw = b1["decode_bytes"] / (b1["batch_step_ms"] / 1000.0)
    flops_rate = b1["decode_flops"] / (b1["batch_step_ms"] / 1000.0)
    return FittedModel("Q_roofline", {"effective_compute_flops_per_s": flops_rate,
                                       "effective_bandwidth_bytes_per_s": bw, "fixed_overhead_ms": 0.0,
                                       "source": "calibrated_this_slice_batch1"},
                        ["decode_flops", "decode_bytes"], 1, [1])


def predict_model_q(params: dict, *, decode_flops: float, decode_bytes: float, **_) -> float | None:
    if params.get("effective_compute_flops_per_s") is None:
        return None
    compute_ms = (decode_flops / params["effective_compute_flops_per_s"]) * 1000.0
    memory_ms = (decode_bytes / params["effective_bandwidth_bytes_per_s"]) * 1000.0
    return max(compute_ms, memory_ms) + params["fixed_overhead_ms"]


# --- Model R: component model (only instantiate timed components with direct evidence) ---

def build_model_r(profiler_component_ms: dict[str, float] | None) -> FittedModel:
    if not profiler_component_ms:
        return FittedModel("R_component", {"status": "unsupported_no_profiler_evidence"}, [], 0, [])
    return FittedModel("R_component", {**profiler_component_ms, "status": "profiler_derived"},
                        list(profiler_component_ms.keys()), 1, [])


def predict_model_r(params: dict, **_) -> float | None:
    if params.get("status") != "profiler_derived":
        return None
    return sum(v for k, v in params.items() if k.endswith("_ms"))


# --- Model S: special batch-1 path ---

def fit_model_s(rows: list[dict]) -> FittedModel:
    usable = _target_rows(rows)
    b1 = [r["batch_step_ms"] for r in usable if r["batch_size"] == 1]
    multi = [r for r in usable if r["batch_size"] >= 2]
    special = statistics.median(b1) if b1 else 0.0
    if multi:
        X = [[1.0, r["batch_size"]] for r in multi]
        y = [r["batch_step_ms"] for r in multi]
        coeffs = _lstsq(X, y)
        general = {"C_fixed": float(coeffs[0]), "C_sequence": float(coeffs[1])}
    else:
        general = {"C_fixed": 0.0, "C_sequence": 0.0}
    return FittedModel("S_special_batch1_path", {"special_single_sequence_ms": special, **general},
                        ["batch_eq_1_special", "batch_ge_2_linear"], len(usable), _cal_sizes(usable))


def predict_model_s(params: dict, *, batch_size: int, **_) -> float:
    if batch_size == 1:
        return params["special_single_sequence_ms"]
    return params["C_fixed"] + batch_size * params["C_sequence"]


MODELS: dict[str, tuple[Callable, Callable]] = {
    "N_binary_transition": (fit_model_n, predict_model_n),
    "O_linear_scaling": (fit_model_o, predict_model_o),
    "S_special_batch1_path": (fit_model_s, predict_model_s),
}


def evaluate(predict_fn: Callable, fitted: FittedModel, rows: list[dict]) -> dict:
    usable = _target_rows(rows)
    errors = []
    for r in usable:
        kwargs = {k: v for k, v in r.items() if k != "batch_step_ms"}
        pred = predict_fn(fitted.params, **kwargs)
        if pred is None:
            continue
        errors.append({"batch_size": r["batch_size"], "predicted": pred, "actual": r["batch_step_ms"],
                        "abs_error": abs(pred - r["batch_step_ms"])})
    if not errors:
        return {"mae": None, "n": 0}
    abs_errors = [e["abs_error"] for e in errors]
    rel_errors = [e["abs_error"] / abs(e["actual"]) for e in errors if e["actual"]]
    return {
        "mae": statistics.mean(abs_errors), "max_error": max(abs_errors),
        "median_relative_error": statistics.median(rel_errors) if rel_errors else None,
        "n": len(errors), "per_row": errors,
    }
