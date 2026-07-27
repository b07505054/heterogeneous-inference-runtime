"""E2E-5: scheduler capacity-deficit models (G-M).

Source-level grounding (vllm/v1/core/sched/scheduler.py, 0.24.0):
  self.max_num_running_reqs = scheduler_config.max_num_seqs   (line 108)
  if len(self.running) == self.max_num_running_reqs: break     (line 630, admission loop)
  assert len(self.running) <= self.max_num_running_reqs        (line 994)
max_num_seqs caps len(self.running) -- requests currently admitted (decode OR
in-progress prefill), NOT a cap on waiting requests. This directly motivates
`admission_deficit` below as the scheduler-accurate feature, not a guess.

As in interference_scaling_models.py, all fitting is plain OLS
(numpy.linalg.lstsq), not Ridge/GBDT/neural.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


def requested_deficit(workload_concurrency: int, max_num_seqs: int) -> int:
    return workload_concurrency - max_num_seqs


def admission_deficit(active_decode_requests: int, newly_admitted_requests: int, max_num_seqs: int) -> int:
    return active_decode_requests + newly_admitted_requests - max_num_seqs


def positive_deficit(deficit: int) -> int:
    return max(deficit, 0)


def capacity_utilization(active_or_scheduled_sequences: int, max_num_seqs: int) -> float:
    if max_num_seqs <= 0:
        return float("inf")
    return active_or_scheduled_sequences / max_num_seqs


def observed_running_deficit(peak_requests_running: int | None, max_num_seqs: int) -> int | None:
    if peak_requests_running is None:
        return None
    return max(peak_requests_running - max_num_seqs, 0)


@dataclass(frozen=True)
class FittedModel:
    name: str
    params: dict[str, Any]
    feature_names: list[str]
    n_calibration_rows: int
    calibration_max_num_seqs: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "feature_names": self.feature_names,
                "n_calibration_rows": self.n_calibration_rows,
                "calibration_max_num_seqs": self.calibration_max_num_seqs}


def _rows_with_target(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("interference_ms") is not None]


def _lstsq(X: list[list[float]], y: list[float]) -> np.ndarray:
    coeffs, *_ = np.linalg.lstsq(np.array(X, dtype=float), np.array(y, dtype=float), rcond=None)
    return coeffs


def _cal_lens(rows: list[dict]) -> list[int]:
    return sorted({r["max_num_seqs"] for r in rows})


# --- Model G: binary admission deficit ---

def fit_model_g(rows: list[dict]) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if r["admission_deficit"] > 0]
    c = statistics.median([r["interference_ms"] for r in usable]) if usable else 0.0
    return FittedModel("G_binary_deficit", {"C_deficit": c}, ["deficit_gt_0"], len(usable), _cal_lens(usable))


def predict_model_g(params: dict, *, admission_deficit: int, **_) -> float:
    return params["C_deficit"] if admission_deficit > 0 else 0.0


# --- Model H: linear positive deficit (through origin) ---

def fit_model_h(rows: list[dict]) -> FittedModel:
    usable = _rows_with_target(rows)
    X = [[max(r["admission_deficit"], 0)] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0])
    return FittedModel("H_linear_deficit", {"C_deficit_unit": float(coeffs[0])}, ["positive_deficit"],
                        len(usable), _cal_lens(usable))


def predict_model_h(params: dict, *, admission_deficit: int, **_) -> float:
    return params["C_deficit_unit"] * max(admission_deficit, 0)


# --- Model I: capacity-utilization threshold ---

def fit_model_i(rows: list[dict], threshold: float = 1.0) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if r["capacity_utilization"] > threshold]
    X = [[r["capacity_utilization"] - threshold] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0])
    return FittedModel("I_utilization_threshold", {"T": threshold, "C_utilization": float(coeffs[0])},
                        ["utilization_over_threshold"], len(usable), _cal_lens(usable))


def predict_model_i(params: dict, *, capacity_utilization: float, **_) -> float:
    if capacity_utilization <= params["T"]:
        return 0.0
    return (capacity_utilization - params["T"]) * params["C_utilization"]


# --- Model J: piecewise deficit (deficit==1 special-cased) ---

def fit_model_j(rows: list[dict]) -> FittedModel:
    d1 = [r["interference_ms"] for r in _rows_with_target(rows) if r["admission_deficit"] == 1]
    dmore = [r for r in _rows_with_target(rows) if r["admission_deficit"] > 1]
    c1 = statistics.median(d1) if d1 else 0.0
    if dmore:
        X = [[r["admission_deficit"] - 1] for r in dmore]
        y = [r["interference_ms"] - c1 for r in dmore]
        c_more = float(_lstsq(X, y)[0])
    else:
        c_more = 0.0
    usable = [r for r in _rows_with_target(rows) if r["admission_deficit"] >= 1]
    return FittedModel("J_piecewise_deficit", {"C1": c1, "C_more": c_more}, ["deficit_eq_1", "deficit_gt_1"],
                        len(usable), _cal_lens(usable))


def predict_model_j(params: dict, *, admission_deficit: int, **_) -> float:
    if admission_deficit <= 0:
        return 0.0
    if admission_deficit == 1:
        return params["C1"]
    return params["C1"] + (admission_deficit - 1) * params["C_more"]


# --- Model K: observed-running deficit (diagnostic; not compiler-visible pre-launch) ---

def fit_model_k(rows: list[dict]) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if r.get("observed_running_deficit") is not None]
    X = [[r["observed_running_deficit"]] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0])
    return FittedModel("K_observed_running_deficit", {"C_observed": float(coeffs[0])},
                        ["observed_running_deficit"], len(usable), _cal_lens(usable))


def predict_model_k(params: dict, *, observed_running_deficit: int | None, **_) -> float:
    if observed_running_deficit is None:
        return 0.0
    return params["C_observed"] * observed_running_deficit


# --- Model L: scheduler-token regime (binary, gated on an iteration-tokens threshold crossing) ---

def fit_model_l(rows: list[dict], token_threshold: float = 40.0) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if (r.get("iteration_tokens_mean") or 0) > token_threshold]
    c = statistics.median([r["interference_ms"] for r in usable]) if usable else 0.0
    return FittedModel("L_scheduler_token_regime", {"token_threshold": token_threshold, "C_scheduler_regime": c},
                        ["iteration_tokens_mean_gt_threshold"], len(usable), _cal_lens(usable))


def predict_model_l(params: dict, *, iteration_tokens_mean: float | None, **_) -> float:
    if iteration_tokens_mean is None or iteration_tokens_mean <= params["token_threshold"]:
        return 0.0
    return params["C_scheduler_regime"]


MODELS: dict[str, tuple[Callable, Callable]] = {
    "G_binary_deficit": (fit_model_g, predict_model_g),
    "H_linear_deficit": (fit_model_h, predict_model_h),
    "I_utilization_threshold": (fit_model_i, predict_model_i),
    "J_piecewise_deficit": (fit_model_j, predict_model_j),
    "K_observed_running_deficit": (fit_model_k, predict_model_k),
    "L_scheduler_token_regime": (fit_model_l, predict_model_l),
}


def predict_model_m_peak_stall(sustained_ms: float, measured_prefill_ms: float | None, c_transient: float) -> float | None:
    """Model M: peak_stall = sustained_interference + measured_prefill_ms * C_transient.
    Explicitly separate from the sustained-TPOT target -- never mixed into one number."""
    if measured_prefill_ms is None:
        return None
    return sustained_ms + measured_prefill_ms * c_transient


def fit_model_m_transient_term(rows: list[dict]) -> FittedModel:
    """Fits C_transient against (peak_stall - sustained_interference) ~ measured_prefill_ms,
    i.e. isolates the transient component the sustained-deficit models don't cover."""
    usable = [r for r in rows if r.get("peak_stall_ms") is not None and r.get("interference_ms") is not None
              and r.get("measured_new_request_prefill_ms")]
    X = [[r["measured_new_request_prefill_ms"]] for r in usable]
    y = [r["peak_stall_ms"] - r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0])
    return FittedModel("M_capacity_plus_transient", {"C_transient": float(coeffs[0])},
                        ["measured_new_request_prefill_ms"], len(usable), _cal_lens(usable))


def evaluate(predict_fn: Callable, fitted: FittedModel, rows: list[dict]) -> dict:
    usable = _rows_with_target(rows)
    errors = []
    for r in usable:
        pred = predict_fn(fitted.params, **{k: v for k, v in r.items() if k != "interference_ms"})
        errors.append({"row": r, "predicted": pred, "actual": r["interference_ms"], "abs_error": abs(pred - r["interference_ms"])})
    if not errors:
        return {"mae": None, "n": 0}
    abs_errors = [e["abs_error"] for e in errors]
    ordered = sorted(abs_errors)
    p95_idx = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    positive_rows = [e for e in errors if e["actual"] > 30]
    negative_rows = [e for e in errors if e["actual"] <= 30]
    return {
        "mae": statistics.mean(abs_errors), "max_error": max(abs_errors), "p95_abs_error": ordered[p95_idx],
        "n": len(errors),
        "positive_row_mae": statistics.mean(e["abs_error"] for e in positive_rows) if positive_rows else None,
        "negative_row_mae": statistics.mean(e["abs_error"] for e in negative_rows) if negative_rows else None,
        "worst_positive_row_error": max((e["abs_error"] for e in positive_rows), default=None),
        "n_positive_rows": len(positive_rows), "n_negative_rows": len(negative_rows),
        "per_row": errors,
    }


def classification_metrics(predict_fn: Callable, fitted: FittedModel, rows: list[dict], threshold_ms: float = 30.0) -> dict:
    usable = _rows_with_target(rows)
    tp = fp = tn = fn = 0
    for r in usable:
        pred = predict_fn(fitted.params, **{k: v for k, v in r.items() if k != "interference_ms"})
        pred_positive = pred > threshold_ms
        actual_positive = r["interference_ms"] > threshold_ms
        if pred_positive and actual_positive:
            tp += 1
        elif pred_positive and not actual_positive:
            fp += 1
        elif not pred_positive and actual_positive:
            fn += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / n if n else None,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) else None}
