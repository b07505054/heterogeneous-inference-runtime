"""E2E-4: candidate scaling equations for the admission/prefill interference
term (Models A-F from the slice spec), fit by plain ordinary least squares
(numpy.linalg.lstsq) -- explicitly NOT Ridge/GBDT/neural: every model here
has at most 3 linear coefficients solved in closed form, same class of tool
already used and accepted in tp_cost_model.py.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class FittedModel:
    name: str
    params: dict[str, float]
    feature_names: list[str]
    n_calibration_rows: int
    calibration_prompt_lengths: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "feature_names": self.feature_names,
                "n_calibration_rows": self.n_calibration_rows,
                "calibration_prompt_lengths": self.calibration_prompt_lengths}


def _lstsq(X: list[list[float]], y: list[float]) -> np.ndarray:
    coeffs, *_ = np.linalg.lstsq(np.array(X, dtype=float), np.array(y, dtype=float), rcond=None)
    return coeffs


def _rows_with_target(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("interference_ms") is not None]


# --- Model A: fixed regime penalty ---

def fit_model_a(rows: list[dict]) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if (r.get("max_num_seqs") or 1) > 1]
    c_fixed = statistics.median([r["interference_ms"] for r in usable]) if usable else 0.0
    return FittedModel("A_fixed_regime", {"C_fixed": c_fixed}, ["max_num_seqs_gt_1"], len(usable),
                        sorted({r["admitted_prompt_tokens"] for r in usable}))


def predict_model_a(params: dict, *, max_num_seqs: int | None, **_) -> float:
    return params["C_fixed"] if (max_num_seqs or 1) > 1 else 0.0


# --- Model B: prompt-token linear (through origin) ---

def fit_model_b(rows: list[dict]) -> FittedModel:
    usable = _rows_with_target(rows)
    X = [[r["admitted_prompt_tokens"]] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0])
    return FittedModel("B_prompt_token_linear", {"C_token": float(coeffs[0])}, ["admitted_prompt_tokens"],
                        len(usable), sorted({r["admitted_prompt_tokens"] for r in usable}))


def predict_model_b(params: dict, *, admitted_prompt_tokens: int, **_) -> float:
    return params["C_token"] * admitted_prompt_tokens


# --- Model C: per-request + per-token ---

def fit_model_c(rows: list[dict]) -> FittedModel:
    usable = _rows_with_target(rows)
    X = [[r["admitted_request_count"], r["admitted_prompt_tokens"]] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0, 0.0])
    return FittedModel("C_request_plus_token", {"C_request": float(coeffs[0]), "C_token": float(coeffs[1])},
                        ["admitted_request_count", "admitted_prompt_tokens"], len(usable),
                        sorted({r["admitted_prompt_tokens"] for r in usable}))


def predict_model_c(params: dict, *, admitted_request_count: int, admitted_prompt_tokens: int, **_) -> float:
    return params["C_request"] * admitted_request_count + params["C_token"] * admitted_prompt_tokens


# --- Model D: prefill-FLOP scaled (through origin) ---

def fit_model_d(rows: list[dict]) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if r.get("admitted_prefill_flops") is not None]
    X = [[r["admitted_prefill_flops"]] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0])
    return FittedModel("D_prefill_flop_scaled", {"C_flops": float(coeffs[0])}, ["admitted_prefill_flops"],
                        len(usable), sorted({r["admitted_prompt_tokens"] for r in usable}))


def predict_model_d(params: dict, *, admitted_prefill_flops: float, **_) -> float:
    return params["C_flops"] * admitted_prefill_flops


# --- Model E: measured-prefill-time scaled (through origin) ---

def fit_model_e(rows: list[dict]) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if r.get("measured_new_request_prefill_ms") is not None]
    X = [[r["measured_new_request_prefill_ms"]] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0])
    return FittedModel("E_measured_prefill_scaled", {"C_overlap": float(coeffs[0])},
                        ["measured_new_request_prefill_ms"], len(usable),
                        sorted({r["admitted_prompt_tokens"] for r in usable}))


def predict_model_e(params: dict, *, measured_new_request_prefill_ms: float, **_) -> float:
    return params["C_overlap"] * measured_new_request_prefill_ms


# --- Model F: piecewise regime (transition + per-token + per-request) ---

def fit_model_f(rows: list[dict]) -> FittedModel:
    usable = [r for r in _rows_with_target(rows) if (r.get("max_num_seqs") or 1) > 1]
    X = [[1.0, r["admitted_prompt_tokens"], r["admitted_request_count"]] for r in usable]
    y = [r["interference_ms"] for r in usable]
    coeffs = _lstsq(X, y) if usable else np.array([0.0, 0.0, 0.0])
    return FittedModel("F_piecewise_regime",
                        {"C_transition": float(coeffs[0]), "C_token": float(coeffs[1]), "C_request": float(coeffs[2])},
                        ["max_num_seqs_gt_1", "admitted_prompt_tokens", "admitted_request_count"], len(usable),
                        sorted({r["admitted_prompt_tokens"] for r in usable}))


def predict_model_f(params: dict, *, max_num_seqs: int | None, admitted_prompt_tokens: int,
                     admitted_request_count: int, **_) -> float:
    if (max_num_seqs or 1) == 1:
        return 0.0
    return params["C_transition"] + params["C_token"] * admitted_prompt_tokens + params["C_request"] * admitted_request_count


MODELS: dict[str, tuple[Callable, Callable]] = {
    "A_fixed_regime": (fit_model_a, predict_model_a),
    "B_prompt_token_linear": (fit_model_b, predict_model_b),
    "C_request_plus_token": (fit_model_c, predict_model_c),
    "D_prefill_flop_scaled": (fit_model_d, predict_model_d),
    "E_measured_prefill_scaled": (fit_model_e, predict_model_e),
    "F_piecewise_regime": (fit_model_f, predict_model_f),
}

FIXED_151MS_BASELINE = FittedModel("baseline_fixed_151ms", {"C_fixed": 151.0}, ["max_num_seqs_gt_1"], 6, [128])


def evaluate(predict_fn: Callable, fitted: FittedModel, rows: list[dict]) -> dict:
    usable = _rows_with_target(rows)
    errors = []
    for r in usable:
        pred = predict_fn(fitted.params, **{k: v for k, v in r.items() if k != "interference_ms"})
        errors.append({"row": r, "predicted": pred, "actual": r["interference_ms"], "abs_error": abs(pred - r["interference_ms"])})
    if not errors:
        return {"mae": None, "median_relative_error": None, "p95_abs_error": None, "max_error": None, "r2": None, "n": 0}
    abs_errors = [e["abs_error"] for e in errors]
    rel_errors = [e["abs_error"] / abs(e["actual"]) for e in errors if e["actual"]]
    ordered_abs = sorted(abs_errors)
    p95_idx = min(len(ordered_abs) - 1, max(0, round(0.95 * (len(ordered_abs) - 1))))
    actual_mean = statistics.mean(e["actual"] for e in errors)
    ss_tot = sum((e["actual"] - actual_mean) ** 2 for e in errors)
    ss_res = sum(e["abs_error"] ** 2 for e in errors)
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    return {
        "mae": statistics.mean(abs_errors), "median_relative_error": statistics.median(rel_errors) if rel_errors else None,
        "p95_abs_error": ordered_abs[p95_idx], "max_error": max(abs_errors), "r2": r2, "n": len(errors),
        "per_row": errors,
    }
