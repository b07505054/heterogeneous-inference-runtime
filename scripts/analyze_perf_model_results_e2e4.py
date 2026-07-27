#!/usr/bin/env python3
"""E2E-4 analysis: per-group interference labels, kernel-level correlation,
Model A-F fitting on a calibration split with held-out evaluation, hypothesis
tests H1-H6, and candidate-ranking regret before/after.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perf_model import compute_model, interference_labels as il
from perf_model import interference_scaling_models as ism
from deployment.vllm_adapter import server_info_client
from scripts.analyze_perf_model_results import build_model_features, resolved_facts, metrics_delta_mean_ms

CALIBRATION_PROMPT_LENGTHS = {64, 128, 512}
HELD_OUT_PROMPT_LENGTHS = {256}


def load_raws(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*.json"))]


def per_round_labels(raw: dict) -> list[dict]:
    rounds = []
    for round_idx, tl in enumerate(raw.get("anchor_timelines", [])):
        arrivals = tl.get("token_arrival_times") or []
        admission_rows = [r for r in raw.get("admission_pooled_rows", []) if r.get("round") == round_idx and r.get("ok")]
        if len(arrivals) < 3 or not admission_rows:
            continue
        admission_time = min(r["submit_time"] for r in admission_rows)
        split = il.split_timeline(arrivals, admission_time)
        baseline = il.baseline_gap_ms(split)
        rounds.append({
            "round": round_idx, "baseline_gap_ms": baseline,
            "peak_stall_ms": il.peak_stall_ms(split, baseline),
            "total_stall_area_ms": il.total_stall_area_ms(split, baseline),
            "recovery_time_ms": il.recovery_time_ms(split, baseline),
            "sustained_slowdown_ratio": il.sustained_slowdown_ratio(split, baseline),
            "sustained_additive_ms": (
                statistics.median(split.post_gaps_ms) - baseline
                if baseline is not None and split.post_gaps_ms else None
            ),
            "post_admission_percentiles": il.post_admission_percentiles(split),
            "affected_token_count": il.affected_token_count(split, baseline),
            "interference_e2e_ms": il.interference_e2e_ms(
                measured_anchor_e2e_ms=(
                    (tl["completion_time"] - tl["submit_time"]) * 1000.0 if tl.get("completion_time") else None
                ),
                ttft_ms=(tl["first_token_time"] - tl["submit_time"]) * 1000.0 if tl.get("first_token_time") else None,
                baseline_gap=baseline, output_tokens=tl.get("output_tokens", 0),
            ),
        })
    return rounds


def median_or_none(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def build_row(raw: dict, model) -> dict:
    rounds = per_round_labels(raw)
    facts = resolved_facts(raw)
    prompt_length = raw["prompt_length"]
    multiplicity = raw["multiplicity"]
    max_num_seqs = raw["max_num_seqs_requested"]

    admitted_prefill_flops = compute_model.prefill_op_counts(model, prompt_length).total_flops * multiplicity
    measured_prefill_ms = None
    if raw.get("post_warmup_metrics_text") and raw.get("final_metrics_text"):
        measured_prefill_ms = metrics_delta_mean_ms(raw["post_warmup_metrics_text"], raw["final_metrics_text"],
                                                      "vllm:request_prefill_time_seconds")

    adherence = None
    if facts is not None:
        adherence = server_info_client.compare_requested_vs_resolved(
            raw["fixed_configuration"], facts, requested_max_num_seqs=max_num_seqs,
        )

    return {
        "prompt_length": prompt_length, "admitted_prompt_tokens": prompt_length * multiplicity,
        "admitted_request_count": multiplicity, "max_num_seqs": max_num_seqs,
        "admitted_prefill_flops": admitted_prefill_flops,
        "measured_new_request_prefill_ms": measured_prefill_ms,
        "baseline_gap_ms": median_or_none([r["baseline_gap_ms"] for r in rounds]),
        "peak_stall_ms": median_or_none([r["peak_stall_ms"] for r in rounds]),
        "total_stall_area_ms": median_or_none([r["total_stall_area_ms"] for r in rounds]),
        "recovery_time_ms": median_or_none([r["recovery_time_ms"] for r in rounds]),
        "sustained_slowdown_ratio": median_or_none([r["sustained_slowdown_ratio"] for r in rounds]),
        "interference_ms": median_or_none([r["sustained_additive_ms"] for r in rounds]),  # fit target
        "interference_e2e_ms": median_or_none([r["interference_e2e_ms"] for r in rounds]),
        "n_rounds_with_labels": len(rounds),
        "adherence": adherence,
        "chunked_prefill_resolved": facts.enable_chunked_prefill if facts else None,
        "resolved_num_gpu_blocks": facts.num_gpu_blocks if facts else None,
        "classification": raw["classification"], "reference_match": raw.get("reference_match"),
        "oom_detected": raw.get("oom_detected_in_log"), "process_cleanup_status": raw.get("process_cleanup_status"),
        "peak_gpu_memory_mib": raw.get("peak_gpu_memory_mib"),
    }


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x_arr = np.array([p[0] for p in pairs], dtype=float)
    y_arr = np.array([p[1] for p in pairs], dtype=float)
    if x_arr.std() == 0 or y_arr.std() == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def correlations(rows: list[dict]) -> dict:
    targets = {"peak_stall_ms": [r["peak_stall_ms"] for r in rows],
               "total_stall_area_ms": [r["total_stall_area_ms"] for r in rows],
               "interference_ms": [r["interference_ms"] for r in rows]}
    features = {"admitted_prompt_tokens": [r["admitted_prompt_tokens"] for r in rows],
                "admitted_prefill_flops": [r["admitted_prefill_flops"] for r in rows],
                "measured_new_request_prefill_ms": [r["measured_new_request_prefill_ms"] for r in rows],
                "admitted_request_count": [r["admitted_request_count"] for r in rows]}
    return {t: {f: pearson_r(fv, tv) for f, fv in features.items()} for t, tv in targets.items()}


def fit_and_evaluate_all(rows: list[dict]) -> dict:
    calib_rows = [r for r in rows if r["prompt_length"] in CALIBRATION_PROMPT_LENGTHS and (r["max_num_seqs"] or 1) > 1]
    held_out_rows = [r for r in rows if r["prompt_length"] in HELD_OUT_PROMPT_LENGTHS and (r["max_num_seqs"] or 1) > 1]

    results = {}
    for name, (fit_fn, predict_fn) in ism.MODELS.items():
        fitted = fit_fn(calib_rows)
        calib_eval = ism.evaluate(predict_fn, fitted, calib_rows)
        held_out_eval = ism.evaluate(predict_fn, fitted, held_out_rows)
        for e in (calib_eval, held_out_eval):
            e.pop("per_row", None)
        results[name] = {"fitted": fitted.to_dict(), "calibration_error": calib_eval, "held_out_error": held_out_eval}

    baseline_eval_calib = ism.evaluate(ism.predict_model_a, ism.FIXED_151MS_BASELINE, calib_rows)
    baseline_eval_held = ism.evaluate(ism.predict_model_a, ism.FIXED_151MS_BASELINE, held_out_rows)
    for e in (baseline_eval_calib, baseline_eval_held):
        e.pop("per_row", None)
    results["baseline_fixed_151ms"] = {"fitted": ism.FIXED_151MS_BASELINE.to_dict(),
                                        "calibration_error": baseline_eval_calib, "held_out_error": baseline_eval_held}
    return {"n_calibration_rows": len(calib_rows), "n_held_out_rows": len(held_out_rows),
            "calibration_prompt_lengths": sorted(CALIBRATION_PROMPT_LENGTHS),
            "held_out_prompt_lengths": sorted(HELD_OUT_PROMPT_LENGTHS), "models": results}


def evaluate_hypotheses(rows: list[dict], corr: dict, model_eval: dict) -> dict:
    h = {}

    multi = [r for r in rows if (r["max_num_seqs"] or 1) > 1 and r["interference_ms"] is not None]
    token_r = corr["interference_ms"]["admitted_prompt_tokens"]
    h["H1"] = {"result": "SUPPORTED" if token_r and token_r > 0.5 else
               ("REJECTED" if token_r is not None and token_r < 0.2 else "INCONCLUSIVE"),
               "evidence": {"pearson_r_tokens_vs_interference": token_r}}

    # H2: request-count effect holding tokens ~constant -- compare same total admitted tokens via
    # different (length, multiplicity) combos is not guaranteed in this matrix; use partial evidence:
    # compare interference at same prompt_length across multiplicity=1 vs 2.
    by_len = {}
    for r in multi:
        by_len.setdefault(r["prompt_length"], {})[r["admitted_request_count"]] = r["interference_ms"]
    deltas = [d[2] - d[1] for d in by_len.values() if 1 in d and 2 in d and d[1] is not None and d[2] is not None]
    h["H2"] = {"result": ("SUPPORTED" if deltas and statistics.median(deltas) > 20 else
                          "REJECTED" if deltas and statistics.median(deltas) <= 0 else "INCONCLUSIVE"),
               "evidence": {"interference_delta_mult2_minus_mult1_by_length": deltas}}

    fixed_model = model_eval["models"]["baseline_fixed_151ms" if False else "A_fixed_regime"]
    held_mae_fixed_151 = model_eval["models"]["baseline_fixed_151ms"]["held_out_error"]["mae"]
    calib_mae_fixed_151 = model_eval["models"]["baseline_fixed_151ms"]["calibration_error"]["mae"]
    h["H3"] = {"result": "SUPPORTED" if (held_mae_fixed_151 or 0) > 40 else "REJECTED",
               "evidence": {"fixed_151ms_calibration_mae": calib_mae_fixed_151, "fixed_151ms_held_out_mae": held_mae_fixed_151}}

    flop_r = corr["interference_ms"]["admitted_prefill_flops"]
    prefill_time_r = corr["interference_ms"]["measured_new_request_prefill_ms"]
    best_alt = max([x for x in (flop_r, prefill_time_r) if x is not None], default=None)
    h["H4"] = {"result": ("SUPPORTED" if best_alt is not None and token_r is not None and best_alt > token_r + 0.05 else
                          "REJECTED" if best_alt is not None and token_r is not None else "INCONCLUSIVE"),
               "evidence": {"tokens_r": token_r, "flops_r": flop_r, "measured_prefill_r": prefill_time_r}}

    by_len_1 = {r["prompt_length"]: r["interference_ms"] for r in rows if (r["max_num_seqs"] or 1) == 1}
    present_transition = all(v is not None and abs(v) < 30 for v in by_len_1.values()) if by_len_1 else False
    multi_present = all(r["interference_ms"] is not None and r["interference_ms"] > 30 for r in multi) if multi else False
    h["H5"] = {"result": "SUPPORTED" if present_transition and multi_present else "INCONCLUSIVE",
               "evidence": {"max_num_seqs_1_interference_by_length": by_len_1,
                            "all_multi_seq_rows_show_interference": multi_present}}

    h["H6"] = {"result": "PENDING_RANKING_EVAL", "evidence": {}}  # filled in by ranking_and_regret
    return h


def ranking_and_regret(rows: list[dict], model_eval: dict) -> dict:
    """Compiler-selection usefulness on the HELD-OUT prompt length only."""
    held = [r for r in rows if r["prompt_length"] in HELD_OUT_PROMPT_LENGTHS]
    if not held:
        return {}
    best_model_name = min(
        (name for name in ism.MODELS if model_eval["models"][name]["held_out_error"]["mae"] is not None),
        key=lambda n: model_eval["models"][n]["held_out_error"]["mae"], default=None,
    )
    return {
        "held_out_prompt_length": sorted(HELD_OUT_PROMPT_LENGTHS)[0],
        "held_out_rows_interference_ms": {f"cand{r['max_num_seqs']}_mult{r['admitted_request_count']}": r["interference_ms"] for r in held},
        "best_model_by_held_out_mae": best_model_name,
        "fixed_151ms_held_out_mae": model_eval["models"]["baseline_fixed_151ms"]["held_out_error"]["mae"],
        "best_model_held_out_mae": model_eval["models"][best_model_name]["held_out_error"]["mae"] if best_model_name else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raws = load_raws(args.raw_dir)
    model = build_model_features()
    rows = [build_row(r, model) for r in raws]

    corr = correlations(rows)
    model_eval = fit_and_evaluate_all(rows)
    ranking = ranking_and_regret(rows, model_eval)
    hypotheses = evaluate_hypotheses(rows, corr, model_eval)
    if ranking:
        fixed_mae = ranking["fixed_151ms_held_out_mae"]
        best_mae = ranking["best_model_held_out_mae"]
        hypotheses["H6"]["result"] = (
            "SUPPORTED" if best_mae is not None and fixed_mae is not None and best_mae < fixed_mae * 0.8
            else "INCONCLUSIVE" if best_mae is None or fixed_mae is None else "REJECTED"
        )
        hypotheses["H6"]["evidence"] = {"fixed_151ms_held_out_mae": fixed_mae, "best_model_held_out_mae": best_mae}

    n_adherent = sum(1 for r in rows if r["adherence"] and r["adherence"]["derived_config_adherent"])
    args.out.write_text(json.dumps({
        "n_groups": len(raws), "n_fully_adherent": n_adherent, "rows": rows, "correlations": corr,
        "model_evaluation": model_eval, "ranking": ranking, "hypotheses": hypotheses,
    }, indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out} groups={len(raws)} adherent={n_adherent}")


if __name__ == "__main__":
    main()
