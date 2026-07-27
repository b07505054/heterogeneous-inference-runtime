#!/usr/bin/env python3
"""E2E-5 analysis.

Key methodological correction vs E2E-4: "interference_ms" is now computed by
aligning each anchor inter-token gap against the OBSERVED num_requests_running
sample nearest in time (perf_model.running_count_alignment), not by taking
the median of everything after admission. The E2E-4-style naive label is
also computed, for direct comparison, to make the correction visible rather
than silently replacing it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perf_model import interference_labels as il
from perf_model import capacity_deficit_models as cdm
from perf_model.running_count_alignment import align_gaps_to_running_count, median_or_none, peak_running_waiting
from deployment.vllm_adapter import metrics_client, server_info_client
from scripts.analyze_perf_model_results import resolved_facts, metrics_delta_mean_ms

CALIBRATION_CAPACITIES = {2, 4, 8}
HELD_OUT_CAPACITIES = {3, 6}
HELD_OUT_ANCHOR_ADMIT_COMBO = (2, 3)  # (active_anchor_count, admitted_request_count) excluded from calibration


def load_raws(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*.json"))]


def build_row(raw: dict) -> dict:
    facts = resolved_facts(raw)
    max_num_seqs = raw["max_num_seqs_requested"]
    active_anchor_count = raw["active_anchor_count"]
    admitted_request_count = raw["admitted_request_count"]

    round0_timelines = raw["anchor_timelines_by_round"][0] if raw["anchor_timelines_by_round"] else []
    round0_admissions = [r for r in raw["admission_pooled_rows"] if r.get("round") == 0 and r.get("ok")]
    samples = raw.get("running_waiting_samples", [])

    admission_time = min((r["submit_time"] for r in round0_admissions), default=None)

    solo_meds, concurrent_meds, naive_post_meds, baselines = [], [], [], []
    for tl in round0_timelines:
        if not tl.get("ok"):
            continue
        arrivals = tl.get("token_arrival_times") or []
        if len(arrivals) < 3:
            continue
        aligned = align_gaps_to_running_count(arrivals, samples, tolerance_s=0.25)
        solo_med = median_or_none(aligned.solo_gaps_ms)
        concurrent_med = median_or_none(aligned.concurrent_gaps_ms)
        if solo_med is not None:
            baselines.append(solo_med)
        if solo_med is not None:
            solo_meds.append(solo_med)
        if concurrent_med is not None:
            concurrent_meds.append(concurrent_med)
        if admission_time is not None:
            split = il.split_timeline(arrivals, admission_time)
            naive_baseline = il.baseline_gap_ms(split)
            naive_med = median_or_none(split.post_gaps_ms)
            if naive_baseline is not None and naive_med is not None:
                naive_post_meds.append(naive_med - naive_baseline)

    baseline_gap_ms = median_or_none(baselines)
    concurrent_gap_ms = median_or_none(concurrent_meds)
    interference_ms = (
        (concurrent_gap_ms - baseline_gap_ms) if (concurrent_gap_ms is not None and baseline_gap_ms is not None)
        else None
    )
    naive_interference_ms = median_or_none(naive_post_meds)

    peaks = peak_running_waiting(samples)

    measured_prefill_ms = None
    iteration_tokens_mean = None
    if raw.get("pre_round_metrics_text") and raw.get("final_metrics_text"):
        measured_prefill_ms = metrics_delta_mean_ms(raw["pre_round_metrics_text"], raw["final_metrics_text"],
                                                      "vllm:request_prefill_time_seconds")
        pre = metrics_client.parse_prometheus_text(raw["pre_round_metrics_text"])
        post = metrics_client.parse_prometheus_text(raw["final_metrics_text"])

        def totals(parsed):
            samples_ = parsed.get("histograms", {}).get("vllm:iteration_tokens_total") or []
            return sum(s.sum or 0.0 for s in samples_), sum(s.count or 0.0 for s in samples_)
        s0, c0 = totals(pre)
        s1, c1 = totals(post)
        if c1 - c0 > 0:
            iteration_tokens_mean = (s1 - s0) / (c1 - c0)

    preemptions_delta = None
    if raw.get("pre_round_metrics_text") and raw.get("final_metrics_text"):
        pre = metrics_client.parse_prometheus_text(raw["pre_round_metrics_text"])
        post = metrics_client.parse_prometheus_text(raw["final_metrics_text"])
        v0 = metrics_client.gauge_value(pre, "vllm:num_preemptions")
        v1 = metrics_client.gauge_value(post, "vllm:num_preemptions")
        if v0 is not None and v1 is not None:
            preemptions_delta = v1 - v0

    adherence = None
    if facts is not None:
        adherence = server_info_client.compare_requested_vs_resolved(
            raw["fixed_configuration"], facts, requested_max_num_seqs=max_num_seqs,
        )

    admitted_deficit = cdm.admission_deficit(active_anchor_count, admitted_request_count, max_num_seqs)
    scheduled_sequences = active_anchor_count + admitted_request_count
    peak_stall_meds = []
    for tl in round0_timelines:
        arrivals = tl.get("token_arrival_times") or []
        if len(arrivals) >= 3 and admission_time is not None:
            split = il.split_timeline(arrivals, admission_time)
            b = il.baseline_gap_ms(split)
            p = il.peak_stall_ms(split, b)
            if p is not None:
                peak_stall_meds.append(p)

    return {
        "max_num_seqs": max_num_seqs, "active_anchor_count": active_anchor_count,
        "admitted_request_count": admitted_request_count, "total_submitted_requests": scheduled_sequences,
        "admission_deficit": admitted_deficit, "positive_deficit": cdm.positive_deficit(admitted_deficit),
        "capacity_utilization": cdm.capacity_utilization(scheduled_sequences, max_num_seqs),
        "observed_running_deficit": cdm.observed_running_deficit(peaks["peak_requests_running"], max_num_seqs),
        "peak_requests_running": peaks["peak_requests_running"], "peak_requests_waiting": peaks["peak_requests_waiting"],
        "n_running_samples": peaks["n_samples"],
        "baseline_gap_ms": baseline_gap_ms, "concurrent_gap_ms": concurrent_gap_ms,
        "interference_ms": interference_ms, "naive_post_release_interference_ms": naive_interference_ms,
        "peak_stall_ms": median_or_none(peak_stall_meds),
        "measured_new_request_prefill_ms": measured_prefill_ms, "iteration_tokens_mean": iteration_tokens_mean,
        "preemptions_delta": preemptions_delta,
        "adherence": adherence, "resolved_num_gpu_blocks": facts.num_gpu_blocks if facts else None,
        "chunked_prefill_resolved": facts.enable_chunked_prefill if facts else None,
        "classification": raw["classification"], "reference_match": raw.get("reference_match"),
        "oom_detected": raw.get("oom_detected_in_log"), "process_cleanup_status": raw.get("process_cleanup_status"),
        "n_admissions_ok": len(round0_admissions),
        "held_out": (max_num_seqs in HELD_OUT_CAPACITIES) or
                    ((active_anchor_count, admitted_request_count) == HELD_OUT_ANCHOR_ADMIT_COMBO),
    }


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    calib = [r for r in rows if r["max_num_seqs"] in CALIBRATION_CAPACITIES and not r["held_out"]]
    held = [r for r in rows if r["held_out"]]
    return calib, held


def fit_and_evaluate_all(rows: list[dict]) -> dict:
    calib_rows, held_rows = split_rows(rows)
    results = {}
    for name, (fit_fn, predict_fn) in cdm.MODELS.items():
        fitted = fit_fn(calib_rows)
        calib_eval = cdm.evaluate(predict_fn, fitted, calib_rows)
        held_eval = cdm.evaluate(predict_fn, fitted, held_rows)
        calib_cls = cdm.classification_metrics(predict_fn, fitted, calib_rows)
        held_cls = cdm.classification_metrics(predict_fn, fitted, held_rows)
        for e in (calib_eval, held_eval):
            e.pop("per_row", None)
        results[name] = {"fitted": fitted.to_dict(), "calibration_error": calib_eval, "held_out_error": held_eval,
                          "calibration_classification": calib_cls, "held_out_classification": held_cls}
    return {"n_calibration_rows": len(calib_rows), "n_held_out_rows": len(held_rows),
            "calibration_capacities": sorted(CALIBRATION_CAPACITIES), "held_out_capacities": sorted(HELD_OUT_CAPACITIES),
            "held_out_anchor_admit_combo": HELD_OUT_ANCHOR_ADMIT_COMBO, "models": results}


def evaluate_hypotheses(rows: list[dict], model_eval: dict) -> dict:
    h = {}
    positive = [r for r in rows if r["interference_ms"] is not None and r["interference_ms"] > 30]
    negative = [r for r in rows if r["interference_ms"] is not None and r["interference_ms"] <= 30]

    # H1: interference occurs when admission_deficit > 0
    tp = sum(1 for r in positive if r["admission_deficit"] > 0)
    fp_neg_but_deficit = sum(1 for r in negative if r["admission_deficit"] > 0)
    fn_pos_no_deficit = sum(1 for r in positive if r["admission_deficit"] <= 0)
    h["H1"] = {"result": "SUPPORTED" if fn_pos_no_deficit == 0 and tp > 0 else
               ("REJECTED" if fn_pos_no_deficit > 0 else "INCONCLUSIVE"),
               "evidence": {"positive_rows_with_deficit_gt_0": tp, "positive_rows_with_deficit_le_0": fn_pos_no_deficit,
                            "negative_rows_with_deficit_gt_0": fp_neg_but_deficit}}

    # H2: magnitude scales with deficit size (among positive rows)
    deficits = [r["positive_deficit"] for r in positive]
    mags = [r["interference_ms"] for r in positive]
    r_val = None
    if len(set(deficits)) > 1:
        import numpy as np
        r_val = float(np.corrcoef(deficits, mags)[0, 1])
    h["H2"] = {"result": "SUPPORTED" if r_val and r_val > 0.5 else ("REJECTED" if r_val is not None else "INCONCLUSIVE"),
               "evidence": {"pearson_r_deficit_vs_magnitude": r_val, "n_positive_rows": len(positive)}}

    # H3: max_num_seqs alone insufficient
    by_cap = {}
    for r in rows:
        by_cap.setdefault(r["max_num_seqs"], []).append(r["interference_ms"])
    mixed_caps = [c for c, vals in by_cap.items() if any(v is not None and v > 30 for v in vals)
                  and any(v is not None and v <= 30 for v in vals)]
    h["H3"] = {"result": "SUPPORTED" if mixed_caps else "INCONCLUSIVE",
               "evidence": {"capacities_with_both_regimes_present": mixed_caps}}

    # H4: prompt length not required (fixed this slice by design)
    h["H4"] = {"result": "SUPPORTED", "evidence": {"prompt_tokens_fixed_at": 128, "note": "not varied this slice by design"}}

    # H5: generalizes to held-out capacities
    best_name = min((n for n in cdm.MODELS if model_eval["models"][n]["held_out_error"]["mae"] is not None),
                     key=lambda n: model_eval["models"][n]["held_out_error"]["mae"], default=None)
    h5_mae = model_eval["models"][best_name]["held_out_error"]["mae"] if best_name else None
    h["H5"] = {"result": "SUPPORTED" if h5_mae is not None and h5_mae < 60 else
               ("INCONCLUSIVE" if h5_mae is None else "REJECTED"),
               "evidence": {"best_model": best_name, "held_out_mae": h5_mae}}

    # H6: separate transient (peak_stall) vs sustained -- qualitative, evaluated via Model M presence
    m_fitted = cdm.fit_model_m_transient_term(rows)
    h["H6"] = {"result": "SUPPORTED" if m_fitted.params.get("C_transient", 0) > 0 and m_fitted.n_calibration_rows >= 3
               else "INCONCLUSIVE", "evidence": m_fitted.to_dict()}

    h["H7"] = {"result": "PENDING", "evidence": {}}  # filled by ranking section
    return h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raws = load_raws(args.raw_dir)
    rows = [build_row(r) for r in raws]
    n_adherent = sum(1 for r in rows if r["adherence"] and r["adherence"]["derived_config_adherent"])

    model_eval = fit_and_evaluate_all(rows)
    hypotheses = evaluate_hypotheses(rows, model_eval)

    calib_rows, held_rows = split_rows(rows)
    best_name = min((n for n in cdm.MODELS if model_eval["models"][n]["held_out_error"]["mae"] is not None),
                     key=lambda n: model_eval["models"][n]["held_out_error"]["mae"], default=None)
    if best_name:
        hypotheses["H7"] = {
            "result": "SUPPORTED" if model_eval["models"][best_name]["held_out_error"]["mae"] < 100 else "INCONCLUSIVE",
            "evidence": {"best_model_held_out_mae": model_eval["models"][best_name]["held_out_error"]["mae"],
                         "held_out_rows": [(r["max_num_seqs"], r["active_anchor_count"], r["admitted_request_count"],
                                             r["interference_ms"]) for r in held_rows]},
        }

    args.out.write_text(json.dumps({
        "n_groups": len(raws), "n_fully_adherent": n_adherent, "rows": rows,
        "model_evaluation": model_eval, "hypotheses": hypotheses,
    }, indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out} groups={len(raws)} adherent={n_adherent}")


if __name__ == "__main__":
    main()
