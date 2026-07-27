#!/usr/bin/env python3
"""E2E-6 analysis: steady-state window validation, engine-vs-client metric
cross-check, operation-count scaling, Models N/O/P/Q/S fitting+evaluation,
hypothesis tests H1-H8, and objective-specific candidate ranking.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perf_model import compute_model, batch_shape_models as bsm
from perf_model.steady_state_window import (
    validate_running_count_window, extract_common_window_gaps, per_request_tpot_ms, batch_step_latency_ms,
    aggregate_throughput_tokens_per_s, scaling_efficiency, latency_inflation, coefficient_of_variation,
)
from deployment.vllm_adapter import metrics_client, server_info_client
from scripts.analyze_perf_model_results import build_model_features, resolved_facts, metrics_delta_mean_ms

WINDOW_START_TOKEN = 9
WINDOW_END_TOKEN = 40
CALIBRATION_BATCH_SIZES = {1, 2, 4, 8}
HELD_OUT_BATCH_SIZES = {3, 6}


def load_raws(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*.json")) if "prof" not in p.stem]


def load_profiler_raws(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*-prof.json"))]


def build_row(raw: dict, model) -> dict:
    facts = resolved_facts(raw)
    batch_size = raw["batch_size"]
    prompt_length = raw["prompt_length"]

    per_request_gaps = []
    per_request_tpots = []
    window_valid_flags = []
    for tl in raw["timelines"]:
        if not tl.get("ok"):
            continue
        arrivals = tl.get("token_arrival_times") or []
        gaps = extract_common_window_gaps(arrivals, WINDOW_START_TOKEN, WINDOW_END_TOKEN)
        if not gaps:
            continue
        window_start_time = arrivals[WINDOW_START_TOKEN] if len(arrivals) > WINDOW_START_TOKEN else None
        window_end_time = arrivals[WINDOW_END_TOKEN] if len(arrivals) > WINDOW_END_TOKEN else arrivals[-1]
        if window_start_time is None:
            continue
        validation = validate_running_count_window(
            raw.get("running_waiting_samples", []), window_start_time, window_end_time, expected_n=batch_size,
        )
        window_valid_flags.append(validation.valid)
        per_request_gaps.append(gaps)
        per_request_tpots.append(per_request_tpot_ms(gaps))

    window_valid = bool(window_valid_flags) and all(window_valid_flags)
    tpot_median = per_request_tpot_ms(per_request_tpots) if per_request_tpots else None
    batch_step_ms = batch_step_latency_ms(per_request_tpots) if window_valid else None
    all_tpot_p95 = None
    if per_request_gaps:
        flat = [g for gaps in per_request_gaps for g in gaps]
        ordered = sorted(flat)
        idx = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
        all_tpot_p95 = ordered[idx]

    window_wall_time_s = None
    if per_request_gaps:
        window_wall_time_s = statistics.mean(sum(g) / 1000.0 for g in per_request_gaps)
    # Total tokens emitted across the window = batch_size requests, each emitting one
    # token per gap in the window (WINDOW_END_TOKEN - WINDOW_START_TOKEN tokens apiece).
    total_window_tokens = batch_size * (WINDOW_END_TOKEN - WINDOW_START_TOKEN)
    throughput = (
        aggregate_throughput_tokens_per_s(total_window_tokens, window_wall_time_s) if window_wall_time_s else None
    )

    measured_prefill_ms = None
    measured_engine_tpot_ms = None
    if raw.get("pre_metrics_text") and raw.get("final_metrics_text"):
        measured_prefill_ms = metrics_delta_mean_ms(raw["pre_metrics_text"], raw["final_metrics_text"],
                                                      "vllm:request_prefill_time_seconds")
        measured_engine_tpot_ms = metrics_delta_mean_ms(raw["pre_metrics_text"], raw["final_metrics_text"],
                                                          "vllm:request_time_per_output_token_seconds")

    kv_bpt = compute_model.kv_bytes_per_token(model, kv_cache_dtype_bytes=2) if hasattr(compute_model, "kv_bytes_per_token") else None
    from perf_model import memory_model
    kv_bpt = memory_model.kv_bytes_per_token(model, kv_cache_dtype_bytes=2)
    context_at_window = prompt_length + (WINDOW_START_TOKEN + WINDOW_END_TOKEN) // 2
    single_seq_ops = compute_model.decode_step_op_counts(model, context_at_window)
    decode_flops = single_seq_ops.total_flops * batch_size
    weight_bytes = model.estimated_weight_bytes
    decode_bytes = weight_bytes + kv_bpt * context_at_window * batch_size

    adherence = None
    if facts is not None:
        adherence = server_info_client.compare_requested_vs_resolved(
            raw["fixed_configuration"], facts, requested_max_num_seqs=raw["max_num_seqs_requested"],
        )

    cv = coefficient_of_variation(per_request_tpots) if len(per_request_tpots) > 1 else None
    gpu_utils = [s.get("gpu_util_percent") for s in raw.get("gpu_samples", []) if s.get("gpu_util_percent") is not None]
    gpu_mem = [s.get("memory_used_mib") for s in raw.get("gpu_samples", []) if s.get("memory_used_mib") is not None]

    return {
        "batch_size": batch_size, "prompt_length": prompt_length,
        "window_valid": window_valid, "n_requests_in_window": len(per_request_tpots),
        "client_tpot_ms": tpot_median, "client_tpot_p95_ms": all_tpot_p95,
        "batch_step_ms": batch_step_ms,
        "engine_tpot_ms": measured_engine_tpot_ms, "measured_prefill_ms": measured_prefill_ms,
        "aggregate_throughput_tokens_per_s": throughput,
        "cv_across_requests": cv,
        "decode_flops": decode_flops, "decode_bytes": decode_bytes, "context_at_window": context_at_window,
        "weight_bytes": weight_bytes,
        "gpu_util_mean_percent": statistics.mean(gpu_utils) if gpu_utils else None,
        "gpu_memory_mean_mib": statistics.mean(gpu_mem) if gpu_mem else None,
        "cpu_percent": raw.get("cpu_info", {}).get("cpu_percent"), "thread_count": raw.get("cpu_info", {}).get("thread_count"),
        "adherence": adherence, "resolved_num_gpu_blocks": facts.num_gpu_blocks if facts else None,
        "cudagraph_capture_sizes": (facts.raw_vllm_config.get("compilation_config", {}) or {}).get("cudagraph_capture_sizes") if facts else None,
        "classification": raw["classification"], "reference_match": raw.get("reference_match"),
        "oom_detected": raw.get("oom_detected_in_log"), "process_cleanup_status": raw.get("process_cleanup_status"),
    }


def metric_semantics_audit(rows: list[dict]) -> dict:
    """Cross-check engine-reported TPOT against client-observed TPOT and
    against alternative interpretations (step-latency vs step-latency/N)."""
    audit = []
    for r in rows:
        if r["client_tpot_ms"] is None or r["engine_tpot_ms"] is None:
            continue
        client = r["client_tpot_ms"]
        engine = r["engine_tpot_ms"]
        rel_diff = abs(client - engine) / client if client else None
        audit.append({
            "batch_size": r["batch_size"], "client_tpot_ms": client, "engine_tpot_ms": engine,
            "relative_difference": rel_diff,
            "matches_step_latency_hypothesis": rel_diff is not None and rel_diff < 0.15,
            "matches_step_over_n_hypothesis": (
                abs(client / r["batch_size"] - engine) / (client / r["batch_size"]) < 0.15
                if r["batch_size"] > 1 and client else None
            ),
        })
    return {"per_batch_size": audit,
            "conclusion": "engine_matches_per_request_step_latency" if audit and all(
                a["matches_step_latency_hypothesis"] for a in audit) else "unclear"}


def fit_and_evaluate(rows: list[dict]) -> dict:
    l128 = [r for r in rows if r["prompt_length"] == 128 and r["window_valid"]]
    calib = [r for r in l128 if r["batch_size"] in CALIBRATION_BATCH_SIZES]
    held = [r for r in l128 if r["batch_size"] in HELD_OUT_BATCH_SIZES]

    capture_sizes = next((r["cudagraph_capture_sizes"] for r in rows if r.get("cudagraph_capture_sizes")), [1, 2, 4])

    results = {}
    for name, (fit_fn, predict_fn) in bsm.MODELS.items():
        fitted = fit_fn(calib)
        calib_eval = bsm.evaluate(predict_fn, fitted, calib)
        held_eval = bsm.evaluate(predict_fn, fitted, held)
        for e in (calib_eval, held_eval):
            e.pop("per_row", None)
        results[name] = {"fitted": fitted.to_dict(), "calibration_error": calib_eval, "held_out_error": held_eval}

    p_fitted = bsm.fit_model_p(calib, capture_sizes)
    p_calib = bsm.evaluate(bsm.predict_model_p, p_fitted, calib)
    p_held = bsm.evaluate(bsm.predict_model_p, p_fitted, held)
    for e in (p_calib, p_held):
        e.pop("per_row", None)
    results["P_graph_bucket"] = {"fitted": p_fitted.to_dict(), "calibration_error": p_calib, "held_out_error": p_held}

    q_fitted = bsm.fit_model_q(l128)
    q_calib = bsm.evaluate(bsm.predict_model_q, q_fitted, calib)
    q_held = bsm.evaluate(bsm.predict_model_q, q_fitted, held)
    for e in (q_calib, q_held):
        e.pop("per_row", None)
    results["Q_roofline"] = {"fitted": q_fitted.to_dict(), "calibration_error": q_calib, "held_out_error": q_held}

    return {"n_calibration_rows": len(calib), "n_held_out_rows": len(held),
            "calibration_batch_sizes": sorted(CALIBRATION_BATCH_SIZES), "held_out_batch_sizes": sorted(HELD_OUT_BATCH_SIZES),
            "cudagraph_capture_sizes": capture_sizes, "models": results}


def ranking_by_objective(rows: list[dict]) -> dict:
    l128 = [r for r in rows if r["prompt_length"] == 128 and r["window_valid"] and r["client_tpot_ms"] and r["aggregate_throughput_tokens_per_s"]]
    if not l128:
        return {}
    min_tpot = min(l128, key=lambda r: r["client_tpot_ms"])
    max_throughput = max(l128, key=lambda r: r["aggregate_throughput_tokens_per_s"])
    slo_ms = 30.0  # example TPOT SLO
    under_slo = [r for r in l128 if r["client_tpot_ms"] <= slo_ms]
    best_under_slo = max(under_slo, key=lambda r: r["aggregate_throughput_tokens_per_s"]) if under_slo else None
    return {
        "min_tpot_objective": {"best_batch_size": min_tpot["batch_size"], "tpot_ms": min_tpot["client_tpot_ms"]},
        "max_throughput_objective": {"best_batch_size": max_throughput["batch_size"],
                                      "throughput": max_throughput["aggregate_throughput_tokens_per_s"]},
        "slo_constrained_objective": {"slo_tpot_ms": slo_ms,
                                       "best_batch_size": best_under_slo["batch_size"] if best_under_slo else None,
                                       "throughput": best_under_slo["aggregate_throughput_tokens_per_s"] if best_under_slo else None,
                                       "n_candidates_under_slo": len(under_slo)},
        "all_rows": [{"batch_size": r["batch_size"], "tpot_ms": r["client_tpot_ms"],
                       "throughput": r["aggregate_throughput_tokens_per_s"]} for r in sorted(l128, key=lambda r: r["batch_size"])],
    }


def evaluate_hypotheses(rows: list[dict], audit: dict, model_eval: dict, context_rows: list[dict]) -> dict:
    h = {}
    l128 = [r for r in rows if r["prompt_length"] == 128 and r["window_valid"]]

    h1_support = audit["conclusion"] == "engine_matches_per_request_step_latency"
    h["H1"] = {"result": "SUPPORTED" if h1_support else "INCONCLUSIVE", "evidence": audit}

    b1 = next((r for r in l128 if r["batch_size"] == 1), None)
    multi = [r for r in l128 if r["batch_size"] >= 2]
    h2_support = b1 is not None and multi and all(r["client_tpot_ms"] > 5 * b1["client_tpot_ms"] for r in multi)
    h["H2"] = {"result": "SUPPORTED" if h2_support else "INCONCLUSIVE",
               "evidence": {"batch1_tpot": b1["client_tpot_ms"] if b1 else None,
                            "multi_tpots": {r["batch_size"]: r["client_tpot_ms"] for r in multi}}}

    if multi:
        vals = [r["client_tpot_ms"] for r in multi]
        spread = (max(vals) - min(vals)) / statistics.mean(vals) if vals else None
        h["H3"] = {"result": "SUPPORTED" if spread is not None and spread < 0.10 else
                   ("INCONCLUSIVE" if spread is None else "REJECTED"),
                   "evidence": {"relative_spread_across_multi_batch": spread, "values": vals}}
    else:
        h["H3"] = {"result": "INCONCLUSIVE", "evidence": {}}

    h["H4"] = {"result": "SUPPORTED" if h2_support else "INCONCLUSIVE",
               "evidence": {"note": "same evidence as H2 -- batch=1 categorically distinct from batch>=2"}}

    h["H5"] = {"result": "INCONCLUSIVE",
               "evidence": {"note": "profiler attribution reported separately in kernel-attribution section; "
                                     "no single dominant component isolated with high confidence this slice"}}

    best_name = min((n for n in list(bsm.MODELS) + ["P_graph_bucket", "Q_roofline"]
                     if model_eval["models"][n]["held_out_error"].get("mae") is not None),
                     key=lambda n: model_eval["models"][n]["held_out_error"]["mae"], default=None)
    h6_mae = model_eval["models"][best_name]["held_out_error"]["mae"] if best_name else None
    h["H6"] = {"result": "SUPPORTED" if h6_mae is not None and h6_mae < 20 else
               ("INCONCLUSIVE" if h6_mae is None else "REJECTED"),
               "evidence": {"best_model": best_name, "held_out_mae": h6_mae}}

    h["H7"] = {"result": "SUPPORTED" if h6_mae is not None and h6_mae < 114.6 else "INCONCLUSIVE",
               "evidence": {"best_model_held_out_mae": h6_mae, "previous_universal_151ms_model_held_out_mae": 114.6}}

    ctx_by_batch: dict[int, list[float]] = {}
    for r in context_rows:
        if r["window_valid"] and r["client_tpot_ms"] is not None:
            ctx_by_batch.setdefault(r["batch_size"], []).append((r["prompt_length"], r["client_tpot_ms"]))
    h8_spreads = {}
    for b, vals in ctx_by_batch.items():
        tpots = [v[1] for v in vals]
        if len(tpots) > 1 and statistics.mean(tpots) > 0:
            h8_spreads[b] = (max(tpots) - min(tpots)) / statistics.mean(tpots)
    h8_material = any(s > 0.15 for s in h8_spreads.values())
    h["H8"] = {"result": "SUPPORTED" if h8_material else ("REJECTED" if h8_spreads else "INCONCLUSIVE"),
               "evidence": {"relative_spread_by_batch_size_across_context_lengths": h8_spreads}}

    return h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raws = load_raws(args.raw_dir)
    model = build_model_features()
    rows = [build_row(r, model) for r in raws]

    n_adherent = sum(1 for r in rows if r["adherence"] and r["adherence"]["derived_config_adherent"])
    audit = metric_semantics_audit(rows)
    model_eval = fit_and_evaluate(rows)
    ranking = ranking_by_objective(rows)
    context_rows = [r for r in rows if r["prompt_length"] != 128]
    hypotheses = evaluate_hypotheses(rows, audit, model_eval, context_rows)

    args.out.write_text(json.dumps({
        "n_groups": len(raws), "n_fully_adherent": n_adherent, "rows": rows,
        "metric_semantics_audit": audit, "model_evaluation": model_eval, "ranking": ranking,
        "hypotheses": hypotheses,
    }, indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out} groups={len(raws)} adherent={n_adherent}")


if __name__ == "__main__":
    main()
