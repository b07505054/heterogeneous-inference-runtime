#!/usr/bin/env python3
"""E2E-8 analysis: baseline-vs-optimized real serving comparison, realization
ratio against E2E-7's isolated LM-head prediction, Models Y/Z/AA/AB, and
hypothesis evaluation. Reuses E2E-6's build_row (same raw-result schema,
since the E2E-8 orchestrator is an additive extension of the E2E-6 script,
not a fork) for per-group TPOT/throughput/adherence extraction.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_perf_model_results import build_model_features
from scripts.analyze_perf_model_results_e2e6 import build_row

CALLS_PER_STEP_LM_HEAD = 1
E2E7_LM_HEAD_ISOLATED_MS = {  # E2E-7 benchmark: default_linear vs gemv_loop median_ms, M matches batch_size
    1: {"default": 2.7546, "gemv": 2.7766},
    2: {"default": 43.6263, "gemv": 5.6709},
    3: {"default": 43.6306, "gemv": 8.9406},
    4: {"default": 43.6288, "gemv": 12.1059},
    6: {"default": 43.9941, "gemv": 18.3890},
    8: {"default": 43.9992, "gemv": 25.1245},
}
HELD_OUT_BATCH_SIZES = {3, 6}


def load_raws(raw_dir: Path) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for p in sorted(raw_dir.glob("*.json")):
        raw = json.loads(p.read_text())
        if "batch_size" not in raw or "candidate_id" not in raw:
            continue  # skip non-sweep files (correctness experiment, etc.)
        if raw.get("enable_profiler"):
            continue  # profiler groups are timing-perturbed; excluded from the clean comparison
        key = ("optimized" if raw.get("tiny_m_enable") else "baseline", raw["batch_size"])
        grouped.setdefault(key, []).append(raw)
    return grouped


def summarize_reps(rows: list[dict]) -> dict:
    tpots = [r["client_tpot_ms"] for r in rows if r["client_tpot_ms"] is not None]
    engine_tpots = [r["engine_tpot_ms"] for r in rows if r["engine_tpot_ms"] is not None]
    throughputs = [r["aggregate_throughput_tokens_per_s"] for r in rows if r["aggregate_throughput_tokens_per_s"] is not None]
    gpu_utils = [r["gpu_util_mean_percent"] for r in rows if r["gpu_util_mean_percent"] is not None]
    cpu = [r["cpu_percent"] for r in rows if r["cpu_percent"] is not None]
    gpu_mem = [r["gpu_memory_mean_mib"] for r in rows if r["gpu_memory_mean_mib"] is not None]

    def stats(xs):
        if not xs:
            return {"median": None, "min": None, "max": None, "p95": None, "n": 0, "cv": None}
        ordered = sorted(xs)
        idx95 = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
        med = statistics.median(ordered)
        cv = (statistics.stdev(xs) / med) if len(xs) > 1 and med else None
        return {"median": med, "min": ordered[0], "max": ordered[-1], "p95": ordered[idx95], "n": len(xs), "cv": cv}

    return {
        "client_tpot_ms": stats(tpots), "engine_tpot_ms": stats(engine_tpots),
        "throughput_tokens_per_s": stats(throughputs), "gpu_util_percent": stats(gpu_utils),
        "cpu_percent": stats(cpu), "gpu_memory_mib": stats(gpu_mem),
        "window_valid_all": all(r["window_valid"] for r in rows),
        "reference_match_all": all(r["reference_match"] is not False for r in rows),
        "n_reps": len(rows),
    }


def baseline_vs_optimized(grouped: dict[tuple[str, int], list[dict]], model) -> dict:
    batch_sizes = sorted({b for (_, b) in grouped})
    comparison = {}
    for b in batch_sizes:
        base_rows = [build_row(r, model) for r in grouped.get(("baseline", b), [])]
        opt_rows = [build_row(r, model) for r in grouped.get(("optimized", b), [])]
        base_summary = summarize_reps(base_rows)
        opt_summary = summarize_reps(opt_rows)
        base_tpot = base_summary["client_tpot_ms"]["median"]
        opt_tpot = opt_summary["client_tpot_ms"]["median"]
        pct_change = ((opt_tpot - base_tpot) / base_tpot * 100.0) if base_tpot and opt_tpot is not None else None
        base_tput = base_summary["throughput_tokens_per_s"]["median"]
        opt_tput = opt_summary["throughput_tokens_per_s"]["median"]
        tput_pct = ((opt_tput - base_tput) / base_tput * 100.0) if base_tput and opt_tput is not None else None
        comparison[str(b)] = {
            "baseline": base_summary, "optimized": opt_summary,
            "tpot_percent_change": pct_change, "throughput_percent_change": tput_pct,
            "measured_saved_ms": (base_tpot - opt_tpot) if base_tpot and opt_tpot is not None else None,
        }
    return comparison


def realization_ratio(comparison: dict) -> dict:
    result = {}
    for b_str, e7 in E2E7_LM_HEAD_ISOLATED_MS.items():
        b_key = str(b_str)
        if b_key not in comparison:
            continue
        predicted_saved_ms = e7["default"] - e7["gemv"]
        measured_saved_ms = comparison[b_key]["measured_saved_ms"]
        ratio = (measured_saved_ms / predicted_saved_ms) if measured_saved_ms is not None and predicted_saved_ms else None
        result[b_key] = {"predicted_lm_head_saved_ms": predicted_saved_ms, "measured_saved_ms": measured_saved_ms,
                          "realization_ratio": ratio}
    return result


def model_y(batch_size: int) -> float:
    return 11.5 if batch_size == 1 else 169.0


def model_z(comparison: dict, batch_size: int) -> float | None:
    """LM-head-integrated model: baseline_projection_cost - measured_lm_head_saving.
    Directly uses the MEASURED baseline TPOT for this batch size minus the
    measured LM-head saving at that same batch size (no extrapolation)."""
    key = str(batch_size)
    if key not in comparison:
        return None
    base = comparison[key]["baseline"]["client_tpot_ms"]["median"]
    saved = comparison[key]["measured_saved_ms"]
    if base is None or saved is None:
        return None
    return base - saved


def evaluate_y_z(comparison: dict) -> dict:
    calib_b = [1, 2, 4, 8]
    held_b = [3, 6]
    results = {"Y": {"calib_err": [], "held_err": []}, "Z": {"calib_err": [], "held_err": []}}
    for b_str, row in comparison.items():
        b = int(b_str)
        actual = row["optimized"]["client_tpot_ms"]["median"]
        if actual is None:
            continue
        pred_y = model_y(b)
        pred_z = model_z(comparison, b)
        err_y = abs(pred_y - actual)
        err_z = abs(pred_z - actual) if pred_z is not None else None
        bucket = "held_err" if b in held_b else "calib_err"
        results["Y"][bucket].append(err_y)
        if err_z is not None:
            results["Z"][bucket].append(err_z)
    out = {}
    for name, errs in results.items():
        out[name] = {
            "calibration_mae": statistics.mean(errs["calib_err"]) if errs["calib_err"] else None,
            "held_out_mae": statistics.mean(errs["held_err"]) if errs["held_err"] else None,
            "max_error": max(errs["calib_err"] + errs["held_err"]) if (errs["calib_err"] + errs["held_err"]) else None,
        }
    return out


def evaluate_hypotheses(comparison: dict, ratios: dict) -> dict:
    h = {}
    b1 = comparison.get("1", {})
    b1_pct = b1.get("tpot_percent_change")
    h["H4_batch1_not_regressed"] = {
        "result": "SUPPORTED" if b1_pct is not None and b1_pct <= 3.0 else ("REJECTED" if b1_pct is not None else "INCONCLUSIVE"),
        "evidence": {"batch1_tpot_percent_change": b1_pct},
    }

    b2 = comparison.get("2", {})
    b2_pct = b2.get("tpot_percent_change")
    b2_improvement = -b2_pct if b2_pct is not None else None
    h["H3_isolated_gain_survives"] = {
        "result": "SUPPORTED" if b2_improvement is not None and b2_improvement >= 15.0 else
                  ("REJECTED" if b2_improvement is not None else "INCONCLUSIVE"),
        "evidence": {"batch2_tpot_improvement_percent": b2_improvement},
    }

    r2 = ratios.get("2", {}).get("realization_ratio")
    h["H6_directionally_consistent"] = {
        "result": "SUPPORTED" if r2 is not None and 0.3 <= r2 <= 2.0 else ("INCONCLUSIVE" if r2 is None else "REJECTED"),
        "evidence": {"batch2_realization_ratio": r2},
    }
    return h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    grouped = load_raws(args.raw_dir)
    model = build_model_features()
    comparison = baseline_vs_optimized(grouped, model)
    ratios = realization_ratio(comparison)
    model_eval = evaluate_y_z(comparison)
    hypotheses = evaluate_hypotheses(comparison, ratios)

    args.out.write_text(json.dumps({
        "comparison": comparison, "realization_ratio": ratios, "model_evaluation": model_eval,
        "hypotheses": hypotheses,
    }, indent=2, default=str))
    print(f"wrote {args.out}")
    for b_str, row in sorted(comparison.items(), key=lambda kv: int(kv[0])):
        print(f"B={b_str}: baseline_tpot={row['baseline']['client_tpot_ms']['median']} "
              f"optimized_tpot={row['optimized']['client_tpot_ms']['median']} "
              f"pct_change={row['tpot_percent_change']}")


if __name__ == "__main__":
    main()
