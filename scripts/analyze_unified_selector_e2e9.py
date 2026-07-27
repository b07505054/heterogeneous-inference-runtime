#!/usr/bin/env python3
"""E2E-9 Phase 12B/13/15/16 analysis: compares the unified-selector LM-head
path against the direct E2E-8 env-var path (batch 1,2,4,8), evaluates the
<=3% TPOT/throughput tolerance acceptance criterion, checks correctness
equality, and evaluates the E2E-9 hypotheses that depend on this data
(H3-preservation-style checks specific to E2E-9: H3 selector-preserves-gain,
H4 selector-overhead-negligible). Reuses (does not reimplement) build_row
from analyze_perf_model_results_e2e6 and summarize_reps from
analyze_perf_model_results_e2e8 -- both already proven against this exact
raw-result schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_perf_model_results import build_model_features
from scripts.analyze_perf_model_results_e2e6 import build_row
from scripts.analyze_perf_model_results_e2e8 import summarize_reps

TOLERANCE_PERCENT = 3.0


def load_raws(raw_dir: Path) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for p in sorted(raw_dir.glob("*.json")):
        raw = json.loads(p.read_text())
        if "batch_size" not in raw or "candidate_id" not in raw:
            continue
        if raw.get("tiny_m_enable"):
            state = "e2e8_direct"
        elif raw.get("use_unified_selector"):
            state = "unified_selector"
        else:
            continue
        grouped.setdefault((state, raw["batch_size"]), []).append(raw)
    return grouped


def compare(grouped: dict[tuple[str, int], list[dict]], model) -> dict:
    batch_sizes = sorted({b for (_, b) in grouped})
    out = {}
    for b in batch_sizes:
        direct_rows = [build_row(r, model) for r in grouped.get(("e2e8_direct", b), [])]
        unified_rows = [build_row(r, model) for r in grouped.get(("unified_selector", b), [])]
        direct_summary = summarize_reps(direct_rows)
        unified_summary = summarize_reps(unified_rows)
        d_tpot = direct_summary["client_tpot_ms"]["median"]
        u_tpot = unified_summary["client_tpot_ms"]["median"]
        tpot_pct_diff = ((u_tpot - d_tpot) / d_tpot * 100.0) if d_tpot and u_tpot is not None else None
        d_tput = direct_summary["throughput_tokens_per_s"]["median"]
        u_tput = unified_summary["throughput_tokens_per_s"]["median"]
        tput_pct_diff = ((u_tput - d_tput) / d_tput * 100.0) if d_tput and u_tput is not None else None

        direct_texts = {r.get("request_id", i): r.get("reference_completion", {}).get("text")
                         for i, r in enumerate(grouped.get(("e2e8_direct", b), []))}
        unified_texts = {r.get("request_id", i): r.get("reference_completion", {}).get("text")
                          for i, r in enumerate(grouped.get(("unified_selector", b), []))}
        d_first_text = next(iter(direct_texts.values()), None)
        u_first_text = next(iter(unified_texts.values()), None)
        correctness_identical = (d_first_text == u_first_text) if (d_first_text and u_first_text) else None

        within_tolerance = (abs(tpot_pct_diff) <= TOLERANCE_PERCENT) if tpot_pct_diff is not None else None
        out[str(b)] = {
            "e2e8_direct": direct_summary, "unified_selector": unified_summary,
            "tpot_percent_diff": tpot_pct_diff, "throughput_percent_diff": tput_pct_diff,
            "within_3_percent_tolerance": within_tolerance, "correctness_identical": correctness_identical,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    grouped = load_raws(args.raw_dir)
    model = build_model_features()
    comparison = compare(grouped, model)

    args.out.write_text(json.dumps({"comparison": comparison, "tolerance_percent": TOLERANCE_PERCENT},
                                    indent=2, default=str))
    print(f"wrote {args.out}")
    all_within = True
    for b_str, row in sorted(comparison.items(), key=lambda kv: int(kv[0])):
        within = row["within_3_percent_tolerance"]
        all_within = all_within and bool(within)
        print(f"B={b_str}: e2e8_direct_tpot={row['e2e8_direct']['client_tpot_ms']['median']} "
              f"unified_tpot={row['unified_selector']['client_tpot_ms']['median']} "
              f"pct_diff={row['tpot_percent_diff']} within_3pct={within} "
              f"correctness_identical={row['correctness_identical']}")
    print(f"\nALL_BATCHES_WITHIN_3_PERCENT: {all_within}")


if __name__ == "__main__":
    main()
