#!/usr/bin/env python3
"""E2E-11 Phase 6 analysis: baseline vs direct_e2e8 vs unified_selector, batch 1/2/4/8."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_perf_model_results import build_model_features
from scripts.analyze_perf_model_results_e2e6 import build_row
from scripts.analyze_perf_model_results_e2e8 import summarize_reps

def load_raws(raw_dir: Path):
    grouped = {}
    for p in sorted(raw_dir.glob("*.json")):
        raw = json.loads(p.read_text())
        if "batch_size" not in raw or "candidate_id" not in raw:
            continue
        if raw.get("tiny_m_enable"):
            state = "direct_e2e8"
        elif raw.get("use_unified_selector"):
            state = "unified_selector"
        else:
            state = "baseline"
        grouped.setdefault((state, raw["batch_size"]), []).append(raw)
    return grouped

grouped = load_raws(Path("/tmp/perfmodel_e2e11/raw"))
model = build_model_features()
out = {}
for b in (1, 2, 4, 8):
    row = {}
    for state in ("baseline", "direct_e2e8", "unified_selector"):
        rows = [build_row(r, model) for r in grouped.get((state, b), [])]
        row[state] = summarize_reps(rows)
    base_tpot = row["baseline"]["client_tpot_ms"]["median"]
    direct_tpot = row["direct_e2e8"]["client_tpot_ms"]["median"]
    unified_tpot = row["unified_selector"]["client_tpot_ms"]["median"]
    row["unified_vs_baseline_pct"] = ((unified_tpot - base_tpot) / base_tpot * 100.0) if base_tpot else None
    row["unified_vs_direct_pct"] = ((unified_tpot - direct_tpot) / direct_tpot * 100.0) if direct_tpot else None
    row["direct_vs_baseline_pct"] = ((direct_tpot - base_tpot) / base_tpot * 100.0) if base_tpot else None
    out[str(b)] = row
    print(f"B={b}: baseline={base_tpot:.3f} direct_e2e8={direct_tpot:.3f} unified={unified_tpot:.3f} "
          f"unified_vs_baseline={row['unified_vs_baseline_pct']:.2f}% unified_vs_direct={row['unified_vs_direct_pct']:.2f}% "
          f"direct_vs_baseline={row['direct_vs_baseline_pct']:.2f}%")

# correctness: compare reference_completion text across states at each batch
correctness = {}
for b in (1, 2, 4, 8):
    texts = {}
    for state in ("baseline", "direct_e2e8", "unified_selector"):
        rows = grouped.get((state, b), [])
        texts[state] = rows[0].get("reference_completion", {}).get("text") if rows else None
    correctness[str(b)] = {"texts_equal_direct_vs_unified": texts["direct_e2e8"] == texts["unified_selector"],
                             "texts_equal_baseline_vs_unified": texts["baseline"] == texts["unified_selector"], "texts": texts}
    print(f"B={b} correctness: direct==unified={correctness[str(b)]['texts_equal_direct_vs_unified']} "
          f"baseline==unified={correctness[str(b)]['texts_equal_baseline_vs_unified']}")

Path("/tmp/e2e11_lm_head_aggregate.json").write_text(json.dumps(out, indent=2, default=str))
Path("/tmp/e2e11_lm_head_correctness.json").write_text(json.dumps(correctness, indent=2, default=str))
print("wrote /tmp/e2e11_lm_head_aggregate.json and /tmp/e2e11_lm_head_correctness.json")
