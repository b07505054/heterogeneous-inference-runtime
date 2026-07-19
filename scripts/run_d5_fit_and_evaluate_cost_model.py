"""D5: fit the compiler TP cost model on calibration-split data only, freeze
it, then evaluate on held-out-split data against always-TP1, always-TP2, and
an offline oracle. Regret and improvement are computed from real measured
throughput -- never predicted values -- for every row in the comparison.

Reads one or more `{sweep_dir}/tp1_sweep_full.json` + `tp2_sweep_full.json`
pairs (one pair per model), each row already labeled "calibration" or
"held_out" by run_d5_calibration_sweep.py using the pre-declared split rule
in tp_workload_matrix.py. Never re-derives or re-checks the split here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_WORKLOAD_ID_RE = re.compile(r"^in(\d+)_out(\d+)_c(\d+)$")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.vllm_adapter.tp_cost_model import (  # noqa: E402
    MODEL_IDENTITY_FEATURES,
    TPCostModel,
    build_feature_vector,
)

RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d5_compiler_tp_policy"

# Each entry: (label, sweep_dir, real_hf_model_id, gpu_total_mb).
# gpu_total_mb is the real per-GPU total from this project's RTX 4090 host
# (24564 MiB, confirmed via nvidia-smi in every D4B/D5 gpu_inventory
# artifact) -- not a spec-sheet number.
SWEEP_SOURCES: list[tuple[str, Path, str, float, str]] = [
    ("qwen2.5-0.5b", RESULTS_DIR, "Qwen/Qwen2.5-0.5B-Instruct", 24564.0, ""),
    ("qwen2.5-7b", RESULTS_DIR / "7b", "Qwen/Qwen2.5-7B-Instruct", 24564.0, "_7b"),
]


def _load_rows(label: str, sweep_dir: Path, real_hf_model_id: str, suffix: str) -> list[dict[str, Any]]:
    model_features = MODEL_IDENTITY_FEATURES[real_hf_model_id]
    rows = []
    for tp_degree, fname in ((1, f"tp1_sweep_full{suffix}.json"), (2, f"tp2_sweep_full{suffix}.json")):
        path = sweep_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"missing required sweep artifact: {path}")
        bundle = json.loads(path.read_text())
        max_model_len = bundle["launch_spec"]["max_model_len"]
        max_num_seqs = bundle["launch_spec"]["max_num_seqs"]
        gpu_memory_utilization = bundle["launch_spec"]["gpu_memory_utilization"]
        for wr in bundle["workload_results"]:
            # workload shape is authoritative from the workload_id, not from
            # a single request's realized token counts (those can vary by a
            # token or two around the target).
            m = _WORKLOAD_ID_RE.match(wr["workload_id"])
            if not m:
                raise ValueError(f"unparseable workload_id: {wr['workload_id']!r}")
            input_length, output_length, concurrency = int(m.group(1)), int(m.group(2)), int(m.group(3))
            fv = build_feature_vector(model_features, tp_degree, input_length=input_length,
                                       output_length=output_length, concurrency=concurrency)
            rows.append({
                "model_label": label, "real_hf_model_id": real_hf_model_id, "workload_id": wr["workload_id"],
                "tp_degree": tp_degree, "split": wr["split"], "feature_vector": fv,
                "aggregate_throughput_tokens_per_s": wr["aggregate_throughput_tokens_per_s"],
                "mean_e2e_latency_s": wr["mean_e2e_latency_s"],
                "input_length": input_length, "output_length": output_length, "concurrency": concurrency,
                "max_model_len": max_model_len, "max_num_seqs": max_num_seqs,
                "gpu_memory_utilization": gpu_memory_utilization,
            })
    return rows


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    for label, sweep_dir, real_hf_model_id, gpu_total_mb, suffix in SWEEP_SOURCES:
        rows = _load_rows(label, sweep_dir, real_hf_model_id, suffix)
        for r in rows:
            r["gpu_total_mb"] = gpu_total_mb
        all_rows.extend(rows)

    calibration_rows = [r for r in all_rows if r["split"] == "calibration"]
    held_out_rows = [r for r in all_rows if r["split"] == "held_out"]
    print(f"calibration rows: {len(calibration_rows)}, held_out rows: {len(held_out_rows)}")

    model = TPCostModel()
    model.fit(calibration_rows)
    (RESULTS_DIR / "cost_model_fitted.json").write_text(json.dumps(model.to_dict(), indent=2) + "\n")
    print(f"wrote {RESULTS_DIR / 'cost_model_fitted.json'}")
    for tp_degree, reg in model.throughput_models.items():
        print(f"  TP{tp_degree} throughput regression: R^2={reg.r_squared:.4f} n={reg.n_samples}")

    # Held-out evaluation: group held-out rows by (model_label, workload_id)
    # since each workload has one TP1 row and one TP2 row -- both real,
    # measured, and only now (post-freeze) compared against each other.
    held_out_by_cell: dict[tuple[str, str], dict[int, dict]] = {}
    for r in held_out_rows:
        key = (r["model_label"], r["workload_id"])
        held_out_by_cell.setdefault(key, {})[r["tp_degree"]] = r

    evaluation_rows = []
    for (model_label, workload_id), by_tp in held_out_by_cell.items():
        if 1 not in by_tp or 2 not in by_tp:
            continue
        r1, r2 = by_tp[1], by_tp[2]
        model_features = MODEL_IDENTITY_FEATURES[r1["real_hf_model_id"]]
        decision = model.decide(
            model_features=model_features, input_length=r1["input_length"], output_length=r1["output_length"],
            concurrency=r1["concurrency"], gpu_total_mb=r1["gpu_total_mb"],
            gpu_memory_utilization=r1["gpu_memory_utilization"], max_model_len=r1["max_model_len"],
            max_num_seqs=r1["max_num_seqs"],
        )
        actual_tp1_throughput = r1["aggregate_throughput_tokens_per_s"] or 0.0
        actual_tp2_throughput = r2["aggregate_throughput_tokens_per_s"] or 0.0
        oracle_throughput = max(actual_tp1_throughput, actual_tp2_throughput)
        oracle_choice = "tp1" if actual_tp1_throughput >= actual_tp2_throughput else "tp2"
        chosen_throughput = actual_tp1_throughput if decision["decision"] == "tp1" else (
            actual_tp2_throughput if decision["decision"] == "tp2" else 0.0)
        always_tp1_regret_pct = (oracle_throughput - actual_tp1_throughput) / oracle_throughput * 100 if oracle_throughput else 0.0
        always_tp2_regret_pct = (oracle_throughput - actual_tp2_throughput) / oracle_throughput * 100 if oracle_throughput else 0.0
        compiler_regret_pct = (oracle_throughput - chosen_throughput) / oracle_throughput * 100 if oracle_throughput else 0.0
        evaluation_rows.append({
            "model_label": model_label, "workload_id": workload_id, "compiler_decision": decision["decision"],
            "decision_reason": decision["reason"], "oracle_choice": oracle_choice,
            "compiler_matches_oracle": decision["decision"] == oracle_choice,
            "actual_tp1_throughput": actual_tp1_throughput, "actual_tp2_throughput": actual_tp2_throughput,
            "oracle_throughput": oracle_throughput, "chosen_throughput": chosen_throughput,
            "always_tp1_regret_pct": always_tp1_regret_pct, "always_tp2_regret_pct": always_tp2_regret_pct,
            "compiler_regret_pct": compiler_regret_pct,
        })

    n = len(evaluation_rows)
    summary = {
        "n_held_out_cells_evaluated": n,
        "compiler_oracle_match_rate": sum(1 for e in evaluation_rows if e["compiler_matches_oracle"]) / n if n else None,
        "mean_compiler_regret_pct": sum(e["compiler_regret_pct"] for e in evaluation_rows) / n if n else None,
        "mean_always_tp1_regret_pct": sum(e["always_tp1_regret_pct"] for e in evaluation_rows) / n if n else None,
        "mean_always_tp2_regret_pct": sum(e["always_tp2_regret_pct"] for e in evaluation_rows) / n if n else None,
        "per_cell": evaluation_rows,
    }
    (RESULTS_DIR / "held_out_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {RESULTS_DIR / 'held_out_evaluation.json'}")
    print(f"compiler/oracle match rate: {summary['compiler_oracle_match_rate']}")
    print(f"mean regret -- compiler: {summary['mean_compiler_regret_pct']:.3f}% "
          f"always-TP1: {summary['mean_always_tp1_regret_pct']:.3f}% "
          f"always-TP2: {summary['mean_always_tp2_regret_pct']:.3f}%")


if __name__ == "__main__":
    main()
