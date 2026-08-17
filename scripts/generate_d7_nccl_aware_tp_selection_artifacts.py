#!/usr/bin/env python3
"""Generate D7 NCCL-aware TP-selection evidence artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.vllm_adapter.tp_cost_model import (  # noqa: E402
    FittedRegression,
    MODEL_IDENTITY_FEATURES,
    TPCostModel,
    estimated_communication_bytes,
    load_communication_predictor,
)

D6_DIR = REPO_ROOT / "results/runtime_paths/distributed_d6_compiler_owned_tp_selection"
D7_DIR = REPO_ROOT / "results/runtime_paths/distributed_d7_nccl_aware_tp_selection"
NCCL_DIR = REPO_ROOT / "results/runtime_paths/nccl_calibration"
MODEL_BY_LABEL = {
    "qwen05b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
WORKLOAD_RE = re.compile(r"^in(?P<input>\d+)_out(?P<output>\d+)_c(?P<concurrency>\d+)$")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - fail closed with path context.
        raise SystemExit(f"malformed or unreadable JSON input {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_workload(workload_id: str) -> dict[str, int]:
    match = WORKLOAD_RE.match(workload_id)
    if not match:
        raise SystemExit(f"malformed workload id: {workload_id}")
    return {k: int(v) for k, v in match.groupdict().items()}


def load_cost_model() -> TPCostModel:
    fitted = read_json(REPO_ROOT / "results/runtime_paths/distributed_d5_compiler_tp_policy/cost_model_fitted.json")
    model = TPCostModel()
    for tp in (1, 2):
        block = fitted["throughput_models"][str(tp)]
        model.throughput_models[tp] = FittedRegression(
            tp_degree=tp,
            coefficients=block["coefficients"],
            n_samples=block["n_samples"],
            r_squared=block["r_squared"],
        )
    model.frozen = True
    return model


def candidate_evidence(decision: dict[str, Any], model_features: dict[str, Any], tp: int) -> dict[str, Any]:
    comm = decision["communication"]
    prefix = f"tp{tp}"
    before = decision[f"predicted_{prefix}_throughput_before_communication"]
    after = decision[f"predicted_{prefix}_throughput"]
    bytes_value = estimated_communication_bytes(model_features, tp)
    time_us = comm[f"estimated_nccl_comm_time_us_{prefix}"]
    return {
        "candidate_id": prefix,
        "world_size": tp,
        "tensor_parallel_size": tp,
        "estimated_communication_bytes": bytes_value,
        "estimated_nccl_comm_time_us": time_us,
        "communication_collective_kind": comm["communication_collective_kind"],
        "communication_profile_id": comm["communication_profile_id"],
        "communication_predictor_kind": comm["communication_predictor_kind"],
        "topology_class": comm["topology_class"],
        "p2p_available": comm["p2p_available"],
        "nccl_transport": comm["nccl_transport"],
        "predicted_tp_throughput_before_communication": before,
        "predicted_tp_throughput_after_communication": after,
        "communication_changed_tp_decision": decision["communication_changed_decision"],
        "profitability": {
            "predicted_throughput_before_communication_tokens_per_s": before,
            "predicted_throughput_tokens_per_s": after,
            "estimated_communication_bytes": bytes_value,
            "estimated_nccl_comm_time_us": time_us,
            "communication_profile_id": comm["communication_profile_id"],
            "communication_predictor_kind": comm["communication_predictor_kind"],
            "nccl_transport": comm["nccl_transport"],
            "p2p_available": comm["p2p_available"],
        },
    }


def evaluate(decisions: list[dict[str, Any]], measured: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    per_cell = []
    regrets = []
    for row in decisions:
        key = (row["model_label"], row["workload_id"])
        actual = measured[key]
        selected = row["compiler_selected_tp"]
        tp1 = actual["actual_tp1_throughput"]
        tp2 = actual["actual_tp2_throughput"]
        chosen = tp1 if selected == "tp1" else tp2
        best = max(tp1, tp2)
        regret = 0.0 if best <= 0 else 100.0 * (best - chosen) / best
        regrets.append(regret)
        per_cell.append({
            "model_label": row["model_label"],
            "workload_id": row["workload_id"],
            "compiler_selected_tp": selected,
            "oracle_choice": actual["oracle_choice"],
            "match": selected == actual["oracle_choice"],
            "actual_tp1_throughput": tp1,
            "actual_tp2_throughput": tp2,
            "compiler_regret_pct": regret,
            "communication_changed_decision": row["communication_changed_decision"],
        })
    return {
        "n_held_out_cells": len(per_cell),
        "tp1_selections": sum(1 for r in decisions if r["compiler_selected_tp"] == "tp1"),
        "tp2_selections": sum(1 for r in decisions if r["compiler_selected_tp"] == "tp2"),
        "oracle_match_rate": sum(1 for r in per_cell if r["match"]) / len(per_cell),
        "mean_compiler_regret_pct": sum(regrets) / len(regrets),
        "worst_case_compiler_regret_pct": max(regrets),
        "per_cell": per_cell,
    }


def main() -> None:
    predictor = load_communication_predictor(
        read_json(NCCL_DIR / "communication_cost_profile.json"),
        read_json(NCCL_DIR / "fit_report.json"),
    )
    cost_model = load_cost_model()
    d6_decisions = read_json(D6_DIR / "fresh_compilation_decisions.json")
    measured_eval = read_json(D6_DIR / "held_out_evaluation_d6.json")
    measured_by_key = {(r["model_label"], r["workload_id"]): r for r in measured_eval["per_cell"]}

    compilation_dir = D7_DIR / "fresh_compilations"
    compilation_dir.mkdir(parents=True, exist_ok=True)
    d7_decisions = []
    comparison = []
    flips = []

    for old in d6_decisions:
        workload = parse_workload(old["workload_id"])
        model_features = MODEL_IDENTITY_FEATURES[MODEL_BY_LABEL[old["model_label"]]]
        decision = cost_model.decide(
            model_features=model_features,
            input_length=workload["input"],
            output_length=workload["output"],
            concurrency=workload["concurrency"],
            gpu_total_mb=24564.0,
            gpu_memory_utilization=0.9,
            max_model_len=2048,
            max_num_seqs=4,
            communication_calibration=predictor,
        )
        selected = decision["decision"]
        row = {
            "model_label": old["model_label"],
            "workload_id": old["workload_id"],
            "compiler_selected_tp": selected,
            "pre_communication_selected_tp": decision["pre_communication_decision"],
            "selection_reason": "profitable_tp1_selected_nccl_adjusted_throughput_higher"
            if selected == "tp1" else "profitable_tp2_selected_nccl_adjusted_throughput_higher",
            "policy_id": "d7_nccl_aware_profitability_selector_v1",
            "plan_has_distributed_key": selected == "tp2",
            "plan_tensor_parallel_size": 2 if selected == "tp2" else 1,
            "predicted_tp1_throughput_before_communication": decision[
                "predicted_tp1_throughput_before_communication"
            ],
            "predicted_tp2_throughput_before_communication": decision[
                "predicted_tp2_throughput_before_communication"
            ],
            "predicted_tp1_throughput": decision["predicted_tp1_throughput"],
            "predicted_tp2_throughput": decision["predicted_tp2_throughput"],
            "estimated_communication_bytes_tp1": decision["communication"]["estimated_communication_bytes_tp1"],
            "estimated_communication_bytes_tp2": decision["communication"]["estimated_communication_bytes_tp2"],
            "estimated_nccl_comm_time_us_tp1": decision["communication"]["estimated_nccl_comm_time_us_tp1"],
            "estimated_nccl_comm_time_us_tp2": decision["communication"]["estimated_nccl_comm_time_us_tp2"],
            "communication_profile_id": decision["communication"]["communication_profile_id"],
            "communication_predictor_kind": decision["communication"]["communication_predictor_kind"],
            "nccl_transport": decision["communication"]["nccl_transport"],
            "p2p_available": decision["communication"]["p2p_available"],
            "communication_changed_decision": decision["communication_changed_decision"],
        }
        d7_decisions.append(row)

        actual = measured_by_key[(old["model_label"], old["workload_id"])]
        changed_vs_d6 = old["compiler_selected_tp"] != selected
        cmp_row = {
            "model_label": old["model_label"],
            "workload_id": old["workload_id"],
            "d6_selected_tp": old["compiler_selected_tp"],
            "d7_selected_tp": selected,
            "pre_communication_selected_tp": decision["pre_communication_decision"],
            "changed_vs_d6": changed_vs_d6,
            "communication_changed_decision": decision["communication_changed_decision"],
            "measured_winner": actual["oracle_choice"],
            "d7_matches_measured_winner": selected == actual["oracle_choice"],
        }
        comparison.append(cmp_row)
        if changed_vs_d6:
            flips.append(cmp_row)

        stem = f"{old['model_label']}_{old['workload_id']}"
        write_json(compilation_dir / f"{stem}_candidate_evidence.json", {
            "schema_version": "d7_nccl_aware_tp_selection_evidence.v1",
            "policy_id": "d7_nccl_aware_profitability_selector_v1",
            "model_label": old["model_label"],
            "workload_id": old["workload_id"],
            "selected_candidate_id": selected,
            "pre_communication_selected_candidate_id": decision["pre_communication_decision"],
            "communication_changed_decision": decision["communication_changed_decision"],
            "measured_winner": actual["oracle_choice"],
            "d7_matches_measured_winner": selected == actual["oracle_choice"],
            "candidates": [
                candidate_evidence(decision, model_features, 1),
                candidate_evidence(decision, model_features, 2),
            ],
        })
        write_json(compilation_dir / f"{stem}_decision.json", row)

    held_out = evaluate(d7_decisions, measured_by_key)
    write_json(D7_DIR / "fresh_compilation_decisions.json", d7_decisions)
    write_json(D7_DIR / "candidate_evidence.json", [
        read_json(compilation_dir / f"{r['model_label']}_{r['workload_id']}_candidate_evidence.json")
        for r in d7_decisions
    ])
    write_json(D7_DIR / "held_out_evaluation.json", held_out)
    write_json(D7_DIR / "comparison_against_d6_decisions.json", {
        "n_cells": len(comparison),
        "n_decision_flips_vs_d6": len(flips),
        "decision_flips": flips,
        "per_cell": comparison,
    })
    write_json(D7_DIR / "decision_flips_caused_by_nccl_aware_cost.json", {
        "count": len(flips),
        "flips": flips,
        "all_flips_match_measured_winner": all(f["d7_matches_measured_winner"] for f in flips),
    })
    write_json(D7_DIR / "artifact_summary.json", {
        "truth_boundary": (
            "D7 NCCL-aware offline compiler-equivalent validation using Phase 1 measured "
            "nccl-tests calibration; no vLLM source modified or launched."
        ),
        "profile_id": predictor.profile_id,
        "communication_predictor_kind": predictor.predictor_kind,
        "topology_class": predictor.topology_class,
        "p2p_available": predictor.p2p_available,
        "nccl_transport": predictor.nccl_transport,
        "n_held_out_cells": len(d7_decisions),
        "n_decision_flips_vs_d6": len(flips),
        "oracle_match_rate": held_out["oracle_match_rate"],
    })
    print(f"wrote {D7_DIR}")


if __name__ == "__main__":
    main()
