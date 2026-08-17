#!/usr/bin/env python3
"""Phase 4D: experimentally locate the TP1/TP2 break-even boundary.

This wrapper reuses scripts/run_d8_vllm_nccl_attribution.py and its validated
Phase 4B/4C wall-clock NCCL attribution. It does not touch selector logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
D8_PATH = REPO_ROOT / "scripts/run_d8_vllm_nccl_attribution.py"
BREAK_DIR = REPO_ROOT / "results/runtime_paths/distributed_d8_vllm_nccl_attribution/break_even"
RAW_DIR = BREAK_DIR / "raw"
PLAN_DIR = BREAK_DIR / "plans"
D2_DIR = REPO_ROOT / "results/runtime_paths/distributed_d2_qwen_pipeline"

MODEL_RUNS = {
    "qwen2.5-0.5b": {
        "hf_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "plan_model_id": "qwen2.5-0.5b",
        "hidden_size": 896,
        "num_layers": 24,
        "num_attention_heads": 14,
        "num_kv_heads": 2,
        "workloads": ["in32_out32_c1", "in32_out32_c4", "in32_out32_c8"],
        "reuse_multicell_if_available": True,
    },
    "qwen2.5-7b": {
        "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
        "plan_model_id": "qwen2.5-7b",
        "hidden_size": 3584,
        "num_layers": 28,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "workloads": ["in32_out32_c1", "in32_out32_c4", "in32_out32_c8"],
        "reuse_multicell_if_available": False,
    },
}


def load_d8():
    spec = importlib.util.spec_from_file_location("d8_attribution", D8_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_json(name: str, payload: Any) -> None:
    path = BREAK_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {path}")


def write_model_plan(src: Path, dst: Path, cfg: dict[str, Any]) -> Path:
    data = json.loads(src.read_text())
    data["model_identity"].update({
        "model_id": cfg["plan_model_id"],
        "hidden_size": cfg["hidden_size"],
        "num_layers": cfg["num_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg["num_kv_heads"],
        "truth_boundary": "phase4d_temporary_model_identity_for_matched_break_even_experiment",
    })
    data["plan_id"] = f"phase4d-{cfg['plan_model_id']}-{src.stem}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return dst


def configure_d8_for_model(d8, key: str, cfg: dict[str, Any]) -> Path:
    out = RAW_DIR / key
    d8.RESULTS_DIR = out
    d8.LOG_DIR = out / "logs"
    d8.TRACE_DIR = out / "traces"
    d8.MULTICELL_DIR = out / "multicell"
    d8.MODEL_ID = cfg["hf_model_id"]
    if cfg["plan_model_id"] == "qwen2.5-0.5b":
        d8.TP1_PLAN_PATH = D2_DIR / "real_qwen_tp1_execution_plan.json"
        d8.TP2_PLAN_PATH = D2_DIR / "real_qwen_tp2_execution_plan.json"
    else:
        d8.TP1_PLAN_PATH = write_model_plan(
            D2_DIR / "real_qwen_tp1_execution_plan.json",
            PLAN_DIR / key / "real_qwen_tp1_execution_plan.json",
            cfg,
        )
        d8.TP2_PLAN_PATH = write_model_plan(
            D2_DIR / "real_qwen_tp2_execution_plan.json",
            PLAN_DIR / key / "real_qwen_tp2_execution_plan.json",
            cfg,
        )
    d8.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    d8.LOG_DIR.mkdir(parents=True, exist_ok=True)
    d8.TRACE_DIR.mkdir(parents=True, exist_ok=True)
    d8.MULTICELL_DIR.mkdir(parents=True, exist_ok=True)
    return out


def workloads_for(d8, ids: list[str]):
    out = []
    for wid in ids:
        m = d8.re.match(r"^in(\d+)_out(\d+)_c(\d+)$", wid)
        if not m:
            raise ValueError(wid)
        out.append(d8.WorkloadSpec(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return out


def run_model(d8, key: str, cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = configure_d8_for_model(d8, key, cfg)
    workloads = workloads_for(d8, cfg["workloads"])
    ns = argparse.Namespace(
        workload=cfg["workloads"],
        startup_timeout_s=args.startup_timeout_s,
        request_timeout_s=args.request_timeout_s,
        warmup_requests=args.warmup_requests,
        measured_repetitions=args.measured_repetitions,
        profiled_measured_repetitions=args.profiled_measured_repetitions,
        allow_run=True,
    )
    deps = d8.dependency_inventory()
    d8.write_json("environment_manifest.json", d8.environment_manifest(deps))
    d8.write_json("topology_and_transport.json", d8.topology_and_transport())
    d8.write_json("workload_manifest.json", d8.workload_manifest(workloads, deps))
    try:
        d8.run_real_measurement(ns, workloads)
    except Exception as exc:  # noqa: BLE001
        (out / "blocked_reason.json").write_text(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2) + "\n")
        return {"status": "blocked", "model_key": key, "hf_model_id": cfg["hf_model_id"], "output_dir": str(out), "blocked_reason": str(exc)}
    return {"status": "measured", "model_key": key, "hf_model_id": cfg["hf_model_id"], "output_dir": str(out)}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dist(values: list[float]) -> dict[str, Any]:
    mean = statistics.mean(values) if values else None
    sd = statistics.stdev(values) if len(values) >= 2 else None
    vals = sorted(values)
    def pct(p: float):
        if len(vals) < 5:
            return None
        pos = (len(vals)-1)*p
        lo = int(pos); hi = min(len(vals)-1, lo+1); frac = pos-lo
        return vals[lo]*(1-frac)+vals[hi]*frac
    return {"sample_count": len(values), "mean_us": mean, "p50_us": pct(0.5), "p95_us": pct(0.95), "cv": (sd/mean if sd is not None and mean else None)}


def _predicted_raw_step_distribution(pred_rows: list[dict[str, Any]], workload_id: str) -> dict[str, Any]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in pred_rows:
        if row.get("workload_id") == workload_id and row.get("predicted_nccl_us") is not None:
            by_step.setdefault(int(row["step_id"]), []).append(row)
    values = []
    for step_rows in by_step.values():
        rank0 = [r for r in step_rows if r.get("rank") == "rank0"]
        selected = rank0 if rank0 else step_rows
        values.append(sum(float(r["predicted_nccl_us"]) for r in selected))
    return dist(values)


def _bytes_per_call_distribution(pred_rows: list[dict[str, Any]], workload_id: str) -> dict[str, Any]:
    vals = []
    for row in pred_rows:
        if row.get("workload_id") != workload_id:
            continue
        bytes_value = row.get("tensor_bytes") or row.get("prediction_query_bytes")
        if bytes_value is not None:
            vals.append(float(bytes_value))
    return dist(vals)


def aggregate(run_statuses: list[dict[str, Any]]) -> None:
    cells = []
    comm_rows = []
    for status in run_statuses:
        key = status["model_key"]
        out = Path(status["output_dir"])
        if status["status"] != "measured":
            cells.append(status)
            continue
        e2e = read_json(out / "multicell/end_to_end_summary.json")
        attr = read_json(out / "multicell/per_cell_attribution.json")
        pred = read_json(out / "multicell/communication_prediction_validation.json")
        raw_pred = read_json(out / "nccl_prediction_vs_observed.json")
        raw_pred_rows = raw_pred.get("rows", [])
        br = read_json(out / "multicell/break_even_validation.json")
        attr_by = {r["workload_id"]: r for r in attr["rows"]}
        pred_call_by = {r["workload_id"]: r for r in pred["call_count_aware_exposed_prediction"]["per_cell"]}
        pred_kernel_by = {r["workload_id"]: r for r in pred["per_collective_kernel_prediction"]}
        break_by = {r["workload_id"]: r for r in br["rows"]}
        for r in e2e["rows"]:
            wid = r["workload_id"]
            a = attr_by.get(wid, {})
            b = break_by.get(wid, {})
            pcall = pred_call_by.get(wid, {})
            pk = pred_kernel_by.get(wid, {})
            predicted_raw_dist = _predicted_raw_step_distribution(raw_pred_rows, wid)
            bytes_per_call_dist = _bytes_per_call_distribution(raw_pred_rows, wid)
            cell = {
                "model_key": key,
                "hf_model_id": status["hf_model_id"],
                "workload_id": wid,
                "tp1_tpot": r["tp1_tpot"],
                "tp2_tpot": r["tp2_tpot"],
                "measured_winner": r["measured_winner"],
                "profiler_perturbation": r["profiler_perturbation"],
                "tp_compute_savings_us": b.get("compute_saving_us"),
                "predicted_raw_nccl_comm_us_per_decode_step": predicted_raw_dist,
                "collective_calls_per_decode_step": a.get("collective_call_count_per_decode_step"),
                "collective_bytes_per_decode_step": a.get("collective_bytes_per_decode_step"),
                "collective_bytes_per_call": bytes_per_call_dist,
                "measured_exposed_nccl_wall_us": a.get("exposed_nccl_wall_us_per_decode_step"),
                "runtime_residual_us": b.get("runtime_residual_us"),
                "break_even": b,
                "kernel_prediction": pk,
            }
            cells.append(cell)
            comm_rows.append({
                "model_key": key,
                "workload_id": wid,
                "collective_calls_per_step_mean": (a.get("collective_call_count_per_decode_step") or {}).get("mean_us"),
                "collective_bytes_per_step_mean": (a.get("collective_bytes_per_decode_step") or {}).get("mean_us"),
                "collective_bytes_per_call_mean": bytes_per_call_dist.get("mean_us"),
                "predicted_raw_nccl_comm_us_per_step_mean": predicted_raw_dist.get("mean_us"),
                "exposed_nccl_wall_us_mean": (a.get("exposed_nccl_wall_us_per_decode_step") or {}).get("mean_us"),
                "overlap_ratio_mean": (a.get("overlap_ratio_per_decode_step") or {}).get("mean_us"),
            })
    measured = [c for c in cells if c.get("tp1_tpot")]
    first_tp2 = next((c for c in measured if c.get("measured_winner") == "tp2"), None)
    tp1_cells = [c for c in measured if c.get("measured_winner") == "tp1"]
    write_json("workload_matrix.json", {"status": "measured", "runs": run_statuses, "cells": [{k:c.get(k) for k in ("model_key","hf_model_id","workload_id","measured_winner") } for c in cells]})
    write_json("end_to_end_results.json", {"status": "measured", "cells": cells})
    write_json("per_cell_cost_breakdown.json", {"status": "measured", "cells": cells})
    write_json("communication_scaling.json", {"status": "measured", "rows": comm_rows})
    write_json("decision_boundary.json", {
        "status": "boundary_found" if first_tp2 and tp1_cells else "no_boundary_found_within_feasible_range",
        "success_criterion_met": bool(first_tp2 and tp1_cells),
        "first_tp2_favorable_cell": None if first_tp2 is None else {k:first_tp2.get(k) for k in ("model_key","hf_model_id","workload_id","measured_winner","tp_compute_savings_us")},
        "tp1_favorable_cell_count": len(tp1_cells),
        "tp2_favorable_cell_count": len([c for c in measured if c.get("measured_winner") == "tp2"]),
        "analysis": {
            "model_compute_increase": "compare 0.5B vs larger-model TP1/TP2 TPOT and compute_saving_us",
            "collective_count": "reported in communication_scaling.json",
            "bytes_per_call": "reported via collective byte summaries; authoritative per-call metadata is in raw per_decode_step_collectives.json",
            "concurrency": "c1/c4/c8 cells included where feasible",
            "runtime_residual": "reported per cell; current estimator leaves residual at 0 when decomposing TP2 TPOT into compute estimate plus exposed NCCL",
        },
    })
    lines = [
        "# Phase 4D TP1/TP2 Break-Even Boundary",
        "",
        "This run reuses Phase 4B/4C attribution: measured request IDs, decode-step NVTX ranges, cross-rank NCCL wall unions, wall-clock overlap, and exposed NCCL wall time.",
        "",
        f"Boundary status: {'found' if first_tp2 and tp1_cells else 'not found within feasible range'}",
        f"TP1-favorable cells: {len(tp1_cells)}",
        f"TP2-favorable cells: {len([c for c in measured if c.get('measured_winner') == 'tp2'])}",
        "",
        "Selector logic and kernel-selection logic were not modified.",
    ]
    (BREAK_DIR / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {BREAK_DIR / 'README.md'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--measured-repetitions", type=int, default=5)
    parser.add_argument("--profiled-measured-repetitions", type=int, default=1)
    parser.add_argument("--models", nargs="*", default=["qwen2.5-0.5b", "qwen2.5-7b"])
    args = parser.parse_args(argv)
    BREAK_DIR.mkdir(parents=True, exist_ok=True)
    d8 = load_d8()
    statuses = []
    for key in args.models:
        statuses.append(run_model(d8, key, MODEL_RUNS[key], args))
    aggregate(statuses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
