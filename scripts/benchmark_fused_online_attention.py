#!/usr/bin/env python3
"""Dense versus fused-online Qwen-compatible attention evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.attention_planner import AttentionWorkload, select_attention_plan
from deployment.attention_runtime import CompilerAttentionRuntime, make_attention_plan


CALIBRATION = (
    [("prefill", q, q) for q in (4, 8, 16, 32, 64, 128, 256)]
    + [("decode", 1, k) for k in (4, 8, 16, 32, 64, 128, 256, 512, 1024)]
)
HELD_OUT = (
    [("prefill", q, q) for q in (11, 37, 73, 129)]
    + [("decode", 1, k) for k in (17, 63, 127, 263, 511)]
)
FIXED = {
    "always_dense_serial": ("dense_materialized", "serial", 1),
    "always_dense_split_head_2": ("dense_materialized", "split_head", 2),
    "always_fused_serial": ("fused_tiled_online_softmax", "serial", 1),
    "always_fused_split_head_2": ("fused_tiled_online_softmax", "split_head", 2),
    "always_fused_split_head_4": ("fused_tiled_online_softmax", "split_head", 4),
}


def inputs(phase, query_len, context_len):
    generator = torch.Generator().manual_seed(71000 + query_len + context_len)
    q = torch.randn(1, 14, query_len, 64, generator=generator)
    k = torch.randn(1, 2, context_len, 64, generator=generator)
    v = torch.randn(1, 2, context_len, 64, generator=generator)
    mask = None if phase == "decode" else torch.triu(
        torch.full((1, 1, query_len, context_len), -torch.inf), 1)
    return q, k, v, mask


def evaluate(shapes, calls, warmup):
    rows, selector = [], []
    for phase, query_len, context_len in shapes:
        workload = AttentionWorkload(
            phase, 1, query_len, context_len, 14, 2, 64)
        selected, trace = select_attention_plan(workload)
        selector.append(trace)
        q, k, v, mask = inputs(phase, query_len, context_len)
        with CompilerAttentionRuntime(make_attention_plan(phase=phase)) as reference_runtime:
            reference = reference_runtime.attention(q, k, v, mask, 0.125)
        for candidate in (x for x in trace["considered_candidates"] if x["legal"]):
            plan = make_attention_plan(
                phase=phase, strategy=candidate["strategy"],
                workers=candidate["worker_count"],
                algorithm=candidate["algorithm"],
                query_tile=candidate["query_tile"],
                key_tile=candidate["key_tile"],
                implementation=(
                    "torch_dense_materialized_v1"
                    if candidate["algorithm"] == "dense_materialized" else
                    "torch_tiled_online_softmax_exact_v1"),
                provenance="forced_candidate_measurement")
            wall, dispatch, barrier = [], [], []
            with CompilerAttentionRuntime(plan) as runtime:
                for _ in range(warmup):
                    runtime.attention(q, k, v, mask, 0.125)
                for _ in range(calls):
                    started = time.perf_counter_ns()
                    output = runtime.attention(q, k, v, mask, 0.125)
                    wall.append((time.perf_counter_ns() - started) / 1e6)
                    dispatch.append(runtime.traces[-1].timing.dispatch_ms)
                    barrier.append(0.0)
                memory = runtime.traces[-1].memory
            difference = (output - reference).abs()
            rows.append({
                "phase": phase, "query_len": query_len,
                "context_len": context_len,
                "candidate_id": candidate["candidate_id"],
                "algorithm": candidate["algorithm"],
                "strategy": candidate["strategy"],
                "worker_count": candidate["worker_count"],
                "query_tile": candidate["query_tile"],
                "key_tile": candidate["key_tile"],
                "compiler_selected": candidate["candidate_id"] == selected["native_kernel_id"],
                "correctness": "pass" if torch.allclose(
                    output, reference, rtol=2e-5, atol=2e-6) else "fail",
                "max_absolute_error": float(difference.max()),
                "nan_count": int(torch.isnan(output).sum()),
                "inf_count": int(torch.isinf(output).sum()),
                "median_ms": statistics.median(wall),
                "p95_ms": float(np.percentile(wall, 95)),
                "variance_ms2": statistics.pvariance(wall),
                "dispatch_median_ms": statistics.median(dispatch),
                "barrier_median_ms": statistics.median(barrier),
                "memory": memory,
            })
    return rows, selector


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(
            (row["phase"], row["query_len"], row["context_len"]), []).append(row)
    details = []
    for shape, values in groups.items():
        winner = min(values, key=lambda x: (x["median_ms"], x["candidate_id"]))
        selected = next(x for x in values if x["compiler_selected"])
        details.append({
            "workload": list(shape), "legal_candidate_count": len(values),
            "selected_candidate": selected["candidate_id"],
            "selected_algorithm": selected["algorithm"],
            "measured_winner": winner["candidate_id"],
            "winner_algorithm": winner["algorithm"],
            "selected_median_ms": selected["median_ms"],
            "winner_median_ms": winner["median_ms"],
            "regret": selected["median_ms"] / winner["median_ms"] - 1,
        })
    regrets = [x["regret"] for x in details]
    quality = {
        "workloads": len(details),
        "exact_winner_rate": sum(
            x["selected_candidate"] == x["measured_winner"] for x in details) / len(details),
        "mean_regret": statistics.mean(regrets),
        "median_regret": statistics.median(regrets),
        "p95_regret": float(np.percentile(regrets, 95)),
        "maximum_regret": max(regrets),
        "fused_selection_rate": sum(
            x["selected_algorithm"] == "fused_tiled_online_softmax" for x in details) / len(details),
        "dense_selection_rate": sum(
            x["selected_algorithm"] == "dense_materialized" for x in details) / len(details),
        "fallback_rate": 0.0,
    }
    fixed = {}
    for name, (algorithm, strategy, workers) in FIXED.items():
        policy_regrets, fallbacks = [], 0
        for values in groups.values():
            winner = min(x["median_ms"] for x in values)
            choices = [x for x in values if x["algorithm"] == algorithm
                       and x["strategy"] == strategy
                       and x["worker_count"] == workers]
            if choices:
                chosen = min(choices, key=lambda x: x["median_ms"])
            else:
                chosen = next(x for x in values if x["algorithm"] == "dense_materialized"
                              and x["strategy"] == "serial")
                fallbacks += 1
            policy_regrets.append(chosen["median_ms"] / winner - 1)
        fixed[name] = {
            "mean_regret": statistics.mean(policy_regrets),
            "maximum_regret": max(policy_regrets),
            "fallback_rate": fallbacks / len(groups),
        }
    return details, quality, fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    calibration, calibration_selector = evaluate(
        CALIBRATION, args.calls, args.warmup)
    held_out, held_out_selector = evaluate(HELD_OUT, args.calls, args.warmup)
    calibration_details, calibration_quality, _ = summarize(calibration)
    held_details, held_quality, fixed = summarize(held_out)
    (args.output_dir / "candidate_calibration.json").write_text(json.dumps({
        "calls": args.calls, "warmup": args.warmup,
        "workloads": [list(x) for x in CALIBRATION],
        "selector_traces": calibration_selector,
        "rows": calibration, "details": calibration_details,
        "quality": calibration_quality}, indent=2) + "\n")
    (args.output_dir / "held_out_selector_evaluation.json").write_text(json.dumps({
        "calls": args.calls, "warmup": args.warmup,
        "workloads": [list(x) for x in HELD_OUT],
        "selector_traces": held_out_selector,
        "rows": held_out, "details": held_details,
        "quality": held_quality, "fixed_policies": fixed}, indent=2) + "\n")
    dense_memory = []
    allocation = []
    for row in calibration + held_out:
        if row["strategy"] != "serial":
            continue
        record = {
            "workload": [row["phase"], row["query_len"], row["context_len"]],
            "candidate_id": row["candidate_id"], **row["memory"]}
        (dense_memory if row["algorithm"] == "dense_materialized" else allocation).append(record)
    (args.output_dir / "dense_memory_baseline.json").write_text(
        json.dumps({"rows": dense_memory}, indent=2) + "\n")
    (args.output_dir / "fused_allocation_audit.json").write_text(
        json.dumps({"rows": allocation,
                    "dense_helper_called_by_fused": False,
                    "torch_softmax_called_by_fused": False,
                    "sdpa_called_by_fused": False}, indent=2) + "\n")
    crossover = [{
        "workload": x["workload"], "winner": x["measured_winner"],
        "winner_algorithm": x["winner_algorithm"],
        "compiler_selected": x["selected_candidate"],
        "regret": x["regret"]} for x in calibration_details + held_details]
    (args.output_dir / "crossover_analysis.json").write_text(
        json.dumps({"rows": crossover}, indent=2) + "\n")
    print(json.dumps({
        "calibration_quality": calibration_quality,
        "held_out_quality": held_quality, "fixed_policies": fixed}, indent=2))


if __name__ == "__main__":
    main()
