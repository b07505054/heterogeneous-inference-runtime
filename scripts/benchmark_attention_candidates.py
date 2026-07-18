#!/usr/bin/env python3
"""Phase-separated attention candidate and selector measurements."""

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

from deployment.attention_runtime import (  # noqa: E402
    CompilerAttentionRuntime, legal_attention_candidates, select_attention_plan,
)


CALIBRATION = [
    ("prefill", 8, 8), ("prefill", 32, 32), ("prefill", 128, 128),
    ("decode", 1, 8), ("decode", 1, 64), ("decode", 1, 256),
]
HELD_OUT = [
    ("prefill", 16, 16), ("prefill", 64, 64), ("prefill", 192, 192),
    ("decode", 1, 16), ("decode", 1, 128), ("decode", 1, 384),
]


def tensors(qlen, context):
    g = torch.Generator().manual_seed(9000 + qlen + context)
    q = torch.randn(1, 14, qlen, 64, generator=g)
    k = torch.randn(1, 2, context, 64, generator=g)
    v = torch.randn(1, 2, context, 64, generator=g)
    mask = None
    if qlen > 1:
        mask = torch.triu(
            torch.full((1, 1, qlen, context), float("-inf")), diagonal=1)
    return q, k, v, mask


def measure(shape, calls, warmup):
    phase, qlen, context = shape
    q, k, v, mask = tensors(qlen, context)
    rows = []
    candidates = legal_attention_candidates(
        phase=phase, batch=1, query_len=qlen, context_len=context,
        query_heads=14, kv_heads=2, head_dim=64)
    for plan in candidates:
        with CompilerAttentionRuntime(plan) as rt:
            for _ in range(warmup):
                rt.attention(q, k, v, mask, 0.125)
            wall, timings = [], []
            for _ in range(calls):
                a = time.perf_counter_ns()
                out = rt.attention(q, k, v, mask, 0.125)
                wall.append((time.perf_counter_ns() - a) / 1e6)
                timings.append(rt.traces[-1].timing)
            assert torch.isfinite(out).all()
            rows.append({
                "phase": phase, "query_len": qlen, "context_len": context,
                "candidate_id": plan["native_kernel_id"],
                "strategy": plan["selected_strategy"],
                "workers": plan["worker_count"],
                "median_ms": statistics.median(wall),
                "p95_ms": float(np.percentile(wall, 95)),
                "variance_ms2": statistics.pvariance(wall),
                "dispatch_median_ms": statistics.median(t.dispatch_ms for t in timings),
                "qk_median_ms": statistics.median(t.qk_ms for t in timings),
                "softmax_median_ms": statistics.median(t.softmax_ms for t in timings),
                "pv_median_ms": statistics.median(t.pv_ms for t in timings),
                "assembly_median_ms": statistics.median(t.assembly_ms for t in timings),
                "calls": calls,
            })
    return rows


def policy_metrics(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["phase"], row["query_len"], row["context_len"]), []).append(row)
    details, regrets = [], []
    for shape, values in groups.items():
        winner = min(values, key=lambda x: x["median_ms"])
        selected_plan = select_attention_plan(
            phase=shape[0], batch=1, query_len=shape[1], context_len=shape[2])
        selected = next(x for x in values
                        if x["candidate_id"] == selected_plan["native_kernel_id"])
        regret = selected["median_ms"] / winner["median_ms"] - 1
        regrets.append(regret)
        details.append({
            "shape": list(shape), "selected": selected["candidate_id"],
            "winner": winner["candidate_id"], "regret": regret,
        })
    return {
        "exact_match_rate": sum(x["selected"] == x["winner"] for x in details) / len(details),
        "mean_regret": statistics.mean(regrets),
        "median_regret": statistics.median(regrets),
        "p95_regret": float(np.percentile(regrets, 95)),
        "max_regret": max(regrets), "fallback_rate": 0.0, "details": details,
    }


def fixed_policy(rows, strategy, workers):
    groups = {}
    for row in rows:
        groups.setdefault((row["phase"], row["query_len"], row["context_len"]), []).append(row)
    regrets, fallbacks = [], 0
    for values in groups.values():
        winner = min(x["median_ms"] for x in values)
        chosen = next((x for x in values
                       if x["strategy"] == strategy and x["workers"] == workers), None)
        if chosen is None:
            chosen = next(x for x in values if x["strategy"] == "serial")
            fallbacks += 1
        regrets.append(chosen["median_ms"] / winner - 1)
    return {"mean_regret": statistics.mean(regrets),
            "max_regret": max(regrets), "fallback_rate": fallbacks / len(groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    calibration = [r for s in CALIBRATION for r in measure(s, args.calls, args.warmup)]
    held_out = [r for s in HELD_OUT for r in measure(s, args.calls, args.warmup)]
    payload = {
        "artifact_type": "cpu_attention_candidate_evaluation",
        "truth_boundary": "single-process persistent CPU threads; not vLLM serving",
        "calibration_shapes": [list(x) for x in CALIBRATION],
        "held_out_shapes": [list(x) for x in HELD_OUT],
        "calibration_rows": calibration, "held_out_rows": held_out,
        "selector": policy_metrics(held_out),
        "fixed_policies": {
            "always_serial": fixed_policy(held_out, "serial", 1),
            "always_4_split_head": fixed_policy(held_out, "split_head", 4),
            "always_8_split_head": fixed_policy(held_out, "split_head", 8),
            "always_4_split_query_when_legal": fixed_policy(
                held_out, "split_query", 4),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selector": payload["selector"],
                      "fixed_policies": payload["fixed_policies"]}, indent=2))


if __name__ == "__main__":
    main()
