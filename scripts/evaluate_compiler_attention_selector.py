#!/usr/bin/env python3
"""Measure compiler attention selection against fixed candidate policies."""

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


WORKLOADS = [
    ("prefill", 4, 4), ("prefill", 16, 16), ("prefill", 64, 64),
    ("prefill", 128, 128), ("decode", 1, 4), ("decode", 1, 32),
    ("decode", 1, 128), ("decode", 1, 512),
]
POLICIES = [
    ("always_serial", "serial", 1),
    ("always_split_head_2", "split_head", 2),
    ("always_split_head_4", "split_head", 4),
    ("always_split_head_8", "split_head", 8),
    ("always_split_query_2_when_legal", "split_query", 2),
    ("always_split_query_4_when_legal", "split_query", 4),
    ("always_split_query_8_when_legal", "split_query", 8),
]


def tensors(phase: str, query_len: int, context_len: int):
    generator = torch.Generator().manual_seed(20260717 + query_len + context_len)
    q = torch.randn(1, 14, query_len, 64, generator=generator)
    k = torch.randn(1, 2, context_len, 64, generator=generator)
    v = torch.randn(1, 2, context_len, 64, generator=generator)
    mask = None
    if phase == "prefill":
        mask = torch.triu(
            torch.full((1, 1, query_len, context_len), float("-inf")), 1)
    return q, k, v, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows, traces = [], []
    for phase, query_len, context_len in WORKLOADS:
        workload = AttentionWorkload(
            phase=phase, batch=1, query_len=query_len, context_len=context_len,
            query_heads=14, kv_heads=2, head_dim=64)
        selected, trace = select_attention_plan(workload)
        traces.append(trace)
        q, k, v, mask = tensors(phase, query_len, context_len)
        reference = None
        for candidate in (x for x in trace["considered_candidates"] if x["legal"]):
            plan = make_attention_plan(
                phase=phase, strategy=candidate["strategy"],
                workers=candidate["worker_count"], provenance="forced_policy_measurement")
            elapsed = []
            with CompilerAttentionRuntime(plan) as runtime:
                for _ in range(args.warmup):
                    runtime.attention(q, k, v, mask, 0.125)
                for _ in range(args.calls):
                    started = time.perf_counter_ns()
                    output = runtime.attention(q, k, v, mask, 0.125)
                    elapsed.append((time.perf_counter_ns() - started) / 1e6)
            if candidate["strategy"] == "serial":
                reference = output
            rows.append({
                "phase": phase, "query_len": query_len,
                "context_len": context_len,
                "candidate_id": candidate["candidate_id"],
                "strategy": candidate["strategy"],
                "worker_count": candidate["worker_count"],
                "median_ms": statistics.median(elapsed),
                "p95_ms": float(np.percentile(elapsed, 95)),
                "variance_ms2": statistics.pvariance(elapsed),
                "compiler_selected": candidate["candidate_id"] == selected["native_kernel_id"],
                "output": output,
            })
        assert reference is not None
        for row in rows:
            if (row["phase"], row["query_len"], row["context_len"]) == (
                    phase, query_len, context_len):
                torch.testing.assert_close(
                    row.pop("output"), reference, rtol=2e-5, atol=2e-6)
                row["correctness"] = "pass"
    groups = {}
    for row in rows:
        groups.setdefault((row["phase"], row["query_len"], row["context_len"]), []).append(row)
    detail = []
    for shape, values in groups.items():
        winner = min(values, key=lambda x: (x["median_ms"], x["candidate_id"]))
        chosen = next(x for x in values if x["compiler_selected"])
        detail.append({
            "workload": list(shape),
            "legal_candidate_count": len(values),
            "selected_candidate": chosen["candidate_id"],
            "measured_winner": winner["candidate_id"],
            "selected_median_ms": chosen["median_ms"],
            "winner_median_ms": winner["median_ms"],
            "regret": chosen["median_ms"] / winner["median_ms"] - 1,
        })
    def metrics(items):
        regrets = [x["regret"] for x in items]
        return {
            "workloads": len(items),
            "exact_match_rate": sum(x["selected_candidate"] == x["measured_winner"] for x in items) / len(items),
            "mean_regret": statistics.mean(regrets),
            "median_regret": statistics.median(regrets),
            "p95_regret": float(np.percentile(regrets, 95)),
            "maximum_regret": max(regrets),
            "fallback_rate": 0.0,
        }
    fixed = {}
    for name, strategy, workers in POLICIES:
        regrets, fallbacks = [], 0
        for values in groups.values():
            winner = min(x["median_ms"] for x in values)
            selected = next((x for x in values if x["strategy"] == strategy
                             and x["worker_count"] == workers), None)
            if selected is None:
                selected = next(x for x in values if x["strategy"] == "serial")
                fallbacks += 1
            regrets.append(selected["median_ms"] / winner - 1)
        fixed[name] = {
            "mean_regret": statistics.mean(regrets),
            "maximum_regret": max(regrets),
            "fallback_rate": fallbacks / len(groups),
        }
    payload = {
        "artifact_type": "compiler_attention_policy_comparison",
        "truth_boundary": "standalone real numerical CPU attention; not full model or vLLM serving",
        "calls": args.calls, "warmup": args.warmup,
        "rows": rows, "details": detail,
        "selector_quality": {
            "all": metrics(detail),
            "prefill": metrics([x for x in detail if x["workload"][0] == "prefill"]),
            "decode": metrics([x for x in detail if x["workload"][0] == "decode"]),
        },
        "fixed_policies": fixed,
    }
    (args.output_dir / "selector_matrix.json").write_text(
        json.dumps({"workloads": traces}, indent=2) + "\n")
    (args.output_dir / "candidate_policy_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "selector_quality": payload["selector_quality"],
        "fixed_policies": fixed,
    }, indent=2))


if __name__ == "__main__":
    main()
