#!/usr/bin/env python3
"""E2E-9 Phase 12A: RMSNorm validation matrix. For every shape in the real
measured sweep (perf_model/evidence/rmsnorm_benchmark_e2e9.json), runs the
full unified pipeline -- candidate generation -> legality -> cost prediction
-> selection -> REAL execution via the runtime dispatcher on the GPU -- and
records: selected block size, predicted latency, measured winner block size
(from the offline sweep), regret, correctness against an eager reference,
and a fresh CUDA-event-timed measurement of the SELECTED candidate (selector
overhead measured separately/outside this timed region, per Phase 13).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from perf_model.cost_model_registry import CostModelRegistry
from perf_model.implementation_decision import select_implementation
from perf_model.operation_descriptor import (
    OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)
from perf_model.rmsnorm_cost_model_adapter import DEFAULT_EVIDENCE_PATH, RMSNormCostModel
from perf_model.runtime_dispatcher import execute_rmsnorm, torch_rmsnorm_eager

WARMUP = 20
RUNS = 100


def timed_run(fn, warmup=WARMUP, runs=RUNS) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {"median_ms": times[len(times) // 2], "min_ms": times[0], "max_ms": times[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "requires CUDA"
    raw = json.loads(DEFAULT_EVIDENCE_PATH.read_text())
    shapes = sorted({(r["tokens"], r["hidden"]) for r in raw["exact_candidates"]})

    registry = CostModelRegistry()
    cost_model = RMSNormCostModel()
    registry.register(OperationFamily.RMS_NORM, cost_model)

    rows = []
    for tokens, hidden in shapes:
        env = OperationEnvelope(
            operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
            dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode",
            logical_shape=(tokens, hidden),
        )
        payload = RMSNormDescriptor(token_count=tokens, hidden_size=hidden, epsilon=1e-6, has_weight=True,
                                     input_contiguous=True, output_contiguous=True)
        op = OperationDescriptor(common=env, payload=payload)

        # Selection happens once, outside any timed kernel loop.
        select_start = time.perf_counter()
        decision = select_implementation(op, registry, target={"dtype": "float32", "device_type": "cuda"})
        select_elapsed_ms = (time.perf_counter() - select_start) * 1e3

        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
        weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

        out = execute_rmsnorm(x, weight, 1e-6, decision)
        ref = torch_rmsnorm_eager(x, weight, 1e-6)
        max_abs_err = (out - ref).abs().max().item()
        correct = torch.allclose(out, ref, rtol=1e-3, atol=1e-4)

        timing = timed_run(lambda: execute_rmsnorm(x, weight, 1e-6, decision))

        winner_bs, winner_ms = cost_model.measured_winner(tokens, hidden)
        selected_bs = decision.selected_candidate.parameters.get("block_size")
        regret_ms = timing["median_ms"] - winner_ms if winner_ms else None

        rows.append({
            "tokens": tokens, "hidden": hidden, "selected_block_size": selected_bs,
            "measured_winner_block_size": winner_bs, "block_size_match": selected_bs == winner_bs,
            "predicted_latency_ms": decision.predicted_cost.predicted_latency_ms,
            "fresh_measured_median_ms": timing["median_ms"], "offline_measured_winner_ms": winner_ms,
            "regret_ms": regret_ms, "selection_time_ms": select_elapsed_ms,
            "correct": correct, "max_abs_error": max_abs_err,
            "predicted_cost_source": decision.predicted_cost.source,
            "legal_candidate_count": len(decision.all_legal_candidates),
        })
        print(f"tokens={tokens:4d} hidden={hidden:5d} selected_bs={selected_bs} winner_bs={winner_bs} "
              f"match={selected_bs==winner_bs} correct={correct} fresh_ms={timing['median_ms']:.4f} "
              f"regret_ms={regret_ms:.4f} select_time_ms={select_elapsed_ms:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2))
    n_match = sum(r["block_size_match"] for r in rows)
    n_correct = sum(r["correct"] for r in rows)
    print(f"\nSUMMARY: block_size_match={n_match}/{len(rows)} correct={n_correct}/{len(rows)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
