#!/usr/bin/env python3
"""Correctness and 1/2/4/8-worker scaling for the CPU sharding prototype."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.cpu_sharding import PersistentCPUShardRuntime, make_plan


SHAPES = [(1, 256, 256), (32, 512, 512), (128, 768, 768),
          (512, 768, 768), (11, 257, 263)]


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(20260717)
    rows = []
    for m, k, n in SHAPES:
        x = rng.normal(size=(m, k)).astype(np.float32)
        w = rng.normal(size=(k, n)).astype(np.float32)
        reference = x @ w
        for workers in (1, 2, 4, 8):
            with PersistentCPUShardRuntime(make_plan(workers, "split_m",
                                                     "cost_model_selected")) as rt:
                for _ in range(args.warmup):
                    rt.linear(x, w)
                timings = [rt.linear(x, w)[1] for _ in range(args.calls)]
                got, _ = rt.linear(x, w)
                np.testing.assert_allclose(got, reference, rtol=1e-5, atol=1e-4)
                total = [t.total_ms for t in timings]
                rows.append({
                    "shape": [m, k, n], "workers": workers,
                    "median_ms": statistics.median(total),
                    "p95_ms": pct(total, 95),
                    "variance_ms2": statistics.pvariance(total),
                    "dispatch_median_ms": statistics.median(t.dispatch_ms for t in timings),
                    "compute_median_ms": statistics.median(t.compute_ms for t in timings),
                    "assembly_median_ms": statistics.median(t.collective_ms for t in timings),
                    "calls": args.calls, "correct": True,
                    "affinity": {str(k): list(v) for k, v in rt.affinity.items()},
                })
    payload = {
        "artifact_type": "single_node_shared_memory_cpu_sharding_benchmark",
        "truth_boundary": (
            "persistent threads on 4 physical/8 logical CPU cores; "
            "not vLLM tensor parallelism and not multi-device inference"),
        "seed": 20260717, "dtype": "float32", "rtol": 1e-5, "atol": 1e-4,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
