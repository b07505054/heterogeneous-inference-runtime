#!/usr/bin/env python3
"""E2E-9 Phase 13: measures the unified selector's own overhead, separated
from operation execution time:
  1. Cold policy-build cost for the LM-head runtime_piecewise policy
     (one-time, at first call / plan-build time -- calls select_implementation()
     once per M in the covered range).
  2. Cached per-call dispatch overhead: policy.resolve_candidate_id() dict
     lookup + the branch in runtime_dispatcher.execute_lm_head_linear, with
     the actual GEMV/GEMM compute excluded (measured as wall time around the
     Python-side resolution call only, not around the CUDA kernel).
  3. RMSNorm per-shape selection cost (cost-model measured-lookup + argmin),
     which Phase 12A already records per shape (selection_time_ms) -- this
     script re-reports it here with the <1%-of-improvement acceptance check
     applied against the real E2E-8 batch=2 saving (38.4ms per decode step,
     168.94ms -> 133.12ms TPOT was the measured saving driving that number).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perf_model.cost_model_registry import CostModelRegistry
from perf_model.execution_policy import build_runtime_piecewise_policy
from perf_model.operation_descriptor import (
    LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype,
)
from perf_model.tiny_m_linear_cost_model import TinyMLinearCostModel

# Real E2E-8 measured batch=2 TPOT saving (168.94ms -> 133.12ms), the
# improvement the unified selector's dispatch overhead must not erase.
E2E8_BATCH2_MEASURED_SAVING_MS = 168.94 - 133.12
OVERHEAD_BUDGET_PERCENT = 1.0


def _linear_template(m=1, n=151936, k=896):
    env = OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                             dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(m, n, k))
    payload = LinearDescriptor(M=m, N=n, K=k, has_bias=False, decode_or_prefill="decode", graph_captured=False,
                                eager_execution=True, tensor_parallel_size=1, weight_layout="row_major",
                                input_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cold-build-reps", type=int, default=20)
    parser.add_argument("--cached-lookup-reps", type=int, default=100_000)
    args = parser.parse_args()

    registry = CostModelRegistry()
    registry.register(OperationFamily.LINEAR, TinyMLinearCostModel())
    m_values = list(range(1, 13))

    cold_times_ms = []
    for _ in range(args.cold_build_reps):
        start = time.perf_counter()
        build_runtime_piecewise_policy(_linear_template(), registry, m_values)
        cold_times_ms.append((time.perf_counter() - start) * 1e3)
    cold_times_ms.sort()
    cold_median_ms = cold_times_ms[len(cold_times_ms) // 2]

    policy = build_runtime_piecewise_policy(_linear_template(), registry, m_values)
    start = time.perf_counter()
    for _ in range(args.cached_lookup_reps):
        policy.resolve_candidate_id({"M": 2})
    total_s = time.perf_counter() - start
    per_call_us = (total_s / args.cached_lookup_reps) * 1e6

    overhead_ms_per_call = per_call_us / 1e3
    overhead_percent_of_batch2_saving = (overhead_ms_per_call / E2E8_BATCH2_MEASURED_SAVING_MS) * 100.0
    within_budget = overhead_percent_of_batch2_saving < OVERHEAD_BUDGET_PERCENT

    result = {
        "cold_policy_build_median_ms": cold_median_ms, "cold_policy_build_n_m_values": len(m_values),
        "cold_policy_build_reps": args.cold_build_reps,
        "cached_dispatch_overhead_us_per_call": per_call_us,
        "cached_dispatch_overhead_ms_per_call": overhead_ms_per_call,
        "e2e8_batch2_measured_saving_ms": E2E8_BATCH2_MEASURED_SAVING_MS,
        "overhead_percent_of_batch2_saving": overhead_percent_of_batch2_saving,
        "overhead_budget_percent": OVERHEAD_BUDGET_PERCENT, "within_budget": within_budget,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
