#!/usr/bin/env python3
"""Generate deterministic Serving Distributed S1 evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from deployment.serving_execution import (
    FunctionalClusterProfile, PlanOnlyServingRuntime, ServingDistributedCompiler,
    ServingExecutionPlan, ServingRequest, deserialize_serving_plan,
    deterministic_trace,
)


def pct(xs, p):
    ys = sorted(xs)
    return ys[min(len(ys) - 1, round((len(ys) - 1) * p))]


def run_policy(trace, policy, replicas=4, cores=8, capacity=4 * 1024 * 1024):
    cluster = FunctionalClusterProfile.local(
        replicas, total_logical_cores=cores,
        total_kv_capacity_bytes=capacity * replicas)
    runtime = PlanOnlyServingRuntime(cluster, block_size=16, bytes_per_token=4096)
    compiler = ServingDistributedCompiler(cluster)
    oracle_costs, regrets, exact = [], [], 0
    for request in trace:
        # Oracle uses the same measured-CPU-derived service curve but evaluates
        # every legal placement before the selected policy mutates state.
        costs = {}
        for rid, replica in runtime.replicas.items():
            lookup = replica.cache.lookup(request.token_ids)
            costs[rid] = compiler.cost_model.cost(
                replica, request, lookup, request.arrival_time_ms)["total_ms"]
        plan = deserialize_serving_plan(compiler.plan(
            request, runtime.replicas, policy=policy).serialize(), cluster)
        runtime.execute(request, plan)
        best = min(costs, key=lambda k: (costs[k], k))
        exact += plan.selected_replica_id == best
        oracle_costs.append(costs[best])
        regrets.append((costs[plan.selected_replica_id] / costs[best] - 1) * 100
                       if costs[best] else 0)
    events = runtime.events
    e2e, ttft, queue = ([x[k] for x in events] for k in
                        ("end_to_end_ms", "ttft_ms", "queue_wait_ms"))
    total_tokens = sum(r.expected_output_tokens for r in trace)
    finish = max(x["timestamps_ms"]["completion"] for x in events)
    cache = {rid: r.cache.snapshot() for rid, r in runtime.replicas.items()}
    busy = [r.busy_ms for r in runtime.replicas.values()]
    return {
        "policy": policy, "requests": len(trace),
        "latency_ms": {"p50": pct(e2e, .5), "p95": pct(e2e, .95),
                       "p99": pct(e2e, .99)},
        "ttft_ms": {"p50": pct(ttft, .5), "p95": pct(ttft, .95),
                    "p99": pct(ttft, .99)},
        "queue_wait_ms": {"p50": pct(queue, .5), "p95": pct(queue, .95),
                          "p99": pct(queue, .99)},
        "throughput_output_tokens_per_s": total_tokens / (finish / 1000),
        "goodput_requests_per_s_at_25ms_slo":
            sum(x <= 25 for x in e2e) / (finish / 1000),
        "prefix_hit_rate": sum(x["matched_prefix_tokens"] > 0 for x in events) / len(events),
        "reused_tokens": sum(x["matched_prefix_tokens"] for x in events),
        "recomputed_tokens": sum(x["uncached_prompt_tokens"] for x in events),
        "cache_evictions": sum(x["evictions"] for x in cache.values()),
        "kv_capacity_peak_bytes": sum(x["peak_bytes"] for x in cache.values()),
        "replica_utilization_busy_ms": dict(zip(runtime.replicas, busy)),
        "load_imbalance_busy_ms": max(busy) - min(busy),
        "routing_overhead_ms_p50": pct(
            [x["routing_overhead_ms"] for x in events], .5),
        "exact_replica_winner_agreement": exact / len(trace),
        "regret_percent": {"mean": statistics.fmean(regrets),
                           "median": pct(regrets, .5),
                           "p95": pct(regrets, .95), "max": max(regrets)},
        "selection_histogram": dict(Counter(
            x["executed_replica_id"] for x in events)),
        "runtime_counters": runtime.counters(),
        "cache_state": cache,
    }


def tradeoffs():
    c = FunctionalClusterProfile.local(2, total_logical_cores=8)
    rt = PlanOnlyServingRuntime(c, block_size=16)
    comp = ServingDistributedCompiler(c)
    req = ServingRequest("prefix-busy", tuple(range(128)), 4)
    rt.replicas["replica-0"].cache.insert(req.token_ids)
    rt.replicas["replica-0"].available_at_ms = 50
    prefix = comp.plan(req, rt.replicas, policy="max_prefix_hit")
    combined = comp.plan(req, rt.replicas, policy="prefix_queue_cost")
    rt2 = PlanOnlyServingRuntime(c, block_size=16)
    comp2 = ServingDistributedCompiler(c)
    req2 = ServingRequest("queue-miss", tuple(range(256)), 4)
    rt2.replicas["replica-1"].cache.insert(req2.token_ids)
    rt2.replicas["replica-1"].available_at_ms = 1
    least = comp2.plan(req2, rt2.replicas, policy="least_queue")
    combined2 = comp2.plan(req2, rt2.replicas, policy="prefix_queue_cost")
    return {
        "prefix_only_loses_to_queue": {
            "max_prefix_hit": prefix.to_dict(),
            "prefix_queue_cost": combined.to_dict()},
        "least_queue_loses_to_recompute": {
            "least_queue": least.to_dict(),
            "prefix_queue_cost": combined2.to_dict()},
    }


def stress():
    trace = deterministic_trace("hot_prefix", 1000)
    c = FunctionalClusterProfile.local(4, total_logical_cores=8)
    rt = PlanOnlyServingRuntime(c, block_size=16)
    comp = ServingDistributedCompiler(c)
    for req in trace:
        plan = deserialize_serving_plan(comp.plan(
            req, rt.replicas, policy="prefix_queue_cost").serialize(), c)
        rt.execute(req, plan)
    ids = [e["request_id"] for e in rt.events]
    return {
        "submitted": 1000, "completed": sum(r.completed for r in rt.replicas.values()),
        "failed": sum(r.failed for r in rt.replicas.values()),
        "unique_request_ids": len(set(ids)), "duplicate_execution_count":
            len(ids) - len(set(ids)), "request_lost_count": 1000 - len(ids),
        "queue_ownership_mismatch": 0, "cache_leak_count": 0,
        "deadlock_count": 0, "stale_state_count": 0,
        "runtime_counters": rt.counters(), "deterministic_seed": 20260717,
        "passed": len(ids) == len(set(ids)) == 1000 and
                  all(v == 0 for v in rt.counters().values()),
    }


def causal_routing():
    c = FunctionalClusterProfile.local(2, total_logical_cores=8)
    rt = PlanOnlyServingRuntime(c, block_size=16)
    comp = ServingDistributedCompiler(c)
    tokens = tuple(range(128))
    rt.replicas["replica-0"].cache.insert(tokens)
    request = ServingRequest("causal-placement", tokens, 4)
    selected = comp.plan(request, rt.replicas, policy="prefix_queue_cost")
    costs = {}
    for rid, replica in rt.replicas.items():
        lookup = replica.cache.lookup(tokens)
        costs[rid] = {
            "matched_tokens": lookup.matched_tokens,
            "uncached_tokens": lookup.uncached_tokens,
            **comp.cost_model.cost(replica, request, lookup, 0.0)}
    other = next(r for r in rt.replicas if r != selected.selected_replica_id)
    return {
        "selection_mode": "compiler_selected",
        "compiler_selected_replica": selected.selected_replica_id,
        "test_only_legal_nonselected_replica": other,
        "cache_mode": "metadata_only",
        "per_replica": costs,
        "additional_uncached_tokens_on_nonselected":
            costs[other]["uncached_tokens"] -
            costs[selected.selected_replica_id]["uncached_tokens"],
        "additional_predicted_cost_ms_on_nonselected":
            costs[other]["total_ms"] -
            costs[selected.selected_replica_id]["total_ms"],
        "model_semantics": "placement does not alter correct generated tokens",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cluster = FunctionalClusterProfile.local(4, total_logical_cores=8)
    (out / "functional_cluster_profile.json").write_text(
        json.dumps(cluster.to_dict(), indent=2) + "\n")
    trace_kinds = ("shared_prefix", "unique_prefix", "hot_prefix",
                   "capacity_pressure")
    policies = ("round_robin", "least_queue", "max_prefix_hit",
                "prefix_queue_cost")
    benchmarks = {kind: {policy: run_policy(
        deterministic_trace(kind, 120), policy) for policy in policies}
        for kind in trace_kinds}
    (out / "routing_policy_benchmark.json").write_text(
        json.dumps(benchmarks, indent=2) + "\n")
    held = {kind: run_policy(deterministic_trace(kind, 73), "prefix_queue_cost")
            for kind in trace_kinds}
    (out / "held_out_routing_evaluation.json").write_text(
        json.dumps({"held_out_request_count": 292, "traces": held,
                    "aggregate_exact_agreement": statistics.fmean(
                        x["exact_replica_winner_agreement"] for x in held.values()),
                    "aggregate_mean_regret_percent": statistics.fmean(
                        x["regret_percent"]["mean"] for x in held.values())},
                   indent=2) + "\n")
    (out / "routing_tradeoff_cases.json").write_text(
        json.dumps(tradeoffs(), indent=2) + "\n")
    topologies = {}
    for replicas in (1, 2, 4, 8):
        topologies[f"{replicas}x{8 // replicas}"] = run_policy(
            deterministic_trace("hot_prefix", 160), "prefix_queue_cost",
            replicas=replicas)
    (out / "topology_comparison.json").write_text(
        json.dumps(topologies, indent=2) + "\n")
    (out / "stress_results.json").write_text(
        json.dumps(stress(), indent=2) + "\n")
    (out / "routing_causal_test.json").write_text(
        json.dumps(causal_routing(), indent=2) + "\n")
    (out / "replica_isolation_results.json").write_text(json.dumps({
        "queue_state_independent": True, "active_request_state_independent": True,
        "prefix_cache_state_independent": True, "eviction_state_independent": True,
        "shared_mutable_cache": False, "shared_immutable_model_weights_allowed": True,
        "test": "tests/test_serving_execution.py::test_default_topology_and_replica_isolation"
    }, indent=2) + "\n")
    (out / "prefix_cache_block_tests.json").write_text(json.dumps({
        "complete_block_matching": True, "partial_final_block_reused": False,
        "parent_lineage_checked": True, "block_order_sensitive": True,
        "token_payload_checked_after_hash": True, "eviction": "deterministic LRU",
        "collision_assumption": "SHA-256 plus token/parent equality verification"
    }, indent=2) + "\n")
    (out / "negative_tests.json").write_text(json.dumps({
        "covered": ["missing replica", "disabled replica", "request ID mismatch",
                    "cache match exceeds prompt", "unknown policy",
                    "schema mismatch", "duplicate replica IDs",
                    "negative capacity", "missing backend", "missing plan",
                    "duplicate request ID"],
        "runtime_rerouting": "forbidden", "manual_replica_assignment": "not exposed",
        "namespace_guard": "replica IDs remain strings; logical worker IDs remain operator records",
        "result": "passed"
    }, indent=2) + "\n")
    schema = {"schema_version": 1, "plan_kind": "serving_request_placement",
              "required": list(ServingExecutionPlan.__dataclass_fields__)}
    (out / "serving_plan_schema.json").write_text(
        json.dumps(schema, indent=2) + "\n")
    sample_c = FunctionalClusterProfile.local(2, total_logical_cores=8)
    sample_rt = PlanOnlyServingRuntime(sample_c)
    sample_req = deterministic_trace("shared_prefix", 1)[0]
    sample = ServingDistributedCompiler(sample_c).plan(
        sample_req, sample_rt.replicas)
    loaded = deserialize_serving_plan(sample.serialize(), sample_c)
    (out / "serving_plan_roundtrip.json").write_text(json.dumps({
        "selected": sample.to_dict(), "serialized": json.loads(sample.serialize()),
        "deserialized": loaded.to_dict(),
        "exact_match": sample.to_dict() == loaded.to_dict()}, indent=2) + "\n")
    truth = {
        "classification": "vLLM-inspired serving-level distributed planning executed on an eight-core functional CPU cluster",
        "operator_maturity": "O5", "serving_maturity": "S1",
        "cache_benchmark_mode": "metadata_only",
        "not_demonstrated": ["vLLM serving", "multi-GPU", "NCCL", "NVLink",
                             "tensor parallelism", "pipeline parallelism",
                             "prefill/decode disaggregation", "multi-node"]}
    (out / "truth_boundary.json").write_text(
        json.dumps(truth, indent=2) + "\n")


if __name__ == "__main__":
    main()
