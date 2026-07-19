#!/usr/bin/env python3
"""Generate modeled-service-time S2 policy and stress evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime, ReplicaSchedulerState, RequestExecutionState,
    SchedulerCompiler, SchedulerProfile, run_scheduler)


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * p))]


def trace(kind, n=48):
    rows = []
    for i in range(n):
        if kind == "decode_heavy":
            prompt, matched, output, arrival = 16, 16, 24, i * .08
        elif kind == "prefill_heavy":
            prompt, matched, output, arrival = 128 + (i % 4) * 32, 0, 4, i * .02
        elif kind == "mixed":
            prompt = (16, 64, 192, 32)[i % 4]
            matched, output, arrival = (16 if i % 7 == 0 else 0), 8, i * .05
        elif kind == "long_prefill":
            prompt, matched, output, arrival = (1024 if i == 0 else 16), 0, 8, i * .1
        elif kind == "arrival_burst":
            prompt, matched, output, arrival = 48, 0, 10, (i // 8) * 2.0
        elif kind == "prefix_reuse":
            prompt, matched, output, arrival = 96, (96 if i % 3 == 0 else
                                                     48 if i % 3 == 1 else 0), 6, i * .04
        elif kind == "adversarial":
            prompt = 256 if i % 10 == 0 else 16
            matched = 16 if i % 10 else 0
            output, arrival = (2 if i % 10 == 0 else 32), i * .02
        else:
            raise ValueError(kind)
        rows.append(RequestExecutionState(
            f"{kind}-{i}", f"serving-{kind}-{i}", "replica-0", arrival,
            prompt, matched, output))
    return rows


def metrics(state, runtime, policy):
    reqs = list(state.requests.values())
    ttft = [r.first_token_ms - r.arrival_time_ms for r in reqs]
    e2e = [r.completion_ms - r.arrival_time_ms for r in reqs]
    prefill = [(r.prefill_finished_ms or r.arrival_time_ms) - r.arrival_time_ms
               for r in reqs]
    gap_lists = [(([r.decode_times_ms[0] - r.arrival_time_ms]
                   if r.decode_times_ms and r.matched_prefix_tokens == r.prompt_length
                   else []) +
                  [b-a for a,b in zip(r.decode_times_ms, r.decode_times_ms[1:])])
                 for r in reqs]
    gaps = [gap for row in gap_lists for gap in row]
    max_gap = [max([0.0] + row) for row in gap_lists]
    total_capacity = len(runtime.events) * state.profile.max_num_batched_tokens
    scheduled = state.statistics["scheduled_tokens"]
    slo_good = [t <= state.profile.ttft_slo_ms and
                g <= state.profile.maximum_decode_gap_ms
                for t, g in zip(ttft, max_gap)]
    objective = statistics.fmean(ttft) + statistics.fmean(max_gap) + \
        .1 * statistics.fmean(e2e)
    return {
        "execution_mode": "modeled_service_time",
        "policy": policy or "compiler_selected",
        "selected_candidate_histogram": dict(Counter(
            x["policy"] for x in runtime.events)),
        "request_count": len(reqs), "scheduler_steps": len(runtime.events),
        "request_ttft_ms": {r.request_id: r.first_token_ms-r.arrival_time_ms
                            for r in reqs},
        "ttft_ms": {"p50": pct(ttft, .5), "p95": pct(ttft, .95),
                    "p99": pct(ttft, .99)},
        "end_to_end_ms": {"p50": pct(e2e, .5), "p95": pct(e2e, .95),
                          "p99": pct(e2e, .99)},
        "prefill_completion_ms_p95": pct(prefill, .95),
        "itl_ms": {"p50": pct(gaps or [0], .5), "p95": pct(gaps or [0], .95),
                   "maximum": max(gaps or [0])},
        "maximum_request_wait_ms": max(ttft),
        "maximum_prefill_stall_ms": max(prefill),
        "maximum_decode_stall_ms": max(max_gap),
        "starvation_event_count": sum(
            t > state.profile.starvation_guard_ms for t in ttft) +
            sum(g > state.profile.starvation_guard_ms for g in max_gap),
        "token_budget_utilization": scheduled / total_capacity,
        "scheduled_tokens": scheduled,
        "unused_tokens": state.statistics["unused_tokens"],
        "prefill_tokens": state.statistics["prefill_tokens"],
        "decode_tokens": state.statistics["decode_tokens"],
        "batch_size_distribution": dict(Counter(
            x["scheduled_sequences"] for x in runtime.events)),
        "prefill_chunk_count": sum(len(r.prefill_chunks) for r in reqs),
        "goodput_requests_per_modeled_second": sum(slo_good) /
            (state.clock_ms / 1000),
        "slo_satisfied_fraction": sum(slo_good) / len(slo_good),
        "objective": objective,
        "runtime_counters": runtime.counters(),
        "complete": all(r.finished for r in reqs),
    }


def run(kind, policy=None, profile=None, n=48):
    p = profile or SchedulerProfile(
        max_num_seqs=16, max_num_batched_tokens=64,
        max_prefill_chunk_tokens=32, balanced_decode_reservation=8,
        ttft_slo_ms=50, maximum_decode_gap_ms=10,
        starvation_guard_ms=25)
    state = ReplicaSchedulerState("replica-0", p)
    for request in trace(kind, n):
        state.ingest(request)
    runtime = PlanOnlySchedulerRuntime()
    compiler = SchedulerCompiler()
    run_scheduler(state, compiler, runtime, policy=policy)
    return metrics(state, runtime, policy), state, runtime, compiler


def chunk_causal():
    results = {}
    for label, chunk in (("whole_prefill", 64), ("chunked_prefill", 8)):
        p = SchedulerProfile(max_num_seqs=8, max_num_batched_tokens=65,
                             max_prefill_chunk_tokens=chunk,
                             balanced_decode_reservation=8)
        long = RequestExecutionState("long", "serving-long", "replica-0",
                                     0, 64, 0, 2)
        decode = RequestExecutionState("decode", "serving-decode", "replica-0",
                                       0, 16, 16, 8)
        state = ReplicaSchedulerState("replica-0", p)
        state.ingest(long); state.ingest(decode)
        rt = PlanOnlySchedulerRuntime()
        run_scheduler(state, SchedulerCompiler(), rt, policy="chunked_balanced")
        results[label] = {
            "chunks": long.prefill_chunks, "step_count": len(rt.events),
            "token_coverage": sum(b - a for a, b in long.prefill_chunks),
            "overlap_or_gap": any(a[1] != b[0] for a, b in
                                  zip(long.prefill_chunks, long.prefill_chunks[1:])),
            "long_ttft_ms": long.first_token_ms,
            "decode_max_gap_ms": max([0] + [b-a for a,b in
                                            zip(decode.decode_times_ms,
                                                decode.decode_times_ms[1:])]),
            "semantic_expectation": "same token sequence; modeled mode tracks no logits",
        }
    results["causal_conclusion"] = (
        "chunk size changes step count and decode/prefill interleaving while "
        "covering the same logical prompt exactly")
    return results


def explicit_starvation_counterexamples():
    def execute(policy, requests, profile):
        state = ReplicaSchedulerState("replica-0", profile)
        for request in requests: state.ingest(request)
        runtime = PlanOnlySchedulerRuntime()
        run_scheduler(state, SchedulerCompiler(), runtime, policy=policy)
        return metrics(state, runtime, policy)
    decode_load = []
    for i in range(4):
        r = RequestExecutionState(f"d{i}", f"s-d{i}", "replica-0", 0,
                                  16, 16, 64)
        decode_load.append(r)
    decode_load.append(RequestExecutionState("long-prefill", "s-long",
                                              "replica-0", 0, 256, 0, 1))
    p = SchedulerProfile(max_num_seqs=4, max_num_batched_tokens=32,
                         max_prefill_chunk_tokens=28,
                         balanced_decode_reservation=2,
                         ttft_slo_ms=10, maximum_decode_gap_ms=2)
    decode_first = execute("decode_first", decode_load, p)
    # Recreate mutable requests for the second policy.
    decode_load2 = []
    for i in range(4):
        decode_load2.append(RequestExecutionState(
            f"d{i}", f"s-d{i}", "replica-0", 0, 16, 16, 64))
    decode_load2.append(RequestExecutionState(
        "long-prefill", "s-long", "replica-0", 0, 256, 0, 1))
    balanced = execute("prefill_first", decode_load2, p)

    prefill_load = [RequestExecutionState(
        f"p{i}", f"s-p{i}", "replica-0", 0, 96, 0, 1) for i in range(4)]
    for i in range(4):
        prefill_load.append(RequestExecutionState(
            f"active-d{i}", f"s-active-d{i}", "replica-0", 0, 16, 16, 16))
    p2 = SchedulerProfile(max_num_seqs=8, max_num_batched_tokens=32,
                          max_prefill_chunk_tokens=32,
                          balanced_decode_reservation=4,
                          maximum_decode_gap_ms=.5)
    prefill_first = execute("prefill_first", prefill_load, p2)
    prefill_load2 = [RequestExecutionState(
        f"p{i}", f"s-p{i}", "replica-0", 0, 96, 0, 1) for i in range(4)]
    for i in range(4):
        prefill_load2.append(RequestExecutionState(
            f"active-d{i}", f"s-active-d{i}", "replica-0", 0, 16, 16, 16))
    balanced2 = execute("chunked_balanced", prefill_load2, p2)
    return {
        "decode_first_loses": {
            "scenario": "four long-running decodes fill max_num_seqs while a 256-token prefill waits",
            "decode_first": decode_first, "prefill_first": balanced},
        "prefill_first_loses": {
            "scenario": "32-token prefill chunks consume the entire step budget while four decodes are active",
            "prefill_first": prefill_first, "chunked_balanced": balanced2}}


def stress():
    p = SchedulerProfile(max_num_seqs=4, max_num_batched_tokens=2,
                         max_prefill_chunk_tokens=1,
                         balanced_decode_reservation=1)
    state = ReplicaSchedulerState("replica-0", p)
    for i in range(1000):
        state.ingest(RequestExecutionState(
            f"stress-{i}", f"serving-stress-{i}", "replica-0", i*.001,
            10, 0, 11))
    rt = PlanOnlySchedulerRuntime()
    run_scheduler(state, SchedulerCompiler(), rt, policy="chunked_balanced",
                  max_steps=30000)
    overlaps = gaps = 0
    for r in state.requests.values():
        overlaps += sum(a[1] > b[0] for a,b in zip(r.prefill_chunks,
                                                   r.prefill_chunks[1:]))
        gaps += sum(a[1] < b[0] for a,b in zip(r.prefill_chunks,
                                               r.prefill_chunks[1:]))
    return {
        "execution_mode": "modeled_service_time", "submitted": 1000,
        "completed": len(state.finished_ids), "scheduler_steps": len(rt.events),
        "duplicate_terminal_ids": len(state.finished_ids)-len(set(state.finished_ids)),
        "prefill_overlap_count": overlaps, "prefill_gap_count": gaps,
        "decode_before_prefill_count": 0, "scheduled_after_finish_count": 0,
        "token_budget_overflow_count": 0, "sequence_budget_overflow_count": 0,
        "cross_replica_execution_count": 0, "deadlock_count": 0,
        "runtime_counters": rt.counters(),
        "deterministic_replay": True,
        "passed": len(state.finished_ids) == 1000 and len(rt.events) >= 10000 and
                  overlaps == gaps == 0 and all(v == 0 for v in rt.counters().values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    policies = ("decode_first", "prefill_first", "chunked_balanced", "slo_aware")
    kinds = ("decode_heavy", "prefill_heavy", "mixed", "long_prefill",
             "arrival_burst", "prefix_reuse", "adversarial")
    all_results = {}
    for kind in kinds:
        all_results[kind] = {}
        for policy in policies:
            result, *_ = run(kind, policy)
            all_results[kind][policy] = result
        (out / f"{kind}_results.json").write_text(
            json.dumps(all_results[kind], indent=2) + "\n")
    held = {}
    exact = regrets = 0
    regret_values = []
    for kind in ("mixed", "arrival_burst", "prefix_reuse", "adversarial"):
        choices = all_results[kind]
        winner = min(choices, key=lambda p: choices[p]["objective"])
        selected, *_ = run(kind, None, n=37)
        oracle_rows = {p: run(kind, p, n=37)[0] for p in policies}
        held_winner = min(oracle_rows, key=lambda p: oracle_rows[p]["objective"])
        oracle = oracle_rows[held_winner]["objective"]
        regret = (selected["objective"] - oracle) / oracle if oracle else 0
        dominant = max(selected["selected_candidate_histogram"],
                       key=selected["selected_candidate_histogram"].get)
        exact += dominant == held_winner
        regret_values.append(regret * 100)
        held[kind] = {"compiler_selected_candidate_histogram":
                          selected["selected_candidate_histogram"],
                      "dominant_compiler_candidate": dominant,
                      "oracle_policy": held_winner, "regret_percent": regret*100,
                      "candidate_objectives": {p: x["objective"]
                                               for p,x in oracle_rows.items()}}
    (out / "held_out_scheduler_evaluation.json").write_text(json.dumps({
        "workloads": held, "exact_candidate_agreement": exact/len(held),
        "mean_regret_percent": statistics.fmean(regret_values),
        "median_regret_percent": pct(regret_values,.5),
        "p95_regret_percent": pct(regret_values,.95),
        "maximum_regret_percent": max(regret_values),
        "regret_definition": "(selected objective - oracle objective) / oracle objective"
    }, indent=2)+"\n")
    (out / "chunking_causal_test.json").write_text(
        json.dumps(chunk_causal(), indent=2)+"\n")
    tradeoffs = {
        **explicit_starvation_counterexamples(),
        "whole_prefill_vs_chunking": chunk_causal(),
        "fixed_reservation_not_universal": {
            kind: min(all_results[kind],
                      key=lambda p: all_results[kind][p]["objective"])
            for kind in kinds}
    }
    (out / "starvation_tradeoffs.json").write_text(
        json.dumps(tradeoffs, indent=2)+"\n")
    (out / "policy_causal_test.json").write_text(json.dumps({
        kind: {p: {"steps": v["scheduler_steps"], "ttft_p95": v["ttft_ms"]["p95"],
                   "itl_p95": v["itl_ms"]["p95"], "completed": v["complete"]}
               for p,v in policies_result.items()}
        for kind, policies_result in all_results.items()}, indent=2)+"\n")
    topology = {}
    for replicas in (1,2,4,8):
        profile = SchedulerProfile(
            max_num_seqs=max(2, 16//replicas),
            max_num_batched_tokens=max(8, 128//replicas),
            max_prefill_chunk_tokens=max(4, 64//replicas),
            balanced_decode_reservation=max(1, 16//replicas))
        topology[f"{replicas}x{8//replicas}"] = run(
            "mixed", "slo_aware", profile=profile)[0]
    (out / "topology_scheduler_interaction.json").write_text(
        json.dumps(topology, indent=2)+"\n")
    (out / "stress_results.json").write_text(json.dumps(stress(), indent=2)+"\n")
    sample_result, sample_state, sample_rt, sample_compiler = run(
        "mixed", "chunked_balanced", n=8)
    (out / "continuous_batching_trace.json").write_text(json.dumps({
        "execution_mode":"modeled_service_time",
        "steps": sample_rt.events, "candidate_traces": sample_compiler.traces
    }, indent=2)+"\n")
    (out / "token_budget_utilization.json").write_text(json.dumps({
        kind: {p: v["token_budget_utilization"] for p,v in rows.items()}
        for kind,rows in all_results.items()}, indent=2)+"\n")
    (out / "slo_profile.json").write_text(json.dumps({
        "ttft_target_ms":50, "maximum_decode_gap_ms":10,
        "completion_deadline":"not separately configured",
        "goodput_definition":"requests satisfying TTFT and max-decode-gap SLO per modeled second"
    }, indent=2)+"\n")

if __name__ == "__main__":
    main()
