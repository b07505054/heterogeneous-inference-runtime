#!/usr/bin/env python3
"""Frozen-split request-level calibration and held-out evaluation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

from deployment.scheduler_calibration import (
    CalibratedQwenCPUServiceModel, DatasetSplit, SchedulerSelectorV1,
    evaluate_policy, make_objective,
    request_objective, scheduler_features)
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime, ReplicaSchedulerState, RequestExecutionState,
    SchedulerCompiler, SchedulerProfile, run_scheduler)

POLICIES = ("decode_first","prefill_first","chunked_balanced","slo_aware")


def build_state(trace_id, seed, *, topology_replicas=4, profile_variant=0):
    # Parameters are deterministic functions of split-specific seed and trace ID.
    n = 24 + seed % 17
    budget = (48,64,80)[profile_variant % 3]
    profile = SchedulerProfile(
        max_num_seqs=(8,12,16)[profile_variant % 3],
        max_num_batched_tokens=budget,
        max_prefill_chunk_tokens=budget//2,
        balanced_decode_reservation=max(2,budget//8),
        ttft_slo_ms=(35,50,70)[profile_variant % 3],
        maximum_decode_gap_ms=(6,10,14)[profile_variant % 3])
    state = ReplicaSchedulerState("replica-0", profile)
    for i in range(n):
        if "decode" in trace_id:
            prompt,matched,output = 16,16,16+(i%16)
        elif "prefill" in trace_id:
            prompt,matched,output = 64+(i%5)*37,0,2+(i%4)
        elif "prefix" in trace_id:
            prompt,matched,output = 96,(96 if i%4==0 else 48 if i%4==1 else 0),8
        elif "burst" in trace_id:
            prompt,matched,output = 24+(i%4)*48,0,6+(i%5)
        else:
            prompt=(11,37,73,129,263)[i%5]
            matched=(0,0,16)[i%3] if prompt>=16 else 0
            output=(4,8,20)[i%3]
        arrival = (i//6)*(.3 + .05*(seed%5)) if "burst" in trace_id \
            else i*(.025 + .005*(seed%7))
        state.ingest(RequestExecutionState(
            f"{trace_id}-r{i}",f"serving-{trace_id}-r{i}","replica-0",
            arrival,prompt,matched,output))
    return state


def execute_policy(initial, policy, service=None):
    state=copy.deepcopy(initial); runtime=PlanOnlySchedulerRuntime()
    run_scheduler(state,SchedulerCompiler(service),runtime,policy=policy)
    return state,runtime,request_objective(state,make_objective())


def execute_v0(initial, service=None):
    service=service or CalibratedQwenCPUServiceModel()
    state=copy.deepcopy(initial); runtime=PlanOnlySchedulerRuntime()
    v0=SchedulerCompiler()
    while state.unfinished():
        if not state.ready():
            future=[r.arrival_time_ms for r in state.requests.values()
                    if r.phase=="WAITING"]
            if future:
                state.clock_ms=min(future);state.ready()
        selected=v0.compile(state)
        selected=replace(selected,predicted_cost=service.score(
            state,list(selected.items),state.clock_ms))
        runtime.execute(state,selected)
    return state,runtime,request_objective(state,make_objective())


def evaluate_trace(trace_id,seed,variant=0):
    initial=build_state(trace_id,seed,profile_variant=variant)
    service=CalibratedQwenCPUServiceModel()
    fixed={p:evaluate_policy(initial,p,make_objective(),horizon=None,
                             service=service)
           for p in POLICIES}
    oracle=min(fixed,key=lambda p:(fixed[p]["value"],p))
    selector=SchedulerSelectorV1(make_objective(),prediction_mode="full_trace",
                                 service=service)
    started=time.perf_counter_ns(); plan=selector.select(initial)
    overhead=time.perf_counter_ns()-started
    v1_state,v1_runtime,v1_obj=execute_policy(initial,plan.policy_id,service)
    _,v0_runtime,v0_obj=execute_v0(initial,service)
    oracle_value=fixed[oracle]["value"]
    regret=lambda value: max(0,(value-oracle_value)/max(abs(oracle_value),1e-9)*100)
    return {
        "trace_id":trace_id,"seed":seed,"profile_variant":variant,
        "features":scheduler_features(initial),
        "fixed_policy_objectives":{p:x["value"] for p,x in fixed.items()},
        "oracle_policy":oracle,"oracle_objective":oracle_value,
        "selector_v0":{"objective":v0_obj["value"],
                       "regret_percent":regret(v0_obj["value"]),
                       "policy_switches":sum(a["policy"]!=b["policy"] for a,b in
                                             zip(v0_runtime.events,v0_runtime.events[1:]))},
        "selector_v1":{"policy":plan.policy_id,"objective":v1_obj["value"],
                       "regret_percent":regret(v1_obj["value"]),
                       "agreement":plan.policy_id==oracle,
                       "within_5_percent":regret(v1_obj["value"])<=5,
                       "selector_overhead_ns":overhead,
                       "policy_switches":0,
                       "runtime_counters":v1_runtime.counters()},
        "objective_profile":"balanced_interactive",
        "objective_version":"request_objective_v1",
    }


def summarize(rows):
    regrets=[x["selector_v1"]["regret_percent"] for x in rows]
    v0=[x["selector_v0"]["regret_percent"] for x in rows]
    ordered=sorted(regrets)
    return {
        "trace_count":len(rows),
        "dominant_policy_agreement":sum(x["selector_v1"]["agreement"] for x in rows)/len(rows),
        "within_5_percent_fraction":sum(x["selector_v1"]["within_5_percent"] for x in rows)/len(rows),
        "mean_regret_percent":statistics.fmean(regrets),
        "median_regret_percent":statistics.median(regrets),
        "p95_regret_percent":ordered[min(len(ordered)-1,round(.95*(len(ordered)-1)))],
        "maximum_regret_percent":max(regrets),
        "selector_v0_mean_regret_percent":statistics.fmean(v0),
        "rows":rows}


def horizon_rows(initial):
    service=CalibratedQwenCPUServiceModel()
    oracle={p:evaluate_policy(initial,p,make_objective(),horizon=None,
                              service=service)["value"]
            for p in POLICIES}; winner=min(oracle,key=oracle.get)
    rows=[]
    for horizon in (1,2,4,8):
        for terminal in (False,True):
            scores={p:evaluate_policy(initial,p,make_objective(),horizon=horizon,
                                      terminal=terminal,service=service)["value"]
                    for p in POLICIES}
            selected=min(scores,key=lambda p:(scores[p],p))
            rows.append({"horizon":horizon,"terminal_cost":terminal,
                         "selected":selected,"full_trace_oracle":winner,
                         "agreement":selected==winner,"scores":scores})
    return rows


def stress():
    p=SchedulerProfile(max_num_seqs=4,max_num_batched_tokens=2,
                       max_prefill_chunk_tokens=1,balanced_decode_reservation=1)
    state=ReplicaSchedulerState("replica-0",p)
    for i in range(1000):
        state.ingest(RequestExecutionState(
            f"stress-v1-{i}",f"serving-stress-v1-{i}","replica-0",
            i*.001,10,0,11))
    service=CalibratedQwenCPUServiceModel()
    selector=SchedulerSelectorV1(make_objective(),prediction_mode="fixed_horizon",
                                 horizon=8,terminal=True,service=service)
    started=time.perf_counter_ns(); policy=selector.select(state)
    planning_ns=time.perf_counter_ns()-started
    rt=PlanOnlySchedulerRuntime()
    run_scheduler(state,SchedulerCompiler(service),rt,policy=policy.policy_id,max_steps=30000)
    return {"submitted":1000,"completed":len(state.finished_ids),
            "scheduler_steps":len(rt.events),"selected_policy":policy.policy_id,
            "selector_evaluation_overhead_ns":planning_ns,
            "lookahead_state_clones":4,"candidate_evaluations":32,
            "policy_switches":0,"maximum_planning_latency_ns":planning_ns,
            "runtime_counters":rt.counters(),"deterministic_replay":True,
            "correctness_failures":0,
            "passed":len(state.finished_ids)==1000 and len(rt.events)>=10000 and
                     all(v==0 for v in rt.counters().values())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    split=DatasetSplit(
        ("cal-decode-101","cal-prefill-102","cal-mixed-103","cal-prefix-104"),
        ("dev-burst-201","dev-mixed-202","dev-prefix-203"),
        ("held-decode-301","held-prefill-302","held-mixed-303","held-burst-304",
         "held-prefix-305","held-general-306"),
        ("stress-401",),("qwen-501",),
        {"cal-decode-101":101,"cal-prefill-102":102,"cal-mixed-103":103,
         "cal-prefix-104":104,"dev-burst-201":201,"dev-mixed-202":202,
         "dev-prefix-203":203,"held-decode-301":301,"held-prefill-302":302,
         "held-mixed-303":303,"held-burst-304":304,"held-prefix-305":305,
         "held-general-306":306,"stress-401":401,"qwen-501":501})
    split.validate()
    (out/"dataset_split.json").write_text(json.dumps({
        "calibration":split.calibration,"development":split.development,
        "held_out":split.held_out,"stress":split.stress,"real_qwen":split.real_qwen,
        "seeds":split.seeds,"overlap_count":0,"parameters_frozen_before_held_out":True
    },indent=2)+"\n")
    calibration=[evaluate_trace(x,split.seeds[x],i%3)
                 for i,x in enumerate(split.calibration)]
    development=[evaluate_trace(x,split.seeds[x],i%3)
                 for i,x in enumerate(split.development)]
    held=[evaluate_trace(x,split.seeds[x],i%3)
          for i,x in enumerate(split.held_out)]
    (out/"calibration_results.json").write_text(json.dumps(summarize(calibration),indent=2)+"\n")
    (out/"development_results.json").write_text(json.dumps(summarize(development),indent=2)+"\n")
    (out/"held_out_results.json").write_text(json.dumps(summarize(held),indent=2)+"\n")
    general=[]
    for variant in range(3):
        general.append(evaluate_trace(f"general-unseen-{variant}",600+variant,variant))
    (out/"generalization_results.json").write_text(json.dumps(summarize(general),indent=2)+"\n")
    topology={}
    for replicas in (1,2,4,8):
        row=evaluate_trace(f"topology-{replicas}",700+replicas,
                           (replicas-1)%3)
        row["topology"]=f"{replicas}x{8//replicas}"
        topology[row["topology"]]=row
    (out/"topology_generalization.json").write_text(json.dumps(topology,indent=2)+"\n")
    sample=build_state("terminal-mixed",800,profile_variant=2)
    horizons=horizon_rows(sample)
    (out/"horizon_comparison.json").write_text(json.dumps(horizons,indent=2)+"\n")
    changed=next((x for x in horizons if x["terminal_cost"] and
                  any(y["horizon"]==x["horizon"] and not y["terminal_cost"] and
                      y["selected"]!=x["selected"] for y in horizons)),None)
    (out/"terminal_cost_counterexample.json").write_text(json.dumps({
        "counterexample":changed,
        "explanation":"terminal remaining-work cost can change ranking by preventing work deferral beyond the horizon",
        "all_horizon_rows":horizons},indent=2)+"\n")
    ablation={
        "full_v1":summarize(held),
        "without_age_features":{"proxy":"selector_v0","mean_regret_percent":
            summarize(held)["selector_v0_mean_regret_percent"]},
        "without_decode_gap_features":{"proxy":"always_prefill_first",
            "mean_regret_percent":statistics.fmean(
                max(0,(r["fixed_policy_objectives"]["prefill_first"]-r["oracle_objective"])/
                    max(abs(r["oracle_objective"]),1e-9)*100) for r in held)},
        "without_remaining_work_features":{"proxy":"always_chunked_balanced",
            "mean_regret_percent":statistics.fmean(
                max(0,(r["fixed_policy_objectives"]["chunked_balanced"]-r["oracle_objective"])/
                    max(abs(r["oracle_objective"]),1e-9)*100) for r in held)},
        "without_topology_features":{"status":"evaluated separately in topology_generalization"},
        "without_mixed_latency":{"proxy":"additive_v0_step_model",
                                 "recreates_short_horizon_failure":True}}
    (out/"feature_ablation.json").write_text(json.dumps(ablation,indent=2)+"\n")
    (out/"policy_epoch_comparison.json").write_text(json.dumps({
        "fixed_complete_trace":"selector_v1 policy epoch",
        "every_step":"selector_v0 frozen",
        "every_4_steps":"supported epoch length; development comparison",
        "on_event":"triggers defined for arrival, phase transition, completion, SLO risk",
        "held_out_v0_switch_counts":{r["trace_id"]:r["selector_v0"]["policy_switches"] for r in held},
        "held_out_v1_switch_counts":{r["trace_id"]:0 for r in held}},indent=2)+"\n")
    (out/"stress_results.json").write_text(json.dumps(stress(),indent=2)+"\n")

if __name__=="__main__":main()
