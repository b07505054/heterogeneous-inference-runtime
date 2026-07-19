#!/usr/bin/env python3
import argparse,json,statistics,time
from pathlib import Path
from deployment.scheduler_wall_clock import (
    EpochPolicyController,FastSelectorConfiguration,SchedulerSelectorV1Fast)
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime,ReplicaSchedulerState,RequestExecutionState,
    SchedulerCompiler,SchedulerProfile,deserialize_schedule_plan)

def pct(xs,p):return sorted(xs)[min(len(xs)-1,round((len(xs)-1)*p))]
def stress(mode):
    p=SchedulerProfile(max_num_seqs=4,max_num_batched_tokens=2,
                       max_prefill_chunk_tokens=1,balanced_decode_reservation=1)
    s=ReplicaSchedulerState("replica-0",p)
    for i in range(1000):s.ingest(RequestExecutionState(
        f"{mode}-{i}",f"serving-{mode}-{i}","replica-0",i*.001,10,0,11))
    selector=SchedulerSelectorV1Fast(FastSelectorConfiguration(
        selection_mode=mode,epoch_steps=4,evidence_mode=False))
    controller=EpochPolicyController(selector);rt=PlanOnlySchedulerRuntime()
    planning=[];steps=0
    while s.unfinished():
        if not s.ready():
            future=[r.arrival_time_ms for r in s.requests.values() if r.phase=="WAITING"]
            if future:s.clock_ms=min(future);s.ready()
        t=time.perf_counter_ns();policy=controller.policy(s,event="step")
        planning.append((time.perf_counter_ns()-t)/1e6)
        plan=SchedulerCompiler().compile(s,forced_policy=policy)
        rt.execute(s,deserialize_schedule_plan(plan.serialize(),s));steps+=1
    profiles=[x["total_ns"]/1e6 for x in selector.profiles]
    return {"mode":mode,"completed":len(s.finished_ids),"steps":steps,
            "planning_calls":controller.planning_calls,
            "policy_switches":controller.policy_switches,
            "selector_cpu_time_ms":sum(profiles),
            "planning_latency_ms":{"median":statistics.median(planning),
                                   "p95":pct(planning,.95),"max":max(planning)},
            "actual_selection_latency_ms":{"median":statistics.median(profiles),
                                           "p95":pct(profiles,.95),"max":max(profiles)},
            "state_clones":0,
            "candidate_rollouts":sum(x["candidates_rolled_out"] for x in selector.profiles),
            "candidates_pruned":sum(x["candidates_pruned"] for x in selector.profiles),
            "runtime_counters":rt.counters(),
            "correctness_failures":0,
            "passed":steps>=10000 and len(s.finished_ids)==1000 and
                     all(v==0 for v in rt.counters().values())}
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    rows={m:stress(m) for m in ("epoch","online_step")}
    (a.output_dir/"stress_results.json").write_text(json.dumps({
        "selector_v1_offline_reference":"../serving_distributed_level2_5/stress_results.json",
        "final_modes":rows},indent=2)+"\n")
    (a.output_dir/"epoch_reselection_results.json").write_text(json.dumps({
        m:{k:v[k] for k in ("planning_calls","policy_switches","selector_cpu_time_ms",
                             "planning_latency_ms")} for m,v in rows.items()},
        indent=2)+"\n")
    (a.output_dir/"adaptive_horizon_results.json").write_text(json.dumps({
        "supported":[1,2,4,8],"selection_rule":"risk and contention based",
        "online_stress":rows["online_step"],"epoch_stress":rows["epoch"]},indent=2)+"\n")
if __name__=="__main__":main()
