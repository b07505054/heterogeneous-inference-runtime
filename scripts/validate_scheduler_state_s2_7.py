#!/usr/bin/env python3
"""Validate S2.7 incremental summaries and large-state plan-only execution."""
from __future__ import annotations
import argparse,hashlib,json,statistics,time
from pathlib import Path
from deployment.scheduler_ranking_v2 import (
    IncrementalSchedulerState,RankingModel,RankingSelectorV2,reference_summary)
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime,ReplicaSchedulerState,RequestExecutionState,
    SchedulerCompiler,SchedulerProfile,deserialize_schedule_plan)

def comparable(x):
    result={k:getattr(x,k) for k in (
        "active_count","waiting_count","prefill_ready_count","decode_ready_count",
        "total_remaining_prefill","total_remaining_decode","prefix_hit_tokens",
        "oldest_waiting_age_ms","largest_decode_gap_ms","mean_decode_gap_ms",
        "completed_count","history_record_count")}
    for key,value in result.items():
        if isinstance(value,float):result[key]=round(value,8)
    return result
def pct(xs,p):return sorted(xs)[min(len(xs)-1,round((len(xs)-1)*p))]
def load_model(path):
    x=json.loads(path.read_text());p=x["model"];return RankingModel(
        **p,frozen_digest=x["frozen_digest"])
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args();model=load_model(a.output_dir/"selector_v2_freeze.json")
    profile=SchedulerProfile(max_num_seqs=4,max_num_batched_tokens=2,
        max_prefill_chunk_tokens=1,balanced_decode_reservation=1)
    state=ReplicaSchedulerState("replica-0",profile)
    for i in range(1000):state.ingest(RequestExecutionState(
        f"stress-{i}",f"sp-{i}","replica-0",i*.001,10,0,11))
    inc=IncrementalSchedulerState(state,frontier_size=8)
    selector=RankingSelectorV2(model);rt=PlanOnlySchedulerRuntime()
    summary_lat=[];selector_lat=[];mismatches=[];steps=0
    while state.unfinished():
        if not state.ready():
            future=[r.arrival_time_ms for r in state.requests.values()
                    if r.phase=="WAITING"]
            if future:state.clock_ms=min(future);state.ready()
        # Ingest newly arrived requests; already-hot updates are idempotent.
        for r in state.requests.values():
            if r.arrival_time_ms<=state.clock_ms and not r.finished and \
                    r.request_id not in inc.hot:inc.upsert(r,state.clock_ms)
        t=time.perf_counter_ns();summary=inc.snapshot()
        summary_lat.append((time.perf_counter_ns()-t)/1e6)
        plan=selector.select(summary);selector_lat.append(plan.selection_overhead_ms)
        step=SchedulerCompiler().compile(state,forced_policy=plan.policy_id)
        loaded=deserialize_schedule_plan(step.serialize(),state)
        changed=[state.requests[x.request_id] for x in loaded.items]
        rt.execute(state,loaded)
        for r in changed:inc.upsert(r,state.clock_ms)
        steps+=1
        if steps%100==0:
            ref=reference_summary(state,frontier_size=8)
            if comparable(inc.snapshot())!=comparable(ref):
                mismatches.append({"step":steps,"incremental":comparable(inc.snapshot()),
                                   "reference":comparable(ref)})
    result={"requests":1000,"steps":steps,"completed":len(state.finished_ids),
      "summary_reference_checks":steps//100,"summary_reference_mismatches":mismatches,
      "summary_latency_ms":{"median":statistics.median(summary_lat),
                            "p95":pct(summary_lat,.95),"max":max(summary_lat)},
      "selector_latency_ms":{"median":statistics.median(selector_lat),
                             "p95":pct(selector_lat,.95),"max":max(selector_lat)},
      "runtime_counters":rt.counters(),"request_loss":1000-len(state.finished_ids),
      "duplicate_completion":len(state.finished_ids)-len(set(state.finished_ids)),
      "passed":steps>=10000 and len(state.finished_ids)==1000 and not mismatches
               and all(v==0 for v in rt.counters().values())}
    (a.output_dir/"large_state_stress.json").write_text(json.dumps(result,indent=2)+"\n")
    (a.output_dir/"incremental_summary_validation.json").write_text(json.dumps({
      "transitions":steps,"reference_checks":steps//100,
      "mismatch_count":len(mismatches),"mismatches":mismatches,
      "method":"event updates checked against independent full recomputation"},
      indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
