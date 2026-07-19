#!/usr/bin/env python3
"""S2.11 frozen-v4 scalability and correctness regression."""
import argparse,json,statistics,time
from pathlib import Path
from deployment.scheduler_policy_v4 import RankingSelectorV4,RISK_V4,freeze_v4
from deployment.scheduler_ranking_v2 import IncrementalSchedulerState,reference_summary
from deployment.serving_scheduler import (PlanOnlySchedulerRuntime,
 ReplicaSchedulerState,RequestExecutionState,SchedulerCompiler,SchedulerProfile,
 deserialize_schedule_plan)
def pct(x,p):return sorted(x)[round((len(x)-1)*p)]
def comp(x):return tuple(getattr(x,k) for k in ("active_count","waiting_count",
 "prefill_ready_count","decode_ready_count","total_remaining_prefill",
 "total_remaining_decode","completed_count"))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
 a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 profile=SchedulerProfile(4,2,1,1,1);state=ReplicaSchedulerState("replica-0",profile)
 for i in range(1000):state.ingest(RequestExecutionState(
  f"s-{i}",f"sp-{i}","replica-0",i*.001,10,0,11))
 inc=IncrementalSchedulerState(state,frontier_size=8)
 sel=RankingSelectorV4(freeze_v4(RISK_V4),True);rt=PlanOnlySchedulerRuntime()
 sl=[];pl=[];mismatch=[];steps=0
 while state.unfinished():
  if not state.ready():
   f=[r.arrival_time_ms for r in state.requests.values() if r.phase=="WAITING"]
   state.clock_ms=min(f);state.ready()
  for r in state.requests.values():
   if r.arrival_time_ms<=state.clock_ms and not r.finished and r.request_id not in inc.hot:
    inc.upsert(r,state.clock_ms)
  t=time.perf_counter_ns();s=inc.snapshot();sl.append((time.perf_counter_ns()-t)/1e6)
  d=sel.select(s);pl.append(d.latency_ms)
  plan=SchedulerCompiler().compile(state,forced_policy=d.policy_id)
  loaded=deserialize_schedule_plan(plan.serialize(),state)
  changed=[state.requests[x.request_id] for x in loaded.items];rt.execute(state,loaded)
  for r in changed:inc.upsert(r,state.clock_ms)
  steps+=1
  if steps%100==0 and comp(inc.snapshot())!=comp(reference_summary(state,frontier_size=8)):
   mismatch.append(steps)
 # 5,000 arrived request snapshot is bounded by K.
 big=ReplicaSchedulerState("replica-0",profile)
 for i in range(5000):big.ingest(RequestExecutionState(f"b-{i}",f"bp-{i}",
  "replica-0",0,10,0,2))
 big.clock_ms=1;big.ready();bi=IncrementalSchedulerState(big,frontier_size=8)
 samples=[]
 for _ in range(101):
  t=time.perf_counter_ns();bi.snapshot();samples.append((time.perf_counter_ns()-t)/1e6)
 result={"requests":1000,"steps":steps,"completed":len(state.finished_ids),
  "summary_latency_ms":{"median":statistics.median(sl),"p95":pct(sl,.95)},
  "risk_selector_latency_ms":{"median":statistics.median(pl),"p95":pct(pl,.95)},
  "requests_5000_summary_median_ms":statistics.median(samples),
  "summary_reference_mismatch_count":len(mismatch),
  "runtime_counters":rt.counters(),"request_loss":1000-len(state.finished_ids),
  "duplicate_completion":len(state.finished_ids)-len(set(state.finished_ids)),
  "passed":steps>=10000 and not mismatch and len(state.finished_ids)==1000 and
   all(v==0 for v in rt.counters().values())}
 (a.output_dir/"stress_results.json").write_text(json.dumps(result,indent=2)+"\n")
 (a.output_dir/"scalability_regression.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
