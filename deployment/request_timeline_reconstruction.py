"""S2.9 physical request-timeline reconstruction from committed step items."""
from __future__ import annotations
from dataclasses import dataclass
import math,statistics
from typing import Any
from deployment.serving_execution import ServingPlanError

RECONSTRUCTION_TOLERANCE_MS=1.0

@dataclass(frozen=True)
class TimestampSemantics:
    clock:str="time.perf_counter_ns"
    domain:str="host_monotonic_wall_clock"
    request_arrival:str="benchmark host start for admitted trace request"
    request_admission:str="first legal ScheduleStepPlan item start"
    prefill_start:str="first prefill callback start"
    prefill_completion:str="commit of final prefill item"
    first_decode_start:str="first decode callback start"
    first_token:str="commit of first decode-token result"
    decode_token:str="commit of each decode-token result"
    completion:str="commit of final requested decode token"
    scheduler_step_start:str="before summary/policy/plan regions"
    scheduler_step_end:str="after state commit"
    state_commit:str="PlanOnlySchedulerRuntime mutation complete"

def reconstruct_run(run:dict[str,Any])->dict[str,Any]:
    if "host_wall_start_ns" not in run:
        raise ServingPlanError("missing physical run clock origin")
    origin=run["host_wall_start_ns"];events={}
    seen_steps=set()
    for step in run["steps"]:
        sid=step["step_id"]
        if sid in seen_steps:raise ServingPlanError("request reconstruction double-counts step")
        seen_steps.add(sid)
        if step["step_end_ns"] is None:raise ServingPlanError("request reconstruction omits scheduler step")
        for item in step["execution"].get("scheduled_items",[]):
            if not step["step_start_ns"]<=item["callback_start_ns"]<=item["commit_ns"]<=step["step_end_ns"]:
                raise ServingPlanError("item timestamp outside scheduler step")
            events.setdefault(item["request_id"],[]).append(item)
    direct=run["direct_request_timestamps"];rows=[]
    for rid,d in direct.items():
        xs=events.get(rid,[])
        if not xs:raise ServingPlanError("request reconstruction omits request")
        prefill=[x for x in xs if x["phase"]=="prefill"]
        decode=[x for x in xs if x["phase"]=="decode"]
        token_ms=[(x["commit_ns"]-origin)/1e6 for x in decode]
        arrival=d["arrival_ms"]
        first=token_ms[0] if token_ms else None
        completion=token_ms[-1] if token_ms else None
        gaps=[b-a for a,b in zip(token_ms,token_ms[1:])]
        direct_tokens=d["decode_token_times_ms"]
        direct_gaps=[b-a for a,b in zip(direct_tokens,direct_tokens[1:])]
        row={"request_id":rid,"queue_wait_ms":
             (min(x["callback_start_ns"] for x in xs)-origin)/1e6-arrival,
             "prefill_start_ms":((prefill[0]["callback_start_ns"]-origin)/1e6
                                 if prefill else None),
             "prefill_completion_ms":((prefill[-1]["commit_ns"]-origin)/1e6
                                      if prefill else 0.0),
             "first_token_ms":first,"decode_token_times_ms":token_ms,
             "max_decode_gap_ms":max([0.0]+gaps),"tpot_ms":
             (statistics.fmean(gaps) if gaps else 0.0),
             "completion_ms":completion,"e2e_ms":completion-arrival,
             "residuals_ms":{
              "ttft":first-d["first_token_ms"],
              "completion":completion-d["completion_ms"],
              "max_decode_gap":max([0.0]+gaps)-max([0.0]+direct_gaps)}}
        rows.append(row)
    return {"run_id":run["run_id"],"trace_id":run["trace_id"],
            "requests":rows,"step_count":len(seen_steps)}

def error_summary(reconstructed:list[dict[str,Any]],
                  tolerance_ms:float=RECONSTRUCTION_TOLERANCE_MS)->dict[str,Any]:
    if tolerance_ms<=0 or not math.isfinite(tolerance_ms):
        raise ServingPlanError("invalid frozen reconstruction tolerance")
    result={"tolerance_ms":tolerance_ms,"metrics":{}}
    for key in ("ttft","completion","max_decode_gap"):
        values=[abs(r["residuals_ms"][key]) for run in reconstructed
                for r in run["requests"]]
        ordered=sorted(values)
        result["metrics"][key]={"mae_ms":statistics.fmean(values),
          "median_ms":statistics.median(values),
          "p95_ms":ordered[min(len(ordered)-1,round(.95*(len(ordered)-1)))],
          "max_ms":max(values)}
    result["passed"]=all(v["p95_ms"]<=tolerance_ms
                         for v in result["metrics"].values())
    return result

