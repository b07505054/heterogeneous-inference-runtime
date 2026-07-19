"""S2.11 semantic policy rollout using the existing legal plan generator.

This diagnostic predictor is deliberately separate from frozen selector-v4.
It duplicates only request progress, never model tensors, provenance history,
or live runtime state.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import hashlib,json
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime, ReplicaSchedulerState, RequestExecutionState, SchedulerCompiler,
    SchedulerProfile,
)

@dataclass(frozen=True)
class RolloutStep:
    step_id: int
    state_hash: str
    items: tuple[tuple[str,str,int,int], ...]
    next_state_hash: str

def _request_copy(r: RequestExecutionState) -> RequestExecutionState:
    return replace(r,prefill_chunks=list(r.prefill_chunks),
                   decode_times_ms=list(r.decode_times_ms),
                   operator_provenance=list(r.operator_provenance))

def clone_scheduler_state(state: ReplicaSchedulerState) -> ReplicaSchedulerState:
    clone=ReplicaSchedulerState(state.replica_id,state.profile)
    clone.clock_ms=state.clock_ms;clone.step_id=state.step_id
    for r in state.requests.values(): clone.ingest(_request_copy(r))
    return clone

def state_hash(state: ReplicaSchedulerState) -> str:
    payload={"replica":state.replica_id,"clock":state.clock_ms,"step":state.step_id,
      "requests":[{"id":r.request_id,"phase":r.phase,
       "prefill":r.prefill_completed_tokens,"decode":r.decode_completed_tokens,
       "finished":r.finished} for r in sorted(state.requests.values(),
                                              key=lambda x:x.request_id)]}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,
      separators=(",",":")).encode()).hexdigest()

def rollout_policy(state:ReplicaSchedulerState,policy:str,horizon:int=8
                   )->tuple[RolloutStep,...]:
    if horizon<1: raise ValueError("rollout horizon must be positive")
    live_hash=state_hash(state);clone=clone_scheduler_state(state)
    compiler=SchedulerCompiler();runtime=PlanOnlySchedulerRuntime();rows=[]
    for _ in range(horizon):
        if not clone.unfinished(): break
        if not clone.ready():
            future=[r.arrival_time_ms for r in clone.requests.values()
                    if r.phase=="WAITING"]
            if not future: break
            clone.clock_ms=min(future);clone.ready()
        before=state_hash(clone)
        plan=compiler.compile(clone,forced_policy=policy)
        items=tuple((x.request_id,x.phase,x.token_start,x.token_count)
                    for x in plan.items)
        runtime.execute(clone,plan)
        rows.append(RolloutStep(plan.step_id,before,items,state_hash(clone)))
    if state_hash(state)!=live_hash:
        raise RuntimeError("rollout-v2 mutated live scheduler state")
    return tuple(rows)

def compare_rollout(predicted:tuple[RolloutStep,...],
                    actual:list[dict])->dict:
    first=None;semantic=equivalent=0;jaccards=[]
    n=max(len(predicted),len(actual))
    for i in range(n):
        p=set(predicted[i].items) if i<len(predicted) else set()
        a=set((x["request_id"],x["phase"],x["token_start"],x["token_count"])
              for x in actual[i].get("execution",{}).get("scheduled_items",[])
              ) if i<len(actual) else set()
        jaccards.append(len(p&a)/len(p|a) if p|a else 1.0)
        if p!=a:
            if first is None:first=i
            semantic+=1
        elif i<len(predicted) and i<len(actual):equivalent+=1
    return {"predicted_steps":len(predicted),"actual_steps":len(actual),
      "first_divergence_step":first,"semantic_divergence_steps":semantic,
      "equivalent_steps":equivalent,
      "mean_selected_work_jaccard":sum(jaccards)/len(jaccards) if jaccards else 1.0}
