#!/usr/bin/env python3
"""Repeated real-Qwen wall-clock scheduler-policy benchmark.

This benchmark uses real eager Qwen model forwards and contiguous per-request
Transformers caches. The separate cross-layer S2.5 runs prove compiler
attention/operator provenance for the same policies.
"""
from __future__ import annotations
import argparse,copy,json,math,statistics,time
from pathlib import Path
import torch

from deployment.scheduler_calibration import SchedulerSelectorV1,make_objective
from deployment.scheduler_wall_clock import (
    EpochPolicyController,FastSelectorConfiguration,SchedulerSelectorV1Fast)
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime,ReplicaSchedulerState,RequestExecutionState,
    SchedulerCompiler,SchedulerProfile,deserialize_schedule_plan)

POLICIES=("decode_first","prefill_first","chunked_balanced","slo_aware",
          "selector_v0","selector_v1_frozen","selector_v1_fast")

def t_ci(xs):
    if len(xs)<2:raise ValueError("confidence interval needs at least two samples")
    mean=statistics.fmean(xs);sd=statistics.stdev(xs);half=2.262*sd/math.sqrt(len(xs))
    return {"n":len(xs),"mean":mean,"median":statistics.median(xs),
            "stdev":sd,"min":min(xs),"max":max(xs),"ci95":[mean-half,mean+half],
            "method":"Student t, df=9"}

def trace(kind,seed):
    common=[151643]*8
    if kind=="mixed_interactive":
        prompts=[common+[100,101],common+[102]*8,[151644]*12,[151645]*20]
        arrivals=[0,.04,.08,.12];outputs=[2,2,2,2]
    else:
        prompts=[[151643]*24,[151644]*8,[151645]*8,[151646]*16]
        arrivals=[0,0,.02,.04];outputs=[2,4,4,2]
    return prompts,arrivals,outputs

def run_once(model,kind,policy,seed):
    prompts,arrivals,outputs=trace(kind,seed)
    profile=SchedulerProfile(max_num_seqs=4,max_num_batched_tokens=10,
                             max_prefill_chunk_tokens=6,
                             balanced_decode_reservation=2)
    state=ReplicaSchedulerState("replica-0",profile)
    local={}
    for i,(tokens,arrival,out) in enumerate(zip(prompts,arrivals,outputs)):
        rid=f"{kind}-{i}"
        state.ingest(RequestExecutionState(rid,f"serving-{rid}","replica-0",
                                           arrival,len(tokens),0,out))
        local[rid]={"prompt":torch.tensor([tokens]),"cache":None,"context":0,
                    "generated":[],"pending":None,"first_wall":None,
                    "complete_wall":None}
    compiler=SchedulerCompiler();runtime=PlanOnlySchedulerRuntime()
    planning_ns=0;planning_calls=0;policy_switches=0;state_clones=0;rollouts=0
    epoch=None;controller=None
    if policy=="selector_v1_frozen":
        t=time.perf_counter_ns()
        epoch=SchedulerSelectorV1(make_objective()).select(state).policy_id
        planning_ns+=time.perf_counter_ns()-t;planning_calls=1;state_clones=4
    elif policy=="selector_v1_fast":
        controller=EpochPolicyController(SchedulerSelectorV1Fast(
            FastSelectorConfiguration(selection_mode="epoch",epoch_steps=4)))
    elif policy in POLICIES[:4]:epoch=policy
    hook_count=0
    def hook(_m,_v):
        nonlocal hook_count;hook_count+=1
    hooks=[layer.self_attn.o_proj.register_forward_pre_hook(hook)
           for layer in model.model.layers]
    started=time.perf_counter();step_policies=[]
    try:
        while state.unfinished():
            if not state.ready():
                future=[r.arrival_time_ms for r in state.requests.values()
                        if r.phase=="WAITING"]
                if future:state.clock_ms=min(future);state.ready()
            t=time.perf_counter_ns()
            if policy=="selector_v0":
                selected=compiler.compile(state);planning_calls+=1
            else:
                if controller:
                    selected_policy=controller.policy(state,event="step")
                else:selected_policy=epoch
                selected=compiler.compile(state,forced_policy=selected_policy)
            planning_ns+=time.perf_counter_ns()-t
            loaded=deserialize_schedule_plan(selected.serialize(),state)
            step_policies.append(loaded.policy)
            def execute(req,item,_plan):
                x=local[req.request_id]
                if item.phase=="decode" and x["pending"] is not None:
                    logits=x["pending"];x["pending"]=None
                    token=int(logits.argmax(-1));x["generated"].append(token)
                    if x["first_wall"] is None:x["first_wall"]=time.perf_counter()
                    if len(x["generated"])==req.expected_output_tokens:
                        x["complete_wall"]=time.perf_counter()
                    return {}
                current=(x["prompt"][:,item.token_start:item.token_start+item.token_count]
                         if item.phase=="prefill" else
                         torch.tensor([[x["generated"][-1]]]))
                with torch.no_grad():
                    out=model(current,past_key_values=x["cache"],use_cache=True,
                              logits_to_keep=1)
                x["cache"]=out.past_key_values
                if item.phase=="prefill" and item.token_start+item.token_count==len(x["prompt"][0]):
                    x["pending"]=out.logits[:,-1].float()
                elif item.phase=="decode":
                    token=int(out.logits[:,-1].argmax(-1));x["generated"].append(token)
                    if x["first_wall"] is None:x["first_wall"]=time.perf_counter()
                    if len(x["generated"])==req.expected_output_tokens:
                        x["complete_wall"]=time.perf_counter()
                return {}
            runtime.execute(state,loaded,execute)
    finally:
        for h in hooks:h.remove()
    end=time.perf_counter();exclusive=(end-started)*1000
    if controller:
        planning_calls=controller.planning_calls;policy_switches=controller.policy_switches
        planning_ns=sum(x["total_ns"] for x in controller.selector.profiles)
        state_clones=0;rollouts=sum(
            controller.current.candidates_rolled_out for _ in [0])
    ttft=[(x["first_wall"]-started)*1000 for x in local.values()]
    e2e=[(x["complete_wall"]-started)*1000 for x in local.values()]
    objective=statistics.fmean(ttft)+.25*statistics.fmean(e2e)
    return {"trace":kind,"policy":policy,"seed":seed,
            "exclusive_execution_ms":exclusive,
            "planning_ms":planning_ns/1e6,
            "inclusive_ms":exclusive+planning_ns/1e6,
            "objective_exclusive":objective,
            "objective_inclusive":objective+planning_ns/1e6,
            "ttft_mean_ms":statistics.fmean(ttft),"e2e_mean_ms":statistics.fmean(e2e),
            "generated":{k:v["generated"] for k,v in local.items()},
            "attention_outputs_entered_o_proj":hook_count,
            "planning_calls":planning_calls,"policy_switches":policy_switches,
            "state_clones":state_clones,"candidate_rollouts":rollouts,
            "step_policies":step_policies,"runtime_counters":runtime.counters()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--warmups",type=int,default=3);ap.add_argument("--runs",type=int,default=10)
    a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModelForCausalLM
    torch.manual_seed(20260717);torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained(
        a.model,local_files_only=True,dtype=torch.float32,
        attn_implementation="eager").eval()
    traces=("mixed_interactive","contention")
    order=[];raw=[]
    for group in range(a.warmups+a.runs):
        policies=POLICIES if group%2==0 else tuple(reversed(POLICIES))
        if group%4 in (2,3):policies=policies[2:]+policies[:2]
        for trace_id in traces:
            for policy in policies:
                order.append({"group":group,"measured":group>=a.warmups,
                              "trace":trace_id,"policy":policy})
                row=run_once(model,trace_id,policy,20260717+group)
                if group>=a.warmups:raw.append(row)
    summary={}
    for trace_id in traces:
        summary[trace_id]={}
        for policy in POLICIES:
            rows=[r for r in raw if r["trace"]==trace_id and r["policy"]==policy]
            summary[trace_id][policy]={
                "objective_exclusive":t_ci([r["objective_exclusive"] for r in rows]),
                "objective_inclusive":t_ci([r["objective_inclusive"] for r in rows]),
                "ttft_mean_ms":t_ci([r["ttft_mean_ms"] for r in rows]),
                "e2e_mean_ms":t_ci([r["e2e_mean_ms"] for r in rows]),
                "planning_ms":t_ci([r["planning_ms"] for r in rows]),
                "outputs_equivalent":all(r["generated"]==rows[0]["generated"] for r in rows),
                "all_runtime_counters_zero":all(all(v==0 for v in r["runtime_counters"].values()) for r in rows),
                "attention_outputs_entered_o_proj":sum(r["attention_outputs_entered_o_proj"] for r in rows)}
    a.output.write_text(json.dumps({
        "execution_mode":"real_qwen_wall_clock","warmups":a.warmups,
        "measured_runs_per_policy_trace":a.runs,
        "run_order":order,"raw_runs":raw,"summary":summary,
        "truth_boundary":"real eager Qwen CPU model-forward; S2.5 separate runs prove compiler-attention provenance"},indent=2)+"\n")
if __name__=="__main__":main()
