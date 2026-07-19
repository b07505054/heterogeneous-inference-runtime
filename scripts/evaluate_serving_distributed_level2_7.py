#!/usr/bin/env python3
"""S2.7 independent CPU wall-clock ranking and state-scaling evaluation."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,random,statistics,time
from pathlib import Path

from deployment.scheduler_ranking_v2 import (
    IncrementalSchedulerState, RANKING_FEATURES, RankingModel,
    RankingSelectorV2, reference_summary, summary_features)
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime,ReplicaSchedulerState,RequestExecutionState,
    SchedulerCompiler,SchedulerProfile,deserialize_schedule_plan)

POLICIES=("decode_first","prefill_first","chunked_balanced","slo_aware")
FAMILIES=("decode_heavy","prefill_heavy","mixed_interactive","contention",
          "arrival_burst","long_prefill","prefix_heavy","low_prefix_reuse",
          "small_token_budget","large_token_budget","small_sequence_budget",
          "large_sequence_budget")

def pct(xs,p): return sorted(xs)[min(len(xs)-1,round((len(xs)-1)*p))]
def stats(xs):
    return {"n":len(xs),"mean":statistics.fmean(xs),
            "median":statistics.median(xs),"p95":pct(xs,.95),
            "min":min(xs),"max":max(xs),
            "stdev":statistics.stdev(xs) if len(xs)>1 else 0.0}

def profile(family):
    tokens=32 if family=="small_token_budget" else 128
    seqs=4 if family=="small_sequence_budget" else 16
    return SchedulerProfile(max_num_seqs=seqs,max_num_batched_tokens=tokens,
        max_prefill_chunk_tokens=min(64,tokens),
        balanced_decode_reservation=min(16,tokens//2))

def make_state(family,seed):
    rng=random.Random(seed);p=profile(family)
    s=ReplicaSchedulerState("replica-0",p)
    n=18 if family in ("decode_heavy","large_sequence_budget") else 12
    for i in range(n):
        if family=="decode_heavy": prompt,out,arrival=8,12,i*.04
        elif family in ("prefill_heavy","long_prefill"):
            prompt,out,arrival=(192 if family=="long_prefill" else 96),2,i*.02
        elif family=="arrival_burst": prompt,out,arrival=32,5,(i//4)*1.5
        elif family=="contention": prompt,out,arrival=(128 if i%3==0 else 8),8,i*.01
        else: prompt,out,arrival=rng.choice((8,24,48,96)),rng.choice((2,5,9)),i*.05
        matched=0
        if family=="prefix_heavy": matched=prompt-(prompt%8)
        elif family!="low_prefix_reuse" and i%4==0: matched=min(prompt,8)
        r=RequestExecutionState(f"{family}-{i}",f"sp-{family}-{i}",
            "replica-0",arrival,prompt,matched,out)
        s.ingest(r)
    return s

def cpu_work(prefill,decode):
    # A real CPU wall-clock workload with non-additive mixed-batch behavior.
    rounds=20+prefill*3+decode*24+(18 if prefill and decode else 0)
    value=b"s2.7"
    for _ in range(rounds):
        value=hashlib.sha256(value).digest()
    return value[0]

def run_policy(family,seed,policy):
    s=make_state(family,seed);rt=PlanOnlySchedulerRuntime()
    wall0=time.perf_counter();first={};done={};step_lat=[];steps=0
    while s.unfinished():
        if not s.ready():
            future=[r.arrival_time_ms for r in s.requests.values()
                    if r.phase=="WAITING"]
            if future:s.clock_ms=min(future);s.ready()
        plan=SchedulerCompiler().compile(s,forced_policy=policy)
        loaded=deserialize_schedule_plan(plan.serialize(),s)
        p=sum(x.token_count for x in loaded.items if x.phase=="prefill")
        d=sum(x.phase=="decode" for x in loaded.items)
        t=time.perf_counter_ns();cpu_work(p,d)
        def callback(req,item,_plan):
            now=(time.perf_counter()-wall0)*1000
            if item.phase=="decode":
                first.setdefault(req.request_id,now)
                if req.decode_completed_tokens+1==req.expected_output_tokens:
                    done[req.request_id]=now
            return {}
        rt.execute(s,loaded,callback)
        step_lat.append((time.perf_counter_ns()-t)/1e6);steps+=1
    elapsed=(time.perf_counter()-wall0)*1000
    ttft=[first[x.request_id] for x in s.requests.values()]
    e2e=[done[x.request_id] for x in s.requests.values()]
    objective=statistics.fmean(ttft)+.25*statistics.fmean(e2e)
    return {"family":family,"seed":seed,"policy":policy,
        "execution_mode":"functional_cpu_wall_clock",
        "objective":objective,"ttft_mean_ms":statistics.fmean(ttft),
        "e2e_mean_ms":statistics.fmean(e2e),"makespan_ms":elapsed,
        "steps":steps,"step_latency_ms":stats(step_lat),
        "runtime_counters":rt.counters()}

def trace_features(family,seed):
    s=make_state(family,seed);s.clock_ms=max(r.arrival_time_ms for r in s.requests.values())
    s.ready()
    return summary_features(reference_summary(s))

def split():
    groups={
      "ranking_train":FAMILIES[:5],"ranking_development":FAMILIES[5:7],
      "ranking_validation":FAMILIES[7:9],"ranking_final_test":FAMILIES[9:],
      "real_qwen_cross_layer":("qwen_mixed_v2","qwen_contention_v2"),
      "scaling_stress":("summary_1000","summary_5000")}
    seeds={f:27000+i for i,f in enumerate(FAMILIES)}
    return groups,seeds

def fit_centroid(rows,groups,seeds):
    winners={p:[] for p in POLICIES}
    for family in groups["ranking_train"]:
        subset=[r for r in rows if r["family"]==family]
        med={p:statistics.median([r["objective"] for r in subset if r["policy"]==p])
             for p in POLICIES}
        winner=min(med,key=lambda p:(med[p],p))
        winners[winner].append(trace_features(family,seeds[family]))
    allx=[trace_features(f,seeds[f]) for f in groups["ranking_train"]]
    means={f:statistics.fmean(x[f] for x in allx) for f in RANKING_FEATURES}
    scales={f:max(statistics.pstdev([x[f] for x in allx]),1.0)
            for f in RANKING_FEATURES}
    global_center={f:0.0 for f in RANKING_FEATURES}
    coefficients={};intercepts={}
    for p in POLICIES:
        source=winners[p]
        center={f:(statistics.fmean((x[f]-means[f])/scales[f] for x in source)
                   if source else global_center[f]) for f in RANKING_FEATURES}
        coefficients[p]={f:-2*center[f] for f in RANKING_FEATURES}
        intercepts[p]=sum(v*v for v in center.values())+(100 if not source else 0)
    payload={"version":"ranking_selector_v2_static_frozen_20260717",
      "coefficients":coefficients,"intercepts":intercepts,
      "feature_means":means,"feature_scales":scales,
      "training_provenance":"independent_functional_cpu_wall_clock_paired_v1",
      "equivalence_margin":.02,"hysteresis_margin":.01,
      "default_policy":"chunked_balanced"}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,
        separators=(",",":")).encode()).hexdigest()
    return RankingModel(**payload,frozen_digest=digest),winners

def evaluate(model,rows,families,seeds,adaptive=False):
    detail=[]
    for family in families:
        subset=[r for r in rows if r["family"]==family]
        med={p:statistics.median([r["objective"] for r in subset if r["policy"]==p])
             for p in POLICIES}
        best=min(med,key=lambda p:(med[p],p))
        state=make_state(family,seeds[family])
        state.clock_ms=max(r.arrival_time_ms for r in state.requests.values())
        state.ready()
        plan=RankingSelectorV2(model,adaptive=adaptive).select(
            reference_summary(state))
        selected=plan.policy_id
        regret=(med[selected]-med[best])/max(med[best],1e-9)
        ordered=sorted(POLICIES,key=lambda p:med[p])
        predicted=sorted(POLICIES,key=lambda p:plan.scores[p])
        pairs=0;correct=0
        for i,a in enumerate(POLICIES):
            for b in POLICIES[i+1:]:
                pairs+=1;correct+=((med[a]-med[b])*(plan.scores[a]-plan.scores[b])>=0)
        detail.append({"family":family,"oracle":best,"selected":selected,
            "objectives":med,"scores":dict(plan.scores),"regret":regret,
            "top1_agreement":selected==best,
            "pairwise_accuracy":correct/pairs,
            "measured_order":ordered,"predicted_order":predicted,
            "selector_overhead_ms":plan.selection_overhead_ms})
    return {"traces":detail,
      "top1_agreement":statistics.fmean(x["top1_agreement"] for x in detail),
      "mean_regret":statistics.fmean(x["regret"] for x in detail),
      "median_regret":statistics.median(x["regret"] for x in detail),
      "p95_regret":pct([x["regret"] for x in detail],.95),
      "pairwise_accuracy":statistics.fmean(x["pairwise_accuracy"] for x in detail)}

def scaling():
    rows=[]
    for n in (4,16,64,256,1000,5000):
        s=make_state("mixed_interactive",30000)
        s.requests.clear();s.version=0
        for i in range(n):s.ingest(RequestExecutionState(
            f"scale-{i}",f"sp-{i}","replica-0",0,32,0,4,phase="PREFILL"))
        inc=IncrementalSchedulerState(s,frontier_size=8)
        old=[];new=[]
        for _ in range(25):
            t=time.perf_counter_ns();copy.deepcopy(s);old.append((time.perf_counter_ns()-t)/1e6)
            t=time.perf_counter_ns();inc.snapshot();new.append((time.perf_counter_ns()-t)/1e6)
        rows.append({"requests":n,"old_snapshot_ms":stats(old),
            "new_summary_ms":stats(new),
            "speedup":statistics.median(old)/statistics.median(new)})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--repetitions",type=int,default=5)
    a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    groups,seeds=split()
    dataset={"groups":groups,"seeds":seeds,
      "freeze_sequence":["train","development","model_freeze","validation",
                         "final_test"],"s2_6_traces_usage":"historical_failure_only"}
    (a.output_dir/"wall_clock_ranking_dataset_split.json").write_text(
        json.dumps(dataset,indent=2)+"\n")
    rows=[]
    for family in FAMILIES:
        order=list(POLICIES)
        for rep in range(a.repetitions):
            if rep%2:order=list(reversed(order))
            for policy in order:rows.append(run_policy(family,seeds[family],policy))
    model,winners=fit_centroid(rows,groups,seeds)
    freeze={"freeze_before_validation_and_final_test":True,
            "model":model.payload(),"frozen_digest":model.frozen_digest,
            "training_winner_groups":{p:[FAMILIES.index(f) for f in FAMILIES
                if trace_features(f,seeds[f]) in xs] for p,xs in winners.items()}}
    (a.output_dir/"selector_v2_freeze.json").write_text(json.dumps(freeze,indent=2)+"\n")
    for name in ("ranking_train","ranking_development","ranking_validation",
                 "ranking_final_test"):
        result=evaluate(model,rows,groups[name],seeds)
        (a.output_dir/f"{name}_results.json").write_text(json.dumps(result,indent=2)+"\n")
    final=evaluate(model,rows,groups["ranking_final_test"],seeds)
    (a.output_dir/"wall_clock_final_test_runs.json").write_text(json.dumps(
        [r for r in rows if r["family"] in groups["ranking_final_test"]],indent=2)+"\n")
    (a.output_dir/"wall_clock_final_results.json").write_text(json.dumps(final,indent=2)+"\n")
    (a.output_dir/"wall_clock_training_runs.json").write_text(json.dumps(
        [r for r in rows if r["family"] in groups["ranking_train"]],indent=2)+"\n")
    scale=scaling()
    (a.output_dir/"new_state_scaling_results.json").write_text(json.dumps(scale,indent=2)+"\n")
    (a.output_dir/"snapshot_scaling_baseline.json").write_text(json.dumps(
        [{"requests":x["requests"],"old_snapshot_ms":x["old_snapshot_ms"]} for x in scale],
        indent=2)+"\n")
    print(json.dumps({"final":final,"scaling":scale},indent=2))
if __name__=="__main__":main()
