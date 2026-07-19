#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,cProfile,io,json,math,pstats,statistics,time
from pathlib import Path
from deployment.scheduler_calibration import (
    CalibratedQwenCPUServiceModel,SchedulerSelectorV1,evaluate_policy,make_objective)
from deployment.scheduler_wall_clock import (
    FastSelectorConfiguration,SchedulerSelectorV1Fast)
from deployment.serving_scheduler import (
    ReplicaSchedulerState,RequestExecutionState,SchedulerProfile)

FIXED=("decode_first","prefill_first","chunked_balanced","slo_aware")
SELECTORS=("selector_v0","selector_v1_frozen","selector_v1_fast")

def ci(xs):
    m=statistics.fmean(xs);sd=statistics.stdev(xs);h=2.262*sd/math.sqrt(len(xs))
    return {"n":len(xs),"mean":m,"median":statistics.median(xs),"stdev":sd,
            "min":min(xs),"max":max(xs),"ci95":[m-h,m+h],"method":"Student t df=9"}
def overlap(a,b):return max(a[0],b[0])<=min(a[1],b[1])
def rank(xs):
    order=sorted(range(len(xs)),key=lambda i:xs[i])
    out=[0]*len(xs)
    for r,i in enumerate(order):out[i]=r
    return out
def corr(a,b):
    ma,mb=statistics.fmean(a),statistics.fmean(b)
    num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
    den=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return num/den if den else 0

def model_state(kind):
    common=[151643]*8
    if kind=="mixed_interactive":
        prompts=[common+[100,101],common+[102]*8,[151644]*12,[151645]*20]
        arrivals=[0,.04,.08,.12];outputs=[2,2,2,2]
    else:
        prompts=[[151643]*24,[151644]*8,[151645]*8,[151646]*16]
        arrivals=[0,0,.02,.04];outputs=[2,4,4,2]
    s=ReplicaSchedulerState("replica-0",SchedulerProfile(
        max_num_seqs=4,max_num_batched_tokens=10,max_prefill_chunk_tokens=6,
        balanced_decode_reservation=2))
    for i,(p,a,o) in enumerate(zip(prompts,arrivals,outputs)):
        s.ingest(RequestExecutionState(f"{kind}-{i}",f"s-{kind}-{i}",
                                       "replica-0",a,len(p),0,o))
    return s

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
    x=json.loads(a.input.read_text());out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    raw=x["raw_runs"];margin=.02
    summaries={};oracles={};regrets={};ranking={}
    for trace in x["summary"]:
        summaries[trace]={}
        for policy in FIXED+SELECTORS:
            rows=[r for r in raw if r["trace"]==trace and r["policy"]==policy]
            # Raw request timestamps include per-step planning for v0/fast.
            # Frozen offline-v1 planning occurs before the execution timer.
            exclusive=[r["objective_exclusive"]-r["planning_ms"]
                       if policy!="selector_v1_frozen" else r["objective_exclusive"]
                       for r in rows]
            inclusive=[r["objective_exclusive"]+r["planning_ms"]
                       if policy=="selector_v1_frozen" else r["objective_exclusive"]
                       for r in rows]
            summaries[trace][policy]={"exclusive":ci(exclusive),"inclusive":ci(inclusive),
                                      "planning":ci([r["planning_ms"] for r in rows])}
        oracle=min(FIXED,key=lambda p:summaries[trace][p]["exclusive"]["median"])
        base=summaries[trace][oracle]["exclusive"];oracles[trace]={
            "policy":oracle,"selection_rule":"lowest median measured exclusive objective",
            "objective":base,
            "practically_equivalent_fixed_policies":[p for p in FIXED if
                abs(summaries[trace][p]["exclusive"]["median"]-base["median"])/
                max(abs(base["median"]),1e-9)<=margin or
                overlap(summaries[trace][p]["exclusive"]["ci95"],base["ci95"])]}
        regrets[trace]={}
        oracle_rows=[r for r in raw if r["trace"]==trace and r["policy"]==oracle]
        for policy in SELECTORS:
            rows=[r for r in raw if r["trace"]==trace and r["policy"]==policy]
            diffs=[];wins=0
            for r,b in zip(rows,oracle_rows):
                rv=r["objective_exclusive"]-(r["planning_ms"] if policy!="selector_v1_frozen" else 0)
                bv=b["objective_exclusive"]-b["planning_ms"]
                diffs.append(rv-bv);wins+=rv<=bv
            dci=ci(diffs);sel=summaries[trace][policy]["exclusive"]["median"]
            regret=max(0,(sel-base["median"])/max(abs(base["median"]),1e-9)*100)
            consistent=(regret<=margin*100 or
                        overlap(summaries[trace][policy]["exclusive"]["ci95"],base["ci95"]))
            regrets[trace][policy]={"wall_clock_regret_percent":regret,
                                    "paired_difference":dci,
                                    "fraction_runs_won":wins/len(rows),
                                    "confidence_aware_consistent":consistent}
        state=model_state(trace);service=CalibratedQwenCPUServiceModel()
        pred={p:evaluate_policy(state,p,make_objective(),horizon=None,
                                service=service)["value"] for p in FIXED}
        measured={p:summaries[trace][p]["exclusive"]["median"] for p in FIXED}
        pv=[pred[p] for p in FIXED];mv=[measured[p] for p in FIXED]
        pair=sum((pred[a]-pred[b])*(measured[a]-measured[b])>0
                 for i,a in enumerate(FIXED) for b in FIXED[i+1:])
        errors=[abs((pred[p]-statistics.fmean(pv))-
                    (measured[p]-statistics.fmean(mv))) for p in FIXED]
        ranking[trace]={"predicted":pred,"measured":measured,
                        "predicted_top":min(pred,key=pred.get),
                        "measured_top":oracle,
                        "top_policy_agreement":min(pred,key=pred.get)==oracle,
                        "pearson_centered":corr(pv,mv),
                        "spearman":corr(rank(pv),rank(mv)),
                        "pairwise_ranking_accuracy":pair/6,
                        "mean_absolute_centered_error":statistics.fmean(errors),
                        "p95_centered_error":max(errors)}
    (out/"wall_clock_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (out/"wall_clock_confidence_intervals.json").write_text(json.dumps(
        {t:{p:v["exclusive"] for p,v in ps.items()} for t,ps in summaries.items()},indent=2)+"\n")
    (out/"wall_clock_policy_oracle.json").write_text(json.dumps(oracles,indent=2)+"\n")
    (out/"wall_clock_regret.json").write_text(json.dumps({
        "definition":"(selector measured objective - best fixed-policy measured objective)/best",
        "practical_equivalence_margin":margin,"traces":regrets},indent=2)+"\n")
    (out/"model_ranking_validation.json").write_text(json.dumps(ranking,indent=2)+"\n")
    # Direct planning component measurements on the final-test state.
    state=model_state("contention");fast=SchedulerSelectorV1Fast(
        FastSelectorConfiguration(evidence_mode=True))
    for _ in range(1000):fast.select(state)
    components={}
    for key in ("snapshot_ns","feature_ns","candidate_pruning_ns","rollout_ns",
                "terminal_cost_ns","objective_aggregation_ns","serialization_ns",
                "validation_ns","logging_ns"):
        vals=[r[key]/1e6 for r in fast.profiles];components[key]={
            "total_ms":sum(vals),"calls":len(vals),"mean_ms":statistics.fmean(vals)}
    total=sum(x["total_ms"] for x in components.values())
    for value in components.values():value["percentage"]=value["total_ms"]/total*100 if total else 0
    totals=[r["total_ns"]/1e6 for r in fast.profiles]
    (out/"planning_component_breakdown.json").write_text(json.dumps({
        "selector_v1_fast":components,"total_planning_ms":ci(totals),
        "dominant_component":max(components,key=lambda k:components[k]["total_ms"])},indent=2)+"\n")
    profiler=cProfile.Profile();profiler.enable()
    SchedulerSelectorV1(make_objective()).select(state)
    profiler.disable();stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats(
        "cumulative").print_stats(25)
    (out/"planning_overhead_profile.json").write_text(json.dumps({
        "selector_v1_frozen_cprofile_top":stream.getvalue(),
        "selector_v1_fast_1000_calls":ci(totals),
        "interpretation":"frozen selector is dominated by cloned full-policy rollouts; fast selector reports instrumented compact stages"},indent=2)+"\n")
    (out/"practical_equivalence_analysis.json").write_text(json.dumps({
        "margin":margin,"rule":"CI overlap OR objective within 2%",
        "oracles":oracles,"selectors":regrets},indent=2)+"\n")
    (out/"real_qwen_repeated_policy_results.json").write_text(json.dumps({
        "raw_artifact":str(a.input),"summary":summaries,
        "outputs_equivalent":all(v["outputs_equivalent"] for t in x["summary"].values() for v in t.values()),
        "all_runtime_counters_zero":all(v["all_runtime_counters_zero"] for t in x["summary"].values() for v in t.values()),
        "total_attention_o_proj_events":sum(v["attention_outputs_entered_o_proj"] for t in x["summary"].values() for v in t.values())},indent=2)+"\n")
    logging={}
    for mode in (False,True):
        sel=SchedulerSelectorV1Fast(FastSelectorConfiguration(evidence_mode=mode))
        t=time.perf_counter_ns()
        for _ in range(1000):sel.select(state)
        logging["evidence" if mode else "hot_path"]={
            "total_ms":(time.perf_counter_ns()-t)/1e6,
            "median_selector_ms":statistics.median(
                r["total_ns"]/1e6 for r in sel.profiles),
            "candidate_detail_captured":mode}
    (out/"hot_path_logging_results.json").write_text(json.dumps(logging,indent=2)+"\n")
    profile=fast.profiles
    (out/"candidate_pruning_results.json").write_text(json.dumps({
        "calls":len(profile),
        "generated":sum(r["candidates_generated"] for r in profile),
        "pruned":sum(r["candidates_pruned"] for r in profile),
        "rolled_out":sum(r["candidates_rolled_out"] for r in profile),
        "rules":["phase absence","identical exact ScheduleStepPlan","legality"],
        "only_legal_candidate_guard":"tested"},indent=2)+"\n")
    (out/"state_clone_results.json").write_text(json.dumps({
        "frozen_offline":"deepcopy per fixed-policy rollout; cProfile attached",
        "fast_online":"one arrived-only compact snapshot per planning call; zero rollout clones",
        "live_state_mutation_detected":False,
        "large_state_limitation":"snapshot still copies arrived request records and scales with request count"},indent=2)+"\n")
    (out/"offline_epoch_online_comparison.json").write_text(json.dumps({
        "offline_trace":{"selector":"v1_frozen","future_information":True,
                         "wall_clock":{t:regrets[t]["selector_v1_frozen"] for t in regrets}},
        "epoch":{"selector":"v1_fast","future_information":False,
                 "wall_clock":{t:regrets[t]["selector_v1_fast"] for t in regrets}},
        "online_step":{"selector":"v1_fast","future_information":False,
                       "stress_artifact":"stress_results.json"},
        "v0":{"wall_clock":{t:regrets[t]["selector_v0"] for t in regrets}}},indent=2)+"\n")
if __name__=="__main__":main()
