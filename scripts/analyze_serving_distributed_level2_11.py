#!/usr/bin/env python3
"""Analyze frozen S2.11 replication, noise, ordering, and rollout fidelity."""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics
from pathlib import Path
from scripts.benchmark_qwen_observability_s2_8 import trace
from deployment.scheduler_rollout_v2 import rollout_policy,compare_rollout
from deployment.serving_scheduler import (
 ReplicaSchedulerState,RequestExecutionState,SchedulerProfile)

FIXED=("decode_first","prefill_first","chunked_balanced","slo_aware")
RISK="ranking_selector_v4_risk_aware"
def quantile(xs,q):
 x=sorted(xs);return x[min(len(x)-1,max(0,round(q*(len(x)-1))))]
def ci_boot(xs,seed=2110,n=10000):
 rng=random.Random(seed);means=[]
 for _ in range(n):means.append(statistics.fmean(rng.choice(xs) for _ in xs))
 return [quantile(means,.025),quantile(means,.975)]
def state_for(kind):
 prompts,arrivals,outputs=trace(kind)
 p=SchedulerProfile(4,10,6,1,2);s=ReplicaSchedulerState("replica-0",p)
 for i,(tokens,a,o) in enumerate(zip(prompts,arrivals,outputs)):
  rid=f"{kind}-{i}";s.ingest(RequestExecutionState(rid,f"sp-{rid}","replica-0",
    a,len(tokens),0,o))
 return s
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
 x=json.loads(a.input.read_text());a.output_dir.mkdir(parents=True,exist_ok=True)
 summary=x["summary"];trace_rows=[];risk_reg=[];base_reg=[];paired=[]
 for t,d in summary.items():
  values={p:d[p]["objective"]["mean"] for p in d}
  oracle=min(values.values())
  rr=(values[RISK]-oracle)/oracle;br=(values["decode_first"]-oracle)/oracle
  risk_reg.append(rr);base_reg.append(br);paired.append(rr-br)
  trace_rows.append({"trace":t,"fixed_oracle":min(FIXED,key=lambda p:values[p]),
   "selector_regret":rr,"baseline_regret":br,"paired_difference":rr-br,
   "selector_tie_aware":values[RISK]<=oracle*1.02})
 result={"trace_rows":trace_rows,
  "selector":{"mean_regret":statistics.fmean(risk_reg),
   "median_regret":statistics.median(risk_reg),"p90_regret":quantile(risk_reg,.9),
   "p95_regret":quantile(risk_reg,.95),"maximum_regret":max(risk_reg),
   "tie_aware_agreement":statistics.fmean(r["selector_tie_aware"] for r in trace_rows)},
  "baseline":{"mean_regret":statistics.fmean(base_reg),
   "p95_regret":quantile(base_reg,.95),"maximum_regret":max(base_reg)},
  "paired_selector_minus_baseline":{"mean":statistics.fmean(paired),
   "bootstrap_ci95":ci_boot(paired),"fraction_selector_better":
    statistics.fmean(v<0 for v in paired)}}
 ci=result["paired_selector_minus_baseline"]["bootstrap_ci95"]
 result["classification"]=("replicated" if ci[1]<0 and
  result["selector"]["mean_regret"]<result["baseline"]["mean_regret"] and
  result["selector"]["p95_regret"]<=result["baseline"]["p95_regret"]
  else "inconclusive" if result["selector"]["mean_regret"]<
  result["baseline"]["mean_regret"] else "failed")
 (a.output_dir/"replication_final_results.json").write_text(json.dumps(result,indent=2)+"\n")
 (a.output_dir/"replication_baseline_comparison.json").write_text(json.dumps({
  "selector":result["selector"],"baseline":result["baseline"],
  "paired":result["paired_selector_minus_baseline"],
  "classification":result["classification"]},indent=2)+"\n")
 # Join measured run group to its frozen order position.
 order={(z["group"],z["trace"],z["policy"]):i for i,z in enumerate(x["order"])
        if z["measured"]}
 position=[];noise=[]
 for r in x["raw_runs"]:
  block=[z for z in x["order"] if z["group"]==r["run_id"] and
         z["trace"]==r["trace_id"]]
  pos=next(i for i,z in enumerate(block) if z["policy"]==r["requested_policy"])
  position.append({"trace":r["trace_id"],"policy":r["requested_policy"],
   "run_id":r["run_id"],"order_position":pos,"objective":r["objective"]})
  noise.append({"trace":r["trace_id"],"policy":r["requested_policy"],
   "run_id":r["run_id"],**r.get("runtime_noise",{}),"makespan_ms":r["makespan_ms"]})
 bypos={str(i):[z["objective"] for z in position if z["order_position"]==i]
        for i in range(len(FIXED)+1)}
 (a.output_dir/"execution_order_analysis.json").write_text(json.dumps({
  "rows":position,"mean_objective_by_order_position":
   {k:statistics.fmean(v) for k,v in bypos.items()},
  "order_mode":x.get("order_mode"),"order_seed":x.get("order_seed")},indent=2)+"\n")
 cvs=[]
 for t in summary:
  for p in summary[t]:
   vals=[r["objective"] for r in x["raw_runs"] if r["trace_id"]==t and
         r["requested_policy"]==p]
   cvs.append({"trace":t,"policy":p,"n":len(vals),
    "coefficient_of_variation":statistics.stdev(vals)/statistics.fmean(vals)})
 (a.output_dir/"runtime_noise_analysis.json").write_text(json.dumps({
  "run_noise":noise,"within_cell":cvs,
  "mean_within_cell_cv":statistics.fmean(z["coefficient_of_variation"] for z in cvs),
  "maximum_within_cell_cv":max(z["coefficient_of_variation"] for z in cvs)},indent=2)+"\n")
 # Compare exact semantic rollout to first measured fixed-policy execution.
 comps=[]
 for t in summary:
  for p in FIXED:
   actual=next(r["steps"] for r in x["raw_runs"] if r["trace_id"]==t and
               r["requested_policy"]==p)
   predicted=rollout_policy(state_for(t),p,128)
   comps.append({"trace":t,"policy":p,**compare_rollout(predicted,actual)})
 (a.output_dir/"prospective_rollout_raw_comparison.json").write_text(json.dumps(comps,indent=2)+"\n")
 causes={}
 for z in comps:
  cause=("no_divergence" if z["first_divergence_step"] is None else
   "arrival_visibility_or_clock_alignment" if z["first_divergence_step"]>0 else
   "initial_request_order_or_plan_item_capture")
  causes[cause]=causes.get(cause,0)+1;z["classified_cause"]=cause
 (a.output_dir/"first_rollout_divergence_analysis.json").write_text(json.dumps({
  "rows":comps,"cause_frequency":causes},indent=2)+"\n")
 fidelity={"original_s2_10":{"kind":"aggregate step-count feature",
   "semantic_plan_prediction":False},
  "policy_rollout_v2":{"exact_first_step_agreement":
   statistics.fmean(z["first_divergence_step"] not in (0,) for z in comps),
   "exact_full_sequence_agreement":statistics.fmean(z["semantic_divergence_steps"]==0 for z in comps),
   "mean_selected_work_jaccard":statistics.fmean(z["mean_selected_work_jaccard"] for z in comps),
   "mean_absolute_step_count_error":statistics.fmean(abs(z["predicted_steps"]-z["actual_steps"]) for z in comps)},
  "selector_v4_changed":False}
 (a.output_dir/"rollout_fidelity_results.json").write_text(json.dumps(fidelity,indent=2)+"\n")
 (a.output_dir/"semantic_vs_timing_divergence.json").write_text(json.dumps({
  "semantic_divergence_steps":sum(z["semantic_divergence_steps"] for z in comps),
  "equivalent_plan_steps":sum(z["equivalent_steps"] for z in comps),
  "timing_divergence":"measured separately by within-cell variance; rollout-v2 does not predict time"},indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
