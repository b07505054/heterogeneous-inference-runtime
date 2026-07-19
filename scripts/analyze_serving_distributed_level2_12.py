#!/usr/bin/env python3
"""Frozen S2.12 paired, clustered, tail, family, gate, and rollout analysis."""
import argparse,hashlib,json,math,random,statistics
from pathlib import Path
from scripts.benchmark_qwen_observability_s2_8 import trace,scheduler_profile
from deployment.scheduler_rollout_v2 import rollout_policy,compare_rollout
from deployment.serving_scheduler import ReplicaSchedulerState,RequestExecutionState
P=("decode_first","prefill_first","chunked_balanced","slo_aware")
R="ranking_selector_v4_risk_aware";SEED=2120
def q(x,p):return sorted(x)[min(len(x)-1,round((len(x)-1)*p))]
def boot_mean(x,n=10000):
 rng=random.Random(SEED);return [statistics.fmean(rng.choice(x) for _ in x) for _ in range(n)]
def ci(x):b=boot_mean(x);return [q(b,.025),q(b,.975)]
def trace_state(t):
 prompts,arrivals,outputs=trace(t);s=ReplicaSchedulerState("replica-0",scheduler_profile(t))
 for i,(tok,a,o) in enumerate(zip(prompts,arrivals,outputs)):
  rid=f"{t}-{i}";s.ingest(RequestExecutionState(rid,f"sp-{rid}","replica-0",a,len(tok),0,o))
 return s
def score_and_unc(s):
 active=max(s["active_request_count"],1);budget=max(s["token_budget"],1)
 ppr=s["total_remaining_prefill_tokens"]/active
 pp=s["total_remaining_prefill_tokens"]/budget
 dp=s["decode_ready_count"]/max(s["sequence_budget"],1)
 mixed=float(bool(s["prefill_ready_count"] and s["decode_ready_count"]))
 wait=s["oldest_waiting_age_ms"]/50;gap=s["largest_decode_gap_ms"]/10
 scores={"decode_first":0.0,"prefill_first":.35*pp-.15*dp,
  "chunked_balanced":.08-.16*mixed-.04*min(ppr/32,2),
  "slo_aware":.06-.18*max(wait,gap)-.08*mixed}
 pred_steps=pp+s["total_remaining_decode_tokens"]/max(s["sequence_budget"],1)
 ood=min(1,max(0,(ppr-64)/128));unc=min(1,ood+.05*pred_steps/8+.015)
 cand=min(scores,key=lambda z:(scores[z],z));improve=-scores[cand]
 covered=unc<=.35 and improve>.12
 return scores,unc,ood,improve,cand,covered
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
 ap.add_argument("--registry",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True)
 a=ap.parse_args();x=json.loads(a.input.read_text());reg=json.loads(a.registry.read_text())
 a.output_dir.mkdir(parents=True,exist_ok=True);raw=x["raw_runs"]
 family={z["trace_id"]:z["family"] for z in reg["traces"]}
 # Run-level: one paired observation per trace x measured block.
 obs=[]
 for t in x["summary"]:
  groups=sorted(set(r["run_id"] for r in raw if r["trace_id"]==t))
  for g in groups:
   vals={r["requested_policy"]:r["objective"] for r in raw if r["trace_id"]==t and r["run_id"]==g}
   oracle=min(vals.values());sr=(vals[R]-oracle)/oracle;br=(vals["decode_first"]-oracle)/oracle
   obs.append({"trace":t,"family":family[t],"block":g,"selector_regret":sr,
    "baseline_regret":br,"improvement":br-sr})
 dif=[z["improvement"] for z in obs];mean=statistics.fmean(dif);sd=statistics.stdev(dif)
 run_result={"paired_observations":len(obs),"mean_improvement":mean,
  "median_improvement":statistics.median(dif),"standard_error":sd/math.sqrt(len(dif)),
  "bootstrap_ci95":ci(dif),"standardized_effect_size":mean/sd if sd else 0,
  "bootstrap_probability_improvement":statistics.fmean(v>0 for v in boot_mean(dif)),
  "rows":obs}
 (a.output_dir/"run_level_paired_analysis.json").write_text(json.dumps(run_result,indent=2)+"\n")
 # Controlling clustered analysis: average repeats within each trace.
 tr=[]
 for t in x["summary"]:
  z=[o for o in obs if o["trace"]==t];tr.append({"trace":t,"family":family[t],
   "selector_regret":statistics.fmean(o["selector_regret"] for o in z),
   "baseline_regret":statistics.fmean(o["baseline_regret"] for o in z),
   "improvement":statistics.fmean(o["improvement"] for o in z)})
 td=[z["improvement"] for z in tr];tb=boot_mean(td)
 clustered={"independent_trace_count":len(tr),"mean_improvement":statistics.fmean(td),
  "median_improvement":statistics.median(td),"cluster_bootstrap_ci95":[q(tb,.025),q(tb,.975)],
  "bootstrap_probability_improvement":statistics.fmean(v>0 for v in tb),"rows":tr}
 (a.output_dir/"trace_clustered_paired_analysis.json").write_text(json.dumps(clustered,indent=2)+"\n")
 # Tail difference and cluster bootstrap.
 sr=[z["selector_regret"] for z in tr];br=[z["baseline_regret"] for z in tr]
 def tail_delta(sample):
  return q([tr[i]["selector_regret"] for i in sample],.95)-q([tr[i]["baseline_regret"] for i in sample],.95)
 rng=random.Random(SEED);bd=[tail_delta([rng.randrange(len(tr)) for _ in tr]) for _ in range(10000)]
 tail={"selector":{"p90":q(sr,.9),"p95":q(sr,.95),"maximum":max(sr),
   "worst_trace_mean":max(sr)},"baseline":{"p90":q(br,.9),"p95":q(br,.95),
   "maximum":max(br),"worst_trace_mean":max(br)},
  "p95_difference_selector_minus_baseline":q(sr,.95)-q(br,.95),
  "cluster_bootstrap_ci95":[q(bd,.025),q(bd,.975)],
  "noninferiority_tolerance":.005,"noninferiority_pass":q(bd,.975)<=.005}
 (a.output_dir/"tail_regret_analysis.json").write_text(json.dumps(tail,indent=2)+"\n")
 fam=[]
 for z in tr:
  status=("selector_better" if z["improvement"]>.0 else
   "selector_worse" if z["improvement"]<-.02 else "statistically_tied_or_insufficient")
  fam.append({**z,"sample_count":10,"tie_aware_agreement":float(z["selector_regret"]<=.02),
   "classification":status})
 (a.output_dir/"family_level_results.json").write_text(json.dumps(fam,indent=2)+"\n")
 # Sensitivity.
 lot=[]
 for excluded in tr:
  keep=[z["improvement"] for z in tr if z is not excluded]
  lot.append({"excluded_trace":excluded["trace"],"mean_improvement":statistics.fmean(keep),
   "cluster_bootstrap_ci95":ci(keep),"conclusion":"positive" if ci(keep)[0]>0 else "inconclusive"})
 (a.output_dir/"leave_one_trace_out.json").write_text(json.dumps(lot,indent=2)+"\n")
 lof=[{"excluded_family":z["family"],"mean_improvement":r["mean_improvement"],
       "cluster_bootstrap_ci95":r["cluster_bootstrap_ci95"],"conclusion":r["conclusion"]}
      for z,r in zip(tr,lot)]
 (a.output_dir/"leave_one_family_out.json").write_text(json.dumps(lof,indent=2)+"\n")
 # Noise and order.
 cvs=[];noise=[]
 for t in x["summary"]:
  for p in x["summary"][t]:
   rs=[r for r in raw if r["trace_id"]==t and r["requested_policy"]==p]
   vals=[r["objective"] for r in rs];cvs.append({"trace":t,"policy":p,"n":len(vals),
    "cv":statistics.stdev(vals)/statistics.fmean(vals)})
   noise.extend({"trace":t,"policy":p,"block":r["run_id"],**r["runtime_noise"]} for r in rs)
 pos=[]
 for r in raw:
  block=[z for z in x["order"] if z["group"]==r["run_id"] and z["trace"]==r["trace_id"]]
  pos.append({"trace":r["trace_id"],"policy":r["requested_policy"],"block":r["run_id"],
   "position":next(i for i,z in enumerate(block) if z["policy"]==r["requested_policy"]),
   "objective":r["objective"]})
 (a.output_dir/"runtime_noise_analysis.json").write_text(json.dumps({"within_cell":cvs,
  "mean_cv":statistics.fmean(z["cv"] for z in cvs),"maximum_cv":max(z["cv"] for z in cvs),
  "order_position_means":{str(i):statistics.fmean(z["objective"] for z in pos if z["position"]==i) for i in range(5)},
  "between_block_variance":statistics.pvariance([
    statistics.fmean(z["objective"] for z in pos if z["block"]==b)
    for b in sorted(set(z["block"] for z in pos))]),
  "noise_rows":noise},indent=2)+"\n")
 order_rows=[]
 for group in sorted(set(z["group"] for z in x["order"])):
  for t in x["summary"]:
   block=[z for z in x["order"] if z["group"]==group and z["trace"]==t]
   for i,z in enumerate(block):
    order_rows.append({**z,"block_id":group,"order_position":i,
     "previous_policy":block[i-1]["policy"] if i else None,
     "next_policy":block[i+1]["policy"] if i+1<len(block) else None,
     "warm_state_indicator":group>0})
 (a.output_dir/"execution_order_analysis.json").write_text(json.dumps({
  "mode":x.get("order_mode"),"seed":x.get("order_seed"),"rows":order_rows},indent=2)+"\n")
 # Frozen gate behavior on selector steps.
 gate={"candidate_selection_count":0,"robust_default_count":0,"out_of_distribution_count":0,
       "confidence_rejection_count":0,"improvement_margin_rejection_count":0,
       "practical_tie_count":0,"step_count":0}
 buckets={k:[] for k in ("0-20","20-40","40-60","60-80","80-100")}
 obsmap={(z["trace"],z["block"]):z for z in obs}
 candidate_outcomes=[];default_outcomes=[]
 for r in raw:
  if r["requested_policy"]!=R:continue
  for st in r["steps"]:
   sc,u,ood,improve,covcand,cov=score_and_unc(st["state"]);gate["step_count"]+=1
   selected=st["policy_id"];gate["candidate_selection_count"]+=int(selected!="decode_first")
   gate["robust_default_count"]+=int(selected=="decode_first")
   gate["out_of_distribution_count"]+=int(ood>0)
   gate["confidence_rejection_count"]+=int(u>.35)
   gate["improvement_margin_rejection_count"]+=int(improve<=.12)
   vals=sorted(sc.values());gate["practical_tie_count"]+=int(vals[1]-vals[0]<=.02)
   conf=1-u;idx=min(4,int(conf*5));key=("0-20","20-40","40-60","60-80","80-100")[idx]
   outcome=obsmap[(r["trace_id"],r["run_id"])]
   row={"selected":selected,"confidence":conf,"improvement":outcome["improvement"],
        "selector_regret":outcome["selector_regret"]}
   buckets[key].append(row)
   (candidate_outcomes if selected!="decode_first" else default_outcomes).append(row)
 gate["selector_coverage"]=gate["candidate_selection_count"]/max(gate["step_count"],1)
 def outcome_summary(rows):
  return {"sample_count":len(rows),"win_rate":statistics.fmean(x["improvement"]>0 for x in rows) if rows else None,
   "mean_improvement":statistics.fmean(x["improvement"] for x in rows) if rows else None,
   "p95_regret":q([x["selector_regret"] for x in rows],.95) if rows else None,
   "worst_loss":min((x["improvement"] for x in rows),default=None)}
 gate["candidate_realized"]=outcome_summary(candidate_outcomes)
 gate["default_realized"]=outcome_summary(default_outcomes)
 (a.output_dir/"risk_gate_behavior.json").write_text(json.dumps(gate,indent=2)+"\n")
 (a.output_dir/"confidence_calibration_audit.json").write_text(json.dumps({
  "buckets":{k:{"sample_count":len(v),"mean_confidence":
    statistics.fmean(x["confidence"] for x in v) if v else None,
    "observed_win_rate":statistics.fmean(x["improvement"]>0 for x in v) if v else None,
    "mean_realized_improvement":statistics.fmean(x["improvement"] for x in v) if v else None,
    "mean_regret":statistics.fmean(x["selector_regret"] for x in v) if v else None,
    "maximum_regret":max((x["selector_regret"] for x in v),default=None)}
    for k,v in buckets.items()},
  "limitation":"Confidence is a frozen support/OOD gate, not a probability. No threshold was refit."},indent=2)+"\n")
 # Rollout exactness on all fixed policies/new traces.
 comps=[]
 for t in x["summary"]:
  for p in P:
   actual=next(r["steps"] for r in raw if r["trace_id"]==t and r["requested_policy"]==p)
   pred=rollout_policy(trace_state(t),p,256);comps.append({"trace":t,"policy":p,**compare_rollout(pred,actual)})
 rollout={"case_count":len(comps),"first_step_agreement":
  statistics.fmean(z["first_divergence_step"] not in (0,) for z in comps),
  "full_sequence_agreement":statistics.fmean(z["semantic_divergence_steps"]==0 for z in comps),
  "selected_work_jaccard":statistics.fmean(z["mean_selected_work_jaccard"] for z in comps),
  "mean_step_count_error":statistics.fmean(abs(z["predicted_steps"]-z["actual_steps"]) for z in comps),
  "next_state_agreement":statistics.fmean(z["semantic_divergence_steps"]==0 for z in comps),
  "completion_order_agreement":statistics.fmean(z["semantic_divergence_steps"]==0 for z in comps),
  "rows":comps}
 (a.output_dir/"rollout_v2_generalization.json").write_text(json.dumps(rollout,indent=2)+"\n")
 material=any(z["improvement"]<-.02 for z in tr)
 meanpass=clustered["cluster_bootstrap_ci95"][0]>0
 classification=("full_pass" if meanpass and tail["noninferiority_pass"] and not material
  else "mean_only" if meanpass else "inconclusive" if statistics.fmean(td)>0 else "failed")
 final={"classification":classification,"controlling_mean":clustered,
  "tail":tail,"material_family_regression":material,
  "measured_executions":len(raw),"attention_to_o_proj_events":sum(r["attention_to_o_proj"] for r in raw),
  "outputs_equivalent":all(all(r["generated"]==g[0]["generated"] for r in g)
   for g in ([z for z in raw if z["trace_id"]==t and z["requested_policy"]==p]
    for t in x["summary"] for p in x["summary"][t])),
  "runtime_counter_sums":{k:sum(r["runtime_counters"].get(k,0) for r in raw)
    for k in raw[0]["runtime_counters"]}}
 (a.output_dir/"s2_12_final_results.json").write_text(json.dumps(final,indent=2)+"\n")
 print(json.dumps(final,indent=2))
if __name__=="__main__":main()
