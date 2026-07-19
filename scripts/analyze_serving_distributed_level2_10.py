#!/usr/bin/env python3
import argparse,json,statistics,math
from pathlib import Path
POLICIES=("decode_first","prefill_first","chunked_balanced","slo_aware")
SELECTORS=("ranking_selector_v3_static","ranking_selector_v4_pairwise",
           "ranking_selector_v4_risk_aware")
def p95(xs):return sorted(xs)[min(len(xs)-1,round(.95*(len(xs)-1)))]
def autocorr(xs):
 if len(xs)<2:return 0.0
 m=statistics.fmean(xs);den=sum((x-m)**2 for x in xs)
 return sum((a-m)*(b-m) for a,b in zip(xs,xs[1:]))/den if den else 0.0
def phase(s):
 p=s["selected_work"]["scheduled_prefill_tokens"];d=s["selected_work"]["scheduled_decode_sequences"]
 return "mixed" if p and d else ("prefill" if p else "decode")
def vec(s):
 w=s["selected_work"];sh=s["execution"]["shapes"];p=w["scheduled_prefill_tokens"];d=w["scheduled_decode_sequences"]
 return [1,p,d,w["scheduled_sequence_count"],s["execution"]["model_forward_count"],
  max([0]+[q["query_length"] for q in sh]),max([0]+[q["kv_length_before"] for q in sh]),
  s["transitions"]["prefill_to_decode_transition_count"],p*d]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
 x=json.loads(a.input.read_text());a.output_dir.mkdir(parents=True,exist_ok=True)
 names=("intercept","prefill_tokens","decode_sequences","sequence_count","model_forward_count","max_query_length","max_kv_length","phase_transitions","prefill_decode_interaction")
 models={k:json.load(open(a.output_dir.parent/"serving_distributed_level2_9"/f"service_model_v2_{k}.json"))["coefficients"] for k in ("prefill","decode","mixed")}
 propagation=[]
 for run in x["raw_runs"]:
  rs=[s["accounting"]["measured_step_ms"]-
      sum(models[phase(s)][n]*v for n,v in zip(names,vec(s))) for s in run["steps"]]
  propagation.append({"trace":run["trace_id"],"policy":run["requested_policy"],
   "run_id":run["run_id"],"steps":len(rs),"signed_sum_ms":sum(rs),
   "absolute_sum_ms":sum(abs(v) for v in rs),"mean_ms":statistics.fmean(rs),
   "variance":statistics.pvariance(rs),"lag1_autocorrelation":autocorr(rs),
   "first_quartile_mean_ms":statistics.fmean(rs[:max(1,len(rs)//4)]),
   "last_quartile_mean_ms":statistics.fmean(rs[-max(1,len(rs)//4):])})
 (a.output_dir/"policy_error_propagation.json").write_text(json.dumps(propagation,indent=2)+"\n")
 summary=x["summary"];rows=[];baseline=[]
 for trace,d in summary.items():
  values={p:d[p]["objective"]["mean"] for p in d};fixed={p:values[p] for p in POLICIES}
  oracle=min(values,key=values.get);fixed_oracle=min(fixed,key=fixed.get)
  for selector in SELECTORS:
   regret=(values[selector]-values[oracle])/values[oracle]
   rows.append({"trace":trace,"selector":selector,"oracle":oracle,
    "fixed_oracle":fixed_oracle,"objective":values[selector],
    "regret":regret,"exact":selector==oracle,
    "tie_aware":values[selector]<=values[oracle]*1.02})
  baseline.append({"trace":trace,"policy":"decode_first",
   "regret":(values["decode_first"]-values[oracle])/values[oracle]})
 metrics={}
 for s in SELECTORS:
  z=[r for r in rows if r["selector"]==s];reg=[r["regret"] for r in z]
  metrics[s]={"exact_winner_rate":statistics.fmean(r["exact"] for r in z),
   "tie_aware_rate":statistics.fmean(r["tie_aware"] for r in z),
   "mean_regret":statistics.fmean(reg),"median_regret":statistics.median(reg),
   "p95_regret":p95(reg),"maximum_regret":max(reg)}
 br=[r["regret"] for r in baseline]
 base={"policy":"decode_first","mean_regret":statistics.fmean(br),
       "p95_regret":p95(br),"maximum_regret":max(br)}
 result={"rows":rows,"selector_metrics":metrics,"robust_baseline":base,
  "beats_baseline":{s:(metrics[s]["mean_regret"]<base["mean_regret"] and
    metrics[s]["p95_regret"]<=base["p95_regret"]) for s in SELECTORS}}
 (a.output_dir/"preregistered_final_results.json").write_text(json.dumps(result,indent=2)+"\n")
 (a.output_dir/"wall_clock_regret.json").write_text(json.dumps(
  {"selectors":metrics,"baseline":base},indent=2)+"\n")
 (a.output_dir/"baseline_comparison.json").write_text(json.dumps(
  {"baseline":base,"beats_baseline":result["beats_baseline"]},indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
