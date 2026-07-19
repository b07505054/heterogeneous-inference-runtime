#!/usr/bin/env python3
"""State-local diagnostic accuracy without invisible future arrivals."""
import argparse,hashlib,itertools,json,statistics
from pathlib import Path
P=("decode_first","prefill_first","chunked_balanced","slo_aware")
def scores(s):
 active=max(s["active_request_count"],1);budget=max(s["token_budget"],1)
 f={"ppr":s["total_remaining_prefill_tokens"]/active,
  "pp":s["total_remaining_prefill_tokens"]/budget,
  "dp":s["decode_ready_count"]/max(s["sequence_budget"],1),
  "mixed":float(bool(s["prefill_ready_count"] and s["decode_ready_count"])),
  "wait":s["oldest_waiting_age_ms"]/50,"gap":s["largest_decode_gap_ms"]/10}
 return {"decode_first":0.0,"prefill_first":.35*f["pp"]-.15*f["dp"],
  "chunked_balanced":.08-.16*f["mixed"]-.04*min(f["ppr"]/32,2),
  "slo_aware":.06-.18*max(f["wait"],f["gap"])-.08*f["mixed"]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
 x=json.loads(a.input.read_text());groups={}
 for r in x["raw_runs"]:
  if r["requested_policy"] not in P:continue
  for i,step in enumerate(r["steps"]):
   state=step["state"];h=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
   future=sum(z["accounting"]["measured_step_ms"] for z in r["steps"][i:i+4])
   groups.setdefault((r["trace_id"],h,i,r["requested_policy"]),[]).append(future)
 bystate={}
 for (t,h,i,p),v in groups.items():
  bystate.setdefault((t,h,i),{})[p]=statistics.fmean(v)
 rows=[];correct=ties=total=0
 for (t,h,i),outcomes in bystate.items():
  if len(outcomes)<2:continue
  exemplar=next(r["steps"][i]["state"] for r in x["raw_runs"] if
   r["trace_id"]==t and r["requested_policy"] in outcomes and len(r["steps"])>i and
   hashlib.sha256(json.dumps(r["steps"][i]["state"],sort_keys=True).encode()).hexdigest()==h)
  sc=scores(exemplar);pairs=[]
  for left,right in itertools.combinations(sorted(outcomes),2):
   measured=outcomes[left]-outcomes[right];pred=sc[left]-sc[right]
   scale=min(outcomes[left],outcomes[right]);tie=abs(measured)/max(scale,1e-9)<=.02
   ok=tie or (pred<0)==(measured<0);correct+=ok;ties+=tie;total+=1
   pairs.append({"left":left,"right":right,"predicted_delta":pred,
    "realized_four_step_ms_delta":measured,"practical_tie":tie,"correct":ok})
  rows.append({"trace":t,"step_index":i,"state_hash":h,"candidate_outcomes":outcomes,
   "scores":sc,"pairs":pairs})
 result={"realized_horizon":"next 4 scheduler steps; no invisible arrivals injected",
  "state_count":len(rows),"pair_count":total,
  "dynamic_exact_or_tie_aware_pairwise_accuracy":correct/total if total else None,
  "practical_tie_fraction":ties/total if total else None,"rows":rows,
  "limitation":"Only state hashes shared by at least two fixed-policy executions are comparable; this is diagnostic, not trace-level profitability."}
 (a.output_dir/"dynamic_decision_dataset.json").write_text(json.dumps(rows,indent=2)+"\n")
 (a.output_dir/"dynamic_pairwise_accuracy.json").write_text(json.dumps(
  {k:v for k,v in result.items() if k!="rows"},indent=2)+"\n")
 # Frozen v4 confidence was thresholded, not calibrated as a probability.
 (a.output_dir/"confidence_calibration_audit.json").write_text(json.dumps({
  "status":"diagnostic_only","frozen_uncertainty_threshold":.35,
  "probability_calibration_available":False,
  "reason":"v4 uncertainty is a conservative support/OOD gate, not a calibrated probability; replication data was not used to refit it.",
  "dynamic_pairwise_accuracy":result["dynamic_exact_or_tie_aware_pairwise_accuracy"]},indent=2)+"\n")
 print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))
if __name__=="__main__":main()
