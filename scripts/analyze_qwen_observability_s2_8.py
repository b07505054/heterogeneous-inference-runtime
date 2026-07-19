#!/usr/bin/env python3
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
from deployment.scheduler_calibration import CalibratedQwenCPUServiceModel

def pct(xs,p):return sorted(xs)[min(len(xs)-1,round((len(xs)-1)*p))]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",type=Path,required=True)
 ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
 p=json.loads(a.input.read_text());a.output_dir.mkdir(parents=True,exist_ok=True)
 steps=[];groups=defaultdict(list);account=[]
 model=CalibratedQwenCPUServiceModel()
 for run in p["raw_runs"]:
  previous=None
  for row in run["steps"]:
   w=row["selected_work"];pf=w["scheduled_prefill_tokens"];ds=w["scheduled_decode_sequences"]
   if pf and ds:pred=model.mixed_intercept_ms+pf*model.mixed_prefill_token_ms+ds*model.mixed_decode_sequence_ms;kind="mixed"
   elif pf:pred=model.prefill_intercept_ms+pf*model.prefill_token_ms;kind="prefill"
   else:pred=model.decode_intercept_ms+ds*model.decode_sequence_ms;kind="decode"
   actual=row["accounting"]["measured_step_ms"];res=actual-pred
   shapes=row["execution"]["shapes"];kv=max([0]+[x["kv_length_before"] for x in shapes])
   x={"trace_id":run["trace_id"],"run_id":run["run_id"],"requested_policy":run["requested_policy"],
      "executed_policy":row["policy_id"],"step_id":row["step_id"],"kind":kind,
      "prefill_tokens":pf,"decode_sequences":ds,"sequence_count":w["scheduled_sequence_count"],
      "model_forward_count":row["execution"]["model_forward_count"],
      "attention_invocation_count":row["execution"]["attention_invocation_count"],
      "attention_to_o_proj_count":row["execution"]["attention_to_o_proj_count"],
      "max_kv_length":kv,"previous_composition":previous,
      "measured_step_latency_ms":actual,"predicted_step_latency_ms":pred,
      "residual_ms":res,"absolute_error_ms":abs(res),
      "attention_path_ms":row["execution"]["attention_path_ms"],
      "non_attention_and_runtime_ms":max(0,actual-row["execution"]["attention_path_ms"]),
      "transitions":row["transitions"],"state":row["state"],
      "accounting":row["accounting"]}
   steps.append(x);previous=kind;groups[(kind,run["requested_policy"])].append(x)
   account.append(row["accounting"])
 residual=[]
 for (kind,policy),xs in groups.items():
  rs=[x["residual_ms"] for x in xs];ae=[abs(x) for x in rs]
  residual.append({"kind":kind,"policy":policy,"n":len(xs),
   "mean_residual_ms":statistics.fmean(rs),"median_residual_ms":statistics.median(rs),
   "mae_ms":statistics.fmean(ae),"p95_absolute_error_ms":pct(ae,.95)})
 shape={}
 for policy in sorted({r["requested_policy"] for r in p["raw_runs"]}):
  xs=[x for x in steps if x["requested_policy"]==policy]
  shape[policy]={"steps":len(xs),"mixed_fraction":sum(x["kind"]=="mixed" for x in xs)/len(xs),
   "mean_sequences":statistics.fmean(x["sequence_count"] for x in xs),
   "mean_model_forwards":statistics.fmean(x["model_forward_count"] for x in xs),
   "attention_invocations":sum(x["attention_invocation_count"] for x in xs),
   "mean_kv_length":statistics.fmean(x["max_kv_length"] for x in xs),
   "mean_unused_tokens":statistics.fmean(x["state"]["token_budget"]-x["prefill_tokens"]-x["decode_sequences"] for x in xs)}
 final=[];summary=p["summary"]
 for trace,d in summary.items():
  means={k:v["objective"]["mean"] for k,v in d.items()};best=min(means,key=means.get)
  for selector in ("ranking_selector_v3_static","ranking_selector_v3_adaptive"):
   final.append({"trace":trace,"oracle_available_policy":best,"selector":selector,
    "objective":means[selector],"oracle_objective":means[best],
    "regret":(means[selector]-means[best])/means[best]})
 result={"rows":final,"mean_regret":{s:statistics.fmean(x["regret"] for x in final if x["selector"]==s)
   for s in ("ranking_selector_v3_static","ranking_selector_v3_adaptive")},
   "exact_winner_rate":{s:statistics.fmean(x["oracle_available_policy"]==s for x in final if x["selector"]==s)
   for s in ("ranking_selector_v3_static","ranking_selector_v3_adaptive")}}
 (a.output_dir/"step_latency_dataset.json").write_text(json.dumps(steps,indent=2)+"\n")
 (a.output_dir/"step_latency_residuals.json").write_text(json.dumps(residual,indent=2)+"\n")
 (a.output_dir/"policy_execution_shape_analysis.json").write_text(json.dumps(shape,indent=2)+"\n")
 (a.output_dir/"attention_non_attention_cost_analysis.json").write_text(json.dumps({
  "method":"synchronous PyTorch module hooks around Qwen self_attn; non-attention is step remainder and includes runtime",
  "overlap_warning":"self_attn timer includes o_proj; no precise per-op attribution claimed",
  "attention_ms":sum(x["attention_path_ms"] for x in steps),
  "step_ms":sum(x["measured_step_latency_ms"] for x in steps)},indent=2)+"\n")
 (a.output_dir/"timing_component_accounting.json").write_text(json.dumps({
  "steps":len(account),"max_unaccounted_fraction":max(x["unaccounted_fraction"] for x in account),
  "mean_unaccounted_fraction":statistics.fmean(x["unaccounted_fraction"] for x in account),
  "failed_steps":sum(x["unaccounted_ms"]>x["tolerance_ms"] for x in account)},indent=2)+"\n")
 (a.output_dir/"wall_clock_final_results.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
