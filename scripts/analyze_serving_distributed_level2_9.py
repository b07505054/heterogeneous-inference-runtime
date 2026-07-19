#!/usr/bin/env python3
import argparse,json,statistics
from pathlib import Path
import numpy as np
from deployment.request_timeline_reconstruction import reconstruct_run,error_summary
from deployment.scheduler_calibration import CalibratedQwenCPUServiceModel

FEATURES=("intercept","prefill_tokens","decode_sequences","sequence_count",
          "model_forward_count","max_query_length","max_kv_length",
          "phase_transitions","prefill_decode_interaction")
def vector(step):
 w=step["selected_work"];sh=step["execution"]["shapes"]
 p=w["scheduled_prefill_tokens"];d=w["scheduled_decode_sequences"]
 return [1,p,d,w["scheduled_sequence_count"],step["execution"]["model_forward_count"],
  max([0]+[x["query_length"] for x in sh]),max([0]+[x["kv_length_before"] for x in sh]),
  step["transitions"]["prefill_to_decode_transition_count"],p*d]
def phase(step):
 p=step["selected_work"]["scheduled_prefill_tokens"];d=step["selected_work"]["scheduled_decode_sequences"]
 return "mixed" if p and d else ("prefill" if p else "decode")
def fit(rows):
 result={}
 for kind in ("prefill","decode","mixed"):
  xs=[];ys=[]
  for run in rows:
   for s in run["steps"]:
    if phase(s)==kind:xs.append(vector(s));ys.append(s["accounting"]["measured_step_ms"])
  coef=np.linalg.lstsq(np.asarray(xs,float),np.asarray(ys,float),rcond=None)[0]
  result[kind]={"version":f"service_model_v2_{kind}","features":FEATURES,
    "coefficients":dict(zip(FEATURES,map(float,coef))),"training_rows":len(xs),
    "fit_objective":"least_squares_step_latency_ms","regularization":0.0}
 return result
def predict(step,models):
 m=models[phase(step)]["coefficients"];return sum(m[f]*v for f,v in zip(FEATURES,vector(step)))
def old_predict(step):
 m=CalibratedQwenCPUServiceModel();w=step["selected_work"]
 p=w["scheduled_prefill_tokens"];d=w["scheduled_decode_sequences"]
 if p and d:return m.mixed_intercept_ms+p*m.mixed_prefill_token_ms+d*m.mixed_decode_sequence_ms
 if p:return m.prefill_intercept_ms+p*m.prefill_token_ms
 return m.decode_intercept_ms+d*m.decode_sequence_ms
def modeled_objective(run,predictor):
 origin=run["host_wall_start_ns"];clock=0.0;first={};done={}
 for step in run["steps"]:
  actual=step["accounting"]["measured_step_ms"];pred=max(.001,predictor(step))
  for item in step["execution"]["scheduled_items"]:
   if item["phase"]!="decode":continue
   fraction=(item["commit_ns"]-step["step_start_ns"])/max(step["step_end_ns"]-step["step_start_ns"],1)
   t=clock+fraction*pred;first.setdefault(item["request_id"],t);done[item["request_id"]]=t
  clock+=pred
 return statistics.fmean(first.values())+.25*statistics.fmean(done.values())
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--training",type=Path,required=True)
 ap.add_argument("--final",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True)
 a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 train=json.loads(a.training.read_text());final=json.loads(a.final.read_text())
 models=fit(train["raw_runs"])
 for kind,m in models.items():(a.output_dir/f"service_model_v2_{kind}.json").write_text(json.dumps(m,indent=2)+"\n")
 reconstructed=[reconstruct_run(r) for r in final["raw_runs"]]
 errors=error_summary(reconstructed,1.0)
 (a.output_dir/"measured_step_request_reconstruction.json").write_text(json.dumps(reconstructed,indent=2)+"\n")
 (a.output_dir/"reconstruction_error_analysis.json").write_text(json.dumps(errors,indent=2)+"\n")
 residual={};all_rows=[]
 for kind in ("prefill","decode","mixed"):
  old=[];new=[]
  for run in final["raw_runs"]:
   for s in run["steps"]:
    if phase(s)!=kind:continue
    actual=s["accounting"]["measured_step_ms"];old.append(abs(actual-old_predict(s)));new.append(abs(actual-predict(s,models)))
  residual[kind]={"rows":len(old),"old_mae_ms":statistics.fmean(old),
    "v2_mae_ms":statistics.fmean(new),"old_p95_ms":sorted(old)[round(.95*(len(old)-1))],
    "v2_p95_ms":sorted(new)[round(.95*(len(new)-1))]}
 (a.output_dir/"service_model_validation.json").write_text(json.dumps(residual,indent=2)+"\n")
 pipes={}
 for trace in sorted({r["trace_id"] for r in final["raw_runs"]}):
  pipes[trace]={}
  for policy in ("decode_first","prefill_first","chunked_balanced","slo_aware"):
   rs=[r for r in final["raw_runs"] if r["trace_id"]==trace and r["requested_policy"]==policy]
   pipes[trace][policy]={"A_original_model":statistics.fmean(modeled_objective(r,old_predict) for r in rs),
    "B_shape_model":statistics.fmean(modeled_objective(r,lambda s:predict(s,models)) for r in rs),
    "C_measured_reconstruction":statistics.fmean(
      statistics.fmean(q["first_token_ms"] for q in reconstruct_run(r)["requests"])+
      .25*statistics.fmean(q["completion_ms"] for q in reconstruct_run(r)["requests"]) for r in rs),
    "D_direct":statistics.fmean(r["objective"] for r in rs)}
 for trace,x in pipes.items():
  for pipe in ("A_original_model","B_shape_model","C_measured_reconstruction","D_direct"):
   x[f"{pipe}_ranking"]=sorted(
    ("decode_first","prefill_first","chunked_balanced","slo_aware"),
    key=lambda p:x[p][pipe])
 (a.output_dir/"oracle_substitution_experiment.json").write_text(json.dumps(pipes,indent=2)+"\n")
 summary=final["summary"];rows=[]
 for trace,d in summary.items():
  values={p:v["objective"]["mean"] for p,v in d.items()}
  oracle=min(values,key=values.get)
  for selector in ("ranking_selector_v3_static","ranking_selector_v3_adaptive"):
   rows.append({"trace":trace,"oracle":oracle,"selector":selector,
    "regret":(values[selector]-values[oracle])/values[oracle],
    "objective":values[selector],"oracle_objective":values[oracle]})
 result={"rows":rows,"mean_regret":{s:statistics.fmean(x["regret"] for x in rows if x["selector"]==s)
  for s in ("ranking_selector_v3_static","ranking_selector_v3_adaptive")},
  "exact_winner_rate":{s:statistics.fmean(x["oracle"]==s for x in rows if x["selector"]==s)
  for s in ("ranking_selector_v3_static","ranking_selector_v3_adaptive")}}
 (a.output_dir/"wall_clock_final_results.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"reconstruction":errors,"service":residual,"final":result},indent=2))
if __name__=="__main__":main()
