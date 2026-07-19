"""Measure compiler-planned shared-memory attention schedules."""
from __future__ import annotations
import argparse,json,math,statistics,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT))
from deployment.attention_planner import AttentionWorkload,force_test_attention_plan,select_attention_plan
from deployment.attention_runtime import CompilerAttentionRuntime

def p95(xs):return sorted(xs)[min(len(xs)-1,math.ceil(.95*len(xs))-1)]
def data(q,k,seed):
 g=torch.Generator().manual_seed(seed)
 return torch.randn(1,14,q,64,generator=g),torch.randn(1,2,k,64,generator=g),torch.randn(1,2,k,64,generator=g)
def candidate(algorithm,workers):
 return {"algorithm":algorithm,"implementation":("torch_dense_materialized_v1" if algorithm=="dense_materialized" else "native_avx2"),
  "strategy":"serial" if workers==1 else "split_head","workers":workers,
  "query_tile":0 if algorithm=="dense_materialized" else 1,"key_tile":0 if algorithm=="dense_materialized" else 32}
def measure(phase,q,k,c,runs,seed):
 w=AttentionWorkload(phase=phase,batch=1,query_len=q,context_len=k,query_heads=14,kv_heads=2,head_dim=64)
 plan,_=force_test_attention_plan(w,**c)
 Q,K,V=data(q,k,seed);mask=None
 if phase=="prefill":mask=torch.triu(torch.full((1,1,q,k),-torch.inf),diagonal=1)
 with CompilerAttentionRuntime(plan) as rt:
  for _ in range(2):rt.attention(Q,K,V,mask,.125)
  samples=[];traces=[]
  for _ in range(runs):
   t=time.perf_counter_ns();out=rt.attention(Q,K,V,mask,.125);samples.append((time.perf_counter_ns()-t)/1e6);traces.append(rt.traces[-1])
  ref=torch.nn.functional.scaled_dot_product_attention(Q,K.repeat_interleave(7,1),V.repeat_interleave(7,1),attn_mask=mask)
  error=float((out-ref).abs().max())
  last=traces[-1];events=last.worker_events
  starts=[e["start_ns"] for e in events];ends=[e["end_ns"] for e in events]
  placement=plan["distributed_execution"];pc=placement["predicted_cost"]
  return {"phase":phase,"q":q,"k":k,"candidate_id":plan["native_kernel_id"],**c,
   "median_ms":statistics.median(samples),"p95_ms":p95(samples),"variance_ms2":statistics.pvariance(samples),
   "kernel_compute_ms":statistics.median([t.timing.qk_ms for t in traces]),
   "dispatch_ms":statistics.median([t.timing.dispatch_ms for t in traces]),
   "barrier_ms":statistics.median([t.timing.barrier_ms for t in traces]),
   "assembly_ms":statistics.median([t.timing.assembly_ms for t in traces]),
   "predicted_latency":pc["total"],"shard_sizes":[x["query_head_range"][1]-x["query_head_range"][0] for x in placement["workers"]],
   "imbalance_heads":max(x["query_head_range"][1]-x["query_head_range"][0] for x in placement["workers"])-min(x["query_head_range"][1]-x["query_head_range"][0] for x in placement["workers"]),
   "temporary_bytes":last.memory["total_temporary_bytes_including_gqa"],"max_abs_error":error,
   "worker_start_skew_ms":(max(starts)-min(starts))/1e6,"worker_finish_skew_ms":(max(ends)-min(ends))/1e6,
   "timeline":events,"fallback":False,"runtime_repartition_count":rt.runtime_repartition_count,
   "runtime_worker_count_override":rt.runtime_worker_count_override,"runtime_strategy_override":rt.runtime_strategy_override}
def evaluate(rows,shapes):
 grouped={}
 for r in rows:
  if (r["phase"],r["q"],r["k"]) in shapes:grouped.setdefault((r["phase"],r["q"],r["k"]),[]).append(r)
 out=[];regrets=[]
 for shape,rs in grouped.items():
  winner=min(rs,key=lambda x:x["median_ms"]);phase,q,k=shape
  p,_=select_attention_plan(AttentionWorkload(phase=phase,batch=1,query_len=q,context_len=k,query_heads=14,kv_heads=2,head_dim=64))
  selected=next((x for x in rs if x["candidate_id"]==p["native_kernel_id"]),None)
  if selected is None: selected=min(rs,key=lambda x:abs(x["workers"]-p["worker_count"]) if x["algorithm"]==p["algorithm"] else 999)
  regret=(selected["median_ms"]/winner["median_ms"]-1)*100;regrets.append(regret)
  out.append({"phase":phase,"q":q,"k":k,"selected":p["native_kernel_id"],"measured_selected":selected["candidate_id"],
   "winner":winner["candidate_id"],"regret_percent":regret,"exact":p["native_kernel_id"]==winner["candidate_id"]})
 return {"rows":out,"workloads":len(out),"exact_match_rate":sum(x["exact"] for x in out)/len(out),
  "mean_regret_percent":statistics.mean(regrets),"median_regret_percent":statistics.median(regrets),
  "p95_regret_percent":p95(regrets),"max_regret_percent":max(regrets),"fallback_rate":0}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--runs",type=int,default=7);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(1)
 cal={("decode",1,k) for k in [16,64,256,1024]}|{("prefill",q,q) for q in [16,64,256]}
 held={("decode",1,k) for k in [24,48,96,192,384,768,1536]}|{("prefill",q,q) for q in [11,24,48,73,96,192]}
 display={("decode",1,k) for k in [32,128,512,2048]}|{("prefill",q,q) for q in [8,32,128]}
 shapes=sorted(cal|held|display)
 candidates=[candidate(alg,w) for alg in ["dense_materialized","fused_tiled_online_softmax"] for w in [1,2,4,8]]
 rows=[]
 for i,(phase,q,k) in enumerate(shapes):
  for c in candidates:rows.append(measure(phase,q,k,c,a.runs,5000+i))
 payload={"rows":rows,"warmups":2,"runs":a.runs,"torch_intraop_threads":1}
 (a.output_dir/"serial_vs_parallel_benchmark.json").write_text(json.dumps(payload,indent=2))
 calout=evaluate(rows,cal);heldout=evaluate(rows,held)
 (a.output_dir/"calibration_results.json").write_text(json.dumps(calout,indent=2))
 (a.output_dir/"held_out_selector_evaluation.json").write_text(json.dumps(heldout,indent=2))
 scaling=[]
 for shape in shapes:
  for alg in ["dense_materialized","fused_tiled_online_softmax"]:
   rs=[r for r in rows if (r["phase"],r["q"],r["k"])==shape and r["algorithm"]==alg];serial=next(r for r in rs if r["workers"]==1)
   scaling.extend([{**{k:r[k] for k in ["phase","q","k","algorithm","workers","candidate_id","median_ms","dispatch_ms","barrier_ms","imbalance_heads"]},
    "speedup":serial["median_ms"]/r["median_ms"],"parallel_efficiency":serial["median_ms"]/r["median_ms"]/r["workers"],
    "dispatch_fraction":r["dispatch_ms"]/r["median_ms"],"barrier_fraction":r["barrier_ms"]/r["median_ms"]} for r in rs])
 (a.output_dir/"worker_scaling_analysis.json").write_text(json.dumps({"rows":scaling},indent=2))
 representative=min(rows,key=lambda r:abs(r["q"]-64)+abs(r["workers"]-4) if r["phase"]=="prefill" else 999)
 (a.output_dir/"worker_timeline.json").write_text(json.dumps(representative,indent=2))
 print(json.dumps({"rows":len(rows),"calibration":calout,"held_out":heldout},indent=2))
if __name__=="__main__":main()
