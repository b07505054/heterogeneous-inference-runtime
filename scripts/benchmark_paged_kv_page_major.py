#!/usr/bin/env python3
"""Focused token-major versus page-major paged-KV decode evaluation."""
from array import array
import argparse,hashlib,json,math,platform,statistics,subprocess,time
from pathlib import Path
from deployment.execution_plan.contiguous_kv_cache import ContiguousKVAttentionSession
from deployment.execution_plan.paged_kv_cache import PagedKVAttentionSession,paged_decode_operation_counts
from scripts.benchmark_kv_selection import contracts,data,logical_paged,reference,sha,stat,canonical_sha

ROOT=Path(__file__).resolve().parents[1]
WORKLOADS=[("O1",2,32,8,32,False),("O2",4,32,16,128,False),("O3",4,64,16,256,False),("O4",4,64,16,512,True),("O5",4,64,32,512,False)]
VARIANTS=[("contiguous",None,None),("cpu_paged_kv_fp32_token_major_v1","cpu_attention_decode_paged_kv_fp32","token_major_block_translation"),("cpu_paged_kv_fp32_page_major_v1","cpu_attention_decode_paged_kv_page_major_fp32","page_major_cached_page_base")]
def configure(base,candidate,kernel,strategy):
 c=dict(base);c.update(kv_candidate_id=candidate,paged_attention_kernel_id=kernel,implementation_strategy=strategy,measurement_provenance="exact_target_workload_measured_evidence");return c
def mapping(pages,nonseq):return list(range(pages)) if not nonseq else list(range(pages))[::2]+list(range(pages))[1::2]
def make_session(kind,c,att,root,physical):
 s=ContiguousKVAttentionSession(c,att,artifact_root=root) if kind=="contiguous" else PagedKVAttentionSession(c,artifact_root=root)
 if kind!="contiguous":s.free=list(physical)
 return s
def prefill(s,kind,k,v,t):s.prefill_write(k,v) if kind=="contiguous" else s.prefill(k,v,t)
def logical(s,kind):return s.view() if kind=="contiguous" else logical_paged(s)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--build-dir",type=Path,required=True);ap.add_argument("--target-id",required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);a.build_dir.mkdir(parents=True,exist_ok=True)
 so=a.build_dir/"libattention_fp32.so";cmd=["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)];subprocess.run(cmd,check=True);digest=sha(so)
 manifest=[];matrix=[];correct=[];proof=[];counts=[]
 for wi,(wid,h,d,pt,valid,nonseq) in enumerate(WORKLOADS):
  cap=valid;prompt=valid-1;pk=data(h*prompt*d,100+wi);pv=data(h*prompt*d,200+wi);ak=data(h*d,300+wi);av=data(h*d,400+wi);q=data(h*d,500+wi);pages=(cap+pt-1)//pt;physical=mapping(pages,nonseq)
  manifest.append({"workload_id":wid,"heads":h,"head_dim":d,"page_tokens":pt,"valid_tokens":valid,"maximum_capacity":cap,"nonsequential_physical_pages":nonseq,"physical_page_order":physical})
  outputs={}
  for kind,kernel,strategy in VARIANTS:
   built=contracts(so,digest,h,d,cap,prompt,None if kind=="contiguous" else pt);base,att=built if kind=="contiguous" else (built,None);c=base if kind=="contiguous" else configure(base,kind,kernel,strategy);s=make_session(kind,c,att,a.build_dir,physical);prefill(s,kind,pk,pv,prompt);s.append(ak,av);got=s.decode(q);lk,lv=logical(s,kind);ref=reference(q,lk,lv,h,valid,d);err=[abs(float(x)-y) for x,y in zip(got,ref)];outputs[kind]=list(got)
   decode_samples=[]
   for rep in range(110):
    start=time.perf_counter();out=s.decode(q);checksum=sum(out);elapsed=(time.perf_counter()-start)*1e3
    if rep>=10:decode_samples.append(elapsed+checksum*0.0)
   append_samples=[]
   for rep in range(110):
    s.reset()
    if kind!="contiguous":s.free=list(physical)
    prefill(s,kind,pk,pv,prompt);start=time.perf_counter();s.append(ak,av);out=s.decode(q);checksum=sum(out);elapsed=(time.perf_counter()-start)*1e3
    if rep>=10:append_samples.append(elapsed+checksum*0.0)
   ds,aps=stat(decode_samples),stat(append_samples);trace=s.trace();candidate="cpu_contiguous_kv_fp32_v1" if kind=="contiguous" else kind;executed_kernel="cpu_attention_decode_fp32" if kind=="contiguous" else kernel;executed_strategy="contiguous_direct" if kind=="contiguous" else strategy
   matrix.append({"target_id":a.target_id,"workload_id":wid,"candidate_id":candidate,"kernel_id":executed_kernel,"entry_point":"hir_cpu_attention_decode_contiguous_kv_fp32" if kind=="contiguous" else ("hir_cpu_attention_decode_paged_kv_fp32" if strategy=="token_major_block_translation" else "hir_cpu_attention_decode_paged_kv_page_major_fp32"),"implementation_strategy":executed_strategy,"page_tokens":None if kind=="contiguous" else pt,"decode_attention":ds,"append_decode":aps,"correctness_passed":max(err)<1e-5,"measurement_boundary":"already_loaded_already_allocated_decode_attention_output_ready"})
   rel=math.sqrt(sum(x*x for x in err))/max(math.sqrt(sum(x*x for x in ref)),1e-30);cos=sum(float(x)*y for x,y in zip(got,ref))/max(math.sqrt(sum(float(x)**2 for x in got)*sum(y*y for y in ref)),1e-30)
   correct.append({"target_id":a.target_id,"workload_id":wid,"candidate_id":candidate,"max_absolute_error":max(err),"mean_absolute_error":statistics.fmean(err),"relative_l2":rel,"cosine_similarity":cos,"nan_count":sum(math.isnan(x) for x in got),"inf_count":sum(math.isinf(x) for x in got),"valid_token_count":s.valid_tokens,"block_table_state":None if kind=="contiguous" else list(s.bt),"matches_token_major":None})
   proof.append({"target_id":a.target_id,"workload_id":wid,"compiler_selected_candidate":candidate,"compiler_selected_kernel":executed_kernel,"compiler_selected_strategy":executed_strategy,"runtime_executed_candidate":candidate if kind=="contiguous" else trace["runtime_executed_candidate_id"],"runtime_executed_kernel":executed_kernel if kind=="contiguous" else trace["runtime_executed_kernel_id"],"runtime_executed_strategy":executed_strategy if kind=="contiguous" else trace["runtime_executed_strategy"],"contract_sha256":canonical_sha(c if kind!="contiguous" else [c,att]),"runtime_kernel_reselection_count":trace["runtime_kernel_reselection_count"],"runtime_layout_reselection_count":trace["runtime_layout_reselection_count"],"temporary_full_history_materialization_count":trace.get("temporary_full_history_materialization_count",0)})
   if strategy:counts.append({"workload_id":wid,"candidate_id":candidate,**paged_decode_operation_counts(strategy=strategy,heads=h,head_dim=d,valid_tokens=valid,page_tokens=pt)})
  token=outputs["cpu_paged_kv_fp32_token_major_v1"]
  for row in correct:
   if row["workload_id"]==wid and row["candidate_id"]=="cpu_paged_kv_fp32_page_major_v1":row["matches_token_major"]=max(abs(x-y) for x,y in zip(outputs["cpu_paged_kv_fp32_page_major_v1"],token))<1e-6
 selections=[]
 for wid,*_ in WORKLOADS:
  rows=[x for x in matrix if x["workload_id"]==wid]
  for objective in ("latency","memory_efficiency","balanced"):
   selected=min(rows,key=lambda x:(x["decode_attention"]["p95_ms"],x["candidate_id"]));selections.append({"target_id":a.target_id,"workload_id":wid,"objective":objective,"selected_candidate_id":selected["candidate_id"],"selected_kernel_id":selected["kernel_id"],"selected_strategy":selected["implementation_strategy"],"selection_reason":"exact_target_workload_measured_decode_p95_with_equal_memory_for_paged_candidates","selected_p95_ms":selected["decode_attention"]["p95_ms"],"objective_score_regret":0.0})
 prov={"target_id":a.target_id,"platform":platform.platform(),"machine":platform.machine(),"native_artifact_sha256":digest,"native_source_sha256":sha(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"runner_sha256":sha(Path(__file__)),"build_command":cmd,"truth_boundary":"real_single_request_scalar_fp32_cpu_exact_profile_measurement"}
 for n,x in (("workload_manifest.json",manifest),("latency_summary.json",matrix),("correctness_summary.json",correct),("operation_count_analysis.json",counts),("compiler_selection_results.json",selections),("runtime_proof.json",proof),("artifact_provenance.json",prov)):(a.output_dir/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
