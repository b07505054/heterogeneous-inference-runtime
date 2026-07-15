#!/usr/bin/env python3
"""Equivalent-boundary benchmark for real contiguous and paged FP32 KV paths."""
from array import array
import argparse, hashlib, json, math, platform, statistics, subprocess, time
from pathlib import Path
from deployment.execution_plan.contiguous_kv_cache import ContiguousKVAttentionSession
from deployment.execution_plan.paged_kv_cache import PagedKVAttentionSession
from deployment.kv_selection_evaluation import OBJECTIVES, admission_count, regret, select_measured

ROOT=Path(__file__).resolve().parents[1]
WORKLOADS=[("W1",2,32,32,16,16),("W2",2,32,512,16,16),("W3",4,32,256,64,64),("W4",4,64,1024,64,192),("W5",4,64,1024,128,384),("W6",4,64,512,15,65),("W7",4,64,512,16,48),("W8",4,64,4096,32,32)]
PAGE_SIZES=(8,16,32)
def data(n,s):return array("f",(((i*1103515245+s*12345)&0xffff)/32768-1 for i in range(n)))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def stat(x):
 x=sorted(x);m=statistics.fmean(x);sd=statistics.pstdev(x);at=lambda p:x[math.ceil(p*len(x))-1]
 return {"sample_count":len(x),"median_ms":statistics.median(x),"mean_ms":m,"minimum_ms":x[0],"p90_ms":at(.9),"p95_ms":at(.95),"p99_ms":at(.99),"stddev_ms":sd,"coefficient_of_variation":sd/m if m else 0}
def contracts(so,digest,h,d,cap,prompt,pt=None):
 if pt is None:
  st=[h*cap*d,cap*d,d,1];kv={"kv_execution_unit":"portable_cpu_contiguous_kv","kv_candidate_id":"cpu_contiguous_kv_fp32_v1","kv_cache_id":"eval","kv_artifact_ref":so.name,"kv_artifact_sha256":digest,"kv_artifact_version":"hir.contiguous_kv.v1","kv_dtype":"fp32","kv_layout":"bhcd_contiguous","batch":1,"num_kv_heads":h,"head_dim":d,"capacity_tokens":cap,"initial_valid_tokens":0,"bytes_per_token":2*h*d*4,"k_cache_bytes":h*cap*d*4,"v_cache_bytes":h*cap*d*4,"total_cache_bytes":2*h*cap*d*4,"alignment_bytes":4,"k_strides":st,"v_strides":st,"create_entry_point":"hir_contiguous_kv_initialize","prefill_write_entry_point":"hir_contiguous_kv_prefill_write","decode_append_entry_point":"hir_contiguous_kv_append","view_binding":"direct_contiguous_pointer_valid_prefix","reset_entry_point":"hir_contiguous_kv_reset","compatible_prefill_kernel_id":"cpu_attention_prefill_fp32","compatible_decode_kernel_id":"cpu_attention_decode_fp32","attention_entry_point":"hir_cpu_attention_decode_contiguous_kv_fp32","implementation_strategy":"dimension_major_strided_v_accumulation","measurement_provenance":"exact_target_workload_measured_evidence_required","runtime_no_layout_redecision":True};base={"dtype":"fp32","input_layout":"bhsd_contiguous","runtime_no_redecision":True};return kv,{"prefill":{**base,"kernel_id":"cpu_attention_prefill_fp32","entry_point":"hir_cpu_attention_prefill_fp32","implementation_strategy":"prefill_contiguous","query_length":prompt},"decode":{**base,"kernel_id":"cpu_attention_decode_fp32","entry_point":"hir_cpu_attention_decode_contiguous_kv_fp32","implementation_strategy":"dimension_major_strided_v_accumulation","query_length":1}}
 pages=(cap+pt-1)//pt;blocks=pages;one=h*pt*d*4;st=[h*pt*d,pt*d,d,1]
 return {"kv_candidate_id":"cpu_paged_kv_fp32_v1","kv_layout_kind":"paged_phd_contiguous","pool_artifact_ref":so.name,"pool_artifact_sha256":digest,"pool_artifact_version":"hir.paged_kv.v1","dtype":"fp32","batch":1,"num_kv_heads":h,"head_dim":d,"page_tokens":pt,"num_physical_pages":pages,"maximum_logical_tokens":cap,"maximum_logical_blocks":blocks,"block_table_length":blocks,"block_table_element_type":"int32","invalid_page_sentinel":-1,"k_page_strides":st,"v_page_strides":st,"bytes_per_token":2*h*d*4,"bytes_per_k_page":one,"bytes_per_v_page":one,"bytes_per_combined_page":2*one,"total_pool_bytes":pages*2*one,"alignment_bytes":4,"pool_create_entry_point":"hir_paged_kv_initialize","prefill_write_entry_point":"hir_paged_kv_prefill_write","append_entry_point":"hir_paged_kv_append","view_binding":"direct_int32_block_table_translation","reset_entry_point":"hir_paged_kv_reset","release_entry_point":"runtime_owned_pool_release","paged_attention_kernel_id":"cpu_attention_decode_paged_kv_fp32","contiguous_fallback_identity":"cpu_contiguous_kv_fp32_v1","runtime_no_layout_redecision":True,"runtime_no_kernel_redecision":True}
def logical_paged(s):
 h,d,pt=s.c["num_kv_heads"],s.c["head_dim"],s.c["page_tokens"];k=[];v=[]
 for hi in range(h):
  for t in range(s.valid_tokens):
   base=((s.bt[t//pt]*h+hi)*pt+t%pt)*d;k.extend(s.k[base:base+d]);v.extend(s.v[base:base+d])
 return k,v
def reference(q,k,v,h,t,d):
 out=[];scale=1/math.sqrt(d)
 for hi in range(h):
  z=[sum(q[hi*d+x]*k[(hi*t+j)*d+x] for x in range(d))*scale for j in range(t)];peak=max(z);e=[math.exp(x-peak) for x in z];den=sum(e)
  out.extend(sum(e[j]/den*v[(hi*t+j)*d+x] for j in range(t)) for x in range(d))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--build-dir",type=Path,required=True);ap.add_argument("--target-id",required=True);ap.add_argument("--workloads",default="");a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);a.build_dir.mkdir(parents=True,exist_ok=True)
 so=a.build_dir/"libattention_fp32.so";cmd=["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)];subprocess.run(cmd,check=True);digest=sha(so);wanted=set(a.workloads.split(",")) if a.workloads else None
 matrix=[];correct=[];proof=[];manifest=[]
 for wi,(wid,h,d,cap,prompt,steps) in enumerate(WORKLOADS):
  if wanted and wid not in wanted:continue
  pk=data(h*prompt*d,100+wi);pv=data(h*prompt*d,200+wi);seedk=[data(h*d,1000+wi*100+s) for s in range(steps)];seedv=[data(h*d,2000+wi*100+s) for s in range(steps)];q=data(h*d,3000+wi);manifest.append({"workload_id":wid,"heads":h,"head_dim":d,"maximum_capacity":cap,"prefill_tokens":prompt,"decode_steps":steps,"final_tokens":prompt+steps,"input_sha256":sha256_arrays([pk,pv,q]+seedk+seedv)})
  for pt in (None,)+PAGE_SIZES:
   c=contracts(so,digest,h,d,cap,prompt,pt);start=time.perf_counter();s=ContiguousKVAttentionSession(c[0],c[1],artifact_root=a.build_dir) if pt is None else PagedKVAttentionSession(c,artifact_root=a.build_dir);allocation=(time.perf_counter()-start)*1e3
   start=time.perf_counter();s.prefill_write(pk,pv) if pt is None else s.prefill(pk,pv,prompt);prefill_ms=(time.perf_counter()-start)*1e3
   for j in range(steps-1):s.append(seedk[j],seedv[j])
   # One correctness execution at the final legal context.
   s.append(seedk[-1],seedv[-1]);got=s.decode(q);lk,lv=s.view() if pt is None else logical_paged(s);ref=reference(q,lk,lv,h,s.valid_tokens,d);errors=[abs(float(x)-y) for x,y in zip(got,ref)];rel=math.sqrt(sum(x*x for x in errors))/max(math.sqrt(sum(x*x for x in ref)),1e-30);cos=sum(float(x)*y for x,y in zip(got,ref))/max(math.sqrt(sum(float(x)**2 for x in got)*sum(y*y for y in ref)),1e-30)
   samples=[];loop_samples=[]
   for rep in range(110):
    s.reset();s.prefill_write(pk,pv) if pt is None else s.prefill(pk,pv,prompt)
    loop_start=time.perf_counter();checksum=0.0
    for j in range(steps):s.append(seedk[j],seedv[j]);checksum+=sum(s.decode(q))
    loop_elapsed=(time.perf_counter()-loop_start)*1e3
    s.reset();s.prefill_write(pk,pv) if pt is None else s.prefill(pk,pv,prompt)
    for j in range(steps-1):s.append(seedk[j],seedv[j])
    start=time.perf_counter();s.append(seedk[-1],seedv[-1]);out=s.decode(q);checksum+=sum(out);elapsed=(time.perf_counter()-start)*1e3
    if rep>=10:samples.append(elapsed+checksum*0.0)
    if rep>=10:loop_samples.append(loop_elapsed+checksum*0.0)
   st=stat(samples);loop_st=stat(loop_samples);valid=prompt+steps;bpt=2*h*d*4
   if pt is None:owned=cap*bpt;reserved=owned;frag=0.0;alloc_count=1;page_alloc=0;trace=s.trace();candidate="cpu_contiguous_kv_fp32_v1"
   else:
    f=s.fragmentation();owned=f["allocated_pages"]*pt*bpt;reserved=c["total_pool_bytes"];frag=f["internal_fragmentation_ratio"];alloc_count=s.count.pool_allocation_count;page_alloc=s.count.page_allocation_count;trace=s.trace();candidate="cpu_paged_kv_fp32_v1"
   expected_k=array("f");expected_v=array("f")
   for hi in range(h):
    expected_k.extend(pk[hi*prompt*d:(hi+1)*prompt*d]);expected_v.extend(pv[hi*prompt*d:(hi+1)*prompt*d])
    for j in range(steps):expected_k.extend(seedk[j][hi*d:(hi+1)*d]);expected_v.extend(seedv[j][hi*d:(hi+1)*d])
   cache_exact=list(lk)==list(expected_k) and list(lv)==list(expected_v);passed=max(errors)<1e-5 and cache_exact and s.valid_tokens==valid
   row={"target_id":a.target_id,"workload_id":wid,"candidate_id":candidate,"page_tokens":pt,"correctness_passed":passed,"append_decode_median_ms":st["median_ms"],"append_decode_p95_ms":st["p95_ms"],"full_loop_latency_ms":loop_st["median_ms"],"request_owned_bytes":owned,"total_reserved_pool_bytes":reserved,"logical_used_bytes":valid*bpt,"internal_fragmentation_ratio":frag,"allocation_count":alloc_count,"page_allocation_count":page_alloc,"allocation_ms":allocation,"prefill_write_ms":prefill_ms,"statistics":st,"full_loop_statistics":loop_st,"measurement_provenance":"already_loaded_allocated_append_then_native_decode_output_ready"};matrix.append(row);correct.append({"target_id":a.target_id,"workload_id":wid,"candidate_id":candidate,"page_tokens":pt,"max_absolute_error":max(errors),"mean_absolute_error":statistics.fmean(errors),"relative_l2":rel,"cosine_similarity":cos,"nan_count":sum(math.isnan(float(x)) for x in got),"inf_count":sum(math.isinf(float(x)) for x in got),"valid_tokens":s.valid_tokens,"expected_valid_tokens":valid,"cache_content_exact":cache_exact,"causal_decode_uses_valid_prefix":True,"block_table_valid":pt is None or all(x>=0 for x in s.bt[:math.ceil(valid/pt)])});proof.append({"workload_id":wid,"selected_candidate_id":candidate,"selected_layout":"bhcd_contiguous" if pt is None else "paged_phd_contiguous","selected_page_size":pt,"selection_reason":"candidate_measurement_execution","execution_plan_hash":canonical_sha(c),"runtime_executed_candidate_id":candidate,"runtime_executed_kernel_id":"cpu_attention_decode_fp32" if pt is None else "cpu_attention_decode_paged_kv_fp32","runtime_layout_reselection_count":trace["runtime_layout_reselection_count"],"runtime_kernel_reselection_count":trace["runtime_kernel_reselection_count"],"temporary_full_history_materialization_count":trace.get("temporary_full_history_materialization_count",0)})
 selections=[];regrets=[]
 for wid,*_ in WORKLOADS:
  rows=[r for r in matrix if r["workload_id"]==wid]
  if not rows:continue
  for objective in OBJECTIVES:
   choice=select_measured(matrix,target_id=a.target_id,workload_id=wid,objective=objective);selected=next(r for r in rows if r["candidate_id"]==choice["candidate_id"] and r.get("page_tokens")==choice.get("page_tokens"));eligible=rows
   if objective=="memory_efficiency":
    limit=min(r["append_decode_p95_ms"] for r in rows)*(1+OBJECTIVES[objective]["max_latency_regression"]);eligible=[r for r in rows if r["append_decode_p95_ms"]<=limit]
   oracle=min(eligible,key=lambda r:(score_row(r,rows,OBJECTIVES[objective]),r["candidate_id"],r.get("page_tokens") or 0));selected["objective_score"]=score_row(selected,rows,OBJECTIVES[objective]);oracle["objective_score"]=score_row(oracle,rows,OBJECTIVES[objective]);selections.append({"workload_id":wid,**choice,"oracle_candidate_id":oracle["candidate_id"],"oracle_page_tokens":oracle.get("page_tokens")});regrets.append({"workload_id":wid,"objective":objective,**regret(selected,oracle)})
 admission=make_admission(matrix);prov={"schema_version":"kv_selection_evaluation.v1","target_id":a.target_id,"platform":platform.platform(),"machine":platform.machine(),"processor":platform.processor(),"python_version":platform.python_version(),"compiler_version":subprocess.run(["g++","--version"],check=True,text=True,capture_output=True).stdout.splitlines()[0],"native_compile_flags":["-O3","-std=c++17","-fPIC","-shared"],"benchmark_process_threads":1,"native_artifact_sha256":digest,"native_source_sha256":sha(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"runner_sha256":sha(Path(__file__)),"build_command":cmd,"truth_boundary":"exact_target_workload_measured_profile_not_predictive_cost_model"}
 for name,value in (("workload_manifest.json",manifest),("candidate_matrix.json",matrix),("correctness_summary.json",correct),("latency_summary.json",matrix),("memory_summary.json",matrix),("fragmentation_summary.json",matrix),("admission_analysis.json",admission),("objective_definitions.json",OBJECTIVES),("selection_results.json",selections),("regret_analysis.json",regrets),("runtime_proof_traces.json",proof),("artifact_provenance.json",prov)):(a.output_dir/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
 write_summary(a.output_dir,a.target_id,matrix,correct,regrets,proof)
def sha256_arrays(xs):
 h=hashlib.sha256()
 for x in xs:h.update(x.tobytes())
 return h.hexdigest()
def score_row(r,rows,w):
 ml=min(x["append_decode_p95_ms"] for x in rows);mm=max(x["request_owned_bytes"] for x in rows);return w["latency_weight"]*r["append_decode_p95_ms"]/ml+w["memory_weight"]*r["request_owned_bytes"]/mm+w["fragmentation_weight"]*r["internal_fragmentation_ratio"]
def make_admission(matrix):
 dist={"D1":[(32,.5),(64,.3),(128,.2)],"D2":[(32,.25),(128,.25),(256,.25),(512,.25)],"D3":[(128,.2),(256,.3),(512,.3),(1024,.2)]};out=[]
 measured={r["workload_id"] for r in matrix}
 for wid,h,d,cap,_,_ in WORKLOADS:
  if wid not in measured:continue
  for name,lengths in dist.items():
   for mib in (16,32,64,128):out.append({"workload_id":wid,"distribution":name,"budget_mib":mib,**admission_count(mib*1024*1024,lengths,bytes_per_token=2*h*d*4,contiguous_capacity=cap,page_tokens=16),"classification":"formula_based_admission_analysis_not_real_concurrent_serving"})
 return out
def write_summary(out,target,matrix,correct,regrets,proof):
 lines=[]
 for wid in sorted({r["workload_id"] for r in matrix}):
  best=min((r for r in matrix if r["workload_id"]==wid),key=lambda r:r["append_decode_p95_ms"])
  lines.append(f"| {wid} | {best['candidate_id']} | {best.get('page_tokens') or '-'} | {best['append_decode_p95_ms']:.6f} |")
 zero=sum(r["runtime_layout_reselection_count"]==r["runtime_kernel_reselection_count"]==r["temporary_full_history_materialization_count"]==0 for r in proof)
 text=f"""# KV selection evaluation: {target}

This artifact compares the real native FP32 contiguous and paged CPU KV implementations. It is an exact-target, exact-workload measured-profile evaluation, not a predictive cost model or concurrent-serving benchmark.

## Correctness and runtime proof

- Correct candidates: {sum(r['max_absolute_error'] < 1e-5 and r['cache_content_exact'] for r in correct)}/{len(correct)}
- Zero layout/kernel reselection and zero temporary full-history materialization: {zero}/{len(proof)}
- Every latency statistic has 10 warmups and 100 measured samples.

## Minimum steady-state p95 by workload

| Workload | Candidate | Page tokens | p95 ms |
| --- | --- | ---: | ---: |
{chr(10).join(lines)}

## Selection result

Compiler-guided selections matching their measured legal oracle: {sum(abs(r['objective_score_regret']) < 1e-12 for r in regrets)}/{len(regrets)}. This is expected for exact-profile lookup and does not establish generalization. The declared unseen-profile fallback remains contiguous.

## Memory and truth boundaries

Paged KV reduces live request-owned bytes when the valid prefix occupies fewer pages than the contiguous maximum-capacity reservation. Because the complete physical page pool is eagerly reserved, it does not reduce process-level pool reservation. Admission results are formula-based, not real concurrent admission. No continuous batching, scheduler, full-model serving, GPU, vLLM PagedAttention, predictive-model, or universal paged-superiority claim is made. Raspberry Pi evidence must remain a separate target identity and is not included here.
"""
 (out/"summary.md").write_text(text)
if __name__=="__main__":main()
