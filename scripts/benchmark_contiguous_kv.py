#!/usr/bin/env python3
"""Deterministic contiguous-KV correctness and separated-boundary benchmark."""
from array import array
import argparse,hashlib,json,math,platform,statistics,subprocess,time
from pathlib import Path
from deployment.execution_plan.contiguous_kv_cache import ContiguousKVAttentionSession

ROOT=Path(__file__).resolve().parents[1]
CASES=[("A",2,32,16,16,32),("B",4,32,64,64,128),("C",4,64,128,128,256),("D",4,64,128,384,512)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def data(n,s):return array("f",(((i*1103515245+s*12345)&0xffff)/32768-1 for i in range(n)))
def pct(x,p):return sorted(x)[math.ceil(p*len(x))-1]
def stat(x):
 m=statistics.fmean(x);sd=statistics.pstdev(x);return {"samples":len(x),"median_ms":statistics.median(x),"mean_ms":m,"min_ms":min(x),"p90_ms":pct(x,.9),"p95_ms":pct(x,.95),"p99_ms":pct(x,.99),"stddev_ms":sd,"cv":sd/m if m else 0}
def contract(so,sh,h,d,cap,prompt):
 st=[h*cap*d,cap*d,d,1];kv={"kv_execution_unit":"portable_cpu_contiguous_kv","kv_candidate_id":"cpu_contiguous_kv_fp32_v1","kv_cache_id":"benchmark_cache","kv_artifact_ref":so.name,"kv_artifact_sha256":sh,"kv_artifact_version":"hir.contiguous_kv.v1","kv_dtype":"fp32","kv_layout":"bhcd_contiguous","batch":1,"num_kv_heads":h,"head_dim":d,"capacity_tokens":cap,"initial_valid_tokens":0,"bytes_per_token":2*h*d*4,"k_cache_bytes":h*cap*d*4,"v_cache_bytes":h*cap*d*4,"total_cache_bytes":2*h*cap*d*4,"alignment_bytes":4,"k_strides":st,"v_strides":st,"create_entry_point":"hir_contiguous_kv_initialize","prefill_write_entry_point":"hir_contiguous_kv_prefill_write","decode_append_entry_point":"hir_contiguous_kv_append","view_binding":"direct_contiguous_pointer_valid_prefix","reset_entry_point":"hir_contiguous_kv_reset","compatible_prefill_kernel_id":"cpu_attention_prefill_fp32","compatible_decode_kernel_id":"cpu_attention_decode_fp32","runtime_no_layout_redecision":True}
 base={"dtype":"fp32","input_layout":"bhsd_contiguous","runtime_no_redecision":True};return kv,{"prefill":{**base,"kernel_id":"cpu_attention_prefill_fp32","query_length":prompt},"decode":{**base,"kernel_id":"cpu_attention_decode_fp32","query_length":1}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--build-dir",type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);a.build_dir.mkdir(parents=True,exist_ok=True)
 so=a.build_dir/"libattention_fp32.so";cmd=["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)];t=time.perf_counter();subprocess.run(cmd,check=True);build=(time.perf_counter()-t)*1e3;sh=sha(so)
 manifest=[];correct=[];lat=[];traces=[]
 for ci,(name,h,d,prompt,steps,cap) in enumerate(CASES):
  kv,att=contract(so,sh,h,d,cap,prompt);q=data(h*prompt*d,ci+1);k=data(h*prompt*d,ci+2);v=data(h*prompt*d,ci+3)
  t=time.perf_counter();s=ContiguousKVAttentionSession(kv,att,artifact_root=a.build_dir);alloc=(time.perf_counter()-t)*1e3
  t=time.perf_counter();s.prefill_write(k,v);write=(time.perf_counter()-t)*1e3;t=time.perf_counter();po=s.prefill_attention(q,k,v);patt=(time.perf_counter()-t)*1e3
  loop=[];append_times=[];decode_times=[];contexts=[];diffs=[];refs=[];actual=[]
  scale=1/math.sqrt(d)
  for hi in range(h):
   for qi in range(prompt):
    scores=[sum(q[(hi*prompt+qi)*d+x]*k[(hi*prompt+j)*d+x] for x in range(d))*scale for j in range(qi+1)];m=max(scores);e=[math.exp(x-m) for x in scores];z=sum(e)
    for x in range(d):
     ref=sum(e[j]/z*v[(hi*prompt+j)*d+x] for j in range(qi+1));got=float(po[(hi*prompt+qi)*d+x]);actual.append(got);refs.append(ref);diffs.append(abs(got-ref))
  for step in range(steps):
   q1=data(h*d,1000+step);k1=data(h*d,2000+step);v1=data(h*d,3000+step);t=time.perf_counter();s.append(k1,v1);append_times.append((time.perf_counter()-t)*1e3);t=time.perf_counter();got=s.decode(q1);decode_times.append((time.perf_counter()-t)*1e3);loop.append(append_times[-1]+decode_times[-1]);contexts.append(s.valid_tokens)
   ck,cv=s.view();valid=s.valid_tokens
   for hi in range(h):
    scores=[sum(q1[hi*d+x]*ck[(hi*valid+j)*d+x] for x in range(d))*scale for j in range(valid)];m=max(scores);e=[math.exp(x-m) for x in scores];z=sum(e)
    for x in range(d):
     ref=sum(e[j]/z*cv[(hi*valid+j)*d+x] for j in range(valid));actual.append(float(got[hi*d+x]));refs.append(ref);diffs.append(abs(float(got[hi*d+x])-ref))
  # Fixed final context: decode-only steady state, no allocation or mutation.
  append_samples=[];target=prompt+steps
  for sample in range(100):
   s.reset();s.prefill_write(k,v)
   for fill in range(prompt,target-1):s.append(data(h*d,20000+fill),data(h*d,30000+fill))
   kt=data(h*d,40000+sample);vt=data(h*d,50000+sample);t=time.perf_counter();s.append(kt,vt);append_samples.append((time.perf_counter()-t)*1e3)
  qf=data(h*d,9000);samples=[]
  for _ in range(10):s.decode(qf)
  for _ in range(100):t=time.perf_counter();s.decode(qf);samples.append((time.perf_counter()-t)*1e3)
  t=time.perf_counter();s.reset();reset=(time.perf_counter()-t)*1e3
  manifest.append({"case":name,"heads":h,"head_dim":d,"prefill_tokens":prompt,"decode_steps":steps,"capacity_tokens":cap,"bytes_per_token":kv["bytes_per_token"],"total_cache_bytes":kv["total_cache_bytes"],"q_sha256":hashlib.sha256(q.tobytes()).hexdigest(),"k_sha256":hashlib.sha256(k.tobytes()).hexdigest(),"v_sha256":hashlib.sha256(v.tobytes()).hexdigest()})
  rel=math.sqrt(sum(x*x for x in diffs))/max(math.sqrt(sum(x*x for x in refs)),1e-30);cos=sum(x*y for x,y in zip(actual,refs))/max(math.sqrt(sum(x*x for x in actual)*sum(y*y for y in refs)),1e-30)
  correct.append({"case":name,"max_absolute_error":max(diffs),"mean_absolute_error":statistics.fmean(diffs),"relative_l2":rel,"cosine_similarity":cos,"nan_count":sum(math.isnan(x) for x in actual+list(po)),"inf_count":sum(math.isinf(x) for x in actual+list(po)),"cache_content_tolerance":0.0})
  buckets={str(target):decode_times[min(range(len(contexts)),key=lambda i:abs(contexts[i]-target))] for target in (16,32,64,128,256,512) if target>=contexts[0] and target<=contexts[-1]}
  lat.append({"case":name,"cache_allocation_ms":alloc,"prefill_write_ms":write,"prefill_attention_ms":patt,"single_decode_append":stat(append_samples),"decode_attention_by_context_ms":buckets,"decode_attention_final_context":stat(samples),"combined_append_decode_loop":stat(loop),"full_decode_loop_ms":sum(loop),"reset_ms":reset,"peak_cache_bytes":kv["total_cache_bytes"],"bytes_per_token":kv["bytes_per_token"]})
  traces.append({"case":name,**s.trace()})
 prov={"schema_version":"contiguous_kv_evaluation.v1","host":platform.platform(),"build_ms":build,"build_command":cmd,"artifact_sha256":sh,"source_sha256":sha(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"runner_sha256":sha(ROOT/"deployment/execution_plan/contiguous_kv_cache.py"),"truth_boundary":"operator_level_runtime_owned_contiguous_fp32_kv"}
 for n,x in (("workload_manifest.json",manifest),("correctness_summary.json",correct),("latency_summary.json",lat),("runtime_trace.json",traces),("artifact_provenance.json",prov)):(a.output_dir/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
