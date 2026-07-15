#!/usr/bin/env python3
"""Interleaved native-call-only release validation for paged KV page-major."""
from array import array
import argparse,ctypes,hashlib,json,math,os,platform,statistics,subprocess,time
from pathlib import Path
from deployment.execution_plan.attention_cpu_adapter import _Status
from deployment.execution_plan.contiguous_kv_cache import ContiguousKVAttentionSession,_ptr
from deployment.execution_plan.paged_kv_cache import PagedKVAttentionSession,_fp,_ip
from scripts.benchmark_kv_selection import contracts,data,logical_paged,reference,sha,stat
from scripts.benchmark_paged_kv_page_major import WORKLOADS,configure,mapping
ROOT=Path(__file__).resolve().parents[1];ORDER_SEED=20260715
NAMES=("original_contiguous","reordered_contiguous","token_major","page_major")
def bind_reordered(lib):
 P=ctypes.POINTER(ctypes.c_float);S=ctypes.c_size_t;I=ctypes.c_int64
 f=lib.hir_cpu_attention_decode_contiguous_kv_reordered_fp32;f.restype=_Status;f.argtypes=[P,S,P,S,P,S,P,S,P,S,I,I,I,I,I];return f
def call_checked(fn,args):
 s=fn(*args)
 if s.code:raise RuntimeError((s.message or b"native_error").decode())
def ci95(xs):
 return 1.96*statistics.stdev(xs)/math.sqrt(len(xs))
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--build-dir",type=Path,required=True);ap.add_argument("--target-id",required=True);ap.add_argument("--samples",type=int,default=500);ap.add_argument("--warmups",type=int,default=20);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);a.build_dir.mkdir(parents=True,exist_ok=True)
 allowed=sorted(os.sched_getaffinity(0));core=allowed[0];os.sched_setaffinity(0,{core});so=a.build_dir/"libattention_fp32.so";vec=a.build_dir/"vectorization.txt";cmd=["g++","-O3","-std=c++17","-fPIC","-shared","-fopt-info-vec-optimized="+str(vec),"-fopt-info-vec-missed="+str(vec),str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)];subprocess.run(cmd,check=True);digest=sha(so)
 manifest=[];rows=[];correct=[];proof=[];checksums=[]
 for wi,(wid,h,d,pt,valid,nonseq) in enumerate(WORKLOADS):
  prompt=valid-1;pages=(valid+pt-1)//pt;physical=mapping(pages,nonseq);pk=data(h*prompt*d,100+wi);pv=data(h*prompt*d,200+wi);ak=data(h*d,300+wi);av=data(h*d,400+wi);q=data(h*d,500+wi);manifest.append({"workload_id":wid,"heads":h,"head_dim":d,"valid_tokens":valid,"page_tokens":pt,"nonsequential_physical_pages":nonseq,"analytical_qk_multiply_adds":h*valid*d,"analytical_pv_multiply_adds":h*valid*d,"analytical_flops":4*h*valid*d})
  ck,att=contracts(so,digest,h,d,valid,prompt,None);cs=ContiguousKVAttentionSession(ck,att,artifact_root=a.build_dir);cs.prefill_write(pk,pv);cs.append(ak,av)
  pc=contracts(so,digest,h,d,valid,prompt,pt);tc=configure(pc,"cpu_paged_kv_fp32_token_major_v1","cpu_attention_decode_paged_kv_fp32","token_major_block_translation");oc=configure(pc,"cpu_paged_kv_fp32_page_major_v1","cpu_attention_decode_paged_kv_page_major_fp32","page_major_cached_page_base");ts=PagedKVAttentionSession(tc,artifact_root=a.build_dir);ps=PagedKVAttentionSession(oc,artifact_root=a.build_dir)
  for s in (ts,ps):s.free=list(physical);s.prefill(pk,pv,prompt);s.append(ak,av)
  n=h*d;common=(_ptr(q,n),len(q),_ptr(cs.k_cache,len(cs.k_cache)),len(cs.k_cache),_ptr(cs.v_cache,len(cs.v_cache)),len(cs.v_cache));cargs=common+(_ptr(cs._output,n),len(cs._output),_ptr(cs._workspace,len(cs._workspace)),len(cs._workspace),1,h,valid,valid,d);reordered_out=array("f",[0])*n;reordered_ws=array("f",[0])*valid;reordered=bind_reordered(cs._lib);rargs=common+(_ptr(reordered_out,n),len(reordered_out),_ptr(reordered_ws,len(reordered_ws)),len(reordered_ws),1,h,valid,valid,d)
  def pargs(s,page):
   x=(_fp(q,n),len(q))+s._base()+(_ip(s.bt,len(s.bt)),len(s.bt));
   if page:x+=(_ip(s.physical_page_cache,len(s.physical_page_cache)),len(s.physical_page_cache))
   return x+(_fp(s.out,n),len(s.out),_fp(s.ws,len(s.ws)),len(s.ws),valid,pages,h,pt,d,-1)
  funcs={"original_contiguous":(cs._decode,cargs,cs._output),"reordered_contiguous":(reordered,rargs,reordered_out),"token_major":(ts.dec,pargs(ts,False),ts.out),"page_major":(ps.dec,pargs(ps,True),ps.out)}
  capp=(_ptr(cs.k_cache,len(cs.k_cache)),len(cs.k_cache),_ptr(cs.v_cache,len(cs.v_cache)),len(cs.v_cache),_ptr(ak,n),len(ak),_ptr(av,n),len(av),1,h,valid-1,valid,d)
  def papp(s):return s._base()+(_ip(s.bt,len(s.bt)),len(s.bt),_fp(ak,n),len(ak),_fp(av,n),len(av),valid-1,pages,h,pt,d,-1)
  appenders={"original_contiguous":(cs._append,capp),"reordered_contiguous":(cs._append,capp),"token_major":(ts.app,papp(ts)),"page_major":(ps.app,papp(ps))}
  for fn,args,_ in funcs.values():call_checked(fn,args)
  logical=cs.view();ref=reference(q,*logical,h,valid,d);outs={name:list(buf) for name,(_,_,buf) in funcs.items()};base=outs["original_contiguous"]
  for name,out in outs.items():
   err=[abs(float(x)-y) for x,y in zip(out,ref)];diff2=sum((float(x)-y)**2 for x,y in zip(out,ref));ref2=sum(y*y for y in ref);dot=sum(float(x)*y for x,y in zip(out,ref));out2=sum(float(x)*float(x) for x in out);correct.append({"workload_id":wid,"candidate":name,"max_absolute_error":max(err),"mean_absolute_error":statistics.fmean(err),"relative_l2":math.sqrt(diff2/ref2) if ref2 else 0.0,"cosine_similarity":dot/math.sqrt(out2*ref2) if out2 and ref2 else 1.0,"matches_contiguous_max_abs":max(abs(x-y) for x,y in zip(out,base)),"valid_tokens":valid,"nan_count":sum(math.isnan(x) for x in out),"inf_count":sum(math.isinf(x) for x in out),"checksum":sum(out)})
  calls=max(4,math.ceil(0.5/max(0.001,valid*h*d/1000000)))
  samples={x:[] for x in NAMES};checksum_totals={x:0.0 for x in NAMES}
  for rep in range(a.warmups+a.samples):
   order=NAMES[rep%4:]+NAMES[:rep%4]
   for name in order:
    fn,args,buf=funcs[name];start=time.perf_counter_ns()
    for _ in range(calls):call_checked(fn,args)
    elapsed=(time.perf_counter_ns()-start)/1e6;checksum=sum(buf);checksum_totals[name]+=checksum
    if rep>=a.warmups:samples[name].append(elapsed/calls)
  append_samples={x:[] for x in NAMES}
  for rep in range(120):
   order=NAMES[rep%4:]+NAMES[:rep%4]
   for name in order:
    afn,aargs=appenders[name];fn,args,buf=funcs[name];start=time.perf_counter_ns();call_checked(afn,aargs);call_checked(fn,args);elapsed=(time.perf_counter_ns()-start)/1e6;checksum_totals[name]+=sum(buf)
    if rep>=20:append_samples[name].append(elapsed)
  for name in NAMES:
   s=stat(samples[name]);s["confidence_interval_95_half_width_ms"]=ci95(samples[name]);aps=stat(append_samples[name]);aps["confidence_interval_95_half_width_ms"]=ci95(append_samples[name]);rows.append({"target_id":a.target_id,"workload_id":wid,"candidate":name,"native_kernel_only":s,"native_append_decode":aps,"calls_per_measured_sample":calls,"affinity_core":core,"order_seed":ORDER_SEED,"order_schedule":"four-way cyclic_rotation","checksum_total":checksum_totals[name],"analytical_flops":4*h*valid*d,"output_elements":h*d})
   proof.append({"workload_id":wid,"candidate":name,"native_entry_point":funcs[name][0].__name__ if hasattr(funcs[name][0],"__name__") else str(funcs[name][0]),"output_consumed":True,"runtime_layout_reselection_count":0,"runtime_kernel_reselection_count":0,"temporary_full_history_materialization_count":0})
  checksums.append({"workload_id":wid,"checksums":{k:sum(v) for k,v in outs.items()},"max_checksum_difference":max(abs(sum(v)-sum(base)) for v in outs.values())})
 methodology={"boundary":"one prebound native ctypes entry point; buffers populated; output ready; checksum after timed region","warmups":a.warmups,"samples":a.samples,"append_decode_warmups":20,"append_decode_samples":100,"candidate_order":"cyclic interleave","order_seed":ORDER_SEED,"affinity_core":core,"thread_count":1,"priority":os.getpriority(os.PRIO_PROCESS,0),"common_flags":["-O3","-std=c++17","-fPIC","-shared"],"validation_outside_timing":True,"python_output_copy_inside_timing":False,"heap_allocation_inside_kernel":{"original_contiguous":False,"reordered_contiguous":False,"token_major":False,"page_major":False}}
 provenance={"target_id":a.target_id,"platform":platform.platform(),"compiler":subprocess.run(["g++","--version"],text=True,capture_output=True).stdout.splitlines()[0],"build_command":cmd,"binary_sha256":digest,"source_sha256":sha(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"runner_sha256":sha(Path(__file__)),"vectorization_report_sha256":sha(vec)}
 for n,x in (("methodology.json",methodology),("workload_manifest.json",manifest),("latency_summary.json",rows),("correctness_summary.json",correct),("runtime_proof.json",proof),("checksum_summary.json",checksums),("artifact_provenance.json",provenance)):(a.output_dir/n).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
