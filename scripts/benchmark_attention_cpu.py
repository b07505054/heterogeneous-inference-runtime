#!/usr/bin/env python3
"""Deterministic correctness and already-loaded latency study for CPU attention."""
from array import array
import argparse,hashlib,json,math,platform,statistics,subprocess,time
from pathlib import Path
from deployment.execution_plan.attention_cpu_adapter import PersistentAttentionRunner

ROOT=Path(__file__).resolve().parents[1]
CASES=[("prefill",2,16,16,32),("prefill",4,64,64,32),("prefill",4,128,128,64),
       ("decode",2,1,16,32),("decode",4,1,64,32),("decode",4,1,128,64),("decode",4,1,512,64)]
def sha_bytes(x:array):return hashlib.sha256(x.tobytes()).hexdigest()
def sha_file(p:Path):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor(n,seed):return array("f",[((i*1103515245+seed*12345)&0xffff)/32768-1 for i in range(n)])
def ref(q,k,v,h,ql,cl,d,prefill):
 o=array("d",[0])* (h*ql*d);scl=1/math.sqrt(d)
 for hi in range(h):
  for qi in range(ql):
   valid=qi+1 if prefill else cl;qb=(hi*ql+qi)*d
   s=[sum(float(q[qb+x])*float(k[(hi*cl+ci)*d+x]) for x in range(d))*scl for ci in range(valid)]
   mx=max(s);e=[math.exp(x-mx) for x in s];den=sum(e)
   for x in range(d):o[qb+x]=sum(e[ci]/den*float(v[(hi*cl+ci)*d+x]) for ci in range(valid))
 return o
def pct(x,p):return sorted(x)[math.ceil(p*len(x))-1]
def stats(x):
 mean=statistics.fmean(x);sd=statistics.pstdev(x)
 return {"sample_count":len(x),"median_ms":statistics.median(x),"mean_ms":mean,"minimum_ms":min(x),"p90_ms":pct(x,.9),"p95_ms":pct(x,.95),"p99_ms":pct(x,.99),"stddev_ms":sd,"coefficient_of_variation":sd/mean if mean else 0}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--build-dir",type=Path,required=True);a=ap.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True);a.build_dir.mkdir(parents=True,exist_ok=True);so=a.build_dir/"libattention_fp32.so"
 cmd=["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)]
 t=time.perf_counter();subprocess.run(cmd,check=True);build_ms=(time.perf_counter()-t)*1000;sh=sha_file(so)
 manifest=[];correctness=[];latency=[];traces=[]
 for idx,(phase,h,ql,cl,d) in enumerate(CASES):
  q=tensor(h*ql*d,idx*3+1);k=tensor(h*cl*d,idx*3+2);v=tensor(h*cl*d,idx*3+3)
  cfg={"execution_unit":"portable_cpu_attention","backend":"portable_cpu","phase":phase,"candidate_id":f"cpu_attention_{phase}_fp32","kernel_id":f"cpu_attention_{phase}_fp32","entry_point":f"hir_cpu_attention_{phase}_fp32","artifact_ref":so.name,"artifact_sha256":sh,"artifact_version":"hir.cpu_attention.v1","dtype":"fp32","input_layout":"bhsd_contiguous","output_layout":"bhsd_contiguous","batch":1,"query_length":ql,"context_length":cl,"num_query_heads":h,"num_kv_heads":h,"head_dim":d,"causal":True,"workspace_bytes":cl*4,"alignment_bytes":4,"required_isa":"scalar_fp32","fallback_identity":"unsupported_attention_explicit_failure","runtime_no_redecision":True,"truth_boundary":"real_operator_level_fp32_cpu_attention_not_full_model_or_kv_lifetime"}
  t=time.perf_counter();r=PersistentAttentionRunner(cfg,artifact_root=a.build_dir);load_ms=(time.perf_counter()-t)*1000
  expected=ref(q,k,v,h,ql,cl,d,phase=="prefill");t=time.perf_counter();got=r.invoke(q,k,v);first_ms=(time.perf_counter()-t)*1000
  diff=[abs(float(x)-y) for x,y in zip(got,expected)];rel=math.sqrt(sum(x*x for x in diff))/max(math.sqrt(sum(y*y for y in expected)),1e-30);cos=sum(float(x)*y for x,y in zip(got,expected))/max(math.sqrt(sum(float(x)**2 for x in got)*sum(y*y for y in expected)),1e-30)
  cid=f"{phase}_b1_h{h}_q{ql}_c{cl}_d{d}";manifest.append({"workload_id":cid,"seed":idx,"q_sha256":sha_bytes(q),"k_sha256":sha_bytes(k),"v_sha256":sha_bytes(v),"shape":{"batch":1,"heads":h,"query_length":ql,"context_length":cl,"head_dim":d}})
  correctness.append({"workload_id":cid,"max_absolute_error":max(diff),"mean_absolute_error":statistics.fmean(diff),"relative_l2_error":rel,"cosine_similarity":cos,"nan_count":sum(math.isnan(x) for x in got),"inf_count":sum(math.isinf(x) for x in got)})
  for _ in range(10):r.invoke(q,k,v)
  samples=[]
  for _ in range(100):
   t=time.perf_counter();checksum=0
   for _ in range(3):checksum+=r.invoke(q,k,v)[0]
   samples.append((time.perf_counter()-t)*1000/3)
  latency.append({"workload_id":cid,"warmups":10,"inner_invocations":3,"artifact_load_ms":load_ms,"first_inference_ms":first_ms,"already_loaded_invocation":stats(samples),"checksum":checksum})
  traces.append({"workload_id":cid,**r.trace()})
 compiler_version=subprocess.run(["g++","--version"],text=True,capture_output=True,check=True).stdout.splitlines()[0]
 out={"schema_version":"attention_cpu_evaluation.v1","host":{"machine":platform.machine(),"platform":platform.platform()},"build_command":cmd,"compiler_version":compiler_version,"build_ms":build_ms,"artifact_sha256":sh,"source_sha256":sha_file(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"runner_sha256":sha_file(ROOT/"deployment/execution_plan/attention_cpu_adapter.py"),"truth_boundary":"operator_level_fp32_cpu_attention_not_full_model_serving"}
 for name,val in [("workload_manifest.json",manifest),("correctness_summary.json",correctness),("latency_summary.json",latency),("runtime_trace.json",traces),("artifact_provenance.json",out)]:
  (a.output_dir/name).write_text(json.dumps(val,indent=2,sort_keys=True)+"\n")
 (a.output_dir/"native_build_manifest.json").write_text(json.dumps({"schema_version":"attention_native_build.v1","source":"native/cpu_kernels/attention_fp32.cpp","source_sha256":out["source_sha256"],"compiler_version":compiler_version,"flags":["-O3","-std=c++17","-fPIC","-shared"],"artifact_sha256":sh,"artifact_version":"hir.cpu_attention.v1"},indent=2,sort_keys=True)+"\n")
 (a.output_dir/"target_identity.json").write_text(json.dumps({"schema_version":"attention_target.v1","machine":platform.machine(),"platform":platform.platform(),"required_isa":"scalar_fp32","thread_count":1},indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
