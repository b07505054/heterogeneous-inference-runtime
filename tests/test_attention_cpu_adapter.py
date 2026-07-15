from array import array
import hashlib, math, subprocess
from pathlib import Path
import pytest
from deployment.execution_plan.attention_cpu_adapter import AttentionContractError, PersistentAttentionRunner

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
 d=tmp_path_factory.mktemp("attention_native");so=d/"libattention_fp32.so"
 subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)],check=True)
 return d,so,hashlib.sha256(so.read_bytes()).hexdigest()

def contract(artifact,phase="prefill",q=4,c=4,d=8,h=2):
 root,so,sha=artifact;kernel=f"cpu_attention_{phase}_fp32"
 return root,{"execution_unit":"portable_cpu_attention","backend":"portable_cpu","phase":phase,
  "kernel_id":kernel,"entry_point":f"hir_{kernel}","artifact_ref":so.name,"artifact_sha256":sha,
  "artifact_version":"hir.cpu_attention.v1","dtype":"fp32","input_layout":"bhsd_contiguous","output_layout":"bhsd_contiguous",
  "batch":1,"query_length":q,"context_length":c,"num_query_heads":h,"num_kv_heads":h,"head_dim":d,
  "candidate_id":f"cpu_attention_{phase}_fp32","causal":True,"workspace_bytes":c*4,
  "alignment_bytes":4,"required_isa":"scalar_fp32","fallback_identity":"explicit_failure",
  "runtime_no_redecision":True,"truth_boundary":"real_operator_level_fp32_cpu_attention_not_full_model_or_kv_lifetime"}

def data(n,shift=0):return array("f",[((i*17+shift)%31-15)/17 for i in range(n)])
def reference(q,k,v,b,h,ql,cl,d,prefill):
 out=[0.0]*(b*h*ql*d);scale=1/math.sqrt(d)
 for bi in range(b):
  for hi in range(h):
   for qi in range(ql):
    valid=qi+1 if prefill else cl;qb=((bi*h+hi)*ql+qi)*d
    scores=[sum(q[qb+x]*k[((bi*h+hi)*cl+ci)*d+x] for x in range(d))*scale for ci in range(valid)]
    mx=max(scores);ex=[math.exp(x-mx) for x in scores];den=sum(ex)
    for x in range(d):out[qb+x]=sum(ex[ci]/den*v[((bi*h+hi)*cl+ci)*d+x] for ci in range(valid))
 return out

@pytest.mark.parametrize("phase,ql,cl",[("prefill",4,4),("decode",1,7)])
def test_native_matches_reference_and_persists(artifact,phase,ql,cl):
 root,cfg=contract(artifact,phase,ql,cl);r=PersistentAttentionRunner(cfg,artifact_root=root)
 q=data(1*2*ql*8);k=data(1*2*cl*8,3);v=data(1*2*cl*8,7);got=r.invoke(q,k,v)
 ref=reference(q,k,v,1,2,ql,cl,8,phase=="prefill")
 assert max(abs(a-b) for a,b in zip(got,ref))<2e-6
 r.invoke(q,k,v);t=r.trace();assert t["artifact_load_count"]==1 and t["worker_start_count"]==1
 assert t["buffer_allocation_count"]==2 and t["runtime_reselection_count"]==0 and t["runtime_no_redecision"]

def test_causal_future_values_do_not_change_earlier_outputs(artifact):
 root,cfg=contract(artifact);r=PersistentAttentionRunner(cfg,artifact_root=root);q=data(64);k=data(64,2);v=data(64,5)
 a=r.invoke(q,k,v);k2=array("f",k);v2=array("f",v)
 for i in range(8,32):k2[i]+=10000;v2[i]-=10000
 b=r.invoke(q,k2,v2);assert max(abs(a[i]-b[i]) for i in range(8))<1e-6

@pytest.mark.parametrize("field,value,error",[
 ("phase","decode","kernel_or_entry_point_mismatch"),("kernel_id","wrong","kernel_or_entry_point_mismatch"),
 ("artifact_sha256","0"*64,"artifact_hash_mismatch"),("input_layout","strided","layout_mismatch"),
 ("dtype","fp16","dtype_mismatch"),("head_dim",0,"invalid_dimension"),("query_length",1,"query_length_mismatch")])
def test_invalid_contract_rejected(artifact,field,value,error):
 root,cfg=contract(artifact);cfg[field]=value
 with pytest.raises(AttentionContractError,match=error):PersistentAttentionRunner(cfg,artifact_root=root)

def test_buffer_and_workspace_bounds(artifact):
 root,cfg=contract(artifact);r=PersistentAttentionRunner(cfg,artifact_root=root);q=data(64);k=data(64);v=data(64)
 with pytest.raises(AttentionContractError,match="insufficient_input_buffer"):r.invoke(array("f",q[:-1]),k,v)
 with pytest.raises(AttentionContractError,match="insufficient_output_buffer"):r.invoke(q,k,v,output_capacity=1)
 with pytest.raises(AttentionContractError,match="insufficient_workspace"):r.invoke(q,k,v,workspace_capacity=1)

def test_historical_compiler_execution_plan_round_trips(artifact):
 import json
 from deployment.execution_plan.loader import load_execution_plan
 from deployment.execution_plan.stage_builder import build_execution_stages
 from deployment.execution_plan.path_builder import build_execution_paths
 plan_path=ROOT/"artifacts/attention_cpu/compiler_execution_plan.json"
 plan=load_execution_plan(plan_path);stages=build_execution_stages(plan);paths=build_execution_paths(plan,stages)
 attention=[p for p in paths if p.execution_method.value=="cpu_attention_kernel"]
 assert {p.selected_kernel for p in attention}=={"cpu_attention_prefill_fp32","cpu_attention_decode_fp32"}
 raw=json.loads(plan_path.read_text());contract=next(x["attention_execution"] for f in raw["function_plans"] if f["serving_phase"]=="prefill" for x in f["per_op_decisions"] if "attention_execution" in x)
 assert contract["artifact_sha256"] and contract["runtime_no_redecision"] is True
