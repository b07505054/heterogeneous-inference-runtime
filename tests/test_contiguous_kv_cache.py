from array import array
import hashlib, math, subprocess
from pathlib import Path
import pytest

from deployment.execution_plan.attention_cpu_adapter import AttentionContractError
from deployment.execution_plan.contiguous_kv_cache import ContiguousKVAttentionSession

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root=tmp_path_factory.mktemp("contiguous_kv_native"); so=root/"libattention_fp32.so"
    subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)],check=True)
    return root,so,hashlib.sha256(so.read_bytes()).hexdigest()

def contracts(artifact,capacity=8,prompt=2,h=2,d=4,reordered=False):
    root,so,sha=artifact;b=1;strides=[h*capacity*d,capacity*d,d,1]
    candidate="cpu_contiguous_kv_fp32_reordered_v1" if reordered else "cpu_contiguous_kv_fp32_v1";kernel="cpu_attention_decode_contiguous_kv_reordered_fp32" if reordered else "cpu_attention_decode_fp32";entry="hir_cpu_attention_decode_contiguous_kv_reordered_fp32" if reordered else "hir_cpu_attention_decode_contiguous_kv_fp32";strategy="token_major_contiguous_v_accumulation" if reordered else "dimension_major_strided_v_accumulation"
    kv={"kv_execution_unit":"portable_cpu_contiguous_kv","kv_candidate_id":candidate,"kv_cache_id":"test_cache","kv_artifact_ref":so.name,"kv_artifact_sha256":sha,"kv_artifact_version":"hir.contiguous_kv.v1","kv_dtype":"fp32","kv_layout":"bhcd_contiguous","batch":b,"num_kv_heads":h,"head_dim":d,"capacity_tokens":capacity,"initial_valid_tokens":0,"bytes_per_token":2*b*h*d*4,"k_cache_bytes":b*h*capacity*d*4,"v_cache_bytes":b*h*capacity*d*4,"total_cache_bytes":2*b*h*capacity*d*4,"alignment_bytes":4,"k_strides":strides,"v_strides":strides,"create_entry_point":"hir_contiguous_kv_initialize","prefill_write_entry_point":"hir_contiguous_kv_prefill_write","decode_append_entry_point":"hir_contiguous_kv_append","view_binding":"direct_contiguous_pointer_valid_prefix","reset_entry_point":"hir_contiguous_kv_reset","compatible_prefill_kernel_id":"cpu_attention_prefill_fp32","compatible_decode_kernel_id":kernel,"attention_entry_point":entry,"implementation_strategy":strategy,"measurement_provenance":"exact_target_workload_measured_evidence_required","runtime_no_layout_redecision":True}
    base={"dtype":"fp32","input_layout":"bhsd_contiguous","runtime_no_redecision":True}
    return root,kv,{"prefill":{**base,"kernel_id":"cpu_attention_prefill_fp32","entry_point":"hir_cpu_attention_prefill_fp32","implementation_strategy":"prefill_contiguous","query_length":prompt},"decode":{**base,"kernel_id":kernel,"entry_point":entry,"implementation_strategy":strategy,"query_length":1}}

def data(n,offset=0): return array("f",((i+offset)%13/13-0.5 for i in range(n)))
def decode_ref(q,k,v,h,t,d):
    out=[];scale=1/math.sqrt(d)
    for hi in range(h):
        scores=[sum(q[hi*d+x]*k[(hi*t+ci)*d+x] for x in range(d))*scale for ci in range(t)];m=max(scores);e=[math.exp(x-m) for x in scores];z=sum(e)
        out.extend(sum(e[ci]/z*v[(hi*t+ci)*d+x] for ci in range(t)) for x in range(d))
    return out

def test_prefill_write_and_decode_growth_matches_reference(artifact):
    root,kv,att=contracts(artifact);s=ContiguousKVAttentionSession(kv,att,artifact_root=root);h,d=2,4
    q=data(h*2*d,1);k=data(h*2*d,2);v=data(h*2*d,3);out=s.prefill(q,k,v)
    ck,cv=s.view();assert list(ck)==list(k) and list(cv)==list(v) and s.valid_tokens==2 and len(out)==len(q)
    for step in range(3):
        q1=data(h*d,10+step);k1=data(h*d,20+step);v1=data(h*d,30+step)
        got=s.decode_append_then_attend(q1,k1,v1);allk,allv=s.view();expected=decode_ref(q1,allk,allv,h,3+step,d)
        assert max(abs(a-b) for a,b in zip(got,expected))<1e-6
    ck,cv=s.view();assert list(ck)==pytest.approx(allk,abs=0) and list(cv)==pytest.approx(allv,abs=0)
    tr=s.trace();assert tr["cache_allocation_count"]==1 and tr["decode_append_count"]==3 and tr["runtime_layout_reselection_count"]==0 and tr["runtime_kernel_reselection_count"]==0

@pytest.mark.parametrize("reordered",[False,True])
def test_exact_original_and_reordered_dispatch_match_reference(artifact,reordered):
    root,kv,att=contracts(artifact,reordered=reordered);s=ContiguousKVAttentionSession(kv,att,artifact_root=root);h,d=2,4
    s.prefill(data(h*2*d),data(h*2*d,1),data(h*2*d,2));q=data(h*d,3);s.append(data(h*d,4),data(h*d,5));got=s.decode(q);k,v=s.view();expected=decode_ref(q,k,v,h,3,d)
    assert got==pytest.approx(expected,abs=1e-6);tr=s.trace();assert tr["compiler_selected_candidate"]==tr["runtime_executed_candidate"]==kv["kv_candidate_id"];assert tr["compiler_selected_kernel"]==tr["runtime_executed_kernel"]==att["decode"]["kernel_id"];assert tr["compiler_selected_strategy"]==tr["runtime_executed_strategy"]==kv["implementation_strategy"];assert tr["runtime_kernel_reselection_count"]==tr["runtime_layout_reselection_count"]==tr["temporary_full_history_materialization_count"]==0

def test_reordered_strategy_mismatch_fails_closed(artifact):
    root,kv,att=contracts(artifact,reordered=True);kv["implementation_strategy"]="dimension_major_strided_v_accumulation"
    with pytest.raises(AttentionContractError,match="kv_contract_mismatch:implementation_strategy"):ContiguousKVAttentionSession(kv,att,artifact_root=root)

@pytest.mark.parametrize("prompt,capacity",[(1,9),(4,17)])
def test_reordered_partial_capacity_reset_and_reuse(artifact,prompt,capacity):
    root,kv,att=contracts(artifact,capacity=capacity,prompt=prompt,reordered=True);s=ContiguousKVAttentionSession(kv,att,artifact_root=root);h,d=2,4;n=h*prompt*d
    s.prefill_write(data(n,1),data(n,2));first=s.decode(data(h*d,3));assert all(math.isfinite(x) for x in first);s.reset();s.prefill_write(data(n,5),data(n,6));second=s.decode(data(h*d,7));assert all(math.isfinite(x) for x in second);assert s.valid_tokens==prompt and s.trace()["cache_reset_count"]==1

def test_state_transitions_reset_and_release(artifact):
    root,kv,att=contracts(artifact);s=ContiguousKVAttentionSession(kv,att,artifact_root=root);h,d=2,4
    with pytest.raises(AttentionContractError,match="decode_before_prefill"):s.decode_append_then_attend(data(h*d),data(h*d),data(h*d))
    s.prefill(data(h*2*d),data(h*2*d),data(h*2*d));s.reset();assert s.valid_tokens==0 and s.state=="CREATED"
    s.prefill(data(h*2*d,5),data(h*2*d,6),data(h*2*d,7));s.release();assert s.state=="RELEASED"
    with pytest.raises(AttentionContractError,match="use_after_release"):s.reset()

def test_capacity_boundary_and_read_bounds(artifact):
    root,kv,att=contracts(artifact,capacity=3);s=ContiguousKVAttentionSession(kv,att,artifact_root=root);h,d=2,4
    s.prefill(data(h*2*d),data(h*2*d),data(h*2*d));s.decode_append_then_attend(data(h*d),data(h*d),data(h*d));assert s.valid_tokens==3
    with pytest.raises(AttentionContractError,match="append_out_of_capacity"):s.decode_append_then_attend(data(h*d),data(h*d),data(h*d))
    with pytest.raises(AttentionContractError,match="read_beyond_valid_tokens"):s.view(4)

@pytest.mark.parametrize("field,value,error",[("kv_layout","paged","kv_contract_mismatch"),("kv_dtype","f16","kv_contract_mismatch"),("kv_artifact_sha256","0"*64,"kv_artifact_hash_mismatch"),("compatible_decode_kernel_id","wrong","kv_contract_mismatch"),("k_strides",[1,2,3,4],"kv_stride_mismatch")])
def test_contract_rejections(artifact,field,value,error):
    root,kv,att=contracts(artifact);kv[field]=value
    with pytest.raises(AttentionContractError,match=error):ContiguousKVAttentionSession(kv,att,artifact_root=root)

def test_wrong_shape_and_prefill_twice(artifact):
    root,kv,att=contracts(artifact);s=ContiguousKVAttentionSession(kv,att,artifact_root=root)
    with pytest.raises(AttentionContractError,match="insufficient_input_buffer"):s.prefill(data(1),data(1),data(1))
    n=2*2*4;s.prefill(data(n),data(n),data(n))
    with pytest.raises(AttentionContractError,match="prefill_requires_empty_cache"):s.prefill(data(n),data(n),data(n))

def test_historical_compiler_generated_plan_contains_exact_kv_contract(artifact):
    import json
    raw=json.loads((ROOT/"artifacts/contiguous_kv/compiler_execution_plan.json").read_text())
    attention={};kv=None
    for function in raw["function_plans"]:
        phase=function["serving_phase"]
        for decision in function["per_op_decisions"]:
            if "attention_execution" in decision:
                attention[phase]=decision["attention_execution"]
                if phase=="prefill":kv=decision["kv_cache_execution"]
    assert kv and kv["kv_artifact_sha256"] and kv["runtime_no_layout_redecision"] is True
    assert {attention[p]["kernel_id"] for p in attention}=={"cpu_attention_prefill_fp32","cpu_attention_decode_fp32"}
