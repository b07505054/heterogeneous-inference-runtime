from array import array
import hashlib,math,subprocess
from pathlib import Path
import pytest
from deployment.execution_plan.attention_cpu_adapter import AttentionContractError
from deployment.execution_plan.kv_page_manager import KVPageManager
from deployment.execution_plan.paged_kv_cache import PagedKVAttentionSession,PagedKVStorage,paged_decode_operation_counts
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
 r=tmp_path_factory.mktemp("paged_native");so=r/"libattention_fp32.so";subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"native/cpu_kernels/attention_fp32.cpp"),"-o",str(so)],check=True);return r,so,hashlib.sha256(so.read_bytes()).hexdigest()
def cfg(a,pages=4,pt=4,max_tokens=16,h=2,d=4):
 r,so,sha=a;blocks=(max_tokens+pt-1)//pt;one=h*pt*d*4;st=[h*pt*d,pt*d,d,1]
 return r,{"kv_candidate_id":"cpu_paged_kv_fp32_v1","kv_layout_kind":"paged_phd_contiguous","pool_artifact_ref":so.name,"pool_artifact_sha256":sha,"pool_artifact_version":"hir.paged_kv.v1","dtype":"fp32","batch":1,"num_kv_heads":h,"head_dim":d,"page_tokens":pt,"num_physical_pages":pages,"maximum_logical_tokens":max_tokens,"maximum_logical_blocks":blocks,"block_table_length":blocks,"block_table_element_type":"int32","invalid_page_sentinel":-1,"k_page_strides":st,"v_page_strides":st,"bytes_per_token":2*h*d*4,"bytes_per_k_page":one,"bytes_per_v_page":one,"bytes_per_combined_page":2*one,"total_pool_bytes":pages*2*one,"alignment_bytes":4,"pool_create_entry_point":"hir_paged_kv_initialize","prefill_write_entry_point":"hir_paged_kv_prefill_write","append_entry_point":"hir_paged_kv_append","view_binding":"direct_int32_block_table_translation","reset_entry_point":"hir_paged_kv_reset","release_entry_point":"runtime_owned_pool_release","paged_attention_kernel_id":"cpu_attention_decode_paged_kv_fp32","contiguous_fallback_identity":"cpu_contiguous_kv_fp32_v1","runtime_no_layout_redecision":True,"runtime_no_kernel_redecision":True}
def data(n,s=0):return array("f",((i+s)%17/17-.5 for i in range(n)))
def ref(q,k,v,h,t,d):
 out=[];sc=1/math.sqrt(d)
 for hi in range(h):
  z=[sum(q[hi*d+x]*k[(hi*t+j)*d+x] for x in range(d))*sc for j in range(t)];m=max(z);e=[math.exp(x-m) for x in z];den=sum(e)
  out.extend(sum(e[j]/den*v[(hi*t+j)*d+x] for j in range(t)) for x in range(d))
 return out
def logical(s):
 c=s.c;h,d,pt=c["num_kv_heads"],c["head_dim"],c["page_tokens"];k=[];v=[]
 for hi in range(h):
  for t in range(s.valid_tokens):
   p=s.bt[t//pt];base=((p*h+hi)*pt+t%pt)*d;k.extend(s.k[base:base+d]);v.extend(s.v[base:base+d])
 return k,v
def page_major(c):
 c=dict(c);c.update(kv_candidate_id="cpu_paged_kv_fp32_page_major_v1",paged_attention_kernel_id="cpu_attention_decode_paged_kv_page_major_fp32",implementation_strategy="page_major_cached_page_base");return c
def test_cross_page_prefill_append_and_native_decode(artifact):
 r,c=cfg(artifact);s=PagedKVAttentionSession(c,artifact_root=r);h,d=2,4;k=data(h*7*d,1);v=data(h*7*d,2);s.prefill(k,v,7);assert s.bt[:2]==array("i",[0,1]);assert logical(s)==(list(k),list(v))
 q=data(h*d,3);kt=data(h*d,4);vt=data(h*d,5);s.append(kt,vt);assert s.count.append_existing_page_count==1;s.append(kt,vt);assert s.count.append_new_page_count==1
 lk,lv=logical(s);got=s.decode(q);assert max(abs(x-y) for x,y in zip(got,ref(q,lk,lv,h,9,d)))<1e-6;assert s.count.temporary_full_history_materialization_count==0
def test_one_page_prefill_and_reset_returns_pages(artifact):
 r,c=cfg(artifact);s=PagedKVAttentionSession(c,artifact_root=r);s.prefill(data(2*3*4),data(2*3*4,2),3);assert len(s.owned)==1;s.reset();assert not s.owned and len(s.free)==4 and all(x==-1 for x in s.bt)
def test_repeated_decode_observes_history(artifact):
 r,c=cfg(artifact);s=PagedKVAttentionSession(c,artifact_root=r);s.prefill(data(2*2*4),data(2*2*4,2),2)
 for i in range(5):s.append(data(8,10+i),data(8,20+i));q=data(8,30+i);lk,lv=logical(s);assert list(s.decode(q))==pytest.approx(ref(q,lk,lv,2,s.valid_tokens,4),abs=1e-6)
 assert s.count.pool_allocation_count==1 and s.count.runtime_layout_reselection_count==0 and s.count.runtime_kernel_reselection_count==0
def test_out_of_pages(artifact):
 r,c=cfg(artifact,pages=1,pt=2,max_tokens=4);s=PagedKVAttentionSession(c,artifact_root=r);s.prefill(data(8),data(8),1);s.append(data(8),data(8))
 with pytest.raises(AttentionContractError,match="out_of_physical_pages"):s.append(data(8),data(8))
def test_invalid_live_tables_fail(artifact):
 r,c=cfg(artifact);s=PagedKVAttentionSession(c,artifact_root=r);s.prefill(data(16),data(16),2);s.page_manager._requests[s.request_id].physical_pages[0]=-1
 with pytest.raises(AttentionContractError,match="invalid_live_block_table"):s.decode(data(8))
def test_capacity_and_state_failures(artifact):
 r,c=cfg(artifact,max_tokens=4);s=PagedKVAttentionSession(c,artifact_root=r)
 with pytest.raises(AttentionContractError,match="decode_before_prefill"):s.decode(data(8))
 s.prefill(data(32),data(32),4)
 with pytest.raises(AttentionContractError,match="append_beyond_logical_capacity"):s.append(data(8),data(8))
 with pytest.raises(AttentionContractError,match="prefill_requires_created"):s.prefill(data(8),data(8),1)
@pytest.mark.parametrize("field,value,error",[("kv_layout_kind","wrong","contract_mismatch"),("dtype","f16","contract_mismatch"),("pool_artifact_sha256","0"*64,"artifact_hash_mismatch"),("paged_attention_kernel_id","wrong","contract_mismatch"),("block_table_element_type","int64","contract_mismatch"),("k_page_strides",[1,2,3,4],"page_formula_mismatch")])
def test_contract_failures(artifact,field,value,error):
 r,c=cfg(artifact);c[field]=value
 with pytest.raises(AttentionContractError,match=error):PagedKVAttentionSession(c,artifact_root=r)

@pytest.mark.parametrize("tokens",[1,7,8,9,16])
def test_page_major_matches_token_major_for_partial_and_full_pages(artifact,tokens):
 r,c=cfg(artifact);k=data(2*tokens*4,1);v=data(2*tokens*4,2);q=data(8,3)
 baseline=PagedKVAttentionSession(c,artifact_root=r);optimized=PagedKVAttentionSession(page_major(c),artifact_root=r)
 baseline.prefill(k,v,tokens);optimized.prefill(k,v,tokens)
 assert list(optimized.decode(q))==pytest.approx(list(baseline.decode(q)),abs=1e-6)
 assert optimized.trace()["runtime_executed_strategy"]=="page_major_cached_page_base"
 assert optimized.count.runtime_kernel_reselection_count==optimized.count.runtime_layout_reselection_count==optimized.count.temporary_full_history_materialization_count==0

def test_page_major_pointer_walking_addresses_every_token_head_and_dimension(artifact):
 r,c=cfg(artifact,pages=4,pt=4,max_tokens=16,h=2,d=4);c=page_major(c)
 manager=KVPageManager(total_pages=4,tokens_per_page=4);storage=PagedKVStorage(total_pages=4,num_kv_heads=2,tokens_per_page=4,head_dim=4,workspace_tokens=16)
 manager.reserve_prefill("gap",4)
 s=PagedKVAttentionSession(c,artifact_root=r,request_id="walk",storage=storage,page_manager=manager)
 tokens=10;k=data(2*tokens*4,11);v=data(2*tokens*4,19);q=data(8,23)
 s.prefill(k,v,tokens)
 assert list(s.bt[:3])==[1,2,3]
 lk,lv=logical(s)
 assert lk==list(k)
 assert lv==list(v)
 assert list(s.decode(q))==pytest.approx(ref(q,lk,lv,2,tokens,4),abs=1e-6)

@pytest.mark.parametrize("scores",[
 [0.0]*9,
 [-50.0,-10.0,-1.0,0.0,1.0,10.0,25.0,40.0,50.0],
 [50.0,-50.0,-50.0,-50.0,-50.0,-50.0,-50.0,-50.0,-50.0],
 [-50.0]*9,
])
def test_page_major_softmax_stress_scores_are_stable(artifact,scores):
 r,c=cfg(artifact,pages=4,pt=4,max_tokens=16,h=1,d=4);c=page_major(c)
 tokens=len(scores);q=array("f",[1.0,0.0,0.0,0.0]);scale=1/math.sqrt(4)
 k=array("f");v=array("f")
 for idx,score in enumerate(scores):
  k.extend([score/scale,0.0,0.0,0.0])
  v.extend([float(idx+1),float((idx+1)*2),float((idx%3)-1),float(1-idx)])
 s=PagedKVAttentionSession(c,artifact_root=r);s.prefill(k,v,tokens)
 got=list(s.decode(q));expected=ref(q,*logical(s),1,tokens,4)
 assert all(math.isfinite(x) for x in got)
 assert got==pytest.approx(expected,abs=2e-5,rel=2e-5)

def test_page_major_nonsequential_physical_pages_and_boundary_append(artifact):
 r,c=cfg(artifact);c=page_major(c);manager=KVPageManager(total_pages=4,tokens_per_page=4);storage=PagedKVStorage(total_pages=4,num_kv_heads=2,tokens_per_page=4,head_dim=4,workspace_tokens=16);manager.reserve_prefill("hole",8);s=PagedKVAttentionSession(c,artifact_root=r,request_id="r1",storage=storage,page_manager=manager)
 s.prefill(data(2*7*4,1),data(2*7*4,2),7);assert list(s.bt[:2])==[2,3]
 manager.release("hole")
 s.append(data(8,4),data(8,5));s.append(data(8,6),data(8,7));assert list(s.bt[:3])==[2,3,0]
 q=data(8,8);lk,lv=logical(s);assert list(s.decode(q))==pytest.approx(ref(q,lk,lv,2,9,4),abs=1e-6)
 s.reset();manager.reserve_prefill("hole2",1);s.prefill(data(24,9),data(24,10),3);assert list(s.decode(q))==pytest.approx(ref(q,*logical(s),2,3,4),abs=1e-6)

def test_page_major_invalid_table_fails_closed(artifact):
 r,c=cfg(artifact);s=PagedKVAttentionSession(page_major(c),artifact_root=r);s.prefill(data(16),data(16,2),2);s.page_manager._requests[s.request_id].physical_pages[0]=99
 with pytest.raises(AttentionContractError,match="invalid_live_block_table"):s.decode(data(8))

def test_page_major_decode_uses_per_request_validation_not_global_scan(artifact,monkeypatch):
 r,c=cfg(artifact,pt=4,max_tokens=16);s=PagedKVAttentionSession(page_major(c),artifact_root=r);s.prefill(data(2*9*4,1),data(2*9*4,2),9)
 monkeypatch.setattr(s.page_manager,"validate_invariants",lambda: (_ for _ in ()).throw(AssertionError("global invariant scan should not run during decode")))
 q=data(8,3);assert list(s.decode(q))==pytest.approx(ref(q,*logical(s),2,9,4),abs=1e-6)
 assert s.trace()["decode_validation_strategy"]=="per_request_block_table_only"

def test_page_major_boundary_append_remains_valid_after_validation_hoist(artifact):
 r,c=cfg(artifact,pt=4,max_tokens=16);s=PagedKVAttentionSession(page_major(c),artifact_root=r);s.prefill(data(2*4*4,1),data(2*4*4,2),4);s.append(data(8,9),data(8,10))
 assert list(s.bt[:2])==list(s.page_manager.block_table(s.request_id))
 q=data(8,11);assert list(s.decode(q))==pytest.approx(ref(q,*logical(s),2,5,4),abs=1e-6)

def test_page_major_release_reuse_remains_valid_after_validation_hoist(artifact):
 r,c=cfg(artifact,pt=4,max_tokens=16);manager=KVPageManager(total_pages=4,tokens_per_page=4);storage=PagedKVStorage(total_pages=4,num_kv_heads=2,tokens_per_page=4,head_dim=4,workspace_tokens=16)
 s=PagedKVAttentionSession(page_major(c),artifact_root=r,request_id="a",storage=storage,page_manager=manager);s.prefill(data(2*5*4,1),data(2*5*4,2),5);s.release();assert not manager.has_request("a")
 s2=PagedKVAttentionSession(page_major(c),artifact_root=r,request_id="b",storage=storage,page_manager=manager);s2.prefill(data(2*4,3),data(2*4,4),1)
 assert s2.trace()["block_table"][:1]==list(manager.block_table("b"))

def test_page_major_stale_request_mapping_fails_closed_after_validation_hoist(artifact):
 r,c=cfg(artifact,pt=4,max_tokens=16);s=PagedKVAttentionSession(page_major(c),artifact_root=r);s.prefill(data(2*5*4,1),data(2*5*4,2),5);s.page_manager._requests[s.request_id].physical_pages[0]=99
 with pytest.raises(AttentionContractError,match="invalid_live_block_table"):s.decode(data(8))

def test_page_major_operation_count_reduction():
 base=paged_decode_operation_counts(strategy="token_major_block_translation",heads=4,head_dim=64,valid_tokens=512,page_tokens=16)
 opt=paged_decode_operation_counts(strategy="page_major_cached_page_base",heads=4,head_dim=64,valid_tokens=512,page_tokens=16)
 assert base=={"block_table_lookup_count":133120,"logical_division_count":133120,"logical_modulo_count":133120,"K_page_base_calculation_count":2048,"V_page_base_calculation_count":131072,"token_address_calculation_count":133120}
 assert opt=={"block_table_lookup_count":32,"logical_division_count":0,"logical_modulo_count":0,"K_page_base_calculation_count":128,"V_page_base_calculation_count":128,"token_address_calculation_count":4096}
