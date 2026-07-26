"""Single-request runtime-owned physical FP32 KV pages and int32 block table."""
from __future__ import annotations
import ctypes,hashlib,math
from array import array
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any
from deployment.execution_plan.attention_cpu_adapter import AttentionContractError,_Status
from deployment.execution_plan.kv_page_manager import KVPageManager,KVPageAllocationError,KVPageStateError

@dataclass
class PagedCounters:
 artifact_load_count:int=0;pool_create_count:int=0;pool_allocation_count:int=0;block_table_init_count:int=0
 page_allocation_count:int=0;page_release_count:int=0;prefill_write_count:int=0;append_existing_page_count:int=0;append_new_page_count:int=0
 paged_read_count:int=0;paged_decode_invocation_count:int=0;runtime_layout_reselection_count:int=0;runtime_kernel_reselection_count:int=0;temporary_full_history_materialization_count:int=0
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _prod(xs):
 n=1
 for x in xs:
  if not isinstance(x,int) or isinstance(x,bool) or x<=0:raise AttentionContractError("invalid_dimension")
  if n>((1<<63)-1)//x:raise AttentionContractError("size_overflow")
  n*=x
 return n
def _fp(x,n):
 if not isinstance(x,array) or x.typecode!="f" or len(x)<n:raise AttentionContractError("invalid_fp32_buffer")
 return (ctypes.c_float*len(x)).from_buffer(x)
def _ip(x,n):
 if not isinstance(x,array) or x.typecode!="i" or x.itemsize!=4 or len(x)<n:raise AttentionContractError("invalid_int32_block_table")
 return (ctypes.c_int32*len(x)).from_buffer(x)
def paged_decode_operation_counts(*,strategy:str,heads:int,head_dim:int,valid_tokens:int,page_tokens:int):
 pages=(valid_tokens+page_tokens-1)//page_tokens
 if strategy=="token_major_block_translation":
  translations=heads*valid_tokens*(head_dim+1)
  return {"block_table_lookup_count":translations,"logical_division_count":translations,"logical_modulo_count":translations,"K_page_base_calculation_count":heads*valid_tokens,"V_page_base_calculation_count":heads*head_dim*valid_tokens,"token_address_calculation_count":translations}
 if strategy=="page_major_cached_page_base":
  return {"block_table_lookup_count":pages,"logical_division_count":0,"logical_modulo_count":0,"K_page_base_calculation_count":heads*pages,"V_page_base_calculation_count":heads*pages,"token_address_calculation_count":2*heads*valid_tokens}
 raise ValueError("unknown_paged_decode_strategy")

class PagedKVStorage:
 def __init__(self,*,total_pages:int,num_kv_heads:int,tokens_per_page:int,head_dim:int,dtype:str="fp32",workspace_tokens:int=1):
  if dtype!="fp32":raise AttentionContractError("unsupported_paged_kv_storage_dtype")
  self.total_pages=total_pages;self.num_kv_heads=num_kv_heads;self.tokens_per_page=tokens_per_page;self.head_dim=head_dim;self.dtype=dtype
  n=_prod([total_pages,num_kv_heads,tokens_per_page,head_dim]);self._k_pages=array("f",[0])*n;self._v_pages=array("f",[0])*n
  self.workspace=array("f",[0])*max(1,workspace_tokens)
 @property
 def k_pages(self):return self._k_pages
 @property
 def v_pages(self):return self._v_pages

class PagedKVAttentionSession:
 def __init__(self,c:dict[str,Any],*,artifact_root:Path,request_id:str="__default__",storage:PagedKVStorage|None=None,page_manager:KVPageManager|None=None):
  self.c=dict(c);self.count=PagedCounters();self.state="CREATED";self.request_id=request_id;self._validate(artifact_root)
  self.executed_candidate_id=self.c["kv_candidate_id"];self.executed_kernel_id=self.c["paged_attention_kernel_id"];self.executed_strategy=self.c.get("implementation_strategy","token_major_block_translation")
  path=artifact_root/c["pool_artifact_ref"];self.lib=ctypes.CDLL(str(path));self.count.artifact_load_count=1;self.lib.hir_paged_kv_artifact_version.restype=ctypes.c_char_p
  if self.lib.hir_paged_kv_artifact_version().decode()!="hir.paged_kv.v1":raise AttentionContractError("artifact_version_mismatch")
  p,h,t,d=(c[x] for x in ("num_physical_pages","num_kv_heads","page_tokens","head_dim"))
  if (storage is None)!=(page_manager is None):raise AttentionContractError("storage_and_page_manager_must_be_paired")
  if storage is None:
   storage=PagedKVStorage(total_pages=p,num_kv_heads=h,tokens_per_page=t,head_dim=d,dtype=c["dtype"],workspace_tokens=c["maximum_logical_tokens"])
   page_manager=KVPageManager(total_pages=p,tokens_per_page=t);self.count.pool_allocation_count=1
  self.storage=storage;self.page_manager=page_manager
  if storage.total_pages!=p or page_manager.total_pages!=p:raise AttentionContractError("storage_manager_page_count_mismatch")
  if storage.tokens_per_page!=t or page_manager.tokens_per_page!=t:raise AttentionContractError("storage_manager_tokens_per_page_mismatch")
  if storage.num_kv_heads!=h or storage.head_dim!=d or storage.dtype!=c["dtype"]:raise AttentionContractError("storage_shape_mismatch")
  self.physical_page_cache=array("i",[c["invalid_page_sentinel"]])*c["block_table_length"];self.out=array("f",[0])*_prod([h,d]);self.ws=array("f",[0])*c["maximum_logical_tokens"];self.count.pool_create_count=1;self._bind();self.count.block_table_init_count=1
 def _bind(self):
  P=ctypes.POINTER(ctypes.c_float);I32=ctypes.POINTER(ctypes.c_int32);S=ctypes.c_size_t;I=ctypes.c_int64;J=ctypes.c_int32
  self.init=getattr(self.lib,self.c["pool_create_entry_point"]);self.init.restype=_Status;self.init.argtypes=[P,S,P,S,I32,S,I,I,I,I,J]
  self.native_reset=getattr(self.lib,self.c["reset_entry_point"]);self.native_reset.restype=_Status;self.native_reset.argtypes=[P,S,P,S,I32,S,I,I,I,I,J]
  self.write=getattr(self.lib,self.c["prefill_write_entry_point"]);self.write.restype=_Status;self.write.argtypes=[P,S,P,S,I32,S,P,S,P,S,I,I,I,I,I,J]
  self.app=getattr(self.lib,self.c["append_entry_point"]);self.app.restype=_Status;self.app.argtypes=[P,S,P,S,I32,S,P,S,P,S,I,I,I,I,I,J]
  token_entry=self.c.get("paged_attention_entry_point","hir_cpu_attention_decode_paged_kv_fp32")
  if self.c["paged_attention_kernel_id"]=="cpu_attention_decode_paged_kv_page_major_fp32":token_entry=self.c.get("paged_attention_entry_point","hir_cpu_attention_decode_paged_kv_page_major_fp32")
  self.dec=getattr(self.lib,token_entry);self.dec.restype=_Status
  if token_entry=="hir_cpu_attention_decode_paged_kv_page_major_fp32":self.dec.argtypes=[P,S,P,S,P,S,I32,S,I32,S,P,S,P,S,I,I,I,I,I,J]
  else:self.dec.argtypes=[P,S,P,S,P,S,I32,S,P,S,P,S,I,I,I,I,I,J]
 def _validate(self,root):
  c=self.c;required={"kv_layout_kind":"paged_phd_contiguous","dtype":"fp32","block_table_element_type":"int32","pool_artifact_version":"hir.paged_kv.v1","pool_create_entry_point":"hir_paged_kv_initialize","prefill_write_entry_point":"hir_paged_kv_prefill_write","append_entry_point":"hir_paged_kv_append","view_binding":"direct_int32_block_table_translation","reset_entry_point":"hir_paged_kv_reset","release_entry_point":"runtime_owned_pool_release","contiguous_fallback_identity":"cpu_contiguous_kv_fp32_v1"}
  for k,v in required.items():
   if c.get(k)!=v:raise AttentionContractError("contract_mismatch:"+k)
  variants={
   "cpu_paged_kv_fp32_v1":("cpu_attention_decode_paged_kv_fp32","hir_cpu_attention_decode_paged_kv_fp32","token_major_block_translation"),
   "cpu_paged_kv_fp32_token_major_v1":("cpu_attention_decode_paged_kv_fp32","hir_cpu_attention_decode_paged_kv_fp32","token_major_block_translation"),
   "cpu_paged_kv_fp32_page_major_v1":("cpu_attention_decode_paged_kv_page_major_fp32","hir_cpu_attention_decode_paged_kv_page_major_fp32","page_major_cached_page_base"),
  }
  candidate=c.get("kv_candidate_id");kernel,entry,strategy=variants.get(candidate,(None,None,None))
  if kernel is None or c.get("paged_attention_kernel_id")!=kernel or c.get("paged_attention_entry_point",entry)!=entry or c.get("implementation_strategy",strategy)!=strategy:raise AttentionContractError("candidate_contract_mismatch:kernel_or_strategy")
  p,h,t,d,m=(c.get(x) for x in ("num_physical_pages","num_kv_heads","page_tokens","head_dim","maximum_logical_tokens"));blocks=(m+t-1)//t
  if c.get("batch")!=1 or c.get("maximum_logical_blocks")!=blocks or c.get("block_table_length",0)<blocks:raise AttentionContractError("capacity_or_block_table_mismatch")
  bpt=_prod([h,d,8]);one=_prod([h,t,d,4]);st=[h*t*d,t*d,d,1]
  if c.get("bytes_per_token")!=bpt or c.get("bytes_per_k_page")!=one or c.get("bytes_per_v_page")!=one or c.get("bytes_per_combined_page")!=2*one or c.get("total_pool_bytes")!=2*p*one or c.get("k_page_strides")!=st or c.get("v_page_strides")!=st:raise AttentionContractError("page_formula_mismatch")
  if c.get("invalid_page_sentinel")!=-1 or c.get("alignment_bytes")!=4 or c.get("runtime_no_layout_redecision") is not True or c.get("runtime_no_kernel_redecision") is not True:raise AttentionContractError("sentinel_alignment_or_redecision_mismatch")
  path=root/c.get("pool_artifact_ref","")
  if not path.is_file() or _sha(path)!=c.get("pool_artifact_sha256"):raise AttentionContractError("artifact_hash_mismatch")
 @property
 def k(self):return self.storage.k_pages
 @property
 def v(self):return self.storage.v_pages
 @property
 def valid_tokens(self):return self.page_manager.valid_token_count(self.request_id) if self.page_manager.has_request(self.request_id) else 0
 @property
 def owned(self):return list(self.page_manager.block_table(self.request_id)) if self.page_manager.has_request(self.request_id) else []
 @property
 def free(self):return list(self.page_manager._free_pages)
 @property
 def bt(self):return self._block_table_array()
 def _base(self):return (_fp(self.k,len(self.k)),len(self.k),_fp(self.v,len(self.v)),len(self.v))
 @staticmethod
 def _call(fn,args):
  s=fn(*args)
  if s.code:raise AttentionContractError((s.message or b"native_error").decode())
 def _block_table_array(self,block_table=None):
  if block_table is None:
   block_table=self.page_manager.block_table(self.request_id) if self.page_manager.has_request(self.request_id) else ()
  if len(block_table)>self.c["block_table_length"]:raise AttentionContractError("block_table_overflow")
  bt=array("i",[self.c["invalid_page_sentinel"]])*self.c["block_table_length"]
  for i,page in enumerate(block_table):bt[i]=page
  return bt
 def _validate_live(self):
  needed=(self.valid_tokens+self.c["page_tokens"]-1)//self.c["page_tokens"];refs=list(self.page_manager.block_table(self.request_id))
  if len(refs)!=needed or any(x<0 or x>=self.c["num_physical_pages"] for x in refs) or len(set(refs))!=len(refs):raise AttentionContractError("invalid_live_block_table")
 def prefill(self,k,v,tokens):
  if self.state!="CREATED":raise AttentionContractError("prefill_requires_created")
  if tokens<=0 or tokens>self.c["maximum_logical_tokens"]:raise AttentionContractError("prefill_capacity_error")
  try:block_table=self.page_manager.reserve_prefill(self.request_id,tokens)
  except KVPageAllocationError as e:raise AttentionContractError("out_of_physical_pages") from e
  except KVPageStateError as e:raise AttentionContractError(str(e)) from e
  bt=self._block_table_array(block_table);n=_prod([self.c["num_kv_heads"],tokens,self.c["head_dim"]])
  try:self._call(self.write,self._base()+(_ip(bt,len(bt)),len(bt),_fp(k,n),len(k),_fp(v,n),len(v),tokens,self.c["num_physical_pages"],self.c["num_kv_heads"],self.c["page_tokens"],self.c["head_dim"],-1))
  except Exception:
   if self.page_manager.has_request(self.request_id):self.page_manager.release(self.request_id)
   raise
  self.state="READY";self.count.prefill_write_count+=1;self.count.page_allocation_count+=len(block_table)
 def append(self,k,v):
  if self.state!="READY":raise AttentionContractError("append_before_prefill")
  if self.valid_tokens>=self.c["maximum_logical_tokens"]:raise AttentionContractError("append_beyond_logical_capacity")
  logical_index=self.valid_tokens
  try:reservation=self.page_manager.begin_append_token(self.request_id,logical_index)
  except KVPageAllocationError as e:raise AttentionContractError("out_of_physical_pages") from e
  except KVPageStateError as e:raise AttentionContractError(str(e)) from e
  bt=self._block_table_array(reservation.block_table);n=_prod([self.c["num_kv_heads"],self.c["head_dim"]])
  try:self._call(self.app,self._base()+(_ip(bt,len(bt)),len(bt),_fp(k,n),len(k),_fp(v,n),len(v),logical_index,self.c["num_physical_pages"],self.c["num_kv_heads"],self.c["page_tokens"],self.c["head_dim"],-1))
  except Exception:
   self.page_manager.rollback_append_token(reservation)
   raise
  self.page_manager.commit_append_token(reservation)
  if reservation.allocated_new_page:self.count.append_new_page_count+=1;self.count.page_allocation_count+=1
  else:self.count.append_existing_page_count+=1
 def decode(self,q):
  if self.state!="READY":raise AttentionContractError("decode_before_prefill")
  self._validate_live();n=_prod([self.c["num_kv_heads"],self.c["head_dim"]]);bt=self._block_table_array();args=(_fp(q,n),len(q))+self._base()+(_ip(bt,len(bt)),len(bt));
  if self.executed_strategy=="page_major_cached_page_base":args+=(_ip(self.physical_page_cache,len(self.physical_page_cache)),len(self.physical_page_cache))
  args+=(_fp(self.out,n),len(self.out),_fp(self.ws,len(self.ws)),len(self.ws),self.valid_tokens,self.c["num_physical_pages"],self.c["num_kv_heads"],self.c["page_tokens"],self.c["head_dim"],-1);self._call(self.dec,args);self.count.paged_read_count+=1;self.count.paged_decode_invocation_count+=1;return array("f",self.out)
 def reset(self):
  if self.state=="RELEASED":raise AttentionContractError("use_after_release")
  if self.page_manager.has_request(self.request_id):
   self.count.page_release_count+=len(self.page_manager.block_table(self.request_id));self.page_manager.release(self.request_id)
  self.state="CREATED";self.count.block_table_init_count+=1
 def release(self):
  if self.state=="RELEASED":raise AttentionContractError("double_release")
  self.reset();self.state="RELEASED"
 def fragmentation(self):
  slots=len(self.owned)*self.c["page_tokens"];return {"logical_tokens":self.valid_tokens,"allocated_pages":len(self.owned),"allocated_token_slots":slots,"unused_token_slots":slots-self.valid_tokens,"internal_fragmentation_ratio":(slots-self.valid_tokens)/slots if slots else 0,"physical_pool_utilization":len(self.owned)/self.c["num_physical_pages"]}
 def trace(self):return {**asdict(self.count),"valid_tokens":self.valid_tokens,"owned_pages":list(self.owned),"block_table":list(self.bt),"decode_validation_strategy":"per_request_block_table_only","state":self.state,"runtime_executed_candidate_id":self.executed_candidate_id,"runtime_executed_kernel_id":self.executed_kernel_id,"runtime_executed_strategy":self.executed_strategy}
