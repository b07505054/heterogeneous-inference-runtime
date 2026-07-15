"""Runtime-owned contiguous FP32 KV cache for compiler-selected CPU attention."""
from __future__ import annotations
import ctypes, hashlib
from array import array
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from deployment.execution_plan.attention_cpu_adapter import AttentionContractError, _Status

KV_VERSION = "hir.contiguous_kv.v1"

@dataclass
class KVCounters:
    artifact_load_count:int=0; cache_create_count:int=0; cache_allocation_count:int=0
    prefill_write_count:int=0; decode_append_count:int=0; cache_read_count:int=0
    prefill_attention_invocation_count:int=0; decode_attention_invocation_count:int=0
    cache_reset_count:int=0; cache_release_count:int=0
    runtime_layout_reselection_count:int=0; runtime_kernel_reselection_count:int=0

def _sha(path:Path)->str:
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def _product(values:list[int])->int:
    n=1
    for value in values:
        if not isinstance(value,int) or isinstance(value,bool) or value<=0: raise AttentionContractError("invalid_dimension")
        if n>((1<<63)-1)//value: raise AttentionContractError("tensor_size_overflow")
        n*=value
    return n

def _ptr(value:array, minimum:int):
    if not isinstance(value,array) or value.typecode!="f": raise AttentionContractError("fp32_contiguous_array_required")
    if len(value)<minimum: raise AttentionContractError("insufficient_input_buffer")
    return (ctypes.c_float*len(value)).from_buffer(value)

class ContiguousKVAttentionSession:
    """One allocation and one native artifact serving prefill plus growing decode."""
    def __init__(self, kv:dict[str,Any], attention:dict[str,dict[str,Any]], *, artifact_root:Path):
        self.kv=dict(kv); self.attention={k:dict(v) for k,v in attention.items()}
        self.counters=KVCounters(); self.valid_tokens=0; self.state="CREATED"
        self._validate_contracts(artifact_root)
        path=artifact_root/self.kv["kv_artifact_ref"]; self._lib=ctypes.CDLL(str(path)); self.counters.artifact_load_count=1
        self._lib.hir_contiguous_kv_artifact_version.restype=ctypes.c_char_p
        if self._lib.hir_contiguous_kv_artifact_version().decode()!=KV_VERSION: raise AttentionContractError("kv_artifact_version_mismatch")
        b,h,c,d=(self.kv[x] for x in ("batch","num_kv_heads","capacity_tokens","head_dim")); n=_product([b,h,c,d])
        self.k_cache=array("f",[0.0])*n; self.v_cache=array("f",[0.0])*n
        self._workspace=array("f",[0.0])*c; self._output=array("f",[0.0])*_product([b,h,d])
        self.counters.cache_create_count=1; self.counters.cache_allocation_count=1
        self._bind(); self._call(self._init,self._cache_args())

    def _bind(self):
        P=ctypes.POINTER(ctypes.c_float); S=ctypes.c_size_t; I=ctypes.c_int64
        self._init=self._lib.hir_contiguous_kv_initialize; self._init.restype=_Status; self._init.argtypes=[P,S,P,S,I,I,I,I]
        self._write=self._lib.hir_contiguous_kv_prefill_write; self._write.restype=_Status; self._write.argtypes=[P,S,P,S,P,S,P,S,I,I,I,I,I]
        self._append=self._lib.hir_contiguous_kv_append; self._append.restype=_Status; self._append.argtypes=[P,S,P,S,P,S,P,S,I,I,I,I,I]
        self._reset=self._lib.hir_contiguous_kv_reset; self._reset.restype=_Status; self._reset.argtypes=[P,S,P,S,I,I,I,I]
        self._prefill=self._lib.hir_cpu_attention_prefill_fp32; self._prefill.restype=_Status; self._prefill.argtypes=[P,S,P,S,P,S,P,S,P,S,I,I,I,I,I]
        self._decode=self._lib.hir_cpu_attention_decode_contiguous_kv_fp32; self._decode.restype=_Status; self._decode.argtypes=[P,S,P,S,P,S,P,S,P,S,I,I,I,I,I]

    def _validate_contracts(self,root:Path):
        k=self.kv
        required={"kv_candidate_id":"cpu_contiguous_kv_fp32_v1","kv_dtype":"fp32","kv_layout":"bhcd_contiguous","kv_artifact_version":KV_VERSION,"create_entry_point":"hir_contiguous_kv_initialize","prefill_write_entry_point":"hir_contiguous_kv_prefill_write","decode_append_entry_point":"hir_contiguous_kv_append","view_binding":"direct_contiguous_pointer_valid_prefix","reset_entry_point":"hir_contiguous_kv_reset","compatible_prefill_kernel_id":"cpu_attention_prefill_fp32","compatible_decode_kernel_id":"cpu_attention_decode_fp32"}
        for name,value in required.items():
            if k.get(name)!=value: raise AttentionContractError(f"kv_contract_mismatch:{name}")
        b,h,c,d=(k.get(x) for x in ("batch","num_kv_heads","capacity_tokens","head_dim")); elements=_product([b,h,c,d]); bpt=_product([b,h,d,8])
        if k.get("bytes_per_token")!=bpt or k.get("k_cache_bytes")!=elements*4 or k.get("v_cache_bytes")!=elements*4 or k.get("total_cache_bytes")!=elements*8: raise AttentionContractError("kv_byte_contract_mismatch")
        strides=[h*c*d,c*d,d,1]
        if k.get("k_strides")!=strides or k.get("v_strides")!=strides: raise AttentionContractError("kv_stride_mismatch")
        if k.get("alignment_bytes")!=4 or k.get("runtime_no_layout_redecision") is not True: raise AttentionContractError("kv_alignment_or_redecision_mismatch")
        path=root/k.get("kv_artifact_ref","")
        if not path.is_file(): raise AttentionContractError("missing_kv_artifact")
        if _sha(path)!=k.get("kv_artifact_sha256"): raise AttentionContractError("kv_artifact_hash_mismatch")
        for phase,kernel in (("prefill","cpu_attention_prefill_fp32"),("decode","cpu_attention_decode_fp32")):
            a=self.attention.get(phase,{})
            if a.get("kernel_id")!=kernel or a.get("dtype")!="fp32" or a.get("input_layout")!="bhsd_contiguous" or a.get("runtime_no_redecision") is not True: raise AttentionContractError("attention_kv_kernel_incompatible")

    def _cache_args(self):
        k=self.kv
        return (_ptr(self.k_cache,len(self.k_cache)),len(self.k_cache),_ptr(self.v_cache,len(self.v_cache)),len(self.v_cache),k["batch"],k["num_kv_heads"],k["capacity_tokens"],k["head_dim"])
    @staticmethod
    def _call(fn,args):
        status=fn(*args)
        if status.code: raise AttentionContractError((status.message or b"native_error").decode())

    def prefill(self,q:array,k:array,v:array)->array:
        self.prefill_write(k,v)
        return self.prefill_attention(q,k,v)

    def prefill_write(self,k:array,v:array)->None:
        if self.state!="CREATED" or self.valid_tokens: raise AttentionContractError("prefill_requires_empty_cache")
        c=self.attention["prefill"]; tokens=c["query_length"]
        if tokens>self.kv["capacity_tokens"]: raise AttentionContractError("prefill_exceeds_capacity")
        n=_product([self.kv["batch"],self.kv["num_kv_heads"],tokens,self.kv["head_dim"]]);kp=_ptr(k,n);vp=_ptr(v,n)
        self._call(self._write,(*self._cache_args()[:4],kp,len(k),vp,len(v),self.kv["batch"],self.kv["num_kv_heads"],tokens,self.kv["capacity_tokens"],self.kv["head_dim"]))
        self.valid_tokens=tokens; self.state="READY"; self.counters.prefill_write_count+=1

    def prefill_attention(self,q:array,k:array,v:array)->array:
        if self.state!="READY": raise AttentionContractError("cache_not_ready")
        tokens=self.valid_tokens;n=_product([self.kv["batch"],self.kv["num_kv_heads"],tokens,self.kv["head_dim"]]);qp=_ptr(q,n);kp=_ptr(k,n);vp=_ptr(v,n);self.counters.cache_read_count+=1
        out=array("f",[0.0])*n; ws=array("f",[0.0])*tokens
        self._call(self._prefill,(qp,len(q),kp,len(k),vp,len(v),_ptr(out,n),len(out),_ptr(ws,tokens),len(ws),self.kv["batch"],self.kv["num_kv_heads"],tokens,tokens,self.kv["head_dim"]))
        self.counters.prefill_attention_invocation_count+=1; return out

    def decode_append_then_attend(self,q:array,k_token:array,v_token:array)->array:
        self.append(k_token,v_token)
        return self.decode(q)

    def append(self,k_token:array,v_token:array)->None:
        if self.state!="READY": raise AttentionContractError("decode_before_prefill")
        if self.valid_tokens>=self.kv["capacity_tokens"]: raise AttentionContractError("append_out_of_capacity")
        n=_product([self.kv["batch"],self.kv["num_kv_heads"],self.kv["head_dim"]]);kp=_ptr(k_token,n);vp=_ptr(v_token,n)
        self._call(self._append,(*self._cache_args()[:4],kp,len(k_token),vp,len(v_token),self.kv["batch"],self.kv["num_kv_heads"],self.valid_tokens,self.kv["capacity_tokens"],self.kv["head_dim"]))
        self.valid_tokens+=1;self.counters.decode_append_count+=1

    def decode(self,q:array)->array:
        if self.state!="READY" or self.valid_tokens<=0: raise AttentionContractError("decode_before_prefill")
        n=_product([self.kv["batch"],self.kv["num_kv_heads"],self.kv["head_dim"]]);qp=_ptr(q,n);self.counters.cache_read_count+=1
        self._call(self._decode,(qp,len(q),_ptr(self.k_cache,len(self.k_cache)),len(self.k_cache),_ptr(self.v_cache,len(self.v_cache)),len(self.v_cache),_ptr(self._output,len(self._output)),len(self._output),_ptr(self._workspace,len(self._workspace)),len(self._workspace),self.kv["batch"],self.kv["num_kv_heads"],self.valid_tokens,self.kv["capacity_tokens"],self.kv["head_dim"]))
        self.counters.decode_attention_invocation_count+=1;return array("f",self._output)

    def view(self,tokens:int|None=None)->tuple[array,array]:
        if self.state!="READY": raise AttentionContractError("cache_not_ready")
        tokens=self.valid_tokens if tokens is None else tokens
        if tokens<0 or tokens>self.valid_tokens: raise AttentionContractError("read_beyond_valid_tokens")
        b,h,c,d=(self.kv[x] for x in ("batch","num_kv_heads","capacity_tokens","head_dim")); ko=array("f");vo=array("f")
        for bi in range(b):
            for hi in range(h):
                base=(bi*h+hi)*c*d;ko.extend(self.k_cache[base:base+tokens*d]);vo.extend(self.v_cache[base:base+tokens*d])
        self.counters.cache_read_count+=1;return ko,vo

    def reset(self):
        if self.state=="RELEASED": raise AttentionContractError("use_after_release")
        self._call(self._reset,self._cache_args());self.valid_tokens=0;self.state="CREATED";self.counters.cache_reset_count+=1
    def release(self):
        if self.state=="RELEASED": raise AttentionContractError("double_release")
        self.k_cache=array("f");self.v_cache=array("f");self._workspace=array("f");self._output=array("f");self.valid_tokens=0;self.state="RELEASED";self.counters.cache_release_count+=1
    def trace(self): return {**asdict(self.counters),"valid_tokens":self.valid_tokens,"state":self.state,"runtime_no_layout_redecision":self.counters.runtime_layout_reselection_count==0,"runtime_no_kernel_redecision":self.counters.runtime_kernel_reselection_count==0}
