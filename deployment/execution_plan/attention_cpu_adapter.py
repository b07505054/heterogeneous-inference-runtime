"""Fail-closed persistent runner for compiler-selected FP32 CPU attention."""
from __future__ import annotations
import ctypes, hashlib, platform
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = "hir.cpu_attention.v1"
LAYOUT = "bhsd_contiguous"
KERNELS = {
    "prefill": ("cpu_attention_prefill_fp32", "hir_cpu_attention_prefill_fp32"),
    "decode": ("cpu_attention_decode_fp32", "hir_cpu_attention_decode_fp32"),
}

class AttentionContractError(ValueError): pass
class _Status(ctypes.Structure): _fields_=[("code",ctypes.c_int32),("message",ctypes.c_char_p)]

@dataclass
class AttentionCounters:
    artifact_load_count:int=0; worker_start_count:int=0; buffer_allocation_count:int=0
    prefill_invocation_count:int=0; decode_invocation_count:int=0; runtime_reselection_count:int=0

def _sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def _checked_product(values:list[int])->int:
    out=1
    for value in values:
        if not isinstance(value,int) or isinstance(value,bool) or value<=0: raise AttentionContractError("invalid_dimension")
        if out > ((1<<63)-1)//value: raise AttentionContractError("tensor_size_overflow")
        out*=value
    return out

class PersistentAttentionRunner:
    def __init__(self, contract:dict[str,Any], *, artifact_root:Path):
        self.contract=dict(contract); self.counters=AttentionCounters(worker_start_count=1)
        self._validate_static(artifact_root)
        path=(artifact_root/self.contract["artifact_ref"]).resolve()
        self._lib=ctypes.CDLL(str(path));self.counters.artifact_load_count=1
        self._lib.hir_attention_artifact_version.restype=ctypes.c_char_p
        if self._lib.hir_attention_artifact_version().decode()!=ARTIFACT_VERSION: raise AttentionContractError("artifact_version_mismatch")
        self._fn=getattr(self._lib,self.contract["entry_point"]);self._fn.restype=_Status
        self._fn.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t]*4+[ctypes.POINTER(ctypes.c_float),ctypes.c_size_t,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64]
        n=_checked_product([self.contract["batch"],self.contract["num_query_heads"],self.contract["query_length"],self.contract["head_dim"]])
        self._output=array("f",[0.0])*n;self._workspace=array("f",[0.0])*self.contract["context_length"]
        self.counters.buffer_allocation_count=2

    def _validate_static(self,root:Path)->None:
        c=self.contract; phase=c.get("phase"); expected=KERNELS.get(phase)
        if not expected: raise AttentionContractError("unsupported_phase")
        if (c.get("kernel_id"),c.get("entry_point"))!=expected: raise AttentionContractError("kernel_or_entry_point_mismatch")
        if c.get("candidate_id")!=expected[0]: raise AttentionContractError("candidate_identity_mismatch")
        if c.get("backend")!="portable_cpu" or c.get("execution_unit")!="portable_cpu_attention": raise AttentionContractError("backend_mismatch")
        if c.get("dtype")!="fp32":raise AttentionContractError("dtype_mismatch")
        if c.get("input_layout")!=LAYOUT or c.get("output_layout")!=LAYOUT:raise AttentionContractError("layout_mismatch")
        if c.get("causal") is not True:raise AttentionContractError("causal_required")
        if c.get("num_query_heads")!=c.get("num_kv_heads"):raise AttentionContractError("gqa_mqa_unsupported")
        _checked_product([c[k] for k in ("batch","num_query_heads","query_length","context_length","head_dim")])
        if (phase=="prefill" and (c["query_length"]<=1 or c["query_length"]!=c["context_length"])) or (phase=="decode" and c["query_length"]!=1):raise AttentionContractError("query_length_mismatch")
        if c.get("workspace_bytes")!=c["context_length"]*4:raise AttentionContractError("workspace_size_mismatch")
        if c.get("alignment_bytes")!=ctypes.sizeof(ctypes.c_float):raise AttentionContractError("alignment_mismatch")
        if c.get("artifact_version")!=ARTIFACT_VERSION:raise AttentionContractError("artifact_version_mismatch")
        if c.get("runtime_no_redecision") is not True:raise AttentionContractError("runtime_redecision_forbidden")
        path=root/c.get("artifact_ref","")
        if not path.is_file():raise AttentionContractError("missing_attention_artifact")
        if _sha(path)!=c.get("artifact_sha256"):raise AttentionContractError("artifact_hash_mismatch")
        if c.get("required_isa") not in ("scalar_fp32",):raise AttentionContractError("required_isa_unavailable")

    @staticmethod
    def _ptr(x:array, expected:int):
        if not isinstance(x,array) or x.typecode!="f":raise AttentionContractError("input_must_be_contiguous_float32_array")
        if len(x)<expected:raise AttentionContractError("insufficient_input_buffer")
        view=(ctypes.c_float*len(x)).from_buffer(x)
        if ctypes.addressof(view)%ctypes.sizeof(ctypes.c_float):raise AttentionContractError("input_alignment_mismatch")
        return view

    def invoke(self,q:array,k:array,v:array,*,output_capacity:int|None=None,workspace_capacity:int|None=None)->array:
        c=self.contract;b,h,ql,cl,d=(c[x] for x in ("batch","num_query_heads","query_length","context_length","head_dim"))
        qn=_checked_product([b,h,ql,d]);kn=_checked_product([b,h,cl,d])
        if output_capacity is not None and output_capacity<qn:raise AttentionContractError("insufficient_output_buffer")
        if workspace_capacity is not None and workspace_capacity<cl:raise AttentionContractError("insufficient_workspace")
        qp=self._ptr(q,qn);kp=self._ptr(k,kn);vp=self._ptr(v,kn)
        op=(ctypes.c_float*len(self._output)).from_buffer(self._output);wp=(ctypes.c_float*len(self._workspace)).from_buffer(self._workspace)
        status=self._fn(qp,len(q),kp,len(k),vp,len(v),op,len(self._output),wp,len(self._workspace),b,h,ql,cl,d)
        if status.code:raise AttentionContractError((status.message or b"native_error").decode())
        if c["phase"]=="prefill":self.counters.prefill_invocation_count+=1
        else:self.counters.decode_invocation_count+=1
        return array("f",self._output)

    def trace(self)->dict[str,Any]:
        return {**self.counters.__dict__,"runtime_no_redecision":self.counters.runtime_reselection_count==0,
                "kernel_id":self.contract["kernel_id"],"phase":self.contract["phase"],"host":platform.machine()}
