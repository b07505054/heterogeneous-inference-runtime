"""Correctness, latency, allocation, and selector evidence for native fused attention."""
from __future__ import annotations
import argparse, json, math, statistics, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT))
from deployment.native_fused_attention import ABI_VERSION, NativeFusedAttentionLibrary, sha256
from deployment.attention_runtime import _attention_chunk, _fused_online_attention_chunk
from deployment.attention_planner import AttentionWorkload, select_attention_plan

def pct(xs,p): return sorted(xs)[min(len(xs)-1,math.ceil(p*len(xs))-1)]
def timed(fn,warmup=2,runs=7):
    for _ in range(warmup): fn()
    xs=[]
    for _ in range(runs):
        t=time.perf_counter_ns();fn();xs.append((time.perf_counter_ns()-t)/1e6)
    return {"median_ms":statistics.median(xs),"p95_ms":pct(xs,.95),
            "variance_ms2":statistics.pvariance(xs),"samples_ms":xs}
def tensors(q,k,seed):
    g=torch.Generator().manual_seed(seed)
    return (torch.randn(1,14,q,64,generator=g),
            torch.randn(1,2,k,64,generator=g),
            torch.randn(1,2,k,64,generator=g))
def ref(q,k,v):
    return torch.nn.functional.scaled_dot_product_attention(
        q,k.repeat_interleave(7,1),v.repeat_interleave(7,1),
        is_causal=False,attn_mask=torch.tril(torch.ones(q.shape[2],k.shape[2],
        dtype=torch.bool),diagonal=k.shape[2]-q.shape[2]))
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--runs",type=int,default=7);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    artifact=a.output_dir/"libfused_online_attention.so"
    lib=NativeFusedAttentionLibrary(artifact,sha256(artifact),ABI_VERSION)
    decode=[1,4,7,16,31,32,33,63,64,65,127,128,257,511,1024]
    prefill=[1,4,7,16,31,32,33,63,64,65,127,128,256]
    uneven=[(3,17),(11,37),(37,63),(73,127),(129,263)]
    rows=[];worst_abs=worst_rel=0.;mismatches=nans=infs=0
    for idx,(q,k) in enumerate([(1,x) for x in decode]+[(x,x) for x in prefill]+uneven):
      Q,K,V=tensors(q,k,1000+idx);expected=ref(Q,K,V)
      for impl in ("native_scalar","native_avx2"):
        out,mem=lib.run(impl,Q,K,V,64**-.5,1,32,total_query_heads=14)
        diff=(out-expected).abs();absmax=float(diff.max());relmax=float((diff/(expected.abs()+1e-6)).max())
        bad=int((diff>2e-5).sum());mismatches+=bad;nans+=int(torch.isnan(out).sum());infs+=int(torch.isinf(out).sum())
        worst_abs=max(worst_abs,absmax);worst_rel=max(worst_rel,relmax)
        rows.append({"q":q,"k":k,"implementation":impl,"max_abs_error":absmax,
          "max_relative_error":relmax,"mismatch_count":bad,"nan_count":int(torch.isnan(out).sum()),
          "inf_count":int(torch.isinf(out).sum()),"memory":mem})
    correctness={"rows":rows,"row_count":len(rows),"max_abs_error":worst_abs,
      "max_relative_error":worst_rel,"mismatch_count":mismatches,"nan_count":nans,"inf_count":infs}
    (a.output_dir/"native_correctness.json").write_text(json.dumps(correctness,indent=2))
    workloads=[("decode",1,k) for k in [4,8,16,32,64,128,256,512,1024,2048]]
    workloads += [("prefill",q,q) for q in [4,8,16,32,64,128,256]]
    bench=[]
    torch.set_num_threads(1)
    for idx,(phase,q,k) in enumerate(workloads):
      Q,K,V=tensors(q,k,2000+idx);Kr=K.repeat_interleave(7,1);Vr=V.repeat_interleave(7,1)
      mask=torch.zeros(q,k);mask.masked_fill_(~torch.tril(torch.ones(q,k,dtype=torch.bool),diagonal=k-q),-torch.inf)
      cases={
       "dense":lambda:_attention_chunk(Q,Kr,Vr,mask,64**-.5),
       "python_fused":lambda:_fused_online_attention_chunk(Q,Kr,Vr,mask,64**-.5,1,32),
       "native_scalar":lambda:lib.run("native_scalar",Q,K,V,64**-.5,1,32,total_query_heads=14),
       "native_avx2":lambda:lib.run("native_avx2",Q,K,V,64**-.5,1,32,total_query_heads=14)}
      for name,fn in cases.items():
       timing=timed(fn,runs=a.runs)
       result=fn();memory=result[-1] if name in {"dense","python_fused"} else result[1]
       bench.append({"phase":phase,"q":q,"k":k,"candidate":name,**timing,"temporary_bytes":memory["temporary_bytes"],
         "total_temporary_bytes_including_gqa":memory["temporary_bytes"]+(0 if name.startswith("native") else 2*Kr.numel()*4)})
    (a.output_dir/"python_vs_native_benchmark.json").write_text(json.dumps({"rows":bench},indent=2))
    held=[]
    for phase,q,k in [("decode",1,x) for x in [24,48,96,192,384,768,1536]]+[("prefill",x,x) for x in [11,24,48,73,96,192,384]]:
      w=AttentionWorkload(phase=phase,batch=1,query_len=q,context_len=k,query_heads=14,kv_heads=2,head_dim=64)
      plan,trace=select_attention_plan(w);held.append({"phase":phase,"q":q,"k":k,
        "selected_candidate":plan["native_kernel_id"],"algorithm":plan["algorithm"],
        "implementation":plan["implementation"],"legal_candidate_count":trace["legal_candidate_count"]})
    (a.output_dir/"held_out_selector_evaluation.json").write_text(json.dumps({"rows":held},indent=2))
    summary={"correctness":{k:v for k,v in correctness.items() if k!="rows"},"avx2":lib.has_avx2,
      "artifact_sha256":sha256(artifact),"abi_version":ABI_VERSION,"benchmark_rows":len(bench)}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
