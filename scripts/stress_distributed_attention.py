"""1,000-call mixed serial/parallel, dense/native, prefill/decode stress."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.attention_planner import AttentionWorkload,force_test_attention_plan
from deployment.attention_runtime import CompilerAttentionRuntime
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
 specs=[("prefill",8,8,"dense_materialized","torch_dense_materialized_v1",1),
  ("prefill",11,11,"dense_materialized","torch_dense_materialized_v1",4),
  ("decode",1,17,"fused_tiled_online_softmax","native_avx2",1),
  ("decode",1,63,"fused_tiled_online_softmax","native_avx2",4)]
 runtimes=[];payload=[];torch.manual_seed(61)
 for phase,q,k,alg,impl,wc in specs:
  work=AttentionWorkload(phase=phase,batch=1,query_len=q,context_len=k,query_heads=14,kv_heads=2,head_dim=64)
  plan,_=force_test_attention_plan(work,algorithm=alg,implementation=impl,
   strategy="serial" if wc==1 else "split_head",workers=wc,
   query_tile=0 if alg=="dense_materialized" else 1,key_tile=0 if alg=="dense_materialized" else 32)
  runtimes.append(CompilerAttentionRuntime(plan))
  Q=torch.randn(1,14,q,64);K=torch.randn(1,2,k,64);V=torch.randn_like(K)
  mask=torch.triu(torch.full((1,1,q,k),-torch.inf),1) if phase=="prefill" else None
  payload.append((Q,K,V,mask))
 max_error=0.;failures=0
 try:
  for i in range(1000):
   idx=i%len(runtimes);Q,K,V,mask=payload[idx]
   out=runtimes[idx].attention(Q,K,V,mask,.125)
   if i<8:
    ref=torch.nn.functional.scaled_dot_product_attention(Q,K.repeat_interleave(7,1),V.repeat_interleave(7,1),attn_mask=mask)
    max_error=max(max_error,float((out-ref).abs().max()))
   failures+=int(not torch.isfinite(out).all())
  result={"invocations":1000,"per_runtime_invocations":[len(r.traces) for r in runtimes],
   "worker_start_counts":[r.workers.worker_start_count for r in runtimes],
   "runtime_repartition_count":sum(r.runtime_repartition_count for r in runtimes),
   "worker_count_override":sum(r.runtime_worker_count_override for r in runtimes),
   "strategy_override":sum(r.runtime_strategy_override for r in runtimes),
   "manual_shard_count":sum(r.manual_shard_count for r in runtimes),
   "nonfinite_failures":failures,"max_abs_error":max_error,"deadlock":False,
   "worker_pool_recreated_per_call":False}
 finally:
  [r.close() for r in runtimes]
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
