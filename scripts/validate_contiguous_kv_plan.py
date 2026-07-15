#!/usr/bin/env python3
"""Execute a compiler-generated contiguous-KV prefill/decode contract."""
from array import array
import argparse,json
from pathlib import Path
from deployment.execution_plan.contiguous_kv_cache import ContiguousKVAttentionSession

def values(n,offset): return array("f",((i+offset)%17/17-0.5 for i in range(n)))
def main():
 p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();raw=json.loads(a.plan.read_text());attention={};kv=None
 for f in raw["function_plans"]:
  for d in f["per_op_decisions"]:
   if "attention_execution" in d:
    attention[f["serving_phase"]]=d["attention_execution"]
    if f["serving_phase"]=="prefill":kv=d["kv_cache_execution"]
 if not kv or set(attention)!={"prefill","decode"}:raise SystemExit("incomplete compiler KV/attention contract")
 s=ContiguousKVAttentionSession(kv,attention,artifact_root=a.artifact_root);b,h,d=kv["batch"],kv["num_kv_heads"],kv["head_dim"];t=attention["prefill"]["query_length"]
 s.prefill(values(b*h*t*d,1),values(b*h*t*d,2),values(b*h*t*d,3));out=s.decode_append_then_attend(values(b*h*d,4),values(b*h*d,5),values(b*h*d,6))
 a.out.write_text(json.dumps({"checksum":float(sum(out)),**s.trace()},indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
