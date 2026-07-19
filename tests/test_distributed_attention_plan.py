import copy, math
import pytest, torch
from deployment.attention_planner import AttentionWorkload, force_test_attention_plan
from deployment.attention_runtime import CompilerAttentionRuntime
from deployment.cpu_sharding import ShardingPlanError
from deployment.distributed_attention_plan import (
    CONTRACT_VERSION, build_attention_placement, validate_attention_placement)

def placement(workers=4):
 return build_attention_placement(batch=1,query_length=11,context_length=11,
  query_heads=14,kv_heads=2,head_dim=64,strategy="split_head",worker_count=workers,
  direct_output=True)
def validate(p):
 return validate_attention_placement(p,query_length=11,query_heads=14,
  kv_heads=2,head_dim=64,batch=1)
def test_balanced_deterministic_gqa_placement():
 p=placement()
 assert p==placement()
 assert [w["query_head_range"] for w in p["workers"]]==[[0,4],[4,8],[8,11],[11,14]]
 assert [w["kv_head_read_range"] for w in p["workers"]]==[[0,1],[0,2],[1,2],[1,2]]
 assert p["communication"]["reduction"]=="none"
 assert p["synchronization"]["kind"]=="completion_counter"
@pytest.mark.parametrize("mutator,match",[
 (lambda p:p["workers"][1].update(query_head_range=[3,8]),"overlap"),
 (lambda p:p["workers"][1].update(query_head_range=[5,8]),"gap"),
 (lambda p:p.update(worker_count=3),"worker count"),
 (lambda p:p["workers"][0].update(worker_id=4),"worker ID"),
 (lambda p:p["communication"].update(reduction="sum"),"no reduction"),
 (lambda p:p["synchronization"].update(kind="none"),"completion barrier"),
 (lambda p:p.update(contract_version="bad"),"ABI version"),
])
def test_negative_placement_validation(mutator,match):
 p=placement();mutator(p)
 with pytest.raises(ShardingPlanError,match=match):validate(p)
def test_runtime_executes_exact_planned_workers_and_shards():
 w=AttentionWorkload(phase="prefill",batch=1,query_len=11,context_len=11,
  query_heads=14,kv_heads=2,head_dim=64)
 p,_=force_test_attention_plan(w,algorithm="fused_tiled_online_softmax",
  implementation="native_avx2",strategy="split_head",workers=4,
  query_tile=1,key_tile=32)
 torch.manual_seed(9);q=torch.randn(1,14,11,64);k=torch.randn(1,2,11,64);v=torch.randn_like(k)
 with CompilerAttentionRuntime(p) as r:
  out=r.attention(q,k,v,None,1/math.sqrt(64));events=r.traces[-1].worker_events
  assert r.runtime_repartition_count==r.runtime_worker_count_override==0
  assert r.runtime_strategy_override==r.manual_shard_count==0
 ref=torch.nn.functional.scaled_dot_product_attention(
  q,k.repeat_interleave(7,1),v.repeat_interleave(7,1),is_causal=True)
 torch.testing.assert_close(out,ref,atol=2e-5,rtol=2e-5)
 assert [(e["planned_worker_id"],e["executed_logical_worker_id"]) for e in events]==[(0,0),(1,1),(2,2),(3,3)]
 assert [e["planned_shard"] for e in events]==[e["executed_shard"] for e in events]
def test_compiler_plan_missing_placement_is_rejected():
 w=AttentionWorkload(phase="prefill",batch=1,query_len=11,context_len=11,
  query_heads=14,kv_heads=2,head_dim=64)
 p,_=force_test_attention_plan(w,algorithm="dense_materialized",
  strategy="serial",workers=1,query_tile=0,key_tile=0)
 del p["distributed_execution"]
 with pytest.raises(ShardingPlanError,match="requires distributed_execution"):
  from deployment.attention_runtime import validate_attention_plan
  validate_attention_plan(p)
def test_out_of_range_worker_perturbation_is_rejected():
 w=AttentionWorkload(phase="decode",batch=1,query_len=1,context_len=7,
  query_heads=14,kv_heads=2,head_dim=64)
 p,_=force_test_attention_plan(w,algorithm="fused_tiled_online_softmax",
  implementation="native_avx2",strategy="split_head",workers=2,
  query_tile=1,key_tile=32)
 q=torch.randn(1,14,1,64);k=torch.randn(1,2,7,64);v=torch.randn_like(k)
 with CompilerAttentionRuntime(p,perturbation=5,perturb_worker_id=2) as r:
  with pytest.raises(ShardingPlanError,match="does not own"):
   r.attention(q,k,v,None,.125)
