import copy,json
from deployment.vllm_adapter.policy_executor import command,validate_policy
def plan(v=4):return {"schema_version":"vllm.max_num_seqs.policy.v1","runtime_no_policy_redecision":True,"vllm_version":"1","target_gpu_identity":"GPU-X","model":"m","candidate_id":"vllm_max_num_seqs_default" if v is None else f"vllm_max_num_seqs_{v}","max_num_seqs":v,"value_source":"default" if v is None else "explicit","fixed_configuration":{"model":"m","tokenizer":"m","dtype":"float16","max_model_len":2048,"gpu_memory_utilization":.75,"block_size":16,"max_num_batched_tokens":2048,"enable_chunked_prefill":False,"enable_prefix_caching":False,"tensor_parallel_size":1,"pipeline_parallel_size":1,"served_model_name":"m"}}
def test_explicit_and_default_commands():
 assert command(plan(4)).count("--max-num-seqs")==1 and command(plan(4))[command(plan(4)).index("--max-num-seqs")+1]=="4";assert "--max-num-seqs" not in command(plan(None))
def test_mismatch_fails_closed(tmp_path):
 p=tmp_path/"p.json";p.write_text(json.dumps(plan()));validate_policy(plan(),plan_path=p,identity={"vllm_version":"1","gpu_csv":"GPU-X"})
 for k,val in [("vllm_version","2"),("target_gpu_identity","GPU-Y")]:
  q=plan();q[k]=val
  try:validate_policy(q,plan_path=p,identity={"vllm_version":"1","gpu_csv":"GPU-X"});assert False
  except ValueError:pass


def test_illegal_values_and_redecision_contract_fail_closed(tmp_path):
 p=tmp_path/"p.json";p.write_text(json.dumps(plan()))
 for value in (0,-1,"4"):
  q=plan();q["max_num_seqs"]=value
  try:validate_policy(q,plan_path=p,identity={"vllm_version":"1","gpu_csv":"GPU-X"});assert False
  except ValueError:pass
 q=plan();q["runtime_no_policy_redecision"]=False
 try:validate_policy(q,plan_path=p,identity={"vllm_version":"1","gpu_csv":"GPU-X"});assert False
 except ValueError:pass


def test_fixed_configuration_mismatch_is_rejected(tmp_path):
 q=plan();del q["fixed_configuration"]["max_num_batched_tokens"]
 p=tmp_path/"p.json";p.write_text(json.dumps(q))
 try:validate_policy(q,plan_path=p,identity={"vllm_version":"1","gpu_csv":"GPU-X"});assert False
 except ValueError:pass
 q=plan();q["model"]="other"
 p.write_text(json.dumps(q))
 try:validate_policy(q,plan_path=p,identity={"vllm_version":"1","gpu_csv":"GPU-X"});assert False
 except ValueError:pass
