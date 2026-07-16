"""Fail-closed materialization of compiler-selected vLLM max_num_seqs policy."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
EXPECTED={None:"vllm_max_num_seqs_default",1:"vllm_max_num_seqs_1",2:"vllm_max_num_seqs_2",4:"vllm_max_num_seqs_4",8:"vllm_max_num_seqs_8"}
def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def live_identity(python=".venv/bin/python"):
 gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,memory.total,driver_version,compute_cap","--format=csv,noheader,nounits"],text=True).strip();ver=subprocess.check_output([python,"-c","import vllm;print(vllm.__version__)"],text=True).strip();return {"gpu_csv":gpu,"vllm_version":ver}
def validate_policy(plan,*,plan_path,identity):
 errors=[]
 if plan.get("schema_version")!="vllm.max_num_seqs.policy.v1":errors.append("schema_mismatch")
 if plan.get("runtime_no_policy_redecision") is not True:errors.append("runtime_redecision_not_forbidden")
 if plan.get("vllm_version")!=identity["vllm_version"]:errors.append("vllm_version_mismatch")
 if plan.get("target_gpu_identity") not in identity["gpu_csv"]:errors.append("gpu_identity_mismatch")
 fixed=plan.get("fixed_configuration",{});required={"model","dtype","max_model_len","max_num_batched_tokens","enable_chunked_prefill","enable_prefix_caching","tensor_parallel_size","pipeline_parallel_size"}
 if not required<=set(fixed):errors.append("fixed_configuration_incomplete")
 v=plan.get("max_num_seqs");source=plan.get("value_source")
 if (source=="default" and v is not None) or (source=="explicit" and (not isinstance(v,int) or v<=0)):errors.append("max_num_seqs_contract_mismatch")
 if plan.get("model")!=fixed.get("model"):errors.append("model_identity_mismatch")
 if v not in EXPECTED or plan.get("candidate_id")!=EXPECTED.get(v):errors.append("candidate_identity_mismatch")
 if errors:raise ValueError(";".join(errors))
 return {"plan_sha256":sha256(plan_path),"validated":True}
def command(plan,*,host="127.0.0.1",port=8000,python=".venv/bin/python"):
 f=plan["fixed_configuration"];cmd=[python,"-m","vllm.entrypoints.openai.api_server","--model",f["model"],"--tokenizer",f["tokenizer"],"--dtype",f["dtype"],"--max-model-len",str(f["max_model_len"]),"--gpu-memory-utilization",str(f["gpu_memory_utilization"]),"--block-size",str(f["block_size"]),"--max-num-batched-tokens",str(f["max_num_batched_tokens"]),"--tensor-parallel-size",str(f["tensor_parallel_size"]),"--pipeline-parallel-size",str(f["pipeline_parallel_size"]),"--served-model-name",f["served_model_name"],"--host",host,"--port",str(port)]
 cmd.append("--enable-chunked-prefill" if f["enable_chunked_prefill"] else "--no-enable-chunked-prefill");cmd.append("--enable-prefix-caching" if f["enable_prefix_caching"] else "--no-enable-prefix-caching")
 if plan["value_source"]=="explicit":cmd.extend(["--max-num-seqs",str(plan["max_num_seqs"])])
 return cmd
