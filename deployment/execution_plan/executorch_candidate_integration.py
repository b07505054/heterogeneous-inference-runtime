"""Slice 3G complete-candidate evidence, selection, plan, and fail-closed routing."""
from __future__ import annotations
import hashlib,json,re,statistics,subprocess
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Any
from deployment.execution_plan.slice3c_target_selection import CompleteCandidate,CodegenCapabilities,candidate_legality

REJECTION_REASONS=frozenset({"missing_executorch_runtime","runner_hash_mismatch","missing_pte_artifact","pte_hash_mismatch","workload_manifest_mismatch","shape_mismatch","missing_delegation_proof","delegation_proof_mismatch","xnnpack_not_delegated","measurement_identity_mismatch","correctness_gate_failed","thread_budget_exceeded","unsupported_thread_count","target_mismatch","missing_runtime_package","missing_xnnpack_capability","deployment_size_budget_exceeded","artifact_size_budget_exceeded","peak_rss_budget_exceeded","missing_measurement_artifact","measurement_stability_failed"})
def sha256_file(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical_hash(o:dict[str,Any])->str:return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()

@dataclass(frozen=True)
class SelectionPolicy:
 max_threads:int|None=None;max_runtime_package_size:int|None=None;max_artifact_size:int|None=None;max_peak_rss_kib:int|None=None

def validate_external_evidence(c:CompleteCandidate,caps:CodegenCapabilities,e:dict[str,Any],*,shape:dict[str,int],policy:SelectionPolicy)->list[str]:
 r=candidate_legality(c,caps,has_calibration_artifact=True,has_packed_artifact=True,build_tool_flags=caps.supported_compiler_target_flags)
 if c.backend!='executorch_xnnpack':return r
 clone=dict(e); claimed=clone.pop('evidence_sha256',None)
 if not claimed or canonical_hash(clone)!=claimed:r.append('measurement_identity_mismatch')
 if policy.max_threads is not None and c.thread_count>policy.max_threads:r.append('thread_budget_exceeded')
 root=Path(e.get('artifact_root',''))
 runner=root/e.get('runner_ref','');pte=root/e.get('pte_ref','');manifest=root/e.get('workload_manifest_ref','');proof=root/e.get('delegation_proof_ref','')
 if not runner.exists():r.append('missing_executorch_runtime')
 elif sha256_file(runner)!=e.get('runner_sha256'):r.append('runner_hash_mismatch')
 if not pte.exists():r.append('missing_pte_artifact')
 elif sha256_file(pte)!=e.get('pte_sha256'):r.append('pte_hash_mismatch')
 if not manifest.exists() or sha256_file(manifest)!=e.get('workload_manifest_sha256'):r.append('workload_manifest_mismatch')
 if e.get('shape')!=shape:r.append('shape_mismatch')
 if not proof.exists():r.append('missing_delegation_proof')
 else:
  p=json.loads(proof.read_text())
  if canonical_hash({k:v for k,v in p.items() if k!='proof_sha256'})!=p.get('proof_sha256') or p.get('proof_sha256')!=e.get('delegation_proof_sha256'):r.append('delegation_proof_mismatch')
  if p.get('delegate')!='xnnpack' or int(p.get('delegated_node_count',0))<1 or p.get('portable_fallback_nodes',[])!=[]:r.append('xnnpack_not_delegated')
 expected={'candidate_id':c.candidate_id,'backend':c.backend,'runtime':c.runtime,'delegate':c.delegate,'precision':c.precision,'quantization_scheme':c.quantization_scheme,'thread_count':c.thread_count,'shape':shape,'target_id':caps.target_id,'runner_sha256':e.get('runner_sha256'),'pte_sha256':e.get('pte_sha256'),'workload_manifest_sha256':e.get('workload_manifest_sha256')}
 if any(e.get(k)!=v for k,v in expected.items()):r.append('measurement_identity_mismatch')
 metrics=e.get('correctness_metrics',{})
 if float(metrics.get('cosine_similarity',0))<.99 or float(metrics.get('relative_l2_error',1))>.05:r.append('correctness_gate_failed')
 if int(e.get('session_count',0))<5 or int(e.get('sample_count_per_session',0))<100 or not e.get('randomized_order_provenance') or e.get('boundary_classification')!='slice3f_already_loaded_boundary':r.append('measurement_identity_mismatch')
 if not e.get('variability_gate_passed') or float(e.get('between_session_cv',1))>float(e.get('variability_threshold',0)):r.append('measurement_stability_failed')
 if policy.max_runtime_package_size is not None and int(e.get('runtime_package_bytes',1<<62))>policy.max_runtime_package_size:r.append('deployment_size_budget_exceeded')
 if policy.max_artifact_size is not None and int(e.get('artifact_bytes',1<<62))>policy.max_artifact_size:r.append('artifact_size_budget_exceeded')
 if policy.max_peak_rss_kib is not None and int(e.get('peak_rss_kib',1<<62))>policy.max_peak_rss_kib:r.append('peak_rss_budget_exceeded')
 return sorted(set(r))

def select_complete_candidate(candidates:list[CompleteCandidate],caps:CodegenCapabilities,evidence:dict[str,dict[str,Any]],*,shape:dict[str,int],policy:SelectionPolicy=SelectionPolicy())->dict[str,Any]:
 considered=[];legal=[]
 for c in candidates:
  e=evidence.get(c.candidate_id,{})
  if c.backend=='executorch_xnnpack':reasons=validate_external_evidence(c,caps,e,shape=shape,policy=policy)
  else:
   reasons=[]
   if policy.max_threads is not None and c.thread_count>policy.max_threads:reasons.append('thread_budget_exceeded')
   if not e:reasons.append('missing_measurement_artifact')
   elif e.get('shape')!=shape or e.get('candidate_id')!=c.candidate_id:reasons.append('measurement_identity_mismatch')
   elif float(e.get('correctness_metrics',{}).get('cosine_similarity',0))<.99 or float(e.get('correctness_metrics',{}).get('relative_l2_error',1))>.05:reasons.append('correctness_gate_failed')
   if policy.max_runtime_package_size is not None and int(e.get('runtime_package_bytes',0))>policy.max_runtime_package_size:reasons.append('deployment_size_budget_exceeded')
  latency=e.get('steady_state_invoke_median_ms');item={'candidate_id':c.candidate_id,'backend':c.backend,'precision':c.precision,'thread_count':c.thread_count,'latency_median_ms':latency,'correctness_metrics':e.get('correctness_metrics'),'artifact_bytes':e.get('artifact_bytes'),'peak_rss_kib':e.get('peak_rss_kib'),'legal':not reasons,'rejection_reasons':sorted(set(reasons))};considered.append(item)
  if not reasons and latency is not None:legal.append((float(latency),c.candidate_id,c,e))
 if not legal:raise ValueError('no legal complete implementation candidate')
 legal.sort(key=lambda x:(x[0],x[1]));lat,cid,c,e=legal[0]
 return {'schema_version':'slice3g.selection.v1','selected_candidate_id':cid,'selected_backend':c.backend,'selected_latency_median_ms':lat,'selection_reason':'minimum_steady_state_invoke_median_latency_subject_to_legality_correctness_stability_and_policy','policy':asdict(policy),'considered_candidates':considered,'selected_evidence_sha256':e.get('evidence_sha256')}

def build_executorch_execution_plan(c:CompleteCandidate,e:dict[str,Any],selection:dict[str,Any])->dict[str,Any]:
 if c.backend!='executorch_xnnpack':raise ValueError('not an ExecuTorch candidate')
 rejected=[x for x in selection['considered_candidates'] if x['candidate_id']!=c.candidate_id]
 return {'schema_version':'slice3g.executorch_execution_plan.v1','selected_complete_candidate_id':c.candidate_id,'backend':'executorch_xnnpack','runtime':'executorch','delegate':'xnnpack','precision':c.precision,'quantization_scheme':c.quantization_scheme,'thread_count':c.thread_count,'target_id':e['target_id'],'runner_artifact_ref':e['runner_ref'],'runner_sha256':e['runner_sha256'],'pte_artifact_ref':e['pte_ref'],'pte_sha256':e['pte_sha256'],'shared_workload_manifest_ref':e['workload_manifest_ref'],'shared_workload_manifest_sha256':e['workload_manifest_sha256'],'delegation_proof_ref':e['delegation_proof_ref'],'delegation_proof_sha256':e['delegation_proof_sha256'],'measurement_artifact_ref':e['measurement_ref'],'measurement_artifact_sha256':e['evidence_sha256'],'selection_reason':selection['selection_reason'],'rejected_candidate_provenance':rejected,'execution_stages':[{'stage_id':'load_executorch_program','one_time':True,'dependency_ids':[],'produces':'program_ready'},{'stage_id':'bind_input','one_time':False,'dependency_ids':['program_ready'],'produces':'input_ready'},{'stage_id':'execute_xnnpack_delegate','one_time':False,'dependency_ids':['input_ready'],'produces':'output_ready'},{'stage_id':'return_output','one_time':False,'dependency_ids':['output_ready'],'produces':'return_ready'}]}

def route_executorch_plan(plan:dict[str,Any],*,root:Path,input_path:Path,output_path:Path,warmups:int=10,repeats:int=100)->dict[str,Any]:
 if plan.get('backend')!='executorch_xnnpack' or plan.get('runtime')!='executorch' or plan.get('delegate')!='xnnpack':raise ValueError('unsupported backend package')
 if [x['stage_id'] for x in plan.get('execution_stages',[])]!=['load_executorch_program','bind_input','execute_xnnpack_delegate','return_output']:raise ValueError('invalid ExecuTorch stage contract')
 if int(plan['thread_count'])>int(plan.get('maximum_runtime_threads',4)):raise ValueError('thread_budget_exceeded')
 runner=root/plan['runner_artifact_ref'];pte=root/plan['pte_artifact_ref'];proof=root/plan['delegation_proof_ref']
 if not runner.exists():raise ValueError('missing_executorch_runtime')
 if sha256_file(runner)!=plan['runner_sha256']:raise ValueError('runner_hash_mismatch')
 if not pte.exists():raise ValueError('missing_pte_artifact')
 if sha256_file(pte)!=plan['pte_sha256']:raise ValueError('pte_hash_mismatch')
 if not proof.exists():raise ValueError('missing_delegation_proof')
 threads=int(plan['thread_count']);cmd=[str(runner),'--model_path',str(pte),'--inputs',str(input_path),'--num_executions',str(warmups+repeats),'--cpu_threads',str(threads),'--print_output','none','--output_file',str(output_path)+'.runner']
 p=subprocess.run(cmd,text=True,capture_output=True)
 produced=Path(str(output_path)+'.runner-0.bin')
 if p.returncode or not produced.exists():raise RuntimeError('ExecuTorch route failed explicitly: '+p.stderr[-2000:])
 output_path.write_bytes(produced.read_bytes())
 samples=[float(x) for x in re.findall(r'Iteration \d+ of \d+: ([0-9.]+) ms',p.stdout+p.stderr)][warmups:]
 if len(samples)!=repeats:raise RuntimeError('ExecuTorch route did not report the required samples')
 return {'candidate_id':plan['selected_complete_candidate_id'],'backend':plan['backend'],'precision':plan['precision'],'thread_count':threads,'runner_sha256':plan['runner_sha256'],'pte_sha256':plan['pte_sha256'],'delegate_executed':'xnnpack','runtime_no_redecision':True,'warmups':warmups,'samples':repeats,'steady_state_median_ms':statistics.median(samples),'raw_samples_ms':samples,'output_sha256':sha256_file(output_path)}
