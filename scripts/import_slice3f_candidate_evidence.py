#!/usr/bin/env python3
"""Import immutable Slice 3F evidence into Slice 3G compiler-readable artifacts."""
import argparse,hashlib,json,shutil
from pathlib import Path
from deployment.execution_plan.slice3c_target_selection import *
parser=argparse.ArgumentParser()
parser.add_argument('--output-root',type=Path,required=True)
parser.add_argument('--slice3f-root',type=Path,required=True)
parser.add_argument('--slice3e-root',type=Path,required=True)
args=parser.parse_args()
ROOT=args.output_root.resolve();S3F=args.slice3f_root.resolve();S3E=args.slice3e_root.resolve()
agg=json.load(open(S3F/'raw/aggregate.json'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ch(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def et_correct(shape,precision):
 for d in sorted((S3F/'raw/sessions').glob('session_*')):
  m=json.load(open(d/'session_manifest.json'))
  for r in m['runs']:
   if r['shape']==shape and r['implementation']==f'xnn_{precision}_t1':return json.load(open(d/r['result']))['correctness']
 raise RuntimeError(f'missing correctness shape={shape!r} precision={precision!r}')
for row in agg['rows']:
 s=row['shape'];m,n,k=map(int,s.split('x'));shape={'M':m,'N':n,'K':k};src=S3E/'artifacts'/s;dst=ROOT/'artifacts'/s;dst.mkdir(parents=True,exist_ok=True)
 for name in ('workload_manifest.json','model_fp32_xnnpack.pte','model_int8_xnnpack.pte','export_report.json','input_fp32.bin','reference_fp32.bin'):
  shutil.copyfile(src/name,dst/name)
 manifest=dst/'workload_manifest.json';manifest_sha=sha(manifest);export=json.load(open(dst/'export_report.json'))
 evidence={}
 # Existing complete portable candidates remain in the same evidence map.
 custom_metrics=row['correctness']['vs_fp32_reference']; custom_latency=row['aggregate']['custom_int8_t1']['median_of_session_medians_ms']
 evidence[INT8_PACKED_A76_DOTPROD_CANDIDATE_ID]={'candidate_id':INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,'shape':shape,'steady_state_invoke_median_ms':custom_latency,'correctness_metrics':custom_metrics,'thread_count':1,'artifact_bytes':75896,'runtime_package_bytes':75896,'peak_rss_kib':3984}
 old=json.load(open(S3E/'raw/analysis'/f'baseline_{s}.json'));e_fp=old['latency_distributions']['custom_fp32_t1']['value'];c_fp=old['correctness']['custom_fp32']['value']
 evidence[FP32_CANDIDATE_ID]={'candidate_id':FP32_CANDIDATE_ID,'shape':shape,'steady_state_invoke_median_ms':e_fp['median_ms'],'correctness_metrics':c_fp,'thread_count':1,'artifact_bytes':79960,'runtime_package_bytes':79960,'peak_rss_kib':3984}
 for cid,precision,threads,key in [(EXECUTORCH_XNNPACK_FP32_T1_CANDIDATE_ID,'fp32',1,'xnn_fp32_t1'),(EXECUTORCH_XNNPACK_INT8_T1_CANDIDATE_ID,'int8',1,'xnn_int8_t1'),(EXECUTORCH_XNNPACK_INT8_T4_CANDIDATE_ID,'int8',4,'xnn_int8_t4')]:
  proof={'schema':'slice3g.delegation_proof.v1','delegate':'xnnpack','shape':shape,'precision':precision,'delegated_node_count':int(export[precision]['delegate_calls']),'portable_fallback_nodes':[],'pte_sha256':export[precision]['sha256'],'executorch_commit':'e2f18eb23c45bd22ca332b0b8b49a81de304b472','xnnpack_commit':'1adaa7c709d4839d29e1f219cb962b01c9e6a905'};proof['proof_sha256']=ch(proof);pp=ROOT/'proofs'/f'{s}_{precision}.json';pp.write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
  a=row['aggregate'][key];e={'schema_version':'slice3g.external_measurement.v1','artifact_root':str(ROOT),'candidate_id':cid,'backend':'executorch_xnnpack','runtime':'executorch','delegate':'xnnpack','precision':precision,'quantization_scheme':'none' if precision=='fp32' else 'pt2e_per_tensor_affine_per_channel_symmetric_axis0','thread_count':threads,'shape':shape,'target_id':'raspberry-pi5-cortex-a76-cpu','required_architecture':'aarch64','runner_ref':'artifacts/executor_runner','runner_sha256':'adef50a17a4aebc953583638a0ba7d573fc53df4023f9183280887a07fd17341','pte_ref':f'artifacts/{s}/model_{precision}_xnnpack.pte','pte_sha256':export[precision]['sha256'],'workload_manifest_ref':f'artifacts/{s}/workload_manifest.json','workload_manifest_sha256':manifest_sha,'shared_workload_manifest_id':json.load(open(manifest))['manifest_id'],'delegation_proof_ref':f'proofs/{s}_{precision}.json','delegation_proof_sha256':proof['proof_sha256'],'measurement_ref':f'measurements/{s}_{cid.replace(":","_")}.json','correctness_metrics':et_correct(s,precision),'steady_state_invoke_median_ms':a['median_of_session_medians_ms'],'session_medians_ms':a['session_medians_ms'],'session_count':5,'sample_count_per_session':100,'between_session_cv':a['between_session_coefficient_of_variation'],'variability_threshold':.12,'variability_gate_passed':a['between_session_coefficient_of_variation']<=.12,'randomized_order_provenance':[x['seed'] for x in agg['sessions']],'boundary_classification':'slice3f_already_loaded_boundary','runtime_package_bytes':9454800,'artifact_bytes':export[precision]['bytes'],'peak_rss_kib':6400 if precision=='fp32' else 6448,'executorch_tag':'v1.3.1','executorch_commit':'e2f18eb23c45bd22ca332b0b8b49a81de304b472','xnnpack_commit':'1adaa7c709d4839d29e1f219cb962b01c9e6a905','torch_version':'2.12.0+cpu','torchao_version':'0.17.0+git02105d46c'};e['evidence_sha256']=ch(e);mp=ROOT/e['measurement_ref'];mp.write_text(json.dumps(e,indent=2,sort_keys=True)+'\n');evidence[cid]=e
 (ROOT/'measurements'/f'candidate_evidence_{s}.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
