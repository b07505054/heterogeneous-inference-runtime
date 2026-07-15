#!/usr/bin/env python3
import argparse,json
from dataclasses import replace
from pathlib import Path
from deployment.execution_plan.executorch_candidate_integration import *
from deployment.execution_plan.slice3c_target_selection import *
parser=argparse.ArgumentParser()
parser.add_argument('--evidence-root',type=Path,required=True)
args=parser.parse_args()
ROOT=args.evidence_root.resolve();REPO=Path(__file__).resolve().parents[1]
profile=json.load(open(REPO/'configs/target_profiles/raspberry_pi5_cortex_a76_cpu.json'));caps=load_codegen_capabilities(profile)
keep={FP32_CANDIDATE_ID,INT8_PACKED_A76_DOTPROD_CANDIDATE_ID,EXECUTORCH_XNNPACK_FP32_T1_CANDIDATE_ID,EXECUTORCH_XNNPACK_INT8_T1_CANDIDATE_ID,EXECUTORCH_XNNPACK_INT8_T4_CANDIDATE_ID};candidates=[c for c in enumerate_complete_candidates() if c.candidate_id in keep];by={c.candidate_id:c for c in candidates}
summary={'schema':'slice3g.integration_summary.v1','rows':[]}
for s in ('37x41x29','64x64x64','128x128x128','256x256x256'):
 m,n,k=map(int,s.split('x'));shape={'M':m,'N':n,'K':k};e=json.load(open(ROOT/'measurements'/f'candidate_evidence_{s}.json'))
 default=select_complete_candidate(candidates,caps,e,shape=shape)
 one=select_complete_candidate(candidates,caps,e,shape=shape,policy=SelectionPolicy(max_threads=1))
 unavailable=select_complete_candidate(candidates,replace(caps,supports_executorch_runtime=False),e,shape=shape)
 size=select_complete_candidate(candidates,caps,e,shape=shape,policy=SelectionPolicy(max_runtime_package_size=1_000_000))
 selected=by[default['selected_candidate_id']];plan=build_executorch_execution_plan(selected,e[selected.candidate_id],default);pp=ROOT/'plans'/f'selected_plan_{s}.json';pp.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
 oracle=min((x for x in default['considered_candidates'] if x['legal']),key=lambda x:(x['latency_median_ms'],x['candidate_id']))
 policy_rows={}
 for name,sel in [('default',default),('max_threads_1',one),('executorch_unavailable',unavailable),('deployment_size_limited',size)]:
  legal=[x for x in sel['considered_candidates'] if x['legal']];o=min(legal,key=lambda x:(x['latency_median_ms'],x['candidate_id']));policy_rows[name]={'selected':sel['selected_candidate_id'],'oracle':o['candidate_id'],'normalized_regret':(sel['selected_latency_median_ms']-o['latency_median_ms'])/o['latency_median_ms']}
 summary['rows'].append({'shape':s,'candidate_table':default['considered_candidates'],'selected_candidate':selected.candidate_id,'oracle_candidate':oracle['candidate_id'],'selection_agreement':selected.candidate_id==oracle['candidate_id'],'absolute_latency_regret_ms':default['selected_latency_median_ms']-oracle['latency_median_ms'],'normalized_regret':(default['selected_latency_median_ms']-oracle['latency_median_ms'])/oracle['latency_median_ms'],'policies':policy_rows,'plan_ref':str(pp)})
(ROOT/'raw/selection_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))
