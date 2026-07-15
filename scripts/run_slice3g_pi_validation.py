#!/usr/bin/env python3
import copy,hashlib,json,math,struct
from pathlib import Path
from deployment.execution_plan.executorch_candidate_integration import route_executorch_plan
ROOT=Path(__file__).resolve().parents[1];EVID=ROOT/'evidence';OUT=ROOT/'raw';OUT.mkdir(exist_ok=True)
def vals(p):
 b=Path(p).read_bytes();return struct.unpack('<'+'f'*(len(b)//4),b)
def metrics(a,r):
 d=[x-y for x,y in zip(a,r)];nr=math.sqrt(sum(x*x for x in r));na=math.sqrt(sum(x*x for x in a));cs=sum(x*y for x,y in zip(a,r))/(na*nr);rl=math.sqrt(sum(x*x for x in d))/nr
 return {'max_absolute_error':max(map(abs,d)),'relative_l2_error':rl,'cosine_similarity':cs,'gate_pass':cs>=.99 and rl<=.05}
rows=[]
for s in ('37x41x29','64x64x64','128x128x128','256x256x256'):
 plan=json.load(open(EVID/'plans'/f'selected_plan_{s}.json'));out=OUT/f'{s}.bin';run=route_executorch_plan(plan,root=EVID,input_path=EVID/'artifacts'/s/'input_fp32.bin',output_path=out,warmups=10,repeats=100);run['shape']=s;run['correctness']=metrics(vals(out),vals(EVID/'artifacts'/s/'reference_fp32.bin'));rows.append(run)
negative={};base=json.load(open(EVID/'plans'/'selected_plan_64x64x64.json'))
cases={
 'wrong_pte_sha':dict(pte_sha256='0'*64),
 'missing_runner':dict(runner_artifact_ref='artifacts/does_not_exist'),
 'four_threads_on_1t_target':dict(thread_count=4,maximum_runtime_threads=1),
 'missing_delegation_proof':dict(delegation_proof_ref='proofs/does_not_exist'),
 'unsupported_backend_package':dict(backend='unpackaged_backend'),
}
for name,changes in cases.items():
 p=copy.deepcopy(base);p.update(changes)
 try:route_executorch_plan(p,root=EVID,input_path=EVID/'artifacts/64x64x64/input_fp32.bin',output_path=OUT/f'negative_{name}.bin',warmups=1,repeats=1);negative[name]={'passed':False,'error':'unexpected success'}
 except Exception as e:negative[name]={'passed':True,'error':str(e)}
(OUT/'pi_validation.json').write_text(json.dumps({'schema':'slice3g.pi_validation.v1','rows':rows,'negative_tests':negative},indent=2,sort_keys=True)+'\n');print(json.dumps({'rows':rows,'negative_tests':negative},indent=2))
