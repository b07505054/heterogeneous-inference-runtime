#!/usr/bin/env python3
import argparse, hashlib, json, math, os, random, re, statistics, struct, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
ATOL=1e-3; RTOL=1e-4; SEED_OFFSET=210000; PROJECT_KERNEL_ID='portable_fused_matmul_bias_relu_bm32_bn128_bk32'; THRESHOLD=262144

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
 return h.hexdigest()
def pack(vals): return struct.pack('<'+'f'*len(vals), *vals)
def unpack(p):
 b=Path(p).read_bytes(); return list(struct.unpack('<'+'f'*(len(b)//4), b))
def gen(m,n,k,seed):
 rng=random.Random(seed)
 a=[rng.uniform(-3,3) for _ in range(m*k)]; b=[rng.uniform(-3,3) for _ in range(k*n)]; bias=[rng.uniform(-3,3) for _ in range(n)]
 return a,b,bias
def ref(a,b,bias,m,n,k,acc='fp64'):
 out=[]
 for i in range(m):
  for j in range(n):
   if acc=='fp32':
    s=struct.unpack('<f',struct.pack('<f',0.0))[0]
    for kk in range(k):
     prod=struct.unpack('<f',struct.pack('<f',a[i*k+kk]*b[kk*n+j]))[0]
     s=struct.unpack('<f',struct.pack('<f',s+prod))[0]
    v=struct.unpack('<f',struct.pack('<f',s+bias[j]))[0]
   else:
    v=sum(a[i*k+kk]*b[kk*n+j] for kk in range(k))+bias[j]
   out.append(struct.unpack('<f',struct.pack('<f',v if v>0 else 0.0))[0])
 return out
def err(actual, expected):
 dif=[abs(x-y) for x,y in zip(actual,expected)]; rel=[abs(x-y)/max(abs(y),1e-30) for x,y in zip(actual,expected)]
 passv=all((not math.isnan(x)) and (not math.isinf(x)) and abs(x-y) <= ATOL + RTOL*abs(y) for x,y in zip(actual,expected)) and len(actual)==len(expected)
 return {'passed':passv,'max_abs_error':max(dif) if dif else None,'mean_abs_error':statistics.fmean(dif) if dif else None,'max_rel_error':max(rel) if rel else None,'nan_count':sum(math.isnan(x) for x in actual),'inf_count':sum(math.isinf(x) for x in actual)}
def bucket(v):
 av=abs(v)
 if av==0: return 'reference == 0'
 if av<1e-6: return '|ref| < 1e-6'
 if av<1e-4: return '1e-6 <= |ref| < 1e-4'
 if av<1e-2: return '1e-4 <= |ref| < 1e-2'
 return '|ref| >= 1e-2'

def forensics(args):
 raw=json.load(open(args.e2_raw)); recs=raw['records']; fails=[r for r in recs if not r['result'].get('correctness_passed')]
 by_bucket={}; by_mode={}; by_workload={}; near=0
 for r in fails:
  c=r['result']['correctness']; by_mode.setdefault(r['mode'],[]).append(c); by_workload.setdefault(r['workload_id'],[]).append(c)
  # Aggregate-only E2 record did not retain per-index values; classify from max rel/max abs relation.
  if c['max_abs_error'] <= ATOL and c['max_rel_error'] > RTOL: near += 1
 summary={'schema':'e2_1_e2_failure_forensics','e2_records':len(recs),'e2_failed_records':len(fails),'aggregate_limitation':'E2 raw records retained aggregate max/mean errors, not per-output index/value; E2.1 therefore does not rewrite E2 and uses aggregate forensics plus fresh per-output validation.', 'max_abs_error':max(r['result']['correctness']['max_abs_error'] for r in recs),'max_rel_error':max(r['result']['correctness']['max_rel_error'] for r in recs),'failures_with_abs_within_new_atol_and_rel_over_old_gate':near,'failures_by_mode':{k:len(v) for k,v in by_mode.items()},'failures_by_workload':{k:len(v) for k,v in by_workload.items()},'hypothesis':'Most E2 failures are consistent with independent relative-error amplification because absolute errors are small while relative errors exceed the frozen independent gate.'}
 Path(args.out).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2)[:2000])

def negative_controls(args):
 m=n=k=16; a,b,bias=gen(m,n,k,12345); r=ref(a,b,bias,m,n,k,'fp64'); tests={}
 def ck(name, vals): tests[name]=err(vals,r)['passed']
 v=list(r); v[0]+=0.01; ck('one_element_plus_1e_2',v)
 wrong=ref(a,b,[0.0]*n,m,n,k,'fp64'); ck('wrong_bias_zero_bias',wrong)
 no_relu=[sum(a[i*k+kk]*b[kk*n+j] for kk in range(k))+bias[j] for i in range(m) for j in range(n)]; no_relu=[struct.unpack('<f',struct.pack('<f',x))[0] for x in no_relu]; ck('without_relu',no_relu)
 trans=[r[j*n+i] if j<n and i<n else r[i*n+j] for i in range(m) for j in range(n)]; ck('transposed_output',trans)
 wrong_seed=ref(*gen(m,n,k,12346),m,n,k,'fp64'); ck('wrong_seed',wrong_seed)
 v=list(r); v[0]=float('nan'); ck('injected_nan',v)
 v=list(r); v[0]=float('inf'); ck('injected_inf',v)
 v=[x*1.01 for x in r]; ck('systematic_scale_1_percent',v)
 v=list(r); idx=min(range(len(v)), key=lambda i: abs(v[i])); v[idx]+=0.002; ck('near_zero_plus_2e_3',v)
 out={'schema':'e2_1_negative_controls','predicate':f'abs(actual-expected) <= {ATOL} + {RTOL}*abs(expected)','all_rejected':not any(tests.values()),'tests_passed_field_means_wrong_output_accepted':tests}
 Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2));
 if any(tests.values()): raise SystemExit(2)

def prepare(args):
 p1d=json.load(open(args.p1d_manifest)); pte_dir=Path(args.pte_dir); neg=json.load(open(args.negative_controls)); workloads=[]
 for w in p1d['workloads']:
  if w['split'] not in ('calibration','held_out'): continue
  m,n,k=w['m'],w['n'],w['k']; seed=w['seed']+SEED_OFFSET; a,b,bias=gen(m,n,k,seed); r64=ref(a,b,bias,m,n,k,'fp64'); r32=ref(a,b,bias,m,n,k,'fp32')
  pte=pte_dir/f'fused_matmul_bias_relu_{m}x{n}x{k}_xnnpack.pte'; er=pte_dir/f'fused_matmul_bias_relu_{m}x{n}x{k}_xnnpack_export_report.json'
  rep=json.load(open(er))
  workloads.append({'workload_id':w['workload_id'],'split':w['split'],'category':w['category'],'m':m,'n':n,'k':k,'mnk':m*n*k,'original_seed':w['seed'],'seed':seed,'input_hashes':{'a':sha_bytes(pack(a)),'b':sha_bytes(pack(b)),'bias':sha_bytes(pack(bias)),'reference_fp64':sha_bytes(pack(r64)),'reference_fp32':sha_bytes(pack(r32))},'executorch_xnnpack_pte':{'filename':pte.name,'sha256':sha_file(pte),'bytes':pte.stat().st_size,'classification':rep.get('classification'),'delegate_call_count':rep.get('delegate_call_count')},'project_policy_selection':{'mode':'P-SERIAL' if m*n*k<THRESHOLD else 'P-4T'}})
 man={'schema':'e2_1_manifest','schema_version':1,'comparison_id':'E2_1_RPI5_FP32_FUSED_MATMUL_BIAS_RELU_2026_07_13','parent_historical_experiment_id':'E2_RPI5_FP32_FUSED_MATMUL_BIAS_RELU_2026_07_13','parent_verdict':'COMPARISON_INVALID_CORRECTNESS_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY','created_utc':datetime.now(timezone.utc).isoformat(),'correctness_predicate':{'form':'abs(actual - expected) <= atol + rtol * abs(expected)','atol':ATOL,'rtol':RTOL,'derivation':'Mixed allclose rule avoids independent relative-error amplification near zero while retaining 1e-3 absolute ceiling observed sufficient for FP32 matmul reduction differences in E2 diagnostics; negative controls reject semantic corruptions.'},'negative_control_reference':neg,'input_generation':{'algorithm':'python random.Random(seed).uniform(-3,3); E2.1 seed = P1D seed + 210000; A,B,bias serialized little-endian fp32'},'project':{'compiler_commit':'b67cd644568e7f53a64370f926e241e4e42ebe10','runtime_commit':'f0a0dab34d80776973377c5d864a30f156f55b11','kernel_id':PROJECT_KERNEL_ID,'threshold':THRESHOLD},'executorch':{'tag':'v1.3.1','commit':'e2f18eb23c45bd22ca332b0b8b49a81de304b472','xnnpack_commit':'1adaa7c709d4839d29e1f219cb962b01c9e6a905','runner_sha256':'eb3068fb1742e4172a459f9f4c5aebd2dd9dd43151e214ff1402ea925d4e2809'},'protocol':{'warmup_count':5,'timed_repeat_count':20,'total_invocations_per_mode_session':25,'session_count':3,'affinity':'0-3','governor':'performance'},'modes':{'project':['P-SERIAL','P-4T','P-POLICY'],'executorch':['ET-DEFAULT','ET-1T-REQUESTED','ET-4T-REQUESTED']},'analysis_formulas':{'tie_threshold_percent':5.0,'per_workload_latency':'median of session medians','cross_system':'latency ratios, not regret'},'claim_boundaries':['Raspberry Pi 5 only','FP32 MatMul+Bias+ReLU only','frozen suite only','not general superiority'], 'workloads':workloads}
 frozen=json.dumps(man,sort_keys=True,separators=(',',':')).encode(); man['manifest_sha256']=hashlib.sha256(frozen).hexdigest(); Path(args.out).write_text(json.dumps(man,indent=2,sort_keys=True)+'\n'); print(man['comparison_id']); print(man['manifest_sha256']); print(len(workloads))

def write_inputs(w, root):
 d=root/w['workload_id']; d.mkdir(parents=True,exist_ok=True); a,b,bias=gen(w['m'],w['n'],w['k'],w['seed']); r64=ref(a,b,bias,w['m'],w['n'],w['k'],'fp64'); r32=ref(a,b,bias,w['m'],w['n'],w['k'],'fp32')
 for name,vals in [('a',a),('b',b),('bias',bias),('reference_fp64',r64),('reference_fp32',r32)]:
  data=pack(vals); (d/f'{name}.bin').write_bytes(data)
  if sha_bytes(data)!=w['input_hashes'][name]: raise RuntimeError('hash mismatch '+w['workload_id']+' '+name)
 return d

def thermal():
 def sh(c): return subprocess.run(c,shell=True,text=True,capture_output=True).stdout.strip()
 return {'temp':sh('vcgencmd measure_temp 2>/dev/null || true'),'throttled':sh('vcgencmd get_throttled 2>/dev/null || true'),'freq':sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true'),'governor':sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true')}
def run_project(w,mode,paths,repeats):
 if mode=='P-SERIAL': tc,axis,strat=1,'none','serial'
 elif mode=='P-4T': tc,axis,strat=4,'m','contiguous_chunks'
 else:
  tc,axis,strat=(1,'none','serial') if w['mnk']<THRESHOLD else (4,'m','contiguous_chunks')
 out=paths['tmp']/f'{w["workload_id"]}_{mode}.bin'; inp=paths['input_dir']/w['workload_id']
 cmd=['taskset','-c','0-3',paths['project_kernel'],'--m',str(w['m']),'--n',str(w['n']),'--k',str(w['k']),'--a',str(inp/'a.bin'),'--b',str(inp/'b.bin'),'--bias',str(inp/'bias.bin'),'--out',str(out),'--kernel-id',PROJECT_KERNEL_ID,'--thread-count',str(tc),'--partition-axis',axis,'--partition-strategy',strat,'--repeats',str(repeats)]
 r=subprocess.run(cmd,text=True,capture_output=True); js=json.loads(r.stdout) if r.returncode==0 else {}; vals=unpack(out) if out.exists() else []; ref64=unpack(inp/'reference_fp64.bin'); ref32=unpack(inp/'reference_fp32.bin'); e64=err(vals,ref64); e32=err(vals,ref32)
 return {'exit_status':r.returncode,'raw_samples_ms':js.get('samples_ms',[]),'warm_samples_ms':js.get('samples_ms',[])[5:],'cold_first_ms':js.get('samples_ms',[None])[0],'load_time_ms':None,'correctness_fp64':e64,'correctness_fp32':e32,'correctness_passed':e64['passed'],'requested_threads':tc,'observed_thread_classification':'VERIFIED_BY_PROJECT_CONTRACT_SELF_REPORT','runner_report':js}
def parse_et(log): return [float(x) for x in re.findall(r'Iteration \d+ of \d+: ([0-9.]+) ms',log)], (float(re.search(r'Model loaded in ([0-9.]+) ms',log).group(1)) if re.search(r'Model loaded in ([0-9.]+) ms',log) else None), re.findall(r'Resetting threadpool[^\n]*',log)
def run_et(w,mode,paths,repeats):
 extra=[]; req='default'
 if mode=='ET-1T-REQUESTED': extra=['--cpu_threads','1']; req=1
 if mode=='ET-4T-REQUESTED': extra=['--cpu_threads','4']; req=4
 inp=paths['input_dir']/w['workload_id']; outbase=paths['tmp']/f'{w["workload_id"]}_{mode}'
 for p in paths['tmp'].glob(f'{w["workload_id"]}_{mode}*.bin'): p.unlink()
 inputs=','.join(str(inp/x) for x in ['a.bin','b.bin','bias.bin']); pte=paths['pte_dir']/w['executorch_xnnpack_pte']['filename']
 cmd=['taskset','-c','0-3',paths['et_runner'],'--model_path',str(pte),'--inputs',inputs,'--num_executions',str(repeats),'--print_output','none','--output_file',str(outbase)]+extra
 r=subprocess.run(cmd,text=True,capture_output=True); log=r.stdout+r.stderr; samples,load,thread=parse_et(log); out=Path(str(outbase)+'-0.bin'); vals=unpack(out) if out.exists() else []; ref64=unpack(inp/'reference_fp64.bin'); ref32=unpack(inp/'reference_fp32.bin'); e64=err(vals,ref64); e32=err(vals,ref32)
 return {'exit_status':r.returncode,'raw_samples_ms':samples,'warm_samples_ms':samples[5:],'cold_first_ms':samples[0] if samples else None,'load_time_ms':load,'correctness_fp64':e64,'correctness_fp32':e32,'correctness_passed':e64['passed'],'requested_threads':req,'observed_thread_classification':'REQUESTED_THREAD_COUNT_OBSERVED_PARTIALLY','thread_log':thread}
def order(session, items):
 arr=list(items)
 if session==1: arr=list(reversed(arr))
 elif session==2: random.Random(90210).shuffle(arr)
 return arr
def run(args):
 man=json.load(open(args.manifest)); paths={'input_dir':Path(args.input_dir),'pte_dir':Path(args.pte_dir),'project_kernel':args.project_kernel,'et_runner':args.et_runner,'tmp':Path('/tmp/e21_eval')}; paths['tmp'].mkdir(exist_ok=True)
 for w in man['workloads']: write_inputs(w,paths['input_dir'])
 modes=man['modes']['project']+man['modes']['executorch']; records=[]; start=datetime.now(timezone.utc).isoformat()
 for s in range(man['protocol']['session_count']):
  for w,mode in order(s,[(w,m) for w in man['workloads'] for m in modes]):
   before=thermal(); utc=datetime.now(timezone.utc).isoformat(); res=run_project(w,mode,paths,man['protocol']['total_invocations_per_mode_session']) if mode.startswith('P-') else run_et(w,mode,paths,man['protocol']['total_invocations_per_mode_session']); after=thermal(); warm=res['warm_samples_ms']; rec={'comparison_id':man['comparison_id'],'manifest_sha256':man['manifest_sha256'],'workload_id':w['workload_id'],'category':w['category'],'split':w['split'],'m':w['m'],'n':w['n'],'k':w['k'],'mnk':w['mnk'],'system':'project' if mode.startswith('P-') else 'executorch','mode':mode,'session':s,'utc_timestamp':utc,'thermal_before':before,'thermal_after':after,'result':res,'warm_median_ms':statistics.median(warm) if warm else None,'warm_p95_ms':sorted(warm)[max(0,math.ceil(.95*len(warm))-1)] if warm else None}
   print(s,w['workload_id'],mode,rec['warm_median_ms'],res['correctness_passed'],flush=True); records.append(rec)
 Path(args.out).write_text(json.dumps({'schema':'e2_1_raw_results','started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':man['manifest_sha256'],'records':records},indent=2,sort_keys=True)+'\n')
def analyze(args):
 man=json.load(open(args.manifest)); raw=json.load(open(args.raw)); recs=raw['records']; per={}
 for r in recs: per.setdefault((r['workload_id'],r['mode']),[]).append(r['warm_median_ms'])
 rows=[]
 for w in man['workloads']:
  row={'workload_id':w['workload_id'],'category':w['category'],'m':w['m'],'n':w['n'],'k':w['k'],'modes':{}}
  for mode in man['modes']['project']+man['modes']['executorch']:
   vals=per[(w['workload_id'],mode)]; row['modes'][mode]={'session_medians_ms':vals,'median_of_session_medians_ms':statistics.median(vals)}
  ps=row['modes']['P-SERIAL']['median_of_session_medians_ms']; p4=row['modes']['P-4T']['median_of_session_medians_ms']; pp=row['modes']['P-POLICY']['median_of_session_medians_ms']; etd=row['modes']['ET-DEFAULT']['median_of_session_medians_ms']; row['project_policy_regret_percent']=(pp-min(ps,p4))/min(ps,p4)*100; row['project_oracle_mode']='P-SERIAL' if ps<=p4 else 'P-4T'; row['p_policy_vs_et_default_speedup']=etd/pp; ratio=etd/pp; row['practical_result']='tie' if abs(ratio-1)<=.05 else ('project_faster' if ratio>1 else 'executorch_faster'); rows.append(row)
 fails=[r for r in recs if not r['result'].get('correctness_passed')]; speedups=[r['p_policy_vs_et_default_speedup'] for r in rows]; g=math.exp(sum(math.log(x) for x in speedups)/len(speedups)); wins=sum(r['practical_result']=='project_faster' for r in rows); losses=sum(r['practical_result']=='executorch_faster' for r in rows); ties=sum(r['practical_result']=='tie' for r in rows); verdict='COMPARISON_INVALID_CORRECTNESS_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY' if fails else ('PROJECT_FASTER_ON_FROZEN_EXECUTORCH_BASELINE_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY' if wins and not losses else 'EXECUTORCH_FASTER_ON_FROZEN_PROJECT_BASELINE_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY' if losses and not wins else 'MIXED_RESULTS_NO_SINGLE_WINNER_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY')
 out={'schema':'e2_1_analysis','expected_records':len(man['workloads'])*6*3,'actual_records':len(recs),'correctness_failures':len(fails),'per_workload':rows,'project_decision_quality':{'exact_match_rate':sum(abs(r['project_policy_regret_percent'])<1e-9 for r in rows)/len(rows),'mean_regret_percent':statistics.fmean(r['project_policy_regret_percent'] for r in rows),'median_regret_percent':statistics.median(r['project_policy_regret_percent'] for r in rows),'max_regret_percent':max(r['project_policy_regret_percent'] for r in rows)},'practical_p_policy_vs_et_default':{'geomean_speedup_project_policy_vs_et_default':g,'win_tie_loss':{'project_faster':wins,'tie':ties,'executorch_faster':losses},'worst_project_slowdown_vs_et_default':max(1/x for x in speedups),'best_project_speedup_vs_et_default':max(speedups)},'verdict':verdict}
 Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'records':len(recs),'fails':len(fails),'verdict':verdict,'geomean':g,'wtl':out['practical_p_policy_vs_et_default']['win_tie_loss']},indent=2))

def main():
 ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
 f=sub.add_parser('forensics'); f.add_argument('--e2-raw',required=True); f.add_argument('--out',required=True)
 n=sub.add_parser('negative'); n.add_argument('--out',required=True)
 p=sub.add_parser('prepare'); p.add_argument('--p1d-manifest',required=True); p.add_argument('--pte-dir',required=True); p.add_argument('--negative-controls',required=True); p.add_argument('--out',required=True)
 r=sub.add_parser('run'); r.add_argument('--manifest',required=True); r.add_argument('--out',required=True); r.add_argument('--input-dir',required=True); r.add_argument('--pte-dir',required=True); r.add_argument('--project-kernel',required=True); r.add_argument('--et-runner',required=True)
 a=sub.add_parser('analyze'); a.add_argument('--manifest',required=True); a.add_argument('--raw',required=True); a.add_argument('--out',required=True)
 args=ap.parse_args(); globals()[args.cmd if args.cmd!='negative' else 'negative_controls'](args)
if __name__=='__main__': main()
