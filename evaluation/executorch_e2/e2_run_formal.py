#!/usr/bin/env python3
import argparse, hashlib, json, math, os, random, re, statistics, struct, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_KERNEL_ID='portable_fused_matmul_bias_relu_bm32_bn128_bk32'

def sha_file(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
 return h.hexdigest()

def pack(vals): return struct.pack('<'+'f'*len(vals), *vals)
def unpack_file(p):
 b=Path(p).read_bytes(); return list(struct.unpack('<'+'f'*(len(b)//4), b))
def gen(w):
 rng=random.Random(w['seed']); m,n,k=w['m'],w['n'],w['k']
 a=[rng.uniform(-3,3) for _ in range(m*k)]
 b=[rng.uniform(-3,3) for _ in range(k*n)]
 bias=[rng.uniform(-3,3) for _ in range(n)]
 ref=[]
 for i in range(m):
  for j in range(n):
   s=0.0
   for kk in range(k): s += a[i*k+kk]*b[kk*n+j]
   ref.append(max(0.0, s+bias[j]))
 return a,b,bias,ref

def write_inputs(w, root):
 d=root/w['workload_id']; d.mkdir(parents=True, exist_ok=True)
 a,b,bias,ref=gen(w)
 for name, vals in [('a',a),('b',b),('bias',bias),('reference',ref)]:
  p=d/f'{name}.bin'; data=pack(vals); p.write_bytes(data)
  if hashlib.sha256(data).hexdigest()!=w['input_hashes'][name]: raise RuntimeError(f'hash mismatch {w["workload_id"]} {name}')
 return d

def thermal():
 def sh(c): return subprocess.run(c,shell=True,text=True,capture_output=True).stdout.strip()
 return {'temp': sh('vcgencmd measure_temp 2>/dev/null || true'), 'throttled': sh('vcgencmd get_throttled 2>/dev/null || true'), 'freq': sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || true'), 'governor': sh('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true')}

def err(out, ref):
 dif=[abs(x-y) for x,y in zip(out,ref)]
 rel=[abs(x-y)/max(abs(y),1e-30) for x,y in zip(out,ref)]
 return {'max_abs_error': max(dif) if dif else None, 'mean_abs_error': statistics.fmean(dif) if dif else None, 'max_rel_error': max(rel) if rel else None, 'nan_count': sum(math.isnan(x) for x in out), 'inf_count': sum(math.isinf(x) for x in out)}

def run_project(w, mode, paths, repeats):
 if mode=='P-SERIAL': tc,axis,strat=1,'none','serial'; cores='0-3'
 elif mode=='P-4T': tc,axis,strat=4,'m','contiguous_chunks'; cores='0-3'
 elif mode=='P-POLICY':
  if w['mnk'] < 262144: tc,axis,strat=1,'none','serial'
  else: tc,axis,strat=4,'m','contiguous_chunks'
  cores='0-3'
 else: raise ValueError(mode)
 out=paths['tmp']/f'{w["workload_id"]}_{mode}.bin'
 cmd=['taskset','-c',cores,paths['project_kernel'],'--m',str(w['m']),'--n',str(w['n']),'--k',str(w['k']),'--a',str(paths['input_dir']/w['workload_id']/ 'a.bin'),'--b',str(paths['input_dir']/w['workload_id']/ 'b.bin'),'--bias',str(paths['input_dir']/w['workload_id']/ 'bias.bin'),'--out',str(out),'--kernel-id',PROJECT_KERNEL_ID,'--thread-count',str(tc),'--partition-axis',axis,'--partition-strategy',strat,'--repeats',str(repeats)]
 t0=time.monotonic(); r=subprocess.run(cmd,text=True,capture_output=True); t1=time.monotonic()
 if r.returncode!=0: return {'exit_status': r.returncode, 'stderr': r.stderr, 'command': cmd}
 js=json.loads(r.stdout); samples=js['samples_ms']; vals=unpack_file(out); ref=unpack_file(paths['input_dir']/w['workload_id']/ 'reference.bin')
 return {'exit_status':0,'raw_samples_ms':samples,'cold_first_ms':samples[0],'warm_samples_ms':samples[5:],'load_time_ms':None,'process_elapsed_ms':(t1-t0)*1000,'output_sha256':sha_file(out),'correctness': err(vals,ref), 'correctness_passed': err(vals,ref)['max_abs_error'] <= 1e-3 and err(vals,ref)['max_rel_error'] <= 1e-4, 'requested_threads':tc, 'observed_thread_classification':'VERIFIED_BY_PROJECT_CONTRACT_SELF_REPORT', 'runner_report': js}

def parse_et(log):
 samples=[float(x) for x in re.findall(r'Iteration \d+ of \d+: ([0-9.]+) ms', log)]
 load=re.search(r'Model loaded in ([0-9.]+) ms', log)
 thread=re.findall(r'Resetting threadpool[^\n]*', log)
 return samples, float(load.group(1)) if load else None, thread

def run_et(w, mode, paths, repeats):
 if mode=='ET-DEFAULT': extra=[]; req='default'
 elif mode=='ET-1T-REQUESTED': extra=['--cpu_threads','1']; req=1
 elif mode=='ET-4T-REQUESTED': extra=['--cpu_threads','4']; req=4
 else: raise ValueError(mode)
 pte=paths['pte_dir']/w['executorch_xnnpack_pte']['filename']
 outbase=paths['tmp']/f'{w["workload_id"]}_{mode}'
 for p in paths['tmp'].glob(f'{w["workload_id"]}_{mode}*.bin'): p.unlink()
 inputs=','.join(str(paths['input_dir']/w['workload_id']/x) for x in ['a.bin','b.bin','bias.bin'])
 cmd=['taskset','-c','0-3',paths['et_runner'],'--model_path',str(pte),'--inputs',inputs,'--num_executions',str(repeats),'--print_output','none','--output_file',str(outbase)] + extra
 t0=time.monotonic(); r=subprocess.run(cmd,text=True,capture_output=True); t1=time.monotonic()
 log=r.stdout+r.stderr
 if r.returncode!=0: return {'exit_status': r.returncode, 'log': log, 'command': cmd}
 samples, load_ms, threadlog=parse_et(log); out=Path(str(outbase)+'-0.bin'); vals=unpack_file(out); ref=unpack_file(paths['input_dir']/w['workload_id']/ 'reference.bin'); e=err(vals,ref)
 cls='REQUESTED_THREAD_COUNT_OBSERVED_PARTIALLY'
 if mode=='ET-1T-REQUESTED' and any('1 threads' in s or '= 1' in s for s in threadlog): cls='REQUESTED_THREAD_COUNT_OBSERVED_PARTIALLY'
 if mode in ('ET-DEFAULT','ET-4T-REQUESTED') and any('4 threads' in s or '= 4' in s for s in threadlog): cls='REQUESTED_THREAD_COUNT_OBSERVED_PARTIALLY'
 return {'exit_status':0,'raw_samples_ms':samples,'cold_first_ms':samples[0] if samples else None,'warm_samples_ms':samples[5:],'load_time_ms':load_ms,'process_elapsed_ms':(t1-t0)*1000,'output_sha256':sha_file(out),'correctness': e, 'correctness_passed': e['max_abs_error'] <= 1e-3 and e['max_rel_error'] <= 1e-4, 'requested_threads':req, 'observed_thread_classification': cls, 'thread_log':threadlog}

def order_for(session, items):
 arr=list(items)
 if session==1: arr=list(reversed(arr))
 elif session==2:
  rng=random.Random(90210); rng.shuffle(arr)
 return arr

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--out',required=True); ap.add_argument('--input-dir',required=True); ap.add_argument('--pte-dir',required=True); ap.add_argument('--project-kernel',required=True); ap.add_argument('--et-runner',required=True)
 args=ap.parse_args(); man=json.load(open(args.manifest)); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
 paths={'input_dir':Path(args.input_dir),'pte_dir':Path(args.pte_dir),'project_kernel':args.project_kernel,'et_runner':args.et_runner,'tmp':Path('/tmp/e2_eval')}; paths['tmp'].mkdir(exist_ok=True)
 for w in man['workloads']: write_inputs(w, paths['input_dir'])
 modes=man['modes']['project']+man['modes']['executorch']; records=[]
 started=datetime.now(timezone.utc).isoformat()
 for session in range(man['protocol']['session_count']):
  pairs=[(w,m) for w in man['workloads'] for m in modes]
  for w,mode in order_for(session,pairs):
   before=thermal(); utc=datetime.now(timezone.utc).isoformat()
   if mode.startswith('P-'): res=run_project(w,mode,paths,man['protocol']['total_invocations_per_mode_session'])
   else: res=run_et(w,mode,paths,man['protocol']['total_invocations_per_mode_session'])
   after=thermal()
   warm=res.get('warm_samples_ms') or []
   rec={'comparison_id':man['comparison_id'],'manifest_sha256':man['manifest_sha256'],'workload_id':w['workload_id'],'category':w['category'],'split':w['split'],'m':w['m'],'n':w['n'],'k':w['k'],'mnk':w['mnk'],'system':'project' if mode.startswith('P-') else 'executorch','mode':mode,'session':session,'utc_timestamp':utc,'thermal_before':before,'thermal_after':after,'affinity':'0-3','timing_boundary':'warm_execution','warmup_count':man['protocol']['warmup_count'],'timed_repeat_count':man['protocol']['timed_repeat_count'],'artifact_hash': PROJECT_KERNEL_ID if mode.startswith('P-') else w['executorch_xnnpack_pte']['sha256'], 'result':res, 'warm_median_ms': statistics.median(warm) if warm else None, 'warm_mean_ms': statistics.fmean(warm) if warm else None, 'warm_p95_ms': sorted(warm)[max(0, math.ceil(0.95*len(warm))-1)] if warm else None, 'warm_min_ms': min(warm) if warm else None, 'warm_max_ms': max(warm) if warm else None}
   records.append(rec); print(session,w['workload_id'],mode,rec['warm_median_ms'],res.get('correctness_passed'), flush=True)
 result={'schema':'e2_formal_raw_results','schema_version':1,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':man['manifest_sha256'],'records':records}
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(out)
if __name__=='__main__': main()
