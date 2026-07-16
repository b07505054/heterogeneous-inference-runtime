#!/usr/bin/env python3
import argparse,concurrent.futures,json,os,re,signal,statistics,subprocess,sys,threading,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.backends.openai_compatible import OpenAICompatibleBackend,OpenAICompatibleConfig
from benchmark.metrics import latency_summary_ms
from deployment.vllm_adapter.policy_executor import command,live_identity,validate_policy
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--workload-manifest',type=Path,required=True);ap.add_argument('--workload',required=True);ap.add_argument('--session',type=int,required=True);ap.add_argument('--port',type=int,default=8100);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--log',type=Path,required=True);ap.add_argument('--startup-timeout',type=int,default=180);a=ap.parse_args();plan=json.loads(a.plan.read_text());identity=live_identity();proof=validate_policy(plan,plan_path=a.plan,identity=identity);manifest=json.loads(a.workload_manifest.read_text());w=next(x for x in manifest['workloads'] if x['workload_id']==a.workload);cmd=command(plan,port=a.port);a.log.parent.mkdir(parents=True,exist_ok=True);a.out.parent.mkdir(parents=True,exist_ok=True);idle=mem();start=time.time();log=a.log.open('w');env=os.environ.copy();env['VLLM_HOST_IP']='127.0.0.1';proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,text=True,env=env);peak=idle;stop=False
 def monitor():
  nonlocal peak
  while not stop:peak=max(peak,mem());time.sleep(.2)
 th=threading.Thread(target=monitor,daemon=True);th.start();classification='VALID';rows=[];startup=None
 try:
  deadline=time.time()+a.startup_timeout
  while time.time()<deadline:
   if proc.poll() is not None:raise RuntimeError('SERVER_START_FAILED')
   try:
    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/v1/models',timeout=2) as r:
     if r.status==200:startup=time.time()-start;break
   except Exception:time.sleep(1)
  if startup is None:raise RuntimeError('READINESS_TIMEOUT')
  backend=OpenAICompatibleBackend(OpenAICompatibleConfig(base_url=f'http://127.0.0.1:{a.port}',model=plan['fixed_configuration']['served_model_name'],concurrency=w['concurrency'],timeout_s=120))
  for req in w['requests'][:w['warmup_requests']]:backend.execute(req,True)
  measured=w['requests'][w['warmup_requests']:];t0=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=w['concurrency']) as ex:rows=list(ex.map(backend.execute,measured))
  wall=time.perf_counter()-t0
 except RuntimeError as e:classification=str(e)
 finally:
  stop=True;th.join(1);loaded=mem();
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(30)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  log.close();time.sleep(2);after=mem()
 text=a.log.read_text(errors='replace');oom=len(re.findall(r'out of memory|CUDA OOM',text,re.I));classification='OOM_AT_STARTUP' if oom and startup is None else ('OOM_DURING_WORKLOAD' if oom else classification);ok=[r for r in rows if r.get('ok')];fail=[r for r in rows if not r.get('ok')];tokens=sum(r.get('output_tokens',0) for r in ok);wall=locals().get('wall',0);result={"target_identity":plan['target_gpu_identity'],"model_identity":plan['model'],"workload_id":a.workload,"candidate_id":plan['candidate_id'],"max_num_seqs":plan['max_num_seqs'],"value_source":plan['value_source'],"session":a.session,"classification":classification if not fail else 'REQUEST_FAILURE',"request_count":len(rows),"success_count":len(ok),"failure_count":len(fail),"ttft_ms":latency_summary_ms(r['ttft_ms'] for r in ok if r.get('ttft_ms') is not None),"tpot_ms":latency_summary_ms(r['tpot_ms'] for r in ok if r.get('tpot_ms') is not None),"e2e_ms":latency_summary_ms(r['e2e_latency_ms'] for r in ok),"output_token_throughput":tokens/wall if wall else None,"request_throughput":len(ok)/wall if wall else None,"queue_wait_ms":{"p50":"not_available","p95":"not_available","p99":"not_available"},"prefill_time":"not_available","decode_time":"not_available","kv_cache_usage":"not_available","idle_gpu_memory_mib":idle,"model_loaded_gpu_memory_mib":loaded,"peak_gpu_memory_mib":peak,"after_shutdown_gpu_memory_mib":after,"server_startup_seconds":startup,"model_load_seconds":"not_separately_available","oom_count":oom,"timeout_count":sum('timeout' in str(r).lower() for r in fail),"http_server_error_count":len(fail),"server_pid":proc.pid,"command":cmd,"launch_environment":{"VLLM_HOST_IP":"127.0.0.1"},"plan_sha256":proof['plan_sha256'],"runtime_launched_max_num_seqs":plan['max_num_seqs'],"runtime_policy_reselection_count":0,"effective_default_max_num_seqs":effective(text) if plan['value_source']=='default' else plan['max_num_seqs'],"raw_log_sha256":__import__('hashlib').sha256(a.log.read_bytes()).hexdigest(),"request_results":rows,"wall_seconds":wall};a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
def mem():
 try:
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=used_memory','--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL);return sum(int(x.strip()) for x in out.splitlines() if x.strip())
 except Exception:return -1
def effective(text):
 for pat in (r'max_num_seqs["\s:=]+(\d+)',r'Max num sequences[:\s]+(\d+)'):
  m=re.search(pat,text,re.I)
  if m:return int(m.group(1))
 return 'not_available'
if __name__=='__main__':main()
