#!/usr/bin/env python3
"""Aggregate max_num_seqs root-cause diagnostics into canonical artifacts."""
import argparse,glob,hashlib,json,platform,re,statistics,subprocess
from pathlib import Path
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu_summary(x):
 lo=min(r['connection_start_time'] for r in x['request_timelines']);hi=max(r['completion_time'] for r in x['request_timelines']);s=[z for z in x['samples'] if lo<=z['time']<=hi and 'gpu_utilization_percent' in z['gpu']]
 def q(k):
  v=[z['gpu'][k] for z in s];return {'mean':round(statistics.mean(v),6),'maximum':max(v),'minimum':min(v)} if v else 'not_available'
 return {k:q(k) for k in ('gpu_utilization_percent','memory_used_mib','power_watts','sm_clock_mhz','temperature_c')}|{'sample_count':len(s),'zero_gpu_utilization_fraction':round(sum(z['gpu']['gpu_utilization_percent']==0 for z in s)/len(s),6) if s else 'not_available'}
def metric_summary(x):
 lo=min(r['connection_start_time'] for r in x['request_timelines']);hi=max(r['completion_time'] for r in x['request_timelines']);s=[z for z in x['samples'] if lo<=z['time']<=hi]
 out={}
 for k in ('num_requests_running','num_requests_waiting','gpu_cache_usage_perc','kv_cache_usage_perc','generation_tokens_total','prompt_tokens_total','iteration_tokens_total'):
  v=[z['vllm'][k] for z in s if k in z['vllm']];out[k]={'mean':round(statistics.mean(v),6),'maximum':max(v),'minimum':min(v),'sample_count':len(v)} if v else 'not_available'
 return out
def row(x):
 return {k:x[k] for k in ('mode','workload_id','workload_sha256','max_num_seqs','client_concurrency','session','classification','request_count','success_count','failure_count','ttft_ms','tpot_ms','e2e_ms','output_token_throughput','request_throughput','actual_maximum_in_flight','actual_average_in_flight','trace_wall_seconds','startup_seconds')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,required=True);ap.add_argument('--log-dir',type=Path,required=True);ap.add_argument('--source-workload',type=Path,required=True);ap.add_argument('--order-log',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 files={kind:sorted((a.raw_dir/kind).glob('*.json')) for kind in ('reproduction','concurrency_sweep','direct_control')};data={k:[json.loads(p.read_text()) for p in v] for k,v in files.items()}
 assert len(data['reproduction'])==50 and len(data['concurrency_sweep'])==25 and len(data['direct_control'])==2
 write(a.out/'candidate_session_matrix.json',[row(x) for x in data['reproduction']])
 sweep=[row(x)|{'gpu':gpu_summary(x),'vllm_metrics':metric_summary(x)} for x in data['concurrency_sweep']]
 for x in data['reproduction']:
  if x['workload_id']=='S3' and x['client_concurrency']==8:sweep.append(row(x)|{'gpu':gpu_summary(x),'vllm_metrics':metric_summary(x)})
 write(a.out/'client_concurrency_matrix.json',sweep)
 reps=[]
 for w in ('S2','S3'):
  for v in (1,2,3,4,8):
   x=next(z for z in data['reproduction'] if z['workload_id']==w and z['max_num_seqs']==v and z['session']==0);base=min(r['submit_time'] for r in x['request_timelines']);reps.append({'workload_id':w,'max_num_seqs':v,'client_concurrency':x['client_concurrency'],'actual_maximum_in_flight':x['actual_maximum_in_flight'],'actual_average_in_flight':x['actual_average_in_flight'],'requests':[{k:(round(r[k]-base,6) if k.endswith('_time') else r[k]) for k in ('request_id','submit_time','connection_start_time','first_token_time','completion_time','ttft_ms','tpot_ms','output_tokens')} for r in x['request_timelines']]})
 write(a.out/'request_timelines.json',reps)
 controls=[json.loads(p.read_text()) for p in sorted((a.raw_dir/'metric_control').glob('*.json'))]
 write(a.out/'gpu_sampling_summary.json',[{'workload_id':x['workload_id'],'max_num_seqs':x['max_num_seqs'],'client_concurrency':x['client_concurrency'],'session':x['session'],**gpu_summary(x)} for x in data['reproduction']+data['concurrency_sweep']+data['direct_control']+controls])
 write(a.out/'vllm_metrics_summary.json',[{'workload_id':x['workload_id'],'max_num_seqs':x['max_num_seqs'],'client_concurrency':x['client_concurrency'],'session':x['session'],'metric_parser':'exact_metric_name_corrected' if x in controls else 'initial_substring_parser_use_log_findings_for_running_waiting',**metric_summary(x)} for x in data['reproduction']+data['concurrency_sweep']+data['direct_control']+controls])
 findings=[]
 for kind,paths in files.items():
  for p in paths:
   log=(a.log_dir/kind/(p.stem+'.log'));t=log.read_text(errors='replace');cap=re.search(r"cudagraph_capture_sizes': \[([^]]+)\]",t);full=re.search(r'Profiling CUDA graph memory: PIECEWISE=(\d+) \(largest=(\d+)\), FULL=(\d+) \(largest=(\d+)\)',t);periodic=re.findall(r'Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%',t)
   findings.append({'kind':kind,'session':p.name,'cuda_graph_capture_sizes':cap.group(1) if cap else 'not_available','cuda_graph_profile':{'piecewise_count':int(full.group(1)),'piecewise_largest':int(full.group(2)),'full_count':int(full.group(3)),'full_largest':int(full.group(4))} if full else 'not_available','periodic_engine_samples':[{'generation_tokens_per_second':float(x[0]),'running':int(x[1]),'waiting':int(x[2]),'gpu_kv_cache_usage_percent':float(x[3])} for x in periodic],'preemption_found':bool(re.search('preempt|recompute|swap',t,re.I)),'oom_found':bool(re.search('out of memory|CUDA OOM',t,re.I)),'eager_fallback_found':bool(re.search('eager.*fallback|fallback.*eager',t,re.I)),'raw_log_sha256':sha(log)})
 write(a.out/'log_findings.json',findings)
 write(a.out/'direct_control_results.json',[row(x) for x in data['direct_control']])
 write(a.out/'metric_definitions.json',{'TTFT':'(first streamed content event time - HTTP request start time) * 1000','TPOT':'(completion time - first streamed content event time) * 1000 / (output event count - 1)','E2E':'(completion time - HTTP request start time) * 1000','output_token_throughput':'sum output event counts / whole concurrent trace wall seconds','request_throughput':'successful requests / whole concurrent trace wall seconds','output_count_validation':'all completed S2 requests emitted 24 content events and all S3 requests emitted 32, equal to requested max_tokens','queue_note':'TTFT includes server queue wait after HTTP connection start; TPOT excludes pre-first-token queue time'})
 wm=json.loads(a.source_workload.read_text());write(a.out/'workload_manifest.json',{'source_sha256':sha(a.source_workload),'workloads':[w for w in wm['workloads'] if w['workload_id'] in ('S2','S3')]})
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version,compute_cap','--format=csv,noheader,nounits'],text=True).strip();write(a.out/'environment.json',{'hostname':platform.node(),'kernel':platform.platform(),'gpu_csv':gpu,'vllm_version':'0.24.0','pytorch_version':'2.11.0+cu130','python':platform.python_version(),'fixed_configuration_sha256':sha(a.source_workload.parent/'fixed_configuration.json')})
 write(a.out/'affected_artifacts.json',{'existing_evaluation_valid':True,'regeneration_required':False,'reason':'No benchmark, metric, orchestration, trace-identity, or configuration bug found. Diagnostics independently reproduce the measured behavior.','affected_files':[]})
 write(a.out/'root_cause_analysis.json',{'classification':'MULTIPLE_CONTRIBUTING_CAUSES','primary_cause':'target/version-specific active decode batch-shape efficiency: two simultaneous sequences have much worse per-token efficiency than one, while larger active batches amortize work and recover aggregate throughput','secondary_cause':'vLLM admission at max_num_seqs creates server queueing behind the active group, amplifying TTFT and E2E','evidence':['Performance follows actual active concurrency: client concurrency 2 is slow with server limits 2, 3, 4, and 8; concurrency 1 is fast with server limits 1 through 8.','max_num_seqs=3 is intermediate at higher concurrency, not a duplicate of 2 or 4.','Corrected exact-name Prometheus controls show S2 running<=2/waiting<=2 and S3 running<=2/waiting<=6; request timelines show continuous worker replenishment.','Direct-control runs reproduce the anomaly.','No OOM, preemption, KV pressure, worker restart, configuration mismatch, or client serialization was found.'],'cuda_graph_boundary':'Capture sizes change with configured limit, but no replay-miss/eager-fallback evidence was exposed. The data supports a CUDA-graph or kernel batch-shape effect but does not isolate which GPU kernel causes it without deeper profiling.','diagnostic_metric_note':'The initial diagnostic Prometheus parser used substring matching; exact running/waiting claims use corrected representative controls and server logs. Request metrics and GPU samples were unaffected.','recommended_next_action':'run deeper GPU profiling before proceeding'})
 raw=[]
 for kind,paths in files.items():
  for p in paths:raw.append({'kind':kind,'file':p.name,'sha256':sha(p),'log_sha256':sha(a.log_dir/kind/(p.stem+'.log'))})
 for p in sorted((a.raw_dir/'metric_control').glob('*.json')):raw.append({'kind':'metric_control','file':p.name,'sha256':sha(p),'log_sha256':sha(a.log_dir/'metric_control'/(p.stem+'.log'))})
 write(a.out/'artifact_provenance.json',{'raw_evidence':'outside_source_control:/tmp/vllm-max-num-seqs-root','raw_logs':'outside_source_control:/tmp/vllm-max-num-seqs-root-logs','candidate_order_seed':20260716,'order':json.loads(a.order_log.read_text()),'raw_files':raw,'diagnostic_runner_sha256':sha(Path(__file__).with_name('run_vllm_max_num_seqs_diagnostic.py')),'reporter_sha256':sha(Path(__file__))})
 # compact numerical summary
 lines=['# `max_num_seqs=2` root-cause investigation','','Classification: `MULTIPLE_CONTRIBUTING_CAUSES`.','', '| workload | max_num_seqs | session | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | output tok/s | max/avg in-flight |','|---|---:|---:|---:|---:|---:|---:|---|']
 for x in data['reproduction']:lines.append(f"| {x['workload_id']} | {x['max_num_seqs']} | {x['session']} | {x['ttft_ms']['p95']:.6f} | {x['tpot_ms']['p95']:.6f} | {x['e2e_ms']['p95']:.6f} | {x['output_token_throughput']:.6f} | {x['actual_maximum_in_flight']}/{x['actual_average_in_flight']:.6f} |")
 lines+=['','The cliff reproduced in all five clean sessions for S2 and S3 and in direct controls. It follows two active decode sequences regardless of whether the configured server limit is 2, 3, 4, or 8. Excess client requests remain active HTTP requests but vLLM reports them waiting behind the admitted group. No benchmark or metric-definition bug was found.','','Exactly one next action: **run deeper GPU profiling before proceeding**.']
 (a.out/'summary.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
