#!/usr/bin/env python3
"""Focused repeated real-Qwen S2.8 step observability benchmark."""
from __future__ import annotations
import argparse,json,math,os,random,resource,statistics,time
from pathlib import Path
import torch
from deployment.scheduler_observability import (
 RankingSelectorV3,StepInstrumentation,freeze_v3,summary_state_fields)
from deployment.scheduler_ranking_v2 import IncrementalSchedulerState
from deployment.scheduler_policy_v4 import (
 RankingSelectorV4,freeze_v4,PAIRWISE_V4,RISK_V4)
from deployment.serving_scheduler import (
 PlanOnlySchedulerRuntime,ReplicaSchedulerState,RequestExecutionState,
 SchedulerCompiler,SchedulerProfile,deserialize_schedule_plan)

FIXED=("decode_first","prefill_first","chunked_balanced","slo_aware")
POLICIES=FIXED+("ranking_selector_v3_static","ranking_selector_v3_adaptive",
 PAIRWISE_V4,RISK_V4)

def trace(kind):
 common=[151643]*8
 if kind=="mixed_interactive":
  return [common+[100,101],common+[102]*8,[151644]*12,[151645]*20],[0,.04,.08,.12],[2]*4
 if kind=="contention":
  return [[151643]*24,[151644]*8,[151645]*8,[151646]*16],[0,0,.02,.04],[2,4,4,2]
 if kind=="prefill_heavy":
  return [[151643+i]*n for i,n in enumerate((8,24,32,40))],[0,.01,.02,.03],[2]*4
 if kind=="arrival_burst_s29":
  return [[151660+i]*n for i,n in enumerate((7,13,21,29))],[0,0,.08,.08],[3,2,4,3]
 if kind=="long_prefill_s29":
  return [[151670]*48,[151671]*9,[151672]*17,[151673]*25],[0,.01,.03,.05],[2,3,2,3]
 if kind=="prefix_heavy_s29":
  return [[151680]*16+[i]*4 for i in range(4)],[0,.02,.04,.06],[3,3,3,3]
 if kind=="budget_pressure_s29":
  return [[151690+i]*n for i,n in enumerate((11,19,27,35))],[0,.01,.02,.03],[4,2,3,2]
 if kind=="mixed_final_s210":
  return [[151700]*10,[151701]*18,[151702]*14,[151703]*26],[0,.03,.07,.11],[3,2,4,3]
 if kind=="long_final_s210":
  return [[151710]*56,[151711]*12,[151712]*20,[151713]*30],[0,.02,.04,.06],[2,3,2,3]
 if kind=="burst_final_s210":
  return [[151720]*6,[151721]*15,[151722]*23,[151723]*31],[0,0,.09,.09],[4,3,2,4]
 if kind=="decode_final_s210":
  return [[151730+i]*9 for i in range(4)],[0,.01,.02,.03],[5,5,5,5]
 if kind=="mixed_replica_a_s211":
  return [[151740]*8,[151741]*22,[151742]*13,[151743]*31],[0,.025,.06,.095],[4,2,5,3]
 if kind=="mixed_replica_b_s211":
  return [[151744]*17,[151745]*7,[151746]*28,[151747]*12],[0,.015,.055,.11],[3,5,2,4]
 if kind=="contention_a_s211":
  return [[151748]*35,[151749]*9,[151750]*9,[151751]*20],[0,0,.018,.042],[2,6,6,3]
 if kind=="burst_a_s211":
  return [[151752]*6,[151753]*18,[151754]*27,[151755]*11],[0,0,.075,.075],[5,3,2,5]
 if kind=="long_prefill_a_s211":
  return [[151756]*62,[151757]*14,[151758]*25,[151759]*8],[0,.015,.04,.085],[2,4,3,4]
 if kind=="prefix_lowreuse_a_s211":
  return [[151760+i]*n for i,n in enumerate((12,20,29,37))],[0,.02,.045,.09],[4,3,4,2]
 if kind=="decode_heavy_s212":
  return [[151800+i]*10 for i in range(4)],[0,.012,.027,.051],[7,6,7,6]
 if kind=="prefill_heavy_s212":
  return [[151804+i]*n for i,n in enumerate((27,39,51,63))],[0,.008,.019,.033],[2,2,3,2]
 if kind=="mixed_interactive_s212":
  return [[151808]*11,[151809]*24,[151810]*7,[151811]*36],[0,.021,.057,.103],[4,3,6,2]
 if kind=="contention_s212":
  return [[151812]*41,[151813]*8,[151814]*15,[151815]*22],[0,0,.014,.039],[2,7,5,3]
 if kind=="arrival_burst_s212":
  return [[151816]*9,[151817]*17,[151818]*31,[151819]*13],[0,0,.066,.066],[5,4,2,5]
 if kind=="long_prefill_s212":
  return [[151820]*71,[151821]*16,[151822]*23,[151823]*34],[0,.017,.049,.098],[2,4,3,2]
 if kind=="prefix_heavy_s212":
  common=[151824]*20
  return [common+[201]*6,common+[202]*11,common+[203]*4,common+[204]*16],[0,.018,.046,.083],[3,4,3,2]
 if kind=="low_prefix_reuse_s212":
  return [[151825+i]*n for i,n in enumerate((14,26,33,19))],[0,.026,.052,.107],[4,3,2,5]
 if kind=="small_token_budget_s212":
  return [[151829+i]*n for i,n in enumerate((13,21,29,37))],[0,.01,.031,.072],[4,3,4,3]
 if kind=="large_token_budget_s212":
  return [[151833+i]*n for i,n in enumerate((18,42,25,57))],[0,.019,.043,.088],[3,2,4,3]
 if kind=="small_sequence_budget_s212":
  return [[151837+i]*n for i,n in enumerate((12,19,26,33))],[0,.009,.024,.061],[4,4,3,3]
 if kind=="larger_active_batch_s212":
  return [[151841+i]*n for i,n in enumerate((8,12,16,20,24,28))],[0,.006,.013,.021,.034,.055],[3,4,3,2,4,3]
 return [[151643+i]*8 for i in range(4)],[0,.02,.04,.06],[4,4,4,4]

def scheduler_profile(kind):
 if kind=="small_token_budget_s212":
  return SchedulerProfile(4,6,4,1,1)
 if kind=="large_token_budget_s212":
  return SchedulerProfile(4,24,16,1,4)
 if kind=="small_sequence_budget_s212":
  return SchedulerProfile(2,10,6,1,1)
 if kind=="larger_active_batch_s212":
  return SchedulerProfile(6,16,8,1,3)
 return SchedulerProfile(4,10,6,1,2)

def ci(xs):
 mean=statistics.fmean(xs);sd=statistics.stdev(xs) if len(xs)>1 else 0
 half=2.776*sd/math.sqrt(len(xs)) if len(xs)>1 else 0
 return {"n":len(xs),"mean":mean,"median":statistics.median(xs),
         "stdev":sd,"ci95":[mean-half,mean+half]}

def run_once(model,kind,policy,run_id):
 prompts,arrivals,outputs=trace(kind)
 profile=scheduler_profile(kind)
 state=ReplicaSchedulerState("replica-0",profile);local={}
 for i,(tokens,arrival,out) in enumerate(zip(prompts,arrivals,outputs)):
  rid=f"{kind}-{i}";r=RequestExecutionState(rid,f"sp-{rid}","replica-0",
    arrival,len(tokens),0,out);state.ingest(r)
  local[rid]={"prompt":torch.tensor([tokens]),"cache":None,"generated":[],
              "pending":None,"first":None,"done":None,"token_times":[],
              "prefill_start":None,"prefill_done":None}
 inc=IncrementalSchedulerState(state);compiler=SchedulerCompiler()
 runtime=PlanOnlySchedulerRuntime();selector=None
 if policy.startswith("ranking_selector_v3"):
  selector=RankingSelectorV3(freeze_v3(),policy.endswith("adaptive"))
 elif policy==PAIRWISE_V4:selector=RankingSelectorV4(freeze_v4(PAIRWISE_V4))
 elif policy==RISK_V4:selector=RankingSelectorV4(freeze_v4(RISK_V4),True)
 attention_ns=0;attention_calls=0;oproj=0
 starts={}
 def apre(m,args):starts[id(m)]=time.perf_counter_ns()
 def apost(m,args,out):
  nonlocal attention_ns,attention_calls
  attention_ns+=time.perf_counter_ns()-starts.pop(id(m));attention_calls+=1
 def ohook(m,args):
  nonlocal oproj;oproj+=1
 hooks=[]
 for layer in model.model.layers:
  hooks.extend([layer.self_attn.register_forward_pre_hook(apre),
                layer.self_attn.register_forward_hook(apost),
                layer.self_attn.o_proj.register_forward_pre_hook(ohook)])
 usage0=resource.getrusage(resource.RUSAGE_SELF)
 wall0=time.perf_counter();rows=[];request_events=[]
 try:
  while state.unfinished():
   if not state.ready():
    future=[r.arrival_time_ms for r in state.requests.values() if r.phase=="WAITING"]
    state.clock_ms=min(future);state.ready()
   step_start=time.perf_counter_ns()
   for r in state.requests.values():
    if r.arrival_time_ms<=state.clock_ms and not r.finished and r.request_id not in inc.hot:
     inc.upsert(r,state.clock_ms)
   a=step_start;s=inc.snapshot();b=time.perf_counter_ns()
   selected_policy=policy
   if selector:selected_policy=selector.select(s).policy_id
   c=time.perf_counter_ns()
   plan=compiler.compile(state,forced_policy=selected_policy);d=time.perf_counter_ns()
   loaded=deserialize_schedule_plan(plan.serialize(),state);e=time.perf_counter_ns()
   before_phase={x.request_id:x.phase for x in state.requests.values()}
   attn_before=attention_calls;oproj_before=oproj;attn_ns_before=attention_ns
   shape_rows=[];item_events=[]
   def execute(req,item,_plan):
    x=local[req.request_id]
    callback_start=time.perf_counter_ns()
    if item.phase=="prefill" and x["prefill_start"] is None:
     x["prefill_start"]=(callback_start/1e6)
    if item.phase=="decode" and x["pending"] is not None:
     token=int(x["pending"].argmax(-1));x["pending"]=None;x["generated"].append(token)
     commit=time.perf_counter_ns();now=(time.perf_counter()-wall0)*1000
     x["first"]=x["first"] or now;x["token_times"].append(now)
     if len(x["generated"])==req.expected_output_tokens:x["done"]=now
     item_events.append({"request_id":req.request_id,"phase":item.phase,
       "token_start":item.token_start,"token_count":item.token_count,
       "callback_start_ns":callback_start,"commit_ns":commit})
     return {}
    current=(x["prompt"][:,item.token_start:item.token_start+item.token_count]
      if item.phase=="prefill" else torch.tensor([[x["generated"][-1]]]))
    kv_len=(x["cache"].get_seq_length() if x["cache"] is not None else 0)
    shape_rows.append({"request_id":req.request_id,"batch":int(current.shape[0]),
      "query_length":int(current.shape[1]),"kv_length_before":int(kv_len),
      "signature":f"b{current.shape[0]}_q{current.shape[1]}_kv{kv_len}"})
    with torch.no_grad():
     out=model(current,past_key_values=x["cache"],use_cache=True,logits_to_keep=1)
    x["cache"]=out.past_key_values
    if item.phase=="prefill" and item.token_start+item.token_count==len(x["prompt"][0]):
     x["pending"]=out.logits[:,-1].float()
     x["prefill_done"]=time.perf_counter_ns()/1e6
    elif item.phase=="decode":
     token=int(out.logits[:,-1].argmax(-1));x["generated"].append(token)
     now=(time.perf_counter()-wall0)*1000;x["first"]=x["first"] or now
     x["token_times"].append(now)
     if len(x["generated"])==req.expected_output_tokens:x["done"]=now
    item_events.append({"request_id":req.request_id,"phase":item.phase,
      "token_start":item.token_start,"token_count":item.token_count,
      "callback_start_ns":callback_start,"commit_ns":time.perf_counter_ns()})
    return {}
   runtime.execute(state,loaded,execute);f=time.perf_counter_ns()
   changed=[state.requests[x.request_id] for x in loaded.items]
   for r in changed:inc.upsert(r,state.clock_ms)
   g=time.perf_counter_ns()
   transitions=sum(before_phase[r.request_id]=="PREFILL" and r.phase=="DECODE"
                   for r in changed)
   completions=sum(r.finished for r in changed)
   work={"scheduled_prefill_tokens":sum(x.token_count for x in loaded.items if x.phase=="prefill"),
    "scheduled_decode_tokens":sum(x.token_count for x in loaded.items if x.phase=="decode"),
    "scheduled_decode_sequences":sum(x.phase=="decode" for x in loaded.items),
    "scheduled_sequence_count":len(loaded.items)}
   rec=StepInstrumentation(kind,str(run_id),selected_policy,
    selector.config.version if selector else "fixed_policy","replica-0",
    state.step_id-1,s.state_version,step_start,summary_state_fields(s),work)
   for name,x,y in (("summary_update",a,b),("policy_selection",b,c),
    ("plan_generation",c,d),("plan_validation",d,e),
    ("model_execution",e,f),("state_commit",f,g)):
    rec.add_region(name,x,y)
   rec.execution={"model_forward_count":len(shape_rows),
    "attention_invocation_count":attention_calls-attn_before,
    "attention_to_o_proj_count":oproj-oproj_before,
    "operator_plan_count":0,
    "operator_worker_event_count":0,"shapes":shape_rows,
    "scheduled_items":item_events,
    "attention_path_ms":(attention_ns-attn_ns_before)/1e6}
   rec.transitions={"new_arrival_count":0,
    "prefill_to_decode_transition_count":transitions,"completion_count":completions}
   accounting=rec.finish(g,.15,.5);row=rec.to_dict();row["accounting"]=accounting
   rows.append(row)
  elapsed=(time.perf_counter()-wall0)*1000
 finally:
  for h in hooks:h.remove()
 ttft=[x["first"] for x in local.values()];e2e=[x["done"] for x in local.values()]
 usage1=resource.getrusage(resource.RUSAGE_SELF)
 return {"trace_id":kind,"run_id":run_id,"requested_policy":policy,
 "objective":statistics.fmean(ttft)+.25*statistics.fmean(e2e),
  "ttft_mean_ms":statistics.fmean(ttft),"e2e_mean_ms":statistics.fmean(e2e),
  "makespan_ms":elapsed,"host_wall_start_ns":int(wall0*1e9),
  "direct_request_timestamps":{k:{"arrival_ms":0.0,
    "prefill_start_ms":(v["prefill_start"]-wall0*1000 if v["prefill_start"] else None),
    "prefill_completion_ms":(v["prefill_done"]-wall0*1000 if v["prefill_done"] else 0.0),
    "first_token_ms":v["first"],"decode_token_times_ms":v["token_times"],
    "completion_ms":v["done"]} for k,v in local.items()},
  "steps":rows,"generated":{k:v["generated"] for k,v in local.items()},
  "attention_to_o_proj":oproj,"runtime_counters":runtime.counters(),
  "runtime_noise":{"process_cpu_seconds":
    (usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime),
    "voluntary_context_switches":usage1.ru_nvcsw-usage0.ru_nvcsw,
    "involuntary_context_switches":usage1.ru_nivcsw-usage0.ru_nivcsw,
    "minor_page_faults":usage1.ru_minflt-usage0.ru_minflt,
    "major_page_faults":usage1.ru_majflt-usage0.ru_majflt,
    "load_average_1m":os.getloadavg()[0] if hasattr(os,"getloadavg") else None,
    "model_warm_state":True}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",type=Path,required=True)
 ap.add_argument("--output",type=Path,required=True);ap.add_argument("--warmups",type=int,default=1)
 ap.add_argument("--runs",type=int,default=3)
 ap.add_argument("--traces",default="mixed_interactive,contention,prefill_heavy,decode_heavy")
 ap.add_argument("--policies",default=",".join(POLICIES))
 ap.add_argument("--order-mode",choices=("forward_reverse","counterbalanced"),
                 default="forward_reverse")
 ap.add_argument("--order-seed",type=int,default=0)
 a=ap.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
 from transformers import AutoModelForCausalLM
 torch.manual_seed(20260718);torch.set_num_threads(4)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
  dtype=torch.float32,attn_implementation="eager").eval()
 traces=tuple(x.strip() for x in a.traces.split(","))
 policies_requested=tuple(x.strip() for x in a.policies.split(","))
 if any(x not in POLICIES for x in policies_requested):raise ValueError("unknown policy")
 raw=[];order=[]
 for group in range(a.warmups+a.runs):
  if a.order_mode=="forward_reverse":
   policies=policies_requested if group%2==0 else tuple(reversed(policies_requested))
  else:
   base=list(policies_requested);random.Random(a.order_seed).shuffle(base)
   offset=group%len(base);base=base[offset:]+base[:offset]
   policies=tuple(reversed(base)) if (group//len(base))%2 else tuple(base)
  for t in traces:
   for p in policies:
    order.append({"group":group,"trace":t,"policy":p,"measured":group>=a.warmups})
    r=run_once(model,t,p,group)
    if group>=a.warmups:raw.append(r)
 summary={}
 for t in traces:
  summary[t]={}
  for p in policies_requested:
   rs=[r for r in raw if r["trace_id"]==t and r["requested_policy"]==p]
   summary[t][p]={"objective":ci([r["objective"] for r in rs]),
    "ttft":ci([r["ttft_mean_ms"] for r in rs]),"e2e":ci([r["e2e_mean_ms"] for r in rs]),
    "outputs_equivalent":all(r["generated"]==rs[0]["generated"] for r in rs),
    "oproj_events":sum(r["attention_to_o_proj"] for r in rs)}
 a.output.write_text(json.dumps({"execution_mode":"real_qwen_eager_cpu_observability",
 "truth_boundary":"real Qwen full model; eager attention, not compiler-attention provenance",
  "order_mode":a.order_mode,"order_seed":a.order_seed,
  "order":order,"raw_runs":raw,"summary":summary},indent=2)+"\n")
if __name__=="__main__":main()
