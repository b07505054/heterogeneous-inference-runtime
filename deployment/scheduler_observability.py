"""Serving S2.8 passive step observability and interpretable selector v3."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib,json,math,time
from typing import Any
from deployment.scheduler_ranking_v2 import SchedulerStateSummary
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import SCHEDULING_POLICIES

REGIONS=("summary_update","policy_selection","plan_generation","plan_validation",
         "model_setup","model_execution","state_commit","instrumentation")
V3_STATIC="ranking_selector_v3_static"
V3_ADAPTIVE="ranking_selector_v3_adaptive"

@dataclass(frozen=True)
class TimerRegion:
    name:str;start_ns:int;end_ns:int
    @property
    def latency_ms(self): return (self.end_ns-self.start_ns)/1e6

@dataclass
class StepInstrumentation:
    trace_id:str;run_id:str;policy_id:str;selector_version:str
    replica_id:str;step_id:int;scheduler_state_version:int
    step_start_ns:int
    state:dict[str,Any]
    selected_work:dict[str,Any]
    regions:list[TimerRegion]=field(default_factory=list)
    execution:dict[str,Any]=field(default_factory=dict)
    transitions:dict[str,int]=field(default_factory=dict)
    step_end_ns:int|None=None

    def add_region(self,name,start_ns,end_ns):
        if name not in REGIONS or end_ns<start_ns:
            raise ServingPlanError("invalid timer region")
        self.regions.append(TimerRegion(name,start_ns,end_ns))

    def finish(self,end_ns:int,tolerance_fraction:float=.10,
               tolerance_ms:float=.25)->dict[str,Any]:
        if end_ns<self.step_start_ns: raise ServingPlanError("invalid step timer")
        self.step_end_ns=end_ns
        ordered=sorted(self.regions,key=lambda x:x.start_ns)
        for a,b in zip(ordered,ordered[1:]):
            if b.start_ns<a.end_ns:
                raise ServingPlanError("timer regions overlap illegally")
        measured=(end_ns-self.step_start_ns)/1e6
        accounted=sum(x.latency_ms for x in ordered)
        unaccounted=max(0.0,measured-accounted)
        allowed=max(tolerance_ms,measured*tolerance_fraction)
        if unaccounted>allowed:
            raise ServingPlanError("unaccounted timing exceeds tolerance")
        return {"measured_step_ms":measured,"accounted_ms":accounted,
                "unaccounted_ms":unaccounted,
                "unaccounted_fraction":unaccounted/max(measured,1e-12),
                "tolerance_ms":allowed}

    def to_dict(self):
        result=asdict(self)
        result["regions"]=[{**asdict(x),"latency_ms":x.latency_ms}
                           for x in self.regions]
        return result

def summary_state_fields(s:SchedulerStateSummary)->dict[str,Any]:
    return {
      "active_request_count":s.active_count,"waiting_request_count":s.waiting_count,
      "prefill_ready_count":s.prefill_ready_count,
      "decode_ready_count":s.decode_ready_count,
      "total_remaining_prefill_tokens":s.total_remaining_prefill,
      "total_remaining_decode_tokens":s.total_remaining_decode,
      "oldest_waiting_age_ms":s.oldest_waiting_age_ms,
      "largest_decode_gap_ms":s.largest_decode_gap_ms,
      "mean_decode_gap_ms":s.mean_decode_gap_ms,
      "token_budget":s.token_budget,"sequence_budget":s.sequence_budget,
      "prefix_hit_tokens":s.prefix_hit_tokens}

@dataclass(frozen=True)
class SelectorV3Configuration:
    version:str
    default_policy:str="chunked_balanced"
    equivalence_margin:float=.05
    hysteresis_margin:float=.08
    decode_pressure_threshold:float=1.25
    ttft_risk_ratio:float=.8
    training_provenance:str="real_qwen_step_observability_v1"
    frozen_digest:str=""
    def payload(self):
        return {k:v for k,v in asdict(self).items() if k!="frozen_digest"}
    def validate(self):
        digest=hashlib.sha256(json.dumps(self.payload(),sort_keys=True,
          separators=(",",":")).encode()).hexdigest()
        if digest!=self.frozen_digest:raise ServingPlanError("selector-v3 changed after freeze")
        if self.default_policy not in SCHEDULING_POLICIES:
            raise ServingPlanError("selector-v3 selected unavailable policy")

def freeze_v3(version=V3_STATIC,**kwargs):
    bare=SelectorV3Configuration(version=version,**kwargs)
    digest=hashlib.sha256(json.dumps(bare.payload(),sort_keys=True,
      separators=(",",":")).encode()).hexdigest()
    return SelectorV3Configuration(**bare.payload(),frozen_digest=digest)

@dataclass(frozen=True)
class SelectorV3Decision:
    policy_id:str;scores:dict[str,float];margin:float;confidence:float
    selection_reason:str;uncertain:bool;used_default:bool
    retained_by_hysteresis:bool;selector_version:str;latency_ms:float

class RankingSelectorV3:
    """Small measured-break-even rule set; no future state is accepted."""
    def __init__(self,config:SelectorV3Configuration,adaptive=False):
        config.validate();self.config=config;self.adaptive=adaptive
        self.current_policy=None
    def select(self,s:SchedulerStateSummary)->SelectorV3Decision:
        started=time.perf_counter_ns()
        p=s.prefill_ready_count;d=s.decode_ready_count
        gap=s.largest_decode_gap_ms/10.0;wait=s.oldest_waiting_age_ms/50.0
        mixed_signal=s.recent_mixed_ms/max(s.recent_decode_ms+s.recent_prefill_ms,1e-9)
        scores={"decode_first":float(p)-1.5*d-2*gap,
                "prefill_first":float(d)-1.5*p-2*wait,
                "chunked_balanced":-.7*min(p,d)-.5*(gap+wait),
                "slo_aware":-max(gap,wait)*2+.2*abs(p-d)}
        if self.adaptive and s.recent_mixed_ms:
            # Only completed past-step timings contribute.
            scores["chunked_balanced"]+=max(0.0,mixed_signal-1.0)
        ordered=sorted(scores,key=lambda x:(scores[x],x))
        scale=max(abs(scores[ordered[0]]),1.0)
        margin=scores[ordered[1]]-scores[ordered[0]]
        uncertain=margin/scale<=self.config.equivalence_margin
        selected=self.config.default_policy if uncertain else ordered[0]
        retained=False
        if self.current_policy and selected!=self.current_policy:
            improvement=scores[self.current_policy]-scores[selected]
            if improvement/scale<=self.config.hysteresis_margin:
                selected=self.current_policy;retained=True
        self.current_policy=selected
        reason=("practical_tie_default" if uncertain else
                "measured_break_even_score")
        return SelectorV3Decision(selected,scores,margin,
          min(1.0,max(0.0,margin/scale)),reason,uncertain,uncertain,retained,
          V3_ADAPTIVE if self.adaptive else V3_STATIC,
          (time.perf_counter_ns()-started)/1e6)

