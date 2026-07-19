"""S2.10 conservative policy-level pairwise and risk-aware selectors."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json,math,time
from deployment.scheduler_ranking_v2 import SchedulerStateSummary
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import SCHEDULING_POLICIES

PAIRWISE_V4="ranking_selector_v4_pairwise"
RISK_V4="ranking_selector_v4_risk_aware"

@dataclass(frozen=True)
class SelectorV4Configuration:
 version:str
 robust_default:str="decode_first"
 frontier_size:int=8
 decision_margin:float=.12
 uncertainty_threshold:float=.35
 equivalence_margin:float=.02
 rollout_horizon:int=8
 model_version:str="policy_pairwise_rules_v1"
 frozen_digest:str=""
 def payload(self):return {k:v for k,v in asdict(self).items() if k!="frozen_digest"}
 def validate(self):
  d=hashlib.sha256(json.dumps(self.payload(),sort_keys=True,
    separators=(",",":")).encode()).hexdigest()
  if d!=self.frozen_digest:raise ServingPlanError("selector-v4 changed after freeze")
  if self.robust_default not in SCHEDULING_POLICIES:
   raise ServingPlanError("invalid robust default")
  if self.frontier_size<1 or self.rollout_horizon<1:
   raise ServingPlanError("invalid v4 rollout configuration")

def freeze_v4(version,**kw):
 x=SelectorV4Configuration(version=version,**kw)
 d=hashlib.sha256(json.dumps(x.payload(),sort_keys=True,
   separators=(",",":")).encode()).hexdigest()
 return SelectorV4Configuration(**x.payload(),frozen_digest=d)

@dataclass(frozen=True)
class PolicyUncertainty:
 support_count:int;residual_std_ms:float;ood_score:float
 rollout_uncertainty:float;frontier_truncation:float
 @property
 def total(self):return min(1.0,self.ood_score+self.rollout_uncertainty+
   self.frontier_truncation+min(.25,self.residual_std_ms/1000))

@dataclass(frozen=True)
class V4Decision:
 policy_id:str;scores:dict[str,float];pairwise_preferences:dict[str,float]
 uncertainty:dict[str,float];coverage:bool;default_used:bool
 selection_reason:str;selector_version:str;latency_ms:float

def policy_sequence_features(s:SchedulerStateSummary)->dict[str,float]:
 active=max(s.active_count,1);budget=max(s.token_budget,1)
 return {"prefill_per_request":s.total_remaining_prefill/active,
  "decode_per_request":s.total_remaining_decode/active,
  "prefill_pressure":s.total_remaining_prefill/budget,
  "decode_pressure":s.decode_ready_count/max(s.sequence_budget,1),
  "mixed":float(bool(s.prefill_ready_count and s.decode_ready_count)),
  "wait_risk":s.oldest_waiting_age_ms/50.0,
  "gap_risk":s.largest_decode_gap_ms/10.0,
  "frontier_coverage":len(s.frontier)/active,
  "predicted_steps":(s.total_remaining_prefill/budget+
    s.total_remaining_decode/max(s.sequence_budget,1))}

class RankingSelectorV4:
 def __init__(self,config:SelectorV4Configuration,risk_aware=False):
  config.validate();self.config=config;self.risk_aware=risk_aware
 def select(self,s:SchedulerStateSummary)->V4Decision:
  t=time.perf_counter_ns();f=policy_sequence_features(s)
  scores={"decode_first":0.0,
   "prefill_first":.35*f["prefill_pressure"]-.15*f["decode_pressure"],
   "chunked_balanced":.08-.16*f["mixed"]-.04*min(f["prefill_per_request"]/32,2),
   "slo_aware":.06-.18*max(f["wait_risk"],f["gap_risk"])-.08*f["mixed"]}
  prefs={f"{a}_vs_{b}":scores[a]-scores[b] for i,a in enumerate(SCHEDULING_POLICIES)
         for b in SCHEDULING_POLICIES[i+1:]}
  frontier_trunc=max(0.0,1-f["frontier_coverage"])
  ood=min(1.0,max(0.0,(f["prefill_per_request"]-64)/128))
  base_unc=PolicyUncertainty(8,15.0,ood,.05*f["predicted_steps"]/8,
                             .25*frontier_trunc)
  uncertainty={p:base_unc.total for p in SCHEDULING_POLICIES}
  candidate=min(scores,key=lambda p:(scores[p],p));improvement=scores[
   self.config.robust_default]-scores[candidate]
  covered=(uncertainty[candidate]<=self.config.uncertainty_threshold and
           improvement>self.config.decision_margin)
  selected=candidate
  reason="pairwise_best"
  default=False
  if self.risk_aware and not covered:
   selected=self.config.robust_default;default=True
   reason="no_regression_default"
  return V4Decision(selected,scores,prefs,uncertainty,covered,default,reason,
    RISK_V4 if self.risk_aware else PAIRWISE_V4,
    (time.perf_counter_ns()-t)/1e6)

