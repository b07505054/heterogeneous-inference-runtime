"""S2.6 frozen/fast selector modes and low-overhead online planning."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib,json,math,time
from typing import Any

from deployment.scheduler_calibration import (
    CalibratedQwenCPUServiceModel, ObjectiveProfile, make_objective,
    scheduler_features)
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    ReplicaSchedulerState, SchedulerCompiler, SCHEDULING_POLICIES)

SELECTOR_V1_FROZEN = "scheduler_selector_v1_frozen"
SELECTOR_V1_FAST = "scheduler_selector_v1_fast"
SELECTION_MODES = ("offline_trace","epoch","online_step")
SUPPORTED_HORIZONS = (1,2,4,8)


@dataclass(frozen=True)
class FrozenSelectorConfiguration:
    selector_version: str = SELECTOR_V1_FROZEN
    feature_version: str = "scheduler_features_v1"
    objective_version: str = "request_objective_v1"
    service_model_version: str = "real_qwen_cpu_service_model_v1"
    horizon: str = "full_trace"
    terminal_cost: str = "enabled"
    epoch_policy: str = "complete_trace"
    tie_break: str = "objective_then_policy_id"
    immutable_digest: str = (
        "ffa2b1bee77f9acef0498d132ca05e5e6058124ff9606b70762199d87d02fcaa")

    def validate_immutable(self) -> None:
        payload=(self.selector_version,self.feature_version,self.objective_version,
                 self.service_model_version,self.horizon,self.terminal_cost,
                 self.epoch_policy,self.tie_break)
        digest=hashlib.sha256("|".join(payload).encode()).hexdigest()
        if digest != self.immutable_digest:
            raise ServingPlanError("selector configuration modified after freeze")


@dataclass(frozen=True)
class FastSelectorConfiguration:
    selector_version: str = SELECTOR_V1_FAST
    selection_mode: str = "epoch"
    objective_profile: str = "balanced_interactive"
    adaptive_horizons: tuple[int,...] = SUPPORTED_HORIZONS
    epoch_steps: int = 4
    evidence_mode: bool = False
    practical_equivalence_margin: float = .02

    def __post_init__(self):
        if self.selection_mode not in ("epoch","online_step"):
            raise ServingPlanError("offline selector cannot be used as online mode")
        if any(h not in SUPPORTED_HORIZONS for h in self.adaptive_horizons):
            raise ServingPlanError("adaptive horizon selects unsupported value")
        if self.epoch_steps<1 or not 0<=self.practical_equivalence_margin<1:
            raise ServingPlanError("invalid epoch or equivalence margin")


@dataclass(frozen=True)
class FastSchedulerPolicyPlan:
    plan_id:str
    replica_id:str
    policy_id:str
    selection_mode:str
    horizon:int
    valid_for_steps:int
    objective_profile:str
    candidates_generated:int
    candidates_pruned:int
    candidates_rolled_out:int
    pruning_reasons:dict[str,str]
    predicted_scores:dict[str,float]
    selector_version:str=SELECTOR_V1_FAST


def online_state_view(state:ReplicaSchedulerState)->ReplicaSchedulerState:
    """Return an arrived-only state; future arrivals cannot leak online."""
    import copy
    view=ReplicaSchedulerState(state.replica_id,state.profile,step_id=state.step_id,
                               version=state.version,clock_ms=state.clock_ms)
    for rid,request in state.requests.items():
        if request.arrival_time_ms<=state.clock_ms:
            view.requests[rid]=copy.deepcopy(request)
    view.finished_ids=list(state.finished_ids)
    view.statistics=dict(state.statistics)
    return view


def choose_adaptive_horizon(state:ReplicaSchedulerState)->int:
    f=scheduler_features(state)
    if f["prefill_ready_count"] and f["decode_ready_count"]:
        if f["oldest_waiting_age_ms"]>=.8*state.profile.ttft_slo_ms or \
           f["largest_decode_gap_ms"]>=.8*state.profile.maximum_decode_gap_ms:
            return 8
        return 4
    if f["prefill_ready_count"]+f["decode_ready_count"]>state.profile.max_num_seqs:
        return 2
    return 1


def prune_candidates(state:ReplicaSchedulerState):
    compiler=SchedulerCompiler()
    ready=state.ready();has_p=any(r.phase=="PREFILL" for r in ready)
    has_d=any(r.phase=="DECODE" for r in ready)
    kept=[];reasons={};signatures={}
    for policy in SCHEDULING_POLICIES:
        if policy=="prefill_first" and not has_p:
            reasons[policy]="NO_PREFILL_READY";continue
        if policy=="decode_first" and not has_d:
            reasons[policy]="NO_DECODE_READY";continue
        items,legality=compiler.generate(state,policy)
        if legality!="LEGAL":
            reasons[policy]=legality;continue
        sig=tuple((x.request_id,x.phase,x.token_start,x.token_count) for x in items)
        if sig in signatures:
            reasons[policy]=f"IDENTICAL_PLAN_AS_{signatures[sig]}";continue
        signatures[sig]=policy;kept.append((policy,items))
    if not kept:
        # Re-run all policies and retain the first legal plan; pruning can never
        # remove the only legal candidate.
        for policy in SCHEDULING_POLICIES:
            items,legality=compiler.generate(state,policy)
            if legality=="LEGAL":
                kept=[(policy,items)];reasons.pop(policy,None);break
    if not kept: raise ServingPlanError("candidate pruning removed all legal candidates")
    return kept,reasons


class SchedulerSelectorV1Fast:
    def __init__(self,config:FastSelectorConfiguration,
                 objective:ObjectiveProfile|None=None,
                 service:CalibratedQwenCPUServiceModel|None=None):
        self.config=config;self.objective=objective or make_objective(
            config.objective_profile)
        self.service=service or CalibratedQwenCPUServiceModel()
        self.profiles=[]

    def select(self,live_state:ReplicaSchedulerState)->FastSchedulerPolicyPlan:
        total_start=time.perf_counter_ns()
        t=time.perf_counter_ns();state=online_state_view(live_state)
        snapshot_ns=time.perf_counter_ns()-t
        t=time.perf_counter_ns();h=choose_adaptive_horizon(state)
        feature_ns=time.perf_counter_ns()-t
        t=time.perf_counter_ns();candidates,reasons=prune_candidates(state)
        prune_ns=time.perf_counter_ns()-t
        scores={};detail={}
        t=time.perf_counter_ns()
        for policy,items in candidates:
            step=self.service.score(state,items,state.clock_ms)
            included={x.request_id for x in items}
            unscheduled=[r for r in state.ready() if r.request_id not in included]
            delay=step["predicted_step_compute_ms"]*h
            wait_penalty=sum(delay/self.objective.ttft_scale_ms for r in unscheduled
                             if r.phase=="PREFILL")
            gap_penalty=sum(delay/self.objective.decode_gap_scale_ms for r in unscheduled
                            if r.phase=="DECODE")
            remaining=(sum(r.prefill_remaining_tokens*self.service.prefill_token_ms+
                           r.decode_remaining_tokens*self.service.decode_sequence_ms
                           for r in state.requests.values())/
                       self.objective.e2e_scale_ms)
            score=(step["predicted_step_compute_ms"]/
                   self.objective.e2e_scale_ms+
                   self.objective.weights["ttft"]*wait_penalty+
                   self.objective.weights["decode_gap"]*gap_penalty+
                   self.objective.weights["e2e"]*remaining)
            scores[policy]=score;detail[policy]={"unscheduled":len(unscheduled),
                                                "projected_delay_ms":delay}
        rollout_ns=time.perf_counter_ns()-t
        selected=min(scores,key=lambda p:(scores[p],p))
        total_ns=time.perf_counter_ns()-total_start
        record={"snapshot_ns":snapshot_ns,"feature_ns":feature_ns,
                "candidate_pruning_ns":prune_ns,"rollout_ns":rollout_ns,
                "terminal_cost_ns":0,"objective_aggregation_ns":rollout_ns,
                "serialization_ns":0,"validation_ns":0,
                "logging_ns":0 if not self.config.evidence_mode else 1,
                "total_ns":total_ns,"candidate_detail":detail if
                self.config.evidence_mode else None,
                "candidates_generated":len(SCHEDULING_POLICIES),
                "candidates_pruned":len(reasons),
                "candidates_rolled_out":len(candidates)}
        self.profiles.append(record)
        return FastSchedulerPolicyPlan(
            f"fast-policy-{live_state.replica_id}-{live_state.step_id}",
            live_state.replica_id,selected,self.config.selection_mode,h,
            1 if self.config.selection_mode=="online_step" else self.config.epoch_steps,
            self.objective.profile_id,len(SCHEDULING_POLICIES),
            len(reasons),len(candidates),reasons,scores)


class EpochPolicyController:
    def __init__(self,selector:SchedulerSelectorV1Fast):
        self.selector=selector;self.current=None;self.steps_left=0
        self.policy_switches=0;self.planning_calls=0;self.last_signature=None

    @staticmethod
    def signature(state):
        ready=state.ready()
        return (len(ready),sum(r.phase=="PREFILL" for r in ready),
                sum(r.phase=="DECODE" for r in ready),
                tuple(sorted(r.request_id for r in ready)))

    def policy(self,state,event="step"):
        sig=self.signature(state)
        required=event in ("new_arrival","completion","phase_transition","slo_risk")
        if self.current is None or self.steps_left<=0 or required or \
           (sig!=self.last_signature and event=="composition_change"):
            old=self.current.policy_id if self.current else None
            self.current=self.selector.select(state);self.planning_calls+=1
            if old is not None and old!=self.current.policy_id:self.policy_switches+=1
            self.steps_left=self.current.valid_for_steps;self.last_signature=sig
        self.steps_left-=1
        return self.current.policy_id
