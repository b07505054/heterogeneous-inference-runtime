"""Serving S2.5 request-level calibration without changing S2 execution.

Selector v0 remains SchedulerCompiler.compile() per step. Selector v1 chooses
a policy epoch using cloned-state request-level prediction. ScheduleStepPlan
generation and PlanOnlySchedulerRuntime execution remain unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
import statistics
import time
import uuid
from typing import Any

from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime, ReplicaSchedulerState, SchedulerCompiler,
    SchedulerCostModel, ScheduleStepPlan, SCHEDULING_POLICIES,
    deserialize_schedule_plan)

OBJECTIVE_VERSION = "request_objective_v1"
SELECTOR_V0 = "scheduler_selector_v0_frozen"
SELECTOR_V1 = "scheduler_selector_v1_request_level"


@dataclass(frozen=True)
class CalibratedQwenCPUServiceModel(SchedulerCostModel):
    """Focused Qwen2.5-0.5B FP32 fit; mixed steps are not treated as additive."""
    prefill_intercept_ms: float = 125.95819977982697
    prefill_token_ms: float = 1.4587361719864231
    decode_intercept_ms: float = 112.76543966960162
    decode_sequence_ms: float = 4.69742307905107
    mixed_intercept_ms: float = 230.23138358257717
    mixed_prefill_token_ms: float = 1.9534577932822843
    mixed_decode_sequence_ms: float = 0.2441822241602855
    provenance: str = "measured_cpu_qwen2.5_0.5b_fp32_v1"

    def score(self, state, items, now_ms):
        result = super().score(state, items, now_ms)
        prefill = sum(x.token_count for x in items if x.phase == "prefill")
        decodes = sum(1 for x in items if x.phase == "decode")
        if prefill and decodes:
            compute = (self.mixed_intercept_ms +
                       prefill*self.mixed_prefill_token_ms +
                       decodes*self.mixed_decode_sequence_ms)
        elif prefill:
            compute = self.prefill_intercept_ms + prefill*self.prefill_token_ms
        elif decodes:
            compute = self.decode_intercept_ms + decodes*self.decode_sequence_ms
        else:
            compute = 0.0
        result["total_score"] += compute-result["predicted_step_compute_ms"]
        result["predicted_step_compute_ms"] = compute
        return result


@dataclass(frozen=True)
class ObjectiveProfile:
    profile_id: str
    ttft_scale_ms: float
    decode_gap_scale_ms: float
    e2e_scale_ms: float
    goodput_scale_requests_per_s: float
    weights: dict[str, float]
    provenance: str = "configured_objective_profile_v1"
    objective_version: str = OBJECTIVE_VERSION

    def validate(self) -> None:
        if self.profile_id not in objective_profiles():
            raise ServingPlanError("invalid objective profile")
        scales = (self.ttft_scale_ms, self.decode_gap_scale_ms,
                  self.e2e_scale_ms, self.goodput_scale_requests_per_s)
        if any(not math.isfinite(x) or x <= 0 for x in scales):
            raise ServingPlanError("missing or invalid normalization scale")
        required = {"ttft", "decode_gap", "e2e", "slo", "starvation",
                    "fairness", "goodput"}
        if set(self.weights) != required or any(
                not math.isfinite(x) or x < 0 for x in self.weights.values()):
            raise ServingPlanError("objective weights must be finite and nonnegative")


def objective_profiles() -> dict[str, dict[str, float]]:
    return {
        "balanced_interactive": dict(ttft=1.0, decode_gap=1.0, e2e=.25,
                                     slo=2.0, starvation=1.0, fairness=.5,
                                     goodput=.5),
        "decode_latency_priority": dict(ttft=.5, decode_gap=3.0, e2e=.2,
                                        slo=2.0, starvation=1.0, fairness=.5,
                                        goodput=.25),
        "ttft_priority": dict(ttft=3.0, decode_gap=.5, e2e=.2, slo=2.0,
                              starvation=1.0, fairness=.5, goodput=.25),
        "throughput_priority": dict(ttft=.25, decode_gap=.25, e2e=.5, slo=.5,
                                    starvation=.25, fairness=.25, goodput=3.0),
        "fairness_priority": dict(ttft=.75, decode_gap=.75, e2e=.25, slo=1.0,
                                  starvation=2.0, fairness=3.0, goodput=.25),
    }


def make_objective(profile_id: str = "balanced_interactive") -> ObjectiveProfile:
    if profile_id not in objective_profiles():
        raise ServingPlanError("invalid objective profile")
    result = ObjectiveProfile(profile_id, 50.0, 10.0, 250.0, 50.0,
                              objective_profiles()[profile_id])
    result.validate()
    return result


@dataclass(frozen=True)
class DatasetSplit:
    calibration: tuple[str, ...]
    development: tuple[str, ...]
    held_out: tuple[str, ...]
    stress: tuple[str, ...]
    real_qwen: tuple[str, ...]
    seeds: dict[str, int]

    def validate(self) -> None:
        groups = (self.calibration, self.development, self.held_out,
                  self.stress, self.real_qwen)
        flat = [x for group in groups for x in group]
        if len(flat) != len(set(flat)):
            raise ServingPlanError("duplicate trace across dataset splits")
        used_seeds = [self.seeds[x] for x in flat if x in self.seeds]
        if len(used_seeds) != len(set(used_seeds)):
            raise ServingPlanError("trace seed reused across dataset splits")


def scheduler_features(state: ReplicaSchedulerState,
                       replica_core_budget: int = 2,
                       prefix_hit_tokens: int = 0,
                       kv_pressure: float = 0.0) -> dict[str, float]:
    ready = state.ready()
    prefill = [r for r in ready if r.phase == "PREFILL"]
    decode = [r for r in ready if r.phase == "DECODE"]
    gaps = [state.clock_ms - (r.last_decode_ms if r.last_decode_ms is not None
                              else r.arrival_time_ms) for r in decode]
    waiting = [r for r in state.requests.values() if r.phase == "WAITING"]
    return {
        "waiting_request_count": len(waiting),
        "prefill_ready_count": len(prefill),
        "decode_ready_count": len(decode),
        "total_remaining_prefill_tokens": sum(r.prefill_remaining_tokens
                                               for r in state.requests.values()),
        "total_remaining_decode_tokens": sum(r.decode_remaining_tokens
                                              for r in state.requests.values()),
        "oldest_waiting_age_ms": max(
            [0.0] + [state.clock_ms-r.arrival_time_ms for r in waiting]),
        "largest_decode_gap_ms": max([0.0] + gaps),
        "mean_decode_gap_ms": statistics.fmean(gaps) if gaps else 0.0,
        "token_budget": state.profile.max_num_batched_tokens,
        "sequence_budget": state.profile.max_num_seqs,
        "prefix_hit_token_total": prefix_hit_tokens,
        "kv_pressure": kv_pressure,
        "replica_core_budget": replica_core_budget,
    }


def _request_metrics(state: ReplicaSchedulerState) -> dict[str, float]:
    requests = list(state.requests.values())
    ttft, e2e, max_gaps, starvation = [], [], [], 0
    for request in requests:
        first = (request.first_token_ms if request.first_token_ms is not None
                 else state.clock_ms)
        complete = (request.completion_ms if request.completion_ms is not None
                    else state.clock_ms)
        ttft.append(max(0.0, first-request.arrival_time_ms))
        e2e.append(max(0.0, complete-request.arrival_time_ms))
        gaps = ([request.decode_times_ms[0]-request.arrival_time_ms]
                if request.decode_times_ms and
                request.matched_prefix_tokens == request.prompt_length else [])
        gaps += [b-a for a,b in zip(request.decode_times_ms,
                                    request.decode_times_ms[1:])]
        if request.phase == "DECODE" and not request.finished:
            gaps.append(state.clock_ms-(request.last_decode_ms
                                        if request.last_decode_ms is not None
                                        else request.arrival_time_ms))
        max_gaps.append(max([0.0]+gaps))
        starvation += int(ttft[-1] > state.profile.starvation_guard_ms)
        starvation += int(max_gaps[-1] > state.profile.starvation_guard_ms)
    wall_s = max(state.clock_ms/1000, 1e-9)
    goodput = sum(t <= state.profile.ttft_slo_ms and
                  g <= state.profile.maximum_decode_gap_ms
                  for t,g in zip(ttft,max_gaps))/wall_s
    return {
        "mean_ttft_ms": statistics.fmean(ttft),
        "p95_ttft_ms": sorted(ttft)[min(len(ttft)-1, round(.95*(len(ttft)-1)))],
        "mean_e2e_ms": statistics.fmean(e2e),
        "mean_max_decode_gap_ms": statistics.fmean(max_gaps),
        "maximum_decode_gap_ms": max(max_gaps),
        "slo_violation_fraction": statistics.fmean(
            [t > state.profile.ttft_slo_ms or
             g > state.profile.maximum_decode_gap_ms
             for t,g in zip(ttft,max_gaps)]),
        "starvation_events_per_request": starvation/len(requests),
        "fairness_penalty": (max(e2e)-min(e2e)) /
                            max(statistics.fmean(e2e), 1e-9),
        "goodput_requests_per_s": goodput,
    }


def terminal_cost(state: ReplicaSchedulerState, objective: ObjectiveProfile,
                  service: SchedulerCostModel) -> dict[str, float]:
    """Conservative cost for unfinished work beyond a bounded horizon."""
    unfinished = [r for r in state.requests.values() if not r.finished]
    remaining_prefill_ms = sum(r.prefill_remaining_tokens for r in unfinished) * \
        service.prefill_token_ms
    remaining_decode_ms = sum(r.decode_remaining_tokens for r in unfinished) * \
        service.decode_sequence_ms
    projected_wait = sum(max(0.0, state.clock_ms-r.arrival_time_ms)
                         for r in unfinished)
    projected_gap = sum(max(0.0, state.clock_ms-
                            (r.last_decode_ms if r.last_decode_ms is not None
                             else r.arrival_time_ms))
                        for r in unfinished if r.phase == "DECODE")
    normalized = (
        objective.weights["ttft"] * projected_wait /
        objective.ttft_scale_ms +
        objective.weights["decode_gap"] * projected_gap /
        objective.decode_gap_scale_ms +
        objective.weights["e2e"] *
        (remaining_prefill_ms+remaining_decode_ms) / objective.e2e_scale_ms)
    return {"remaining_prefill_ms": remaining_prefill_ms,
            "remaining_decode_ms": remaining_decode_ms,
            "projected_wait_ms": projected_wait,
            "projected_decode_gap_ms": projected_gap,
            "normalized_terminal_cost": normalized}


def request_objective(state: ReplicaSchedulerState,
                      objective: ObjectiveProfile,
                      *, include_terminal: bool = True,
                      service: SchedulerCostModel | None = None) -> dict[str, Any]:
    objective.validate()
    metrics = _request_metrics(state)
    w = objective.weights
    components = {
        "ttft": metrics["mean_ttft_ms"]/objective.ttft_scale_ms,
        "decode_gap": metrics["mean_max_decode_gap_ms"]/
                      objective.decode_gap_scale_ms,
        "e2e": metrics["mean_e2e_ms"]/objective.e2e_scale_ms,
        "slo": metrics["slo_violation_fraction"],
        "starvation": metrics["starvation_events_per_request"],
        "fairness": metrics["fairness_penalty"],
        "goodput": metrics["goodput_requests_per_s"]/
                   objective.goodput_scale_requests_per_s,
    }
    value = sum(w[k]*components[k] for k in components if k != "goodput") - \
        w["goodput"]*components["goodput"]
    terminal = terminal_cost(state, objective, service or SchedulerCostModel()) \
        if include_terminal else {"normalized_terminal_cost": 0.0}
    return {"evaluation_level": "full_trace_request_level" if not state.unfinished()
            else "horizon_state_request_level",
            "objective_version": objective.objective_version,
            "profile_id": objective.profile_id, "metrics": metrics,
            "normalized_components": components,
            "terminal": terminal,
            "value": value + terminal["normalized_terminal_cost"]}


def _advance_one(clone: ReplicaSchedulerState, policy: str,
                 compiler: SchedulerCompiler) -> ScheduleStepPlan | None:
    ready = clone.ready()
    if not ready:
        arrivals = [r.arrival_time_ms for r in clone.requests.values()
                    if r.phase == "WAITING"]
        if not arrivals:
            return None
        clone.clock_ms = min(arrivals)
    plan = compiler.compile(clone, forced_policy=policy)
    loaded = deserialize_schedule_plan(plan.serialize(), clone)
    PlanOnlySchedulerRuntime().execute(clone, loaded)
    return plan


def evaluate_policy(state: ReplicaSchedulerState, policy: str,
                    objective: ObjectiveProfile, *,
                    horizon: int | None = None,
                    terminal: bool = True,
                    continuation_policy: str | None = None,
                    service: SchedulerCostModel | None = None) -> dict[str, Any]:
    if policy not in SCHEDULING_POLICIES:
        raise ServingPlanError("selector references unavailable candidate")
    if horizon is not None and horizon < 1:
        raise ServingPlanError("horizon must be at least one")
    if continuation_policy is not None and continuation_policy not in SCHEDULING_POLICIES:
        raise ServingPlanError("continuation policy is undefined")
    before = copy.deepcopy(state)
    clone = copy.deepcopy(state)
    service = service or SchedulerCostModel()
    compiler = SchedulerCompiler(service)
    plans, limit = [], horizon if horizon is not None else 100000
    while clone.unfinished() and len(plans) < limit:
        selected_policy = policy if not plans else (continuation_policy or policy)
        plan = _advance_one(clone, selected_policy, compiler)
        if plan is None:
            break
        plans.append(plan)
    if state != before:
        raise ServingPlanError("cloned scheduler state mutated live state")
    result = request_objective(clone, objective, include_terminal=terminal,
                               service=service)
    result.update({
        "policy": policy,
        "evaluation_level": "full_trace_request_level" if horizon is None
                            else "fixed_horizon_predicted_outcome",
        "horizon": horizon, "terminal_cost_enabled": terminal,
        "simulated_steps": len(plans), "state_clones": 1,
        "candidate_evaluations": len(plans),
        "first_plan": plans[0].to_dict() if plans else None,
        "excluded_request_delay": {
            r.request_id: max(0.0, clone.clock_ms-r.arrival_time_ms)
            for r in clone.requests.values()
            if plans and r.request_id not in
            {x.request_id for x in plans[0].items}},
    })
    return result


@dataclass(frozen=True)
class SchedulerPolicyPlan:
    plan_id: str
    replica_id: str
    policy_id: str
    objective_profile: str
    selection_epoch: int
    valid_for_steps: int
    reselection_trigger: str
    cost_model_version: str
    selector_version: str
    predicted_objective: float
    schema_version: int = 1
    plan_kind: str = "scheduler_policy_epoch"

    def __post_init__(self) -> None:
        if self.policy_id not in SCHEDULING_POLICIES:
            raise ServingPlanError("selector references unavailable candidate")
        if self.valid_for_steps < 1:
            raise ServingPlanError("policy epoch must include at least one step")

    def serialize(self) -> str:
        return json.dumps(vars(self), sort_keys=True)


class SchedulerSelectorV1:
    """Interpretable policy-epoch selector using request-level prediction."""
    def __init__(self, objective: ObjectiveProfile, *,
                 prediction_mode: str = "full_trace",
                 horizon: int = 8, terminal: bool = True,
                 service: SchedulerCostModel | None = None):
        objective.validate()
        if prediction_mode not in ("full_trace", "fixed_horizon"):
            raise ServingPlanError("invalid prediction mode")
        if horizon < 1:
            raise ServingPlanError("horizon must be at least one")
        self.objective, self.prediction_mode = objective, prediction_mode
        self.horizon, self.terminal = horizon, terminal
        self.service = service or CalibratedQwenCPUServiceModel()
        self.records: list[dict[str, Any]] = []

    def select(self, state: ReplicaSchedulerState,
               *, epoch: int = 0, valid_for_steps: int = 100000,
               trigger: str = "trace_start") -> SchedulerPolicyPlan:
        started = time.perf_counter_ns()
        horizon = None if self.prediction_mode == "full_trace" else self.horizon
        evaluations = [evaluate_policy(state, p, self.objective,
                                       horizon=horizon, terminal=self.terminal,
                                       service=self.service)
                       for p in SCHEDULING_POLICIES]
        chosen = min(evaluations, key=lambda x: (x["value"], x["policy"]))
        plan = SchedulerPolicyPlan(
            f"policy-{uuid.uuid4().hex[:16]}", state.replica_id,
            chosen["policy"], self.objective.profile_id, epoch,
            valid_for_steps, trigger, "cpu_request_model_v1", SELECTOR_V1,
            chosen["value"])
        self.records.append({
            "features": scheduler_features(state),
            "evaluations": [{"policy": x["policy"], "value": x["value"],
                             "level": x["evaluation_level"],
                             "terminal_cost": x["terminal"]["normalized_terminal_cost"]}
                            for x in evaluations],
            "selected_policy": plan.policy_id,
            "selector_overhead_ns": time.perf_counter_ns()-started,
        })
        return plan
