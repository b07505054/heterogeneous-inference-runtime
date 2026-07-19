"""Serving Distributed S2: compiler-planned replica-local scheduling.

ServingExecutionPlan owns request placement. ScheduleStepPlan owns only the
next token allocation on that already-selected replica. The runtime applies the
exact plan and never moves requests, changes phases, or changes token counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time
import uuid
from typing import Any, Callable

from deployment.serving_execution import ServingPlanError

SCHEDULE_SCHEMA_VERSION = 1
SCHEDULING_POLICIES = ("decode_first", "prefill_first", "chunked_balanced",
                       "slo_aware")
PHASES = ("WAITING", "PREFILL", "DECODE", "FINISHED", "FAILED")


@dataclass
class RequestExecutionState:
    request_id: str
    serving_plan_id: str
    replica_id: str
    arrival_time_ms: float
    prompt_length: int
    matched_prefix_tokens: int
    expected_output_tokens: int
    priority: int = 0
    prefill_completed_tokens: int = 0
    decode_completed_tokens: int = 0
    phase: str = "WAITING"
    first_token_emitted: bool = False
    first_scheduled_ms: float | None = None
    prefill_finished_ms: float | None = None
    first_token_ms: float | None = None
    completion_ms: float | None = None
    last_decode_ms: float | None = None
    prefill_chunks: list[tuple[int, int]] = field(default_factory=list)
    decode_times_ms: list[float] = field(default_factory=list)
    operator_provenance: list[dict[str, Any]] = field(default_factory=list)

    @property
    def uncached_prompt_tokens(self) -> int:
        return self.prompt_length - self.matched_prefix_tokens

    @property
    def prefill_remaining_tokens(self) -> int:
        return self.uncached_prompt_tokens - self.prefill_completed_tokens

    @property
    def decode_remaining_tokens(self) -> int:
        return self.expected_output_tokens - self.decode_completed_tokens

    @property
    def finished(self) -> bool:
        return self.phase == "FINISHED"

    def validate(self) -> None:
        if self.phase not in PHASES or not self.request_id or not self.replica_id:
            raise ServingPlanError("invalid request execution state")
        if not 0 <= self.matched_prefix_tokens <= self.prompt_length:
            raise ServingPlanError("invalid prefix token accounting")
        if not 0 <= self.prefill_completed_tokens <= self.uncached_prompt_tokens:
            raise ServingPlanError("invalid prefill progress")
        if not 0 <= self.decode_completed_tokens <= self.expected_output_tokens:
            raise ServingPlanError("invalid decode progress")
        if self.decode_completed_tokens and self.prefill_remaining_tokens:
            raise ServingPlanError("decode began before prefill completed")
        if self.phase == "FINISHED" and self.decode_remaining_tokens:
            raise ServingPlanError("finished request has decode work remaining")

    def make_ready(self) -> None:
        if self.phase != "WAITING":
            return
        self.phase = "PREFILL" if self.prefill_remaining_tokens else "DECODE"


@dataclass(frozen=True)
class SchedulerProfile:
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 256
    max_prefill_chunk_tokens: int = 128
    decode_token_cost_per_request: int = 1
    balanced_decode_reservation: int = 64
    ttft_slo_ms: float = 50.0
    maximum_decode_gap_ms: float = 10.0
    starvation_guard_ms: float = 25.0

    def __post_init__(self) -> None:
        values = (self.max_num_seqs, self.max_num_batched_tokens,
                  self.max_prefill_chunk_tokens,
                  self.decode_token_cost_per_request)
        if any(x < 1 for x in values):
            raise ServingPlanError("scheduler budgets must be positive")
        if self.max_prefill_chunk_tokens > self.max_num_batched_tokens:
            raise ServingPlanError("prefill chunk exceeds step token budget")
        if not 0 <= self.balanced_decode_reservation <= self.max_num_batched_tokens:
            raise ServingPlanError("invalid decode reservation")


@dataclass
class ReplicaSchedulerState:
    replica_id: str
    profile: SchedulerProfile
    requests: dict[str, RequestExecutionState] = field(default_factory=dict)
    step_id: int = 0
    version: int = 0
    clock_ms: float = 0.0
    finished_ids: list[str] = field(default_factory=list)
    statistics: dict[str, float] = field(default_factory=lambda: {
        "scheduled_tokens": 0, "unused_tokens": 0, "steps": 0,
        "prefill_tokens": 0, "decode_tokens": 0,
        "capacity_rejections": 0,
    })

    def ingest(self, request: RequestExecutionState) -> None:
        request.validate()
        if request.replica_id != self.replica_id:
            raise ServingPlanError("cross-replica request ingestion")
        if request.request_id in self.requests:
            raise ServingPlanError("duplicate request ID")
        self.requests[request.request_id] = request
        self.version += 1

    def ready(self) -> list[RequestExecutionState]:
        for request in self.requests.values():
            if request.phase == "WAITING" and request.arrival_time_ms <= self.clock_ms:
                request.make_ready()
        return sorted((r for r in self.requests.values()
                       if r.phase in ("PREFILL", "DECODE")),
                      key=lambda r: (r.arrival_time_ms, r.request_id))

    def unfinished(self) -> bool:
        return any(not r.finished and r.phase != "FAILED"
                   for r in self.requests.values())


@dataclass(frozen=True)
class ScheduleItem:
    request_id: str
    phase: str
    token_start: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass(frozen=True)
class ScheduleStepPlan:
    plan_id: str
    replica_id: str
    step_id: int
    scheduler_state_version: int
    policy: str
    candidate_id: str
    maximum_tokens: int
    scheduled_tokens: int
    unused_tokens: int
    maximum_sequences: int
    scheduled_sequences: int
    items: tuple[ScheduleItem, ...]
    predicted_cost: dict[str, float]
    selection_mode: str = "compiler_selected"
    schema_version: int = SCHEDULE_SCHEMA_VERSION
    plan_kind: str = "serving_schedule_step"

    def to_dict(self) -> dict[str, Any]:
        value = vars(self).copy()
        value["items"] = [x.to_dict() for x in self.items]
        return value

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any],
                  state: ReplicaSchedulerState) -> "ScheduleStepPlan":
        value = dict(payload)
        try:
            value["items"] = tuple(ScheduleItem(**x) for x in value["items"])
            plan = cls(**value)
        except (KeyError, TypeError) as exc:
            raise ServingPlanError(f"invalid schedule plan: {exc}") from exc
        plan.validate(state)
        return plan

    def validate(self, state: ReplicaSchedulerState) -> None:
        if self.schema_version != SCHEDULE_SCHEMA_VERSION or \
                self.plan_kind != "serving_schedule_step":
            raise ServingPlanError("schedule schema mismatch")
        if self.replica_id != state.replica_id:
            raise ServingPlanError("schedule targets another replica")
        if self.step_id != state.step_id or \
                self.scheduler_state_version != state.version:
            raise ServingPlanError("non-monotonic step or stale state version")
        if self.policy not in SCHEDULING_POLICIES:
            raise ServingPlanError("unknown scheduling policy")
        if len(self.items) != len({x.request_id for x in self.items}):
            raise ServingPlanError("duplicate request in schedule step")
        if len(self.items) > self.maximum_sequences or \
                self.maximum_sequences != state.profile.max_num_seqs:
            raise ServingPlanError("sequence budget exceeded")
        if sum(x.token_count for x in self.items) != self.scheduled_tokens or \
                self.scheduled_tokens > self.maximum_tokens or \
                self.maximum_tokens != state.profile.max_num_batched_tokens or \
                self.unused_tokens != self.maximum_tokens - self.scheduled_tokens:
            raise ServingPlanError("token budget accounting mismatch")
        for item in self.items:
            if item.token_count <= 0 or item.request_id not in state.requests:
                raise ServingPlanError("missing request or nonpositive token count")
            request = state.requests[item.request_id]
            if request.replica_id != self.replica_id or request.finished:
                raise ServingPlanError("foreign or finished request scheduled")
            if item.phase == "prefill":
                if request.phase != "PREFILL" or \
                        item.token_start != request.prefill_completed_tokens:
                    raise ServingPlanError("invalid prefill phase or token offset")
                if item.token_count > request.prefill_remaining_tokens or \
                        item.token_count > state.profile.max_prefill_chunk_tokens:
                    raise ServingPlanError("prefill chunk exceeds legal range")
            elif item.phase == "decode":
                if request.phase != "DECODE" or request.prefill_remaining_tokens:
                    raise ServingPlanError("decode before prefill completion")
                if item.token_count != state.profile.decode_token_cost_per_request or \
                        item.token_start != request.decode_completed_tokens:
                    raise ServingPlanError("decode must schedule exactly one logical token")
            else:
                raise ServingPlanError("invalid schedule item phase")
        if not self.predicted_cost or any(
                not math.isfinite(x) or x < 0 for x in self.predicted_cost.values()):
            raise ServingPlanError("invalid predicted schedule cost")


@dataclass(frozen=True)
class SchedulerCostModel:
    prefill_token_ms: float = 0.055
    decode_sequence_ms: float = 0.18
    ttft_penalty_weight: float = 2.0
    decode_gap_penalty_weight: float = 3.0
    starvation_penalty_weight: float = 2.5
    batching_efficiency_per_sequence_ms: float = 0.02
    provenance: str = "derived_from_measured_cpu"

    def score(self, state: ReplicaSchedulerState, items: list[ScheduleItem],
              now_ms: float) -> dict[str, float]:
        prefill = sum(x.token_count for x in items if x.phase == "prefill")
        decodes = sum(1 for x in items if x.phase == "decode")
        compute = prefill * self.prefill_token_ms + decodes * self.decode_sequence_ms
        scheduled = {x.request_id for x in items}
        ttft = decode_gap = pstarve = dstarve = 0.0
        for request in state.ready():
            age = max(0.0, now_ms - request.arrival_time_ms)
            if request.phase == "PREFILL" and request.request_id not in scheduled:
                ttft += max(0.0, age - state.profile.ttft_slo_ms)
                pstarve += max(0.0, age - state.profile.starvation_guard_ms)
            if request.phase == "DECODE" and request.request_id not in scheduled:
                base = request.last_decode_ms if request.last_decode_ms is not None else now_ms
                gap = max(0.0, now_ms - base)
                decode_gap += max(0.0, gap - state.profile.maximum_decode_gap_ms)
                dstarve += max(0.0, gap - state.profile.starvation_guard_ms)
        benefit = max(0, len(items) - 1) * self.batching_efficiency_per_sequence_ms
        total = (compute + ttft * self.ttft_penalty_weight +
                 decode_gap * self.decode_gap_penalty_weight +
                 (pstarve + dstarve) * self.starvation_penalty_weight - benefit)
        return {
            "predicted_step_compute_ms": compute,
            "predicted_ttft_penalty_ms": ttft * self.ttft_penalty_weight,
            "predicted_decode_gap_penalty_ms": decode_gap * self.decode_gap_penalty_weight,
            "predicted_prefill_starvation_penalty_ms":
                pstarve * self.starvation_penalty_weight,
            "predicted_decode_starvation_penalty_ms":
                dstarve * self.starvation_penalty_weight,
            "predicted_kv_pressure_penalty_ms": 0.0,
            "predicted_batch_efficiency_benefit_ms": benefit,
            "total_score": max(0.0, total),
        }


class SchedulerCompiler:
    def __init__(self, cost_model: SchedulerCostModel | None = None):
        self.cost_model = cost_model or SchedulerCostModel()
        self.traces: list[dict[str, Any]] = []

    @staticmethod
    def _allocate_prefill(requests, budget, profile, items):
        for request in requests:
            if budget <= 0 or len(items) >= profile.max_num_seqs:
                break
            count = min(request.prefill_remaining_tokens,
                        profile.max_prefill_chunk_tokens, budget)
            if count:
                items.append(ScheduleItem(request.request_id, "prefill",
                                          request.prefill_completed_tokens, count))
                budget -= count
        return budget

    @staticmethod
    def _allocate_decode(requests, budget, profile, items, limit=None):
        used = 0
        for request in requests:
            if budget < profile.decode_token_cost_per_request or \
                    len(items) >= profile.max_num_seqs or \
                    (limit is not None and used >= limit):
                break
            items.append(ScheduleItem(request.request_id, "decode",
                                      request.decode_completed_tokens,
                                      profile.decode_token_cost_per_request))
            budget -= profile.decode_token_cost_per_request
            used += profile.decode_token_cost_per_request
        return budget

    def generate(self, state: ReplicaSchedulerState,
                 policy: str) -> tuple[list[ScheduleItem], str]:
        if policy not in SCHEDULING_POLICIES:
            raise ServingPlanError("unknown scheduling policy")
        ready = state.ready()
        prefill = [r for r in ready if r.phase == "PREFILL"]
        decode = [r for r in ready if r.phase == "DECODE"]
        budget, items = state.profile.max_num_batched_tokens, []
        if policy == "decode_first":
            budget = self._allocate_decode(decode, budget, state.profile, items)
            self._allocate_prefill(prefill, budget, state.profile, items)
        elif policy == "prefill_first":
            budget = self._allocate_prefill(prefill, budget, state.profile, items)
            self._allocate_decode(decode, budget, state.profile, items)
        elif policy == "chunked_balanced":
            reserve = min(state.profile.balanced_decode_reservation, len(decode))
            budget = self._allocate_decode(decode, budget, state.profile, items,
                                           limit=reserve)
            self._allocate_prefill(prefill, budget, state.profile, items)
        else:
            urgent = sorted(ready, key=lambda r: (
                -max(0.0, state.clock_ms - r.arrival_time_ms -
                     (state.profile.ttft_slo_ms if r.phase == "PREFILL"
                      else state.profile.maximum_decode_gap_ms)),
                r.arrival_time_ms, r.request_id))
            for request in urgent:
                if budget <= 0 or len(items) >= state.profile.max_num_seqs:
                    break
                if request.phase == "DECODE":
                    budget = self._allocate_decode([request], budget,
                                                   state.profile, items)
                else:
                    budget = self._allocate_prefill([request], budget,
                                                    state.profile, items)
        return items, "LEGAL" if items else "NO_READY_REQUESTS"

    def compile(self, state: ReplicaSchedulerState, *,
                forced_policy: str | None = None) -> ScheduleStepPlan:
        started = time.perf_counter_ns()
        records = []
        for policy in SCHEDULING_POLICIES:
            items, legality = self.generate(state, policy)
            cost = self.cost_model.score(state, items, state.clock_ms)
            records.append({"policy": policy, "items": items, "legality": legality,
                            "predicted_cost": cost})
        legal = [x for x in records if x["legality"] == "LEGAL"]
        if not legal:
            raise ServingPlanError("NO_READY_REQUESTS")
        if forced_policy:
            chosen = next((x for x in legal if x["policy"] == forced_policy), None)
            if chosen is None:
                raise ServingPlanError("forced scheduling policy is not legal")
            mode = "forced_test_override"
        else:
            chosen = min(legal, key=lambda x: (
                x["predicted_cost"]["total_score"], x["policy"]))
            mode = "compiler_selected"
        items = tuple(chosen["items"])
        scheduled = sum(x.token_count for x in items)
        plan = ScheduleStepPlan(
            plan_id=f"schedule-{uuid.uuid4().hex[:16]}",
            replica_id=state.replica_id, step_id=state.step_id,
            scheduler_state_version=state.version,
            policy=chosen["policy"], candidate_id=f"{chosen['policy']}_v1",
            maximum_tokens=state.profile.max_num_batched_tokens,
            scheduled_tokens=scheduled,
            unused_tokens=state.profile.max_num_batched_tokens - scheduled,
            maximum_sequences=state.profile.max_num_seqs,
            scheduled_sequences=len(items), items=items,
            predicted_cost=chosen["predicted_cost"], selection_mode=mode)
        plan.validate(state)
        self.traces.append({
            "replica_id": state.replica_id, "step_id": state.step_id,
            "state_version": state.version,
            "candidate_generation_ns": time.perf_counter_ns() - started,
            "candidates": [{
                "candidate_id": f"{x['policy']}_v1",
                "legality": x["legality"],
                "score": x["predicted_cost"]["total_score"],
                "scheduled_tokens": sum(i.token_count for i in x["items"]),
            } for x in records],
            "selected_candidate_id": plan.candidate_id,
        })
        return plan


class PlanOnlySchedulerRuntime:
    def __init__(self):
        self.runtime_schedule_rebuild_count = 0
        self.runtime_item_override_count = 0
        self.runtime_token_count_override_count = 0
        self.runtime_phase_override_count = 0
        self.runtime_unplanned_request_count = 0
        self.runtime_fallback_schedule_count = 0
        self.events: list[dict[str, Any]] = []

    def execute(self, state: ReplicaSchedulerState, plan: ScheduleStepPlan,
                execute_item: Callable[[RequestExecutionState, ScheduleItem,
                                        ScheduleStepPlan], dict[str, Any]] | None = None
                ) -> dict[str, Any]:
        plan.validate(state)
        started = time.perf_counter_ns()
        item_events = []
        duration = plan.predicted_cost["predicted_step_compute_ms"]
        end_ms = state.clock_ms + max(0.001, duration)
        for item in plan.items:
            request = state.requests[item.request_id]
            result = execute_item(request, item, plan) if execute_item else {}
            if request.first_scheduled_ms is None:
                request.first_scheduled_ms = state.clock_ms
            if item.phase == "prefill":
                request.prefill_chunks.append(
                    (item.token_start, item.token_start + item.token_count))
                request.prefill_completed_tokens += item.token_count
                if not request.prefill_remaining_tokens:
                    request.phase = "DECODE"
                    request.prefill_finished_ms = end_ms
            else:
                request.decode_completed_tokens += item.token_count
                request.decode_times_ms.append(end_ms)
                request.last_decode_ms = end_ms
                if not request.first_token_emitted:
                    request.first_token_emitted = True
                    request.first_token_ms = end_ms
                if not request.decode_remaining_tokens:
                    request.phase = "FINISHED"
                    request.completion_ms = end_ms
                    state.finished_ids.append(request.request_id)
            request.operator_provenance.extend(result.get("operator_provenance", []))
            request.validate()
            item_events.append({
                "request_id": request.request_id, "phase": item.phase,
                "token_start": item.token_start, "token_count": item.token_count,
                "model_invocation_id": result.get("model_invocation_id"),
                "operator_plan_id": result.get("operator_plan_id"),
                "operator_provenance": result.get("operator_provenance", []),
                "generated_token_ids": result.get("generated_token_ids", []),
            })
        state.clock_ms = end_ms
        state.step_id += 1
        state.version += 1
        state.statistics["steps"] += 1
        state.statistics["scheduled_tokens"] += plan.scheduled_tokens
        state.statistics["unused_tokens"] += plan.unused_tokens
        state.statistics["prefill_tokens"] += sum(
            x.token_count for x in plan.items if x.phase == "prefill")
        state.statistics["decode_tokens"] += sum(
            x.token_count for x in plan.items if x.phase == "decode")
        event = {
            "schedule_plan_id": plan.plan_id, "replica_id": plan.replica_id,
            "step_id": plan.step_id, "policy": plan.policy,
            "selected_candidate_id": plan.candidate_id,
            "scheduled_tokens": plan.scheduled_tokens,
            "unused_tokens": plan.unused_tokens,
            "scheduled_sequences": plan.scheduled_sequences,
            "items": item_events, "start_ms": end_ms - duration,
            "end_ms": end_ms,
            "plan_validation_and_execution_overhead_ns":
                time.perf_counter_ns() - started,
        }
        self.events.append(event)
        return event

    def counters(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in (
            "runtime_schedule_rebuild_count", "runtime_item_override_count",
            "runtime_token_count_override_count", "runtime_phase_override_count",
            "runtime_unplanned_request_count", "runtime_fallback_schedule_count")}


def deserialize_schedule_plan(text: str,
                              state: ReplicaSchedulerState) -> ScheduleStepPlan:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ServingPlanError("invalid schedule plan JSON") from exc
    return ScheduleStepPlan.from_dict(payload, state)


def run_scheduler(state: ReplicaSchedulerState, compiler: SchedulerCompiler,
                  runtime: PlanOnlySchedulerRuntime, *,
                  policy: str | None = None,
                  execute_item=None, max_steps: int = 100000) -> list[dict[str, Any]]:
    """Continuous loop; future arrivals join without draining active requests."""
    steps = 0
    while state.unfinished():
        ready = state.ready()
        if not ready:
            arrivals = [r.arrival_time_ms for r in state.requests.values()
                        if r.phase == "WAITING"]
            if not arrivals:
                break
            state.clock_ms = min(arrivals)
            ready = state.ready()
        selected = compiler.compile(state, forced_policy=policy)
        loaded = deserialize_schedule_plan(selected.serialize(), state)
        runtime.execute(state, loaded, execute_item)
        steps += 1
        if steps > max_steps:
            raise RuntimeError("scheduler exceeded deterministic step bound")
    return runtime.events
