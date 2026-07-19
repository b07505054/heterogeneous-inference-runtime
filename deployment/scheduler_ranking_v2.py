"""Serving S2.7 wall-clock ranking and incremental selector state.

This module deliberately does not change ScheduleStepPlan generation or
execution.  A ranking model chooses one existing S2 policy; SchedulerCompiler
still constructs and validates the exact token allocation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import heapq
import json
import math
import time
from typing import Any, Mapping

from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    ReplicaSchedulerState, RequestExecutionState, SCHEDULING_POLICIES)

STATIC_SELECTOR_VERSION = "ranking_selector_v2_static"
ADAPTIVE_SELECTOR_VERSION = "ranking_selector_v2_adaptive"
SUMMARY_VERSION = "scheduler_state_summary_v1"
RANKING_FEATURES = (
    "active_count", "waiting_count", "prefill_ready_count",
    "decode_ready_count", "remaining_prefill_per_budget",
    "remaining_decode_per_sequence", "oldest_waiting_age_ratio",
    "largest_decode_gap_ratio", "mean_decode_gap_ratio",
    "prefix_hit_per_request", "token_budget_utilization",
    "sequence_budget_utilization", "recent_prefill_ms",
    "recent_decode_ms", "recent_mixed_ms", "recent_scheduler_ms",
    "replica_core_budget", "policy_switch_count",
    "time_since_reselection_ms",
)


@dataclass(frozen=True)
class HotRequest:
    request_id: str
    phase: str
    arrival_time_ms: float
    remaining_prefill: int
    remaining_decode: int
    decode_gap_ms: float
    decode_anchor_ms: float
    prefix_hit_tokens: int
    generation: int


@dataclass(frozen=True)
class SchedulerStateSummary:
    state_version: int
    clock_ms: float
    active_count: int
    waiting_count: int
    prefill_ready_count: int
    decode_ready_count: int
    total_remaining_prefill: int
    total_remaining_decode: int
    prefix_hit_tokens: int
    oldest_waiting_age_ms: float
    largest_decode_gap_ms: float
    mean_decode_gap_ms: float
    frontier: tuple[HotRequest, ...]
    token_budget: int
    sequence_budget: int
    recent_prefill_ms: float
    recent_decode_ms: float
    recent_mixed_ms: float
    recent_scheduler_ms: float
    replica_core_budget: int
    policy_switch_count: int
    time_since_reselection_ms: float
    completed_count: int
    history_record_count: int
    summary_version: str = SUMMARY_VERSION


@dataclass
class RollingLatencyStatistics:
    alpha: float = .2
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    mixed_ms: float = 0.0
    scheduler_ms: float = 0.0

    def observe(self, kind: str, latency_ms: float) -> None:
        if kind not in ("prefill", "decode", "mixed", "scheduler") or \
                not math.isfinite(latency_ms) or latency_ms < 0:
            raise ServingPlanError("invalid rolling latency observation")
        name = f"{kind}_ms"
        old = getattr(self, name)
        setattr(self, name, latency_ms if old == 0 else
                self.alpha * latency_ms + (1-self.alpha) * old)


class IncrementalSchedulerState:
    """Event-maintained state; snapshot creation is O(K), not O(history)."""
    def __init__(self, state: ReplicaSchedulerState, *, frontier_size: int = 8,
                 replica_core_budget: int = 2):
        if frontier_size < 1:
            raise ServingPlanError("frontier size must be positive")
        self.replica_id = state.replica_id
        self.profile = state.profile
        self.frontier_size = frontier_size
        self.replica_core_budget = replica_core_budget
        self.hot: dict[str, HotRequest] = {}
        self.latency = RollingLatencyStatistics()
        self.completed_count = 0
        self.history_record_count = 0
        self.policy_switch_count = 0
        self.last_reselection_ms = state.clock_ms
        self._state_version = state.version
        self._clock_ms = state.clock_ms
        self._remaining_prefill = self._remaining_decode = 0
        self._prefix_hits = 0
        self._generation = 0
        self._phase_counts = {"WAITING": 0, "PREFILL": 0, "DECODE": 0}
        self._decode_anchor_sum = 0.0
        self._oldest_heap: list[tuple[float, str, int]] = []
        self._gap_heap: list[tuple[float, str, int]] = []
        self._prefill_heap: list[tuple[int, str, int]] = []
        for request in state.requests.values():
            if request.arrival_time_ms <= state.clock_ms and not request.finished:
                self.upsert(request, state.clock_ms)
            elif request.finished:
                self.completed_count += 1
                self.history_record_count += 1

    @staticmethod
    def _hot(request: RequestExecutionState, now_ms: float,
             generation: int) -> HotRequest:
        gap = 0.0
        anchor = now_ms
        if request.phase == "DECODE":
            anchor = (request.last_decode_ms if request.last_decode_ms is not None
                      else request.arrival_time_ms)
            gap = max(0.0, now_ms-anchor)
        return HotRequest(request.request_id, request.phase,
                          request.arrival_time_ms,
                          request.prefill_remaining_tokens,
                          request.decode_remaining_tokens, gap, anchor,
                          request.matched_prefix_tokens, generation)

    def upsert(self, request: RequestExecutionState, now_ms: float) -> None:
        if request.replica_id != self.replica_id:
            raise ServingPlanError("summary received cross-replica request")
        if request.arrival_time_ms > now_ms:
            raise ServingPlanError("future request exposed to online summary")
        old = self.hot.get(request.request_id)
        if old:
            self._remaining_prefill -= old.remaining_prefill
            self._remaining_decode -= old.remaining_decode
            self._prefix_hits -= old.prefix_hit_tokens
            self._phase_counts[old.phase] -= 1
            if old.phase == "DECODE":
                self._decode_anchor_sum -= old.decode_anchor_ms
        if request.finished or request.phase == "FAILED":
            if old:
                del self.hot[request.request_id]
                self.completed_count += int(request.finished)
                self.history_record_count += 1
        else:
            self._generation += 1
            hot = self._hot(request, now_ms, self._generation)
            self.hot[request.request_id] = hot
            self._remaining_prefill += hot.remaining_prefill
            self._remaining_decode += hot.remaining_decode
            self._prefix_hits += hot.prefix_hit_tokens
            self._phase_counts[hot.phase] += 1
            if hot.phase == "DECODE":
                self._decode_anchor_sum += hot.decode_anchor_ms
                heapq.heappush(self._gap_heap,
                               (hot.decode_anchor_ms, hot.request_id,
                                hot.generation))
            if hot.phase in ("WAITING", "PREFILL"):
                heapq.heappush(self._oldest_heap,
                               (hot.arrival_time_ms, hot.request_id,
                                hot.generation))
            heapq.heappush(self._prefill_heap,
                           (-hot.remaining_prefill, hot.request_id,
                            hot.generation))
        self._clock_ms = now_ms
        self._state_version += 1

    def advance_clock(self, now_ms: float,
                      changed: tuple[RequestExecutionState, ...] = ()) -> None:
        if now_ms < self._clock_ms:
            raise ServingPlanError("summary clock moved backwards")
        self._clock_ms = now_ms
        for request in changed:
            self.upsert(request, now_ms)
        self._state_version += 1

    def record_policy(self, policy: str, now_ms: float) -> None:
        if policy not in SCHEDULING_POLICIES:
            raise ServingPlanError("ranking model selected unavailable policy")
        self.policy_switch_count += 1
        self.last_reselection_ms = now_ms

    def snapshot(self) -> SchedulerStateSummary:
        def take(heap, phases):
            result, popped = [], []
            while heap and len(result) < self.frontier_size:
                item = heapq.heappop(heap)
                rid, generation = item[1], item[2]
                hot = self.hot.get(rid)
                if hot is None or hot.generation != generation or \
                        hot.phase not in phases:
                    continue
                result.append(hot);popped.append(item)
            for item in popped:
                heapq.heappush(heap, item)
            return result
        oldest = take(self._oldest_heap, ("WAITING", "PREFILL"))
        gaps = take(self._gap_heap, ("DECODE",))
        large = take(self._prefill_heap, ("WAITING", "PREFILL", "DECODE"))
        frontier: dict[str, HotRequest] = {}
        for group in (oldest, gaps, large):
            for item in group[:self.frontier_size]:
                frontier[item.request_id] = item
        oldest_age = max([0.0]+[
            self._clock_ms-x.arrival_time_ms for x in oldest[:1]])
        decode_count = self._phase_counts["DECODE"]
        mean_gap = ((decode_count*self._clock_ms-self._decode_anchor_sum) /
                    decode_count if decode_count else 0.0)
        largest_gap = max([0.0]+[
            self._clock_ms-x.decode_anchor_ms for x in gaps[:1]])
        return SchedulerStateSummary(
            self._state_version, self._clock_ms, len(self.hot),
            self._phase_counts["WAITING"], self._phase_counts["PREFILL"],
            decode_count,
            self._remaining_prefill, self._remaining_decode, self._prefix_hits,
            oldest_age, largest_gap, mean_gap,
            tuple(sorted(frontier.values(), key=lambda x: x.request_id)),
            self.profile.max_num_batched_tokens, self.profile.max_num_seqs,
            self.latency.prefill_ms, self.latency.decode_ms,
            self.latency.mixed_ms, self.latency.scheduler_ms,
            self.replica_core_budget, self.policy_switch_count,
            max(0.0, self._clock_ms-self.last_reselection_ms),
            self.completed_count, self.history_record_count)


def reference_summary(state: ReplicaSchedulerState, *, frontier_size: int = 8,
                      replica_core_budget: int = 2) -> SchedulerStateSummary:
    inc = IncrementalSchedulerState(state, frontier_size=frontier_size,
                                    replica_core_budget=replica_core_budget)
    return inc.snapshot()


@dataclass(frozen=True)
class SummaryDelta:
    """Structurally shared candidate change; base is immutable."""
    active_delta: int = 0
    remaining_prefill_delta: int = 0
    remaining_decode_delta: int = 0
    elapsed_ms: float = 0.0

    def apply(self, base: SchedulerStateSummary) -> SchedulerStateSummary:
        if self.elapsed_ms < 0:
            raise ServingPlanError("negative candidate elapsed time")
        return replace(
            base, state_version=base.state_version+1,
            clock_ms=base.clock_ms+self.elapsed_ms,
            active_count=base.active_count+self.active_delta,
            total_remaining_prefill=max(
                0, base.total_remaining_prefill+self.remaining_prefill_delta),
            total_remaining_decode=max(
                0, base.total_remaining_decode+self.remaining_decode_delta))


@dataclass(frozen=True)
class RankingModel:
    version: str
    coefficients: Mapping[str, Mapping[str, float]]
    intercepts: Mapping[str, float]
    feature_means: Mapping[str, float]
    feature_scales: Mapping[str, float]
    training_provenance: str
    frozen_digest: str
    equivalence_margin: float = .02
    hysteresis_margin: float = .01
    default_policy: str = "chunked_balanced"

    def payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "coefficients": {p: dict(v) for p, v in self.coefficients.items()},
            "intercepts": dict(self.intercepts),
            "feature_means": dict(self.feature_means),
            "feature_scales": dict(self.feature_scales),
            "training_provenance": self.training_provenance,
            "equivalence_margin": self.equivalence_margin,
            "hysteresis_margin": self.hysteresis_margin,
            "default_policy": self.default_policy,
        }

    def validate(self) -> None:
        digest = hashlib.sha256(json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != self.frozen_digest:
            raise ServingPlanError("selector model changed after freeze")
        if set(self.coefficients) != set(SCHEDULING_POLICIES):
            raise ServingPlanError("ranking model selects unavailable policy")
        if self.default_policy not in SCHEDULING_POLICIES:
            raise ServingPlanError("invalid ranking default policy")
        for policy in SCHEDULING_POLICIES:
            if not math.isfinite(self.intercepts[policy]) or any(
                    f not in RANKING_FEATURES or not math.isfinite(v)
                    for f, v in self.coefficients[policy].items()):
                raise ServingPlanError("non-finite or unknown ranking coefficient")


def summary_features(summary: SchedulerStateSummary) -> dict[str, float]:
    budget = max(summary.token_budget, 1)
    seqs = max(summary.sequence_budget, 1)
    return {
        "active_count": summary.active_count,
        "waiting_count": summary.waiting_count,
        "prefill_ready_count": summary.prefill_ready_count,
        "decode_ready_count": summary.decode_ready_count,
        "remaining_prefill_per_budget": summary.total_remaining_prefill/budget,
        "remaining_decode_per_sequence": summary.total_remaining_decode/seqs,
        "oldest_waiting_age_ratio": summary.oldest_waiting_age_ms/50.0,
        "largest_decode_gap_ratio": summary.largest_decode_gap_ms/10.0,
        "mean_decode_gap_ratio": summary.mean_decode_gap_ms/10.0,
        "prefix_hit_per_request": summary.prefix_hit_tokens/max(summary.active_count, 1),
        "token_budget_utilization": min(1.0, summary.total_remaining_prefill/budget),
        "sequence_budget_utilization": min(1.0, summary.active_count/seqs),
        "recent_prefill_ms": summary.recent_prefill_ms,
        "recent_decode_ms": summary.recent_decode_ms,
        "recent_mixed_ms": summary.recent_mixed_ms,
        "recent_scheduler_ms": summary.recent_scheduler_ms,
        "replica_core_budget": summary.replica_core_budget,
        "policy_switch_count": summary.policy_switch_count,
        "time_since_reselection_ms": summary.time_since_reselection_ms,
    }


@dataclass(frozen=True)
class RankingPolicyPlan:
    plan_id: str
    policy_id: str
    selector_version: str
    scores: Mapping[str, float]
    score_margin: float
    uncertain: bool
    used_default: bool
    retained_by_hysteresis: bool
    state_version: int
    selection_overhead_ms: float


class RankingSelectorV2:
    def __init__(self, model: RankingModel, *, adaptive: bool = False):
        model.validate()
        self.model = model
        self.adaptive = adaptive
        self.current_policy: str | None = None

    def select(self, summary: SchedulerStateSummary) -> RankingPolicyPlan:
        started = time.perf_counter_ns()
        raw = summary_features(summary)
        x = {f: (raw[f]-self.model.feature_means.get(f, 0.0)) /
             max(self.model.feature_scales.get(f, 1.0), 1e-12)
             for f in RANKING_FEATURES}
        scores = {p: self.model.intercepts[p] + sum(
            c*x[f] for f, c in self.model.coefficients[p].items())
                  for p in SCHEDULING_POLICIES}
        ordered = sorted(scores, key=lambda p: (scores[p], p))
        margin = scores[ordered[1]]-scores[ordered[0]]
        scale = max(abs(scores[ordered[0]]), 1.0)
        uncertain = margin/scale <= self.model.equivalence_margin
        chosen = self.model.default_policy if uncertain else ordered[0]
        used_default = uncertain
        retained = False
        if self.current_policy is not None and self.current_policy != chosen:
            improvement = scores[self.current_policy]-scores[chosen]
            if improvement/scale <= self.model.hysteresis_margin:
                chosen = self.current_policy
                retained = True
        self.current_policy = chosen
        return RankingPolicyPlan(
            f"ranking-v2-{summary.state_version}", chosen,
            ADAPTIVE_SELECTOR_VERSION if self.adaptive else
            STATIC_SELECTOR_VERSION, scores, margin, uncertain, used_default,
            retained, summary.state_version,
            (time.perf_counter_ns()-started)/1e6)
