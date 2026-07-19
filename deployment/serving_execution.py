"""Serving Distributed S1 planning for a functional single-node CPU cluster.

This module deliberately separates CPUReplica from operator LogicalWorker.
Replicas own request queues and prefix-cache state.  Operator workers remain an
implementation detail of the referenced OperatorExecutionPlan.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
import time
import uuid
from typing import Any, Callable, Iterable

SERVING_SCHEMA_VERSION = 1
ROUTING_POLICIES = ("round_robin", "least_queue", "max_prefix_hit",
                    "prefix_queue_cost")
CACHE_MODES = ("metadata_only", "functional_tensor")


class ServingPlanError(ValueError):
    pass


@dataclass(frozen=True)
class CPUReplicaProfile:
    replica_id: str
    logical_core_budget: int
    queue_capacity: int = 64
    kv_capacity_bytes: int = 256 * 1024 * 1024
    enabled: bool = True
    execution_backend: str = "cpu_functional"

    def __post_init__(self) -> None:
        if not self.replica_id or self.logical_core_budget < 1:
            raise ServingPlanError("replica ID and positive core budget required")
        if self.queue_capacity < 1 or self.kv_capacity_bytes < 0:
            raise ServingPlanError("invalid replica queue or KV capacity")
        if not self.execution_backend:
            raise ServingPlanError("replica has no execution backend")


@dataclass(frozen=True)
class FunctionalClusterProfile:
    cluster_id: str
    total_logical_cores: int
    replicas: tuple[CPUReplicaProfile, ...]
    execution_mode: str = "functional_cpu"
    profile_version: int = 1
    affinity_applied: bool = False

    def __post_init__(self) -> None:
        ids = [r.replica_id for r in self.replicas]
        if not self.cluster_id or self.total_logical_cores < 1 or not ids:
            raise ServingPlanError("non-empty functional CPU cluster required")
        if len(ids) != len(set(ids)):
            raise ServingPlanError("duplicate replica IDs")
        if sum(r.logical_core_budget for r in self.replicas) > self.total_logical_cores:
            raise ServingPlanError("replica budgets exceed cluster logical cores")

    @classmethod
    def local(cls, replica_count: int = 4, *, total_logical_cores: int | None = None,
              total_kv_capacity_bytes: int = 1024 * 1024 * 1024) -> "FunctionalClusterProfile":
        cores = total_logical_cores or (os.cpu_count() or 1)
        if replica_count < 1 or replica_count > cores:
            raise ServingPlanError("replica count exceeds available logical cores")
        base, remainder = divmod(cores, replica_count)
        profiles = tuple(CPUReplicaProfile(
            f"replica-{i}", base + (i < remainder),
            kv_capacity_bytes=total_kv_capacity_bytes // replica_count)
            for i in range(replica_count))
        return cls(f"local_cpu_{cores}c_{replica_count}r", cores, profiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id, "profile_version": self.profile_version,
            "execution_mode": self.execution_mode,
            "detected_logical_cores": self.total_logical_cores,
            "affinity_applied": self.affinity_applied,
            "replicas": [vars(r) for r in self.replicas],
        }


@dataclass(frozen=True)
class ServingRequest:
    request_id: str
    token_ids: tuple[int, ...]
    expected_output_tokens: int
    arrival_time_ms: float = 0.0
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.request_id or not self.token_ids or self.expected_output_tokens < 1:
            raise ServingPlanError("request ID, prompt tokens, and output count required")


@dataclass
class CacheLookup:
    matched_tokens: int
    matched_blocks: int
    uncached_tokens: int
    required_new_blocks: int


class ReplicaPrefixCache:
    """Per-replica complete-block prefix metadata with parent-hash lineage."""
    def __init__(self, capacity_bytes: int, *, block_size: int = 16,
                 bytes_per_token: int = 4096, mode: str = "metadata_only"):
        if mode not in CACHE_MODES or capacity_bytes < 0 or block_size < 1:
            raise ServingPlanError("invalid cache configuration")
        self.capacity_bytes, self.block_size = capacity_bytes, block_size
        self.bytes_per_token, self.mode = bytes_per_token, mode
        self._blocks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.used_bytes = self.evictions = self.hits = self.misses = 0
        self.peak_bytes = 0

    @staticmethod
    def _hash(parent: str, tokens: tuple[int, ...]) -> str:
        raw = parent.encode() + b"|" + ",".join(map(str, tokens)).encode()
        return hashlib.sha256(raw).hexdigest()

    def _lineage(self, token_ids: tuple[int, ...]) -> list[tuple[str, str, tuple[int, ...]]]:
        parent, result = "root", []
        complete = len(token_ids) // self.block_size
        for i in range(complete):
            tokens = token_ids[i * self.block_size:(i + 1) * self.block_size]
            digest = self._hash(parent, tokens)
            result.append((digest, parent, tokens))
            parent = digest
        return result

    def lookup(self, token_ids: tuple[int, ...]) -> CacheLookup:
        matched = 0
        for digest, parent, tokens in self._lineage(token_ids):
            entry = self._blocks.get(digest)
            # Token and parent comparison makes the collision assumption explicit.
            if not entry or entry["parent"] != parent or entry["tokens"] != tokens:
                break
            matched += self.block_size
            self._blocks.move_to_end(digest)
        if matched:
            self.hits += 1
        else:
            self.misses += 1
        uncached = len(token_ids) - matched
        return CacheLookup(matched, matched // self.block_size, uncached,
                           uncached // self.block_size)

    def insert(self, token_ids: tuple[int, ...], tensor_refs: dict[str, Any] | None = None) -> None:
        block_bytes = self.block_size * self.bytes_per_token
        for digest, parent, tokens in self._lineage(token_ids):
            if digest in self._blocks:
                self._blocks.move_to_end(digest)
                continue
            while self._blocks and self.used_bytes + block_bytes > self.capacity_bytes:
                self._blocks.popitem(last=False)
                self.used_bytes -= block_bytes
                self.evictions += 1
            if block_bytes > self.capacity_bytes:
                continue
            self._blocks[digest] = {
                "parent": parent, "tokens": tokens, "bytes": block_bytes,
                "tensor_ref": tensor_refs.get(digest) if
                self.mode == "functional_tensor" and tensor_refs else None,
            }
            self.used_bytes += block_bytes
            self.peak_bytes = max(self.peak_bytes, self.used_bytes)

    def snapshot(self) -> dict[str, Any]:
        return {"mode": self.mode, "block_size": self.block_size,
                "entries": len(self._blocks), "used_bytes": self.used_bytes,
                "peak_bytes": self.peak_bytes, "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions}


@dataclass
class CPUReplica:
    profile: CPUReplicaProfile
    cache: ReplicaPrefixCache
    queue: deque[str] = field(default_factory=deque)
    active: set[str] = field(default_factory=set)
    available_at_ms: float = 0.0
    completed: int = 0
    failed: int = 0
    busy_ms: float = 0.0


@dataclass(frozen=True)
class CPUServiceCostModel:
    """CPU-derived functional service curve, not a GPU latency model."""
    prefill_base_ms: float = 0.15
    prefill_token_ms: float = 0.055
    decode_token_ms: float = 0.18
    cache_pressure_weight_ms: float = 0.2
    provenance: str = "derived_from_measured_cpu"

    def cost(self, replica: CPUReplica, request: ServingRequest,
             lookup: CacheLookup, now_ms: float) -> dict[str, float]:
        core_scale = 2.0 / replica.profile.logical_core_budget
        queue = max(0.0, replica.available_at_ms - now_ms)
        prefill = (self.prefill_base_ms + lookup.uncached_tokens *
                   self.prefill_token_ms) * core_scale
        decode = request.expected_output_tokens * self.decode_token_ms * core_scale
        pressure = (replica.cache.used_bytes / max(1, replica.cache.capacity_bytes)
                    * self.cache_pressure_weight_ms)
        return {"queue_wait_ms": queue, "prefill_ms": prefill,
                "decode_ms": decode, "cache_pressure_ms": pressure,
                "routing_overhead_ms": 0.0,
                "total_ms": queue + prefill + decode + pressure}


@dataclass(frozen=True)
class ServingExecutionPlan:
    plan_id: str
    request_id: str
    cluster_id: str
    cluster_profile_version: int
    selected_replica_id: str
    routing_policy: str
    routing_candidate_id: str
    cache_mode: str
    prompt_tokens: int
    matched_tokens: int
    uncached_tokens: int
    matched_blocks: int
    predicted_cost: dict[str, float]
    selection_reason: str
    operator_plan_policy: str = "compiler_selected"
    operator_plan_id: str | None = None
    schema_version: int = SERVING_SCHEMA_VERSION
    plan_kind: str = "serving_request_placement"

    def to_dict(self) -> dict[str, Any]:
        return vars(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any],
                  cluster: FunctionalClusterProfile) -> "ServingExecutionPlan":
        try:
            plan = cls(**payload)
        except (TypeError, KeyError) as exc:
            raise ServingPlanError(f"invalid serving plan: {exc}") from exc
        plan.validate(cluster)
        return plan

    def validate(self, cluster: FunctionalClusterProfile) -> None:
        if self.schema_version != SERVING_SCHEMA_VERSION or \
                self.plan_kind != "serving_request_placement":
            raise ServingPlanError("serving plan schema/kind mismatch")
        if self.cluster_id != cluster.cluster_id or \
                self.cluster_profile_version != cluster.profile_version:
            raise ServingPlanError("cluster profile mismatch")
        replicas = {r.replica_id: r for r in cluster.replicas}
        if self.selected_replica_id not in replicas:
            raise ServingPlanError("selected replica does not exist")
        if not replicas[self.selected_replica_id].enabled:
            raise ServingPlanError("selected replica is disabled")
        if self.routing_policy not in ROUTING_POLICIES or self.cache_mode not in CACHE_MODES:
            raise ServingPlanError("unknown routing policy or cache mode")
        if not self.request_id or self.matched_tokens > self.prompt_tokens or \
                self.uncached_tokens != self.prompt_tokens - self.matched_tokens:
            raise ServingPlanError("inconsistent request/cache fields")
        if not self.predicted_cost or any(not math.isfinite(v) or v < 0
                                          for v in self.predicted_cost.values()):
            raise ServingPlanError("predicted costs must be finite and nonnegative")

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class ServingDistributedCompiler:
    def __init__(self, cluster: FunctionalClusterProfile,
                 cost_model: CPUServiceCostModel | None = None):
        self.cluster, self.cost_model = cluster, cost_model or CPUServiceCostModel()
        self.round_robin_cursor = 0
        self.selection_records: list[dict[str, Any]] = []

    def plan(self, request: ServingRequest, replicas: dict[str, CPUReplica],
             *, policy: str = "prefix_queue_cost",
             operator_plan_id: str | None = None) -> ServingExecutionPlan:
        if policy not in ROUTING_POLICIES:
            raise ServingPlanError("unknown routing policy")
        started = time.perf_counter_ns()
        candidates = []
        ordered = [replicas[p.replica_id] for p in self.cluster.replicas if p.enabled]
        for replica in ordered:
            lookup = replica.cache.lookup(request.token_ids)
            cost = self.cost_model.cost(replica, request, lookup,
                                        request.arrival_time_ms)
            candidates.append((replica, lookup, cost))
        if not candidates:
            raise ServingPlanError("no legal serving replica")
        if policy == "round_robin":
            chosen = candidates[self.round_robin_cursor % len(candidates)]
            self.round_robin_cursor += 1
            key = "deterministic round-robin"
        elif policy == "least_queue":
            chosen = min(candidates, key=lambda x: (x[2]["queue_wait_ms"] +
                                                    x[2]["decode_ms"],
                                                    x[0].profile.replica_id))
            key = "minimum predicted queue completion"
        elif policy == "max_prefix_hit":
            chosen = min(candidates, key=lambda x: (-x[1].matched_tokens,
                                                    x[0].profile.replica_id))
            key = "maximum complete-block prefix match"
        else:
            chosen = min(candidates, key=lambda x: (x[2]["total_ms"],
                                                    x[0].profile.replica_id))
            key = "minimum predicted queue+uncached-prefill+decode+pressure"
        routing_ms = (time.perf_counter_ns() - started) / 1e6
        replica, lookup, cost = chosen
        cost = dict(cost)
        cost["routing_overhead_ms"] = routing_ms
        cost["total_ms"] += routing_ms
        plan = ServingExecutionPlan(
            plan_id=f"serving-{uuid.uuid4().hex[:16]}",
            request_id=request.request_id, cluster_id=self.cluster.cluster_id,
            cluster_profile_version=self.cluster.profile_version,
            selected_replica_id=replica.profile.replica_id,
            routing_policy=policy, routing_candidate_id=f"{policy}_v1",
            cache_mode=replica.cache.mode, prompt_tokens=len(request.token_ids),
            matched_tokens=lookup.matched_tokens,
            uncached_tokens=lookup.uncached_tokens,
            matched_blocks=lookup.matched_blocks, predicted_cost=cost,
            selection_reason=key, operator_plan_id=operator_plan_id)
        plan.validate(self.cluster)
        self.selection_records.append({
            "request_id": request.request_id,
            "generated_candidates": len(candidates) * len(ROUTING_POLICIES),
            "legal_replicas": [x[0].profile.replica_id for x in candidates],
            "candidate_costs": {x[0].profile.replica_id: x[2] for x in candidates},
            "selected_replica_id": plan.selected_replica_id,
            "routing_candidate_id": plan.routing_candidate_id,
        })
        return plan


class PlanOnlyServingRuntime:
    """Consumes a deserialized serving plan; never reroutes or picks a replica."""
    def __init__(self, cluster: FunctionalClusterProfile, *,
                 cache_mode: str = "metadata_only", block_size: int = 16,
                 bytes_per_token: int = 4096):
        self.cluster = cluster
        self.replicas = {p.replica_id: CPUReplica(
            p, ReplicaPrefixCache(p.kv_capacity_bytes, block_size=block_size,
                                  bytes_per_token=bytes_per_token, mode=cache_mode))
            for p in cluster.replicas}
        self.runtime_replica_override_count = 0
        self.runtime_reroute_count = 0
        self.missing_plan_count = 0
        self.manual_replica_assignment_count = 0
        self.fallback_routing_count = 0
        self.events: list[dict[str, Any]] = []
        self._completed_ids: set[str] = set()

    def execute(self, request: ServingRequest, plan: ServingExecutionPlan,
                execute_fn: Callable[[CPUReplica, ServingRequest, ServingExecutionPlan],
                                     dict[str, Any]] | None = None) -> dict[str, Any]:
        if plan is None:
            self.missing_plan_count += 1
            raise ServingPlanError("ServingExecutionPlan is required")
        plan.validate(self.cluster)
        if request.request_id != plan.request_id:
            raise ServingPlanError("request ID mismatch")
        if request.request_id in self._completed_ids:
            raise ServingPlanError("duplicate request ID")
        replica = self.replicas[plan.selected_replica_id]
        if len(replica.queue) >= replica.profile.queue_capacity:
            raise ServingPlanError("selected replica queue is full; rerouting forbidden")
        replica.queue.append(request.request_id)
        queue_enter = max(request.arrival_time_ms, 0.0)
        execution_start = max(queue_enter, replica.available_at_ms)
        replica.active.add(request.request_id)
        if replica.queue.popleft() != request.request_id:
            raise ServingPlanError("queue ownership mismatch")
        callback_result = execute_fn(replica, request, plan) if execute_fn else {}
        service = plan.predicted_cost["prefill_ms"] + plan.predicted_cost["decode_ms"]
        measured = callback_result.get("measured_execution_ms")
        duration = float(measured if measured is not None else service)
        first_token = execution_start + float(callback_result.get(
            "measured_first_token_ms", plan.predicted_cost["prefill_ms"]))
        completion = execution_start + duration
        replica.available_at_ms = completion
        replica.busy_ms += duration
        replica.active.remove(request.request_id)
        replica.completed += 1
        replica.cache.insert(request.token_ids,
                             callback_result.get("functional_tensor_refs"))
        self._completed_ids.add(request.request_id)
        event = {
            "request_id": request.request_id, "serving_plan_id": plan.plan_id,
            "planned_replica_id": plan.selected_replica_id,
            "executed_replica_id": replica.profile.replica_id,
            "routing_candidate_id": plan.routing_candidate_id,
            "cache_mode": plan.cache_mode, "matched_prefix_tokens": plan.matched_tokens,
            "uncached_prompt_tokens": plan.uncached_tokens,
            "state_transitions": ["ARRIVED", "QUEUED", "RUNNING_PREFILL",
                                  "RUNNING_DECODE", "FINISHED"],
            "timestamps_ms": {"arrival": request.arrival_time_ms,
                              "plan_created": queue_enter,
                              "queue_enter": queue_enter,
                              "execution_start": execution_start,
                              "first_token": first_token,
                              "completion": completion},
            "routing_overhead_ms": plan.predicted_cost["routing_overhead_ms"],
            "queue_wait_ms": execution_start - request.arrival_time_ms,
            "ttft_ms": first_token - request.arrival_time_ms,
            "end_to_end_ms": completion - request.arrival_time_ms,
            "operator_provenance": callback_result.get("operator_provenance", []),
            "generated_token_ids": callback_result.get("generated_token_ids", []),
        }
        self.events.append(event)
        return event

    def counters(self) -> dict[str, int]:
        return {
            "runtime_replica_override_count": self.runtime_replica_override_count,
            "runtime_reroute_count": self.runtime_reroute_count,
            "missing_plan_count": self.missing_plan_count,
            "manual_replica_assignment_count": self.manual_replica_assignment_count,
            "fallback_routing_count": self.fallback_routing_count,
        }


def deserialize_serving_plan(text: str,
                             cluster: FunctionalClusterProfile) -> ServingExecutionPlan:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ServingPlanError("invalid serving plan JSON") from exc
    return ServingExecutionPlan.from_dict(payload, cluster)


def deterministic_trace(kind: str, count: int, *, block_size: int = 16,
                        seed: int = 20260717) -> list[ServingRequest]:
    """Saved-seed deterministic synthetic token traces for policy evaluation."""
    if count < 1:
        raise ServingPlanError("trace must contain requests")
    common = tuple(range(1000, 1000 + 2 * block_size))
    hot_a = tuple(range(2000, 2000 + 4 * block_size))
    hot_b = tuple(range(3000, 3000 + 3 * block_size))
    rows = []
    for i in range(count):
        if kind == "shared_prefix":
            tokens = common + (4000 + i, 5000 + i)
        elif kind == "hot_prefix":
            prefix = hot_a if i % 5 else hot_b
            tokens = prefix + (4000 + i,)
        elif kind == "unique_prefix":
            tokens = tuple(seed + i * 1000 + j for j in range(34))
        elif kind == "capacity_pressure":
            prefix = tuple(seed + (i % 20) * 1000 + j for j in range(64))
            tokens = prefix + (i,)
        else:
            raise ServingPlanError("unknown deterministic trace kind")
        rows.append(ServingRequest(f"{kind}-{i}", tokens, 4,
                                   arrival_time_ms=i * 0.12))
    return rows
