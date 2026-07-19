"""Compiler-owned shared-memory attention placement and verification.

This is single-process, shared-memory planning.  It is not a network
collective, multi-process tensor parallelism, or vLLM distributed serving.
"""
from __future__ import annotations
from typing import Any

from deployment.cpu_sharding import ShardingPlanError, uneven_ranges

CONTRACT_VERSION = "hir.shared_memory_attention_placement.v1"


def build_attention_placement(
    *, batch: int, query_length: int, context_length: int, query_heads: int,
    kv_heads: int, head_dim: int, strategy: str, worker_count: int,
    predicted: dict[str, float] | None = None, direct_output: bool = False,
) -> dict[str, Any]:
    """Create deterministic, balanced, disjoint logical output assignments."""
    if strategy == "serial":
        axis, size, ranges = "none", query_heads, ((0, query_heads),)
    elif strategy == "split_head":
        axis, size = "query_head", query_heads
        ranges = uneven_ranges(size, worker_count)
    elif strategy == "split_query":
        axis, size = "query_token", query_length
        ranges = uneven_ranges(size, worker_count)
    else:
        raise ShardingPlanError("unsupported distributed attention strategy")
    if worker_count != len(ranges) or any(begin == end for begin, end in ranges):
        raise ShardingPlanError("distributed placement contains an empty shard")
    group = query_heads // kv_heads
    workers: list[dict[str, Any]] = []
    for worker_id, (begin, end) in enumerate(ranges):
        if strategy in {"serial", "split_head"}:
            qh_begin, qh_end, q_begin, q_end = begin, end, 0, query_length
        else:
            qh_begin, qh_end, q_begin, q_end = 0, query_heads, begin, end
        kv_used = sorted({head // group for head in range(qh_begin, qh_end)})
        extent = batch * (qh_end - qh_begin) * (q_end - q_begin) * head_dim
        workers.append({
            "worker_id": worker_id,
            "logical_cpu_id": worker_id,
            "query_head_range": [qh_begin, qh_end],
            "query_token_range": [q_begin, q_end],
            "kv_head_read_range": [min(kv_used), max(kv_used) + 1],
            "q_read": {"kind": "shard_local", "head_range": [qh_begin, qh_end],
                       "query_range": [q_begin, q_end]},
            "k_read": {"kind": "shared_read_only",
                       "kv_head_range": [min(kv_used), max(kv_used) + 1],
                       "context_range": [0, "runtime_context_length"]},
            "v_read": {"kind": "shared_read_only",
                       "kv_head_range": [min(kv_used), max(kv_used) + 1],
                       "context_range": [0, "runtime_context_length"]},
            "output_write": {
                "kind": "disjoint_logical_tensor_slice",
                "head_range": [qh_begin, qh_end],
                "query_range": [q_begin, q_end],
                "extent_elements": extent,
                "offset_elements": (
                    qh_begin * query_length * head_dim
                    if batch == 1 and strategy in {"serial", "split_head"}
                    else None),
            },
            "scratch": {"ownership": "worker_private"},
        })
    costs = predicted or {}
    plan = {
        "contract_version": CONTRACT_VERSION,
        "execution_kind": "compiler_planned_shared_memory_multi_worker",
        "parallel_strategy": strategy,
        "split_dimension": axis,
        "worker_count": worker_count,
        "work_item_count": worker_count,
        "placement_policy": "deterministic_balanced_remainder",
        "workers": workers,
        "communication": {
            "kind": "shared_memory",
            "shared_input_buffers": ["key", "value"],
            "query_access": "shard_local",
            "output_access": "disjoint_logical_slices",
            "copy_count": 0 if direct_output else worker_count,
            "copy_bytes": 0 if direct_output else (
                batch * query_heads * query_length * head_dim * 4),
            "reduction": "none",
        },
        "synchronization": {
            "kind": "completion_counter",
            "synchronization_points": 1,
            "consumer": "o_proj",
            "consumer_requires_all_shards_complete": True,
        },
        "predicted_cost": {
            "kernel_compute": float(costs.get("predicted_kernel_compute", 0.0)),
            "dispatch": float(costs.get("predicted_dispatch", 0.0)),
            "worker_wakeup": float(costs.get("predicted_worker_wakeup", 0.0)),
            "barrier": float(costs.get("predicted_barrier", 0.0)),
            "imbalance": float(costs.get("predicted_imbalance", 0.0)),
            "shared_cache_contention": float(
                costs.get("predicted_shared_cache_contention", 0.0)),
            "memory_bandwidth_contention": float(
                costs.get("predicted_memory_bandwidth_contention", 0.0)),
            "worker_underutilization": float(
                costs.get("predicted_worker_underutilization", 0.0)),
            "total": float(costs.get("predicted_parallel_total", 0.0)),
        },
        "runtime_no_repartition": True,
        "runtime_no_worker_count_override": True,
        "runtime_no_strategy_override": True,
    }
    validate_attention_placement(
        plan, query_length=query_length, query_heads=query_heads,
        kv_heads=kv_heads, head_dim=head_dim, batch=batch)
    return plan


def validate_attention_placement(
    placement: dict[str, Any], *, query_length: int, query_heads: int,
    kv_heads: int, head_dim: int, batch: int,
) -> None:
    if placement.get("contract_version") != CONTRACT_VERSION:
        raise ShardingPlanError("distributed placement ABI version mismatch")
    strategy = placement.get("parallel_strategy")
    if strategy not in {"serial", "split_head", "split_query"}:
        raise ShardingPlanError("unsupported distributed split dimension")
    workers = placement.get("workers")
    count = placement.get("worker_count")
    if not isinstance(workers, list) or count != len(workers) or count < 1:
        raise ShardingPlanError("worker count mismatches placement count")
    if [x.get("worker_id") for x in workers] != list(range(count)):
        raise ShardingPlanError("invalid or non-deterministic worker ID")
    if placement.get("work_item_count") != count:
        raise ShardingPlanError("work item count mismatch")
    if placement.get("communication", {}).get("kind") != "shared_memory":
        raise ShardingPlanError("only shared-memory communication is supported")
    if placement["communication"].get("reduction") != "none":
        raise ShardingPlanError("attention split-head/query requires no reduction")
    sync = placement.get("synchronization", {})
    if (sync.get("kind") != "completion_counter"
            or sync.get("synchronization_points") != 1
            or sync.get("consumer_requires_all_shards_complete") is not True):
        raise ShardingPlanError("one completion barrier before o_proj is required")
    axis_key, size = (
        ("query_token_range", query_length) if strategy == "split_query"
        else ("query_head_range", query_heads))
    cursor = 0
    logical_outputs: list[tuple[int, int, int, int]] = []
    group = query_heads // kv_heads
    for worker in workers:
        begin, end = worker.get(axis_key, [None, None])
        if not isinstance(begin, int) or not isinstance(end, int):
            raise ShardingPlanError("malformed worker shard range")
        if begin != cursor or end <= begin or end > size:
            raise ShardingPlanError("worker shards overlap, contain a gap, or are out of bounds")
        cursor = end
        hb, he = worker.get("query_head_range", [None, None])
        qb, qe = worker.get("query_token_range", [None, None])
        logical_outputs.append((hb, he, qb, qe))
        expected_kv = sorted({h // group for h in range(hb, he)})
        if worker.get("kv_head_read_range") != [min(expected_kv), max(expected_kv)+1]:
            raise ShardingPlanError("worker GQA KV-head read mapping mismatch")
        expected_extent = batch * (he-hb) * (qe-qb) * head_dim
        if worker.get("output_write", {}).get("extent_elements") != expected_extent:
            raise ShardingPlanError("worker output ownership extent mismatch")
    if cursor != size:
        raise ShardingPlanError("worker placement does not cover complete output axis")
    for i, a in enumerate(logical_outputs):
        for b in logical_outputs[i+1:]:
            head_overlap = max(a[0], b[0]) < min(a[1], b[1])
            query_overlap = max(a[2], b[2]) < min(a[3], b[3])
            if head_overlap and query_overlap:
                raise ShardingPlanError("overlapping output shard ranges")
    sizes = [x[1]-x[0] for x in (
        [w[axis_key] for w in workers])]
    if max(sizes)-min(sizes) > 1:
        raise ShardingPlanError("balanced placement imbalance exceeds one shard")
    for flag in ("runtime_no_repartition", "runtime_no_worker_count_override",
                 "runtime_no_strategy_override"):
        if placement.get(flag) is not True:
            raise ShardingPlanError(f"{flag} must be true")
