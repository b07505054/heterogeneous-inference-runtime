"""Plan-driven, single-node CPU sharding prototype.

This module is deliberately not a vLLM tensor-parallel executor.  It executes
one linear subgraph with persistent threads in one process and shared memory.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np


PROVENANCE = {"user_specified", "compiler_inferred", "cost_model_selected",
              "fallback_replicated"}
STRATEGIES = {"replicated", "split_m", "row_parallel", "column_parallel"}
COLLECTIVES = {"shared_memory_reduce", "direct_assembly", "barrier"}


class ShardingPlanError(ValueError):
    pass


def uneven_ranges(size: int, ranks: int) -> tuple[tuple[int, int], ...]:
    if size < 0 or ranks < 1:
        raise ShardingPlanError("size must be nonnegative and ranks positive")
    q, r = divmod(size, ranks)
    return tuple((i * q + min(i, r), (i + 1) * q + min(i + 1, r))
                 for i in range(ranks))


def validate_sharding(plan: dict[str, Any]) -> None:
    mesh = plan.get("mesh", {})
    axes = mesh.get("axes", [])
    if mesh.get("name") != "cpu_mesh" or len(axes) != 1:
        raise ShardingPlanError("one mesh named cpu_mesh is required")
    axis = axes[0]
    if axis.get("name") != "cpu" or not isinstance(axis.get("size"), int):
        raise ShardingPlanError("mesh axis cpu with integer size is required")
    if axis["size"] < 1:
        raise ShardingPlanError("mesh size must be positive")
    spec = plan.get("operator_sharding", {})
    if spec.get("mesh_axis") != "cpu":
        raise ShardingPlanError("operator references nonexistent mesh axis")
    if spec.get("strategy") not in STRATEGIES:
        raise ShardingPlanError("unsupported sharding strategy")
    if spec.get("provenance") not in PROVENANCE:
        raise ShardingPlanError("invalid sharding provenance")
    if spec.get("uneven_policy") not in {"balanced_remainder", "reject"}:
        raise ShardingPlanError("explicit uneven_policy is required")
    for collective in plan.get("collectives", []):
        if collective.get("kind") not in COLLECTIVES:
            raise ShardingPlanError("illegal collective for shared-memory runtime")
    mapping = plan.get("rank_mapping", {})
    cores = mapping.get("logical_cpu_ids", [])
    if mapping.get("policy") != "pinned_logical_cpu" or len(cores) != axis["size"]:
        raise ShardingPlanError("one logical CPU mapping per rank is required")
    if len(set(cores)) != len(cores) or any(not isinstance(c, int) or c < 0 for c in cores):
        raise ShardingPlanError("logical CPU mappings must be unique nonnegative integers")


def make_plan(workers: int = 8, strategy: str = "split_m",
              provenance: str = "user_specified") -> dict[str, Any]:
    collectives: list[dict[str, str]] = []
    dim = 0
    if strategy == "row_parallel":
        dim = 0
        collectives = [{"kind": "shared_memory_reduce",
                        "semantics": "sum partial output tensors in one process"}]
    elif strategy == "column_parallel":
        dim = 1
        collectives = [{"kind": "direct_assembly",
                        "semantics": "write disjoint output slices into shared output"}]
    plan = {
        "mesh": {"name": "cpu_mesh", "axes": [{"name": "cpu", "size": workers}]},
        "operator_sharding": {
            "strategy": strategy, "tensor_dimension": dim, "mesh_axis": "cpu",
            "uneven_policy": "balanced_remainder", "provenance": provenance,
        },
        "collectives": collectives,
        "rank_mapping": {"policy": "pinned_logical_cpu",
                         "logical_cpu_ids": list(range(workers))},
        "fallback": {"strategy": "replicated", "reason": "unsupported_or_ambiguous_op"},
    }
    validate_sharding(plan)
    return plan


def propagate_linear_subgraph(ops: list[dict[str, Any]], input_spec: dict[str, Any]
                              ) -> list[dict[str, Any]]:
    """Propagate through linear/bias/activation/unambiguous reshape.

    Unsupported operations are explicitly fallback-replicated.
    """
    current = dict(input_spec)
    out = []
    for op in ops:
        kind = op["op"]
        annotated = dict(op)
        if kind in {"matmul", "linear", "bias_add", "relu", "gelu"}:
            current = {**current, "provenance": "compiler_inferred"}
        elif kind == "reshape" and op.get("dimension_mapping_unambiguous") is True:
            current = {**current, "provenance": "compiler_inferred"}
        else:
            current = {"strategy": "replicated",
                       "provenance": "fallback_replicated",
                       "reason": f"unsupported_or_ambiguous:{kind}"}
        annotated["sharding"] = dict(current)
        out.append(annotated)
    return out


def materialize_mlir_example(plan: dict[str, Any]) -> str:
    """Emit parseable generic HIR attributes, not a fake dialect."""
    validate_sharding(plan)
    spec = plan["operator_sharding"]
    collective = plan["collectives"][0]["kind"] if plan["collectives"] else "none"
    return f'''module attributes {{
  hir.sharding.mesh = {{name = "cpu_mesh", axis = "cpu", size = {plan["mesh"]["axes"][0]["size"]} : i64}},
  hir.sharding.rank_mapping = "pinned_logical_cpu"
}} {{
  func.func @linear(%x: tensor<?x?xf32>, %w: tensor<?x?xf32>) -> tensor<?x?xf32>
      attributes {{
        hir.sharding.strategy = "{spec["strategy"]}",
        hir.sharding.tensor_dimension = {spec["tensor_dimension"]} : i64,
        hir.sharding.provenance = "{spec["provenance"]}",
        hir.sharding.uneven_policy = "{spec["uneven_policy"]}",
        hir.sharding.materialization = "{collective}"
      }}
}}'''


@dataclass
class Timing:
    total_ms: float
    dispatch_ms: float
    compute_ms: float
    collective_ms: float


class PersistentCPUShardRuntime:
    """Persistent shared-memory worker threads pinned to logical CPU IDs."""

    def __init__(self, plan: dict[str, Any]):
        validate_sharding(plan)
        self.plan = plan
        self.workers = plan["mesh"]["axes"][0]["size"]
        self.cores = tuple(plan["rank_mapping"]["logical_cpu_ids"])
        self._rank_lock = threading.Lock()
        self._next_rank = 0
        self.affinity: dict[int, tuple[int, ...]] = {}
        self.pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="hir-cpu-rank",
            initializer=self._initialize_worker)
        # Force all persistent workers to start before the first measured call.
        gate = threading.Barrier(self.workers)
        list(self.pool.map(lambda _: gate.wait(), range(self.workers)))

    def _initialize_worker(self) -> None:
        with self._rank_lock:
            rank = self._next_rank
            self._next_rank += 1
        core = self.cores[rank]
        try:
            os.sched_setaffinity(0, {core})
            actual = tuple(sorted(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            actual = ()
        self.affinity[rank] = actual
        threading.current_thread().hir_rank = rank  # type: ignore[attr-defined]

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def __enter__(self) -> "PersistentCPUShardRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def linear(self, x: np.ndarray, w: np.ndarray,
               bias: np.ndarray | None = None, activation: str | None = None
               ) -> tuple[np.ndarray, Timing]:
        if x.dtype != np.float32 or w.dtype != np.float32:
            raise ShardingPlanError("runtime supports contiguous fp32 only")
        if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
            raise ShardingPlanError("static rank-2 linear shape mismatch")
        if not x.flags.c_contiguous or not w.flags.c_contiguous:
            raise ShardingPlanError("runtime supports contiguous layout only")
        strategy = self.plan["operator_sharding"]["strategy"]
        t0 = time.perf_counter_ns()
        compute_ns = [0] * self.workers

        def measured(fn: Any, rank: int) -> np.ndarray:
            a = time.perf_counter_ns()
            result = fn()
            compute_ns[rank] = time.perf_counter_ns() - a
            return result

        if strategy in {"split_m", "replicated"}:
            ranges = uneven_ranges(x.shape[0], self.workers)
            futures = [self.pool.submit(measured, lambda s=s, e=e: x[s:e] @ w, r)
                       for r, (s, e) in enumerate(ranges)]
            dispatched = time.perf_counter_ns()
            parts = [f.result() for f in futures]
            c0 = time.perf_counter_ns()
            y = np.concatenate(parts, axis=0)
        elif strategy == "column_parallel":
            ranges = uneven_ranges(w.shape[1], self.workers)
            futures = [self.pool.submit(measured, lambda s=s, e=e: x @ w[:, s:e], r)
                       for r, (s, e) in enumerate(ranges)]
            dispatched = time.perf_counter_ns()
            parts = [f.result() for f in futures]
            c0 = time.perf_counter_ns()
            y = np.concatenate(parts, axis=1)
        elif strategy == "row_parallel":
            ranges = uneven_ranges(w.shape[0], self.workers)
            futures = [self.pool.submit(
                measured, lambda s=s, e=e: x[:, s:e] @ w[s:e, :], r)
                for r, (s, e) in enumerate(ranges)]
            dispatched = time.perf_counter_ns()
            parts = [f.result() for f in futures]
            c0 = time.perf_counter_ns()
            y = np.sum(parts, axis=0)
        else:
            raise ShardingPlanError("unreachable unsupported strategy")
        c1 = time.perf_counter_ns()
        if bias is not None:
            if bias.shape != (w.shape[1],):
                raise ShardingPlanError("bias shape mismatch")
            y += bias
        if activation == "relu":
            np.maximum(y, 0, out=y)
        elif activation not in {None, "identity"}:
            raise ShardingPlanError(f"unsupported activation:{activation}")
        end = time.perf_counter_ns()
        return y, Timing(
            total_ms=(end - t0) / 1e6,
            dispatch_ms=(dispatched - t0) / 1e6,
            compute_ms=max(compute_ns) / 1e6,
            collective_ms=(c1 - c0) / 1e6,
        )


def vllm_metadata_boundary(request: dict[str, Any], plan: dict[str, Any]
                           ) -> dict[str, Any]:
    """Plan selection metadata beside vLLM; it does not alter model outputs."""
    validate_sharding(plan)
    return {
        "request_id": request.get("request_id"),
        "prompt_tokens": request.get("prompt_tokens"),
        "selected_cpu_sharding": plan["operator_sharding"]["strategy"],
        "worker_count": plan["mesh"]["axes"][0]["size"],
        "integration_class": "partial_plan_driven_sidecar",
        "generated_tokens_depend_on_sharded_result": False,
    }
