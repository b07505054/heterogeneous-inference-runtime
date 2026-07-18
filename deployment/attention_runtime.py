"""Compiler-selected causal attention for real CPU model-forward integration.

The implementation uses PyTorch CPU matmul/exp primitives and persistent
affinity-pinned threads. It is an unfused native CPU execution path, not a
simulator, paged-attention implementation, or network-distributed backend.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from deployment.cpu_sharding import (
    PersistentCPUShardRuntime,
    ShardingPlanError,
    uneven_ranges,
)


ATTENTION_STRATEGIES = {"serial", "split_head", "split_query"}
ATTENTION_PHASES = {"prefill", "decode"}
ATTENTION_ALGORITHMS = {"dense_materialized", "fused_tiled_online_softmax"}
NATIVE_IMPLEMENTATIONS = {"native_scalar", "native_avx2"}


def make_attention_plan(
    *,
    phase: str,
    strategy: str = "serial",
    workers: int = 1,
    head_dim: int = 64,
    query_heads: int = 14,
    kv_heads: int = 2,
    provenance: str = "user_specified",
    algorithm: str = "dense_materialized",
    query_tile: int = 0,
    key_tile: int = 0,
    implementation: str = "torch_dense_materialized_v1",
    artifact_ref: str | None = None,
    artifact_sha256: str | None = None,
    abi_version: str | None = None,
    native_symbol: str | None = None,
) -> dict[str, Any]:
    split_dimension = {
        "serial": "none",
        "split_head": "query_head",
        "split_query": "query_token",
    }[strategy]
    if algorithm == "dense_materialized":
        kernel_id = f"torch_cpu_attention_fp32_{strategy}_w{workers}_v1"
        query_tile = key_tile = 0
    elif implementation in NATIVE_IMPLEMENTATIONS:
        kernel_id = (
            f"cpu_fused_online_{implementation}_fp32_q{query_tile}_k{key_tile}_"
            f"{strategy}_w{workers}_v1")
    else:
        kernel_id = (
            f"torch_cpu_fused_online_fp32_q{query_tile}_k{key_tile}_"
            f"{strategy}_w{workers}_v1")
    plan = {
        "decision_kind": "cpu_attention_exact_candidate_selection",
        "phase": phase,
        "selected_strategy": strategy,
        "worker_count": workers,
        "split_dimension": split_dimension,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": "float32",
        "qkv_layout": "batch_head_sequence_dim",
        "kv_cache_layout": "contiguous_layer_sequence_kv_head_dim",
        "causal": True,
        "assembly": "direct_disjoint_output",
        "native_kernel_id": kernel_id,
        "algorithm": algorithm,
        "implementation": implementation,
        "score_materialization": algorithm == "dense_materialized",
        "probability_materialization": algorithm == "dense_materialized",
        "online_softmax": algorithm == "fused_tiled_online_softmax",
        "query_tile": query_tile,
        "key_tile": key_tile,
        "causal_supported": True,
        "gqa_supported": True,
        "selection_provenance": provenance,
        "fallback": {"strategy": "serial", "worker_count": 1},
        "runtime_no_redecision": True,
    }
    if implementation in NATIVE_IMPLEMENTATIONS:
        plan.update({
            "artifact_ref": artifact_ref,
            "artifact_sha256": artifact_sha256,
            "abi_version": abi_version,
            "native_symbol": native_symbol,
            "target_isa": "scalar_fp32" if implementation == "native_scalar" else "x86_avx2_fma",
        })
    validate_attention_plan(plan)
    return plan


def validate_attention_plan(plan: dict[str, Any]) -> None:
    if plan.get("decision_kind") != "cpu_attention_exact_candidate_selection":
        raise ShardingPlanError("attention decision_kind mismatch")
    if plan.get("phase") not in ATTENTION_PHASES:
        raise ShardingPlanError("attention phase must be prefill or decode")
    strategy = plan.get("selected_strategy")
    if strategy not in ATTENTION_STRATEGIES:
        raise ShardingPlanError("unsupported attention strategy")
    workers = plan.get("worker_count")
    if workers not in {1, 2, 4, 8}:
        raise ShardingPlanError("attention worker_count must be 1, 2, 4, or 8")
    if strategy == "serial" and workers != 1:
        raise ShardingPlanError("serial attention requires one worker")
    if strategy == "split_query" and plan.get("phase") == "decode":
        raise ShardingPlanError("split_query is illegal for one-token decode")
    qh, kvh, dim = (plan.get("query_heads"), plan.get("kv_heads"),
                    plan.get("head_dim"))
    if not all(isinstance(v, int) and v > 0 for v in (qh, kvh, dim)):
        raise ShardingPlanError("positive static attention head counts/dimension required")
    if qh % kvh:
        raise ShardingPlanError("query heads must be divisible by KV heads")
    if plan.get("dtype") != "float32":
        raise ShardingPlanError("first attention path supports float32 only")
    if plan.get("qkv_layout") != "batch_head_sequence_dim":
        raise ShardingPlanError("unsupported Q/K/V layout")
    if plan.get("kv_cache_layout") != "contiguous_layer_sequence_kv_head_dim":
        raise ShardingPlanError("unsupported KV-cache layout")
    if plan.get("causal") is not True:
        raise ShardingPlanError("first attention path requires causal mode")
    if plan.get("runtime_no_redecision") is not True:
        raise ShardingPlanError("runtime_no_redecision must be true")
    algorithm = plan.get("algorithm", "dense_materialized")
    if algorithm not in ATTENTION_ALGORITHMS:
        raise ShardingPlanError("unsupported attention algorithm")
    if algorithm == "fused_tiled_online_softmax":
        if strategy == "split_query":
            raise ShardingPlanError("first fused path does not support split_query")
        if not isinstance(plan.get("query_tile"), int) or plan["query_tile"] <= 0:
            raise ShardingPlanError("fused attention query_tile must be positive")
        if not isinstance(plan.get("key_tile"), int) or plan["key_tile"] <= 0:
            raise ShardingPlanError("fused attention key_tile must be positive")
        if plan.get("score_materialization") is not False:
            raise ShardingPlanError("fused attention must not materialize full scores")
        if plan.get("probability_materialization") is not False:
            raise ShardingPlanError("fused attention must not materialize full probabilities")
        if plan.get("online_softmax") is not True:
            raise ShardingPlanError("fused attention requires online_softmax=true")
        implementation = plan.get("implementation")
        if implementation in NATIVE_IMPLEMENTATIONS:
            from deployment.native_fused_attention import ABI_VERSION, SYMBOLS
            if plan.get("abi_version") != ABI_VERSION:
                raise ShardingPlanError("native fused ABI version mismatch")
            if plan.get("native_symbol") != SYMBOLS[implementation]:
                raise ShardingPlanError("native fused symbol/implementation mismatch")
            artifact = plan.get("artifact_ref")
            digest = plan.get("artifact_sha256")
            if not isinstance(artifact, str) or not artifact:
                raise ShardingPlanError("native fused artifact_ref required")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ShardingPlanError("native fused artifact_sha256 required")
    if plan.get("selection_mode") in {"compiler_selected", "forced_test_override"}:
        required = (
            "operator_kind", "selector_version", "target_profile",
            "workload_signature", "workload_domain", "selection_trace",
            "native_kernel_id", "selection_provenance",
        )
        missing = [key for key in required if not plan.get(key)]
        if missing:
            raise ShardingPlanError(
                "compiler-selected attention plan is missing required field(s): "
                + ", ".join(missing))
        if plan["operator_kind"] != "attention":
            raise ShardingPlanError("compiler-selected operator_kind must be attention")
        domain = plan["workload_domain"]
        if not isinstance(domain, dict):
            raise ShardingPlanError("attention workload_domain must be an object")
        for name in ("batch", "query_length_min", "query_length_max",
                     "context_length_min", "context_length_max"):
            if not isinstance(domain.get(name), int) or domain[name] <= 0:
                raise ShardingPlanError(f"invalid attention workload domain field: {name}")
        if (domain["query_length_max"] < domain["query_length_min"]
                or domain["context_length_max"] < domain["context_length_min"]):
            raise ShardingPlanError("attention workload domain range is inverted")
        if algorithm == "dense_materialized":
            expected_id = f"torch_cpu_attention_fp32_{strategy}_w{workers}_v1"
        elif plan.get("implementation") in NATIVE_IMPLEMENTATIONS:
            expected_id = (
                f"cpu_fused_online_{plan['implementation']}_fp32_"
                f"q{plan['query_tile']}_k{plan['key_tile']}_{strategy}_w{workers}_v1")
        else:
            expected_id = (
                f"torch_cpu_fused_online_fp32_q{plan['query_tile']}_k{plan['key_tile']}_"
                f"{strategy}_w{workers}_v1")
        if plan["native_kernel_id"] != expected_id:
            raise ShardingPlanError("native_kernel_id does not match strategy/worker_count")
        expected_split = {"serial": "none", "split_head": "query_head",
                          "split_query": "query_token"}[strategy]
        if plan.get("split_dimension") != expected_split:
            raise ShardingPlanError("strategy/split_dimension mismatch")


def validate_attention_execution(decision: dict[str, Any]) -> None:
    if decision.get("decision_kind") != "cpu_attention_plan_table_v1":
        validate_attention_plan(decision)
        return
    if decision.get("operator_kind") != "attention":
        raise ShardingPlanError("attention plan table operator_kind mismatch")
    if decision.get("selection_mode") not in {
            "compiler_selected", "forced_test_override"}:
        raise ShardingPlanError(
            "attention plan table selection_mode must be compiler_selected or forced_test_override")
    if decision.get("runtime_no_redecision") is not True:
        raise ShardingPlanError("attention plan table requires runtime_no_redecision")
    phases = decision.get("phase_decisions")
    if not isinstance(phases, dict) or set(phases) != {"prefill", "decode"}:
        raise ShardingPlanError("attention plan table requires prefill and decode decisions")
    for phase, plan in phases.items():
        validate_attention_plan(plan)
        if plan["phase"] != phase:
            raise ShardingPlanError("attention phase table key/decision mismatch")
    fallback = decision.get("fallback")
    if fallback != {"policy": "hard_failure", "count": 0}:
        raise ShardingPlanError("Level 5 attention plan requires zero-count hard failure fallback")


def legal_attention_candidates(
    *, phase: str, batch: int, query_len: int, context_len: int,
    query_heads: int, kv_heads: int, head_dim: int, dtype: str = "float32",
) -> list[dict[str, Any]]:
    del batch, context_len
    candidates = []
    for strategy in ATTENTION_STRATEGIES:
        for workers in ((1,) if strategy == "serial" else (2, 4, 8)):
            if dtype != "float32" or head_dim != 64 or query_heads % kv_heads:
                continue
            if strategy == "split_head" and workers > query_heads:
                continue
            if strategy == "split_query" and (
                phase == "decode" or query_len <= 1 or workers > query_len
            ):
                continue
            candidates.append(make_attention_plan(
                phase=phase, strategy=strategy, workers=workers,
                head_dim=head_dim, query_heads=query_heads, kv_heads=kv_heads,
                provenance="cost_model_selected"))
    return candidates


def select_attention_plan(
    *, phase: str, batch: int, query_len: int, context_len: int,
    query_heads: int = 14, kv_heads: int = 2, head_dim: int = 64,
) -> dict[str, Any]:
    """Opt-in static selector calibrated for this 4-physical-core host."""
    candidates = legal_attention_candidates(
        phase=phase, batch=batch, query_len=query_len, context_len=context_len,
        query_heads=query_heads, kv_heads=kv_heads, head_dim=head_dim)
    if phase == "decode" or batch * query_len < 32:
        wanted = ("serial", 1)
    elif query_len >= 128:
        wanted = ("split_query", 4)
    else:
        wanted = ("split_head", 2)
    return next((p for p in candidates
                 if (p["selected_strategy"], p["worker_count"]) == wanted),
                make_attention_plan(phase=phase))


@dataclass
class AttentionTiming:
    total_ms: float
    dispatch_ms: float
    qk_ms: float
    softmax_ms: float
    pv_ms: float
    assembly_ms: float


@dataclass
class AttentionTrace:
    selected_candidate_id: str
    executed_candidate_id: str
    phase: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_sum: float
    invocation: int
    produced_returned_output: bool = True
    timing: AttentionTiming | None = None
    memory: dict[str, Any] | None = None


def _repeat_gqa(x: torch.Tensor, query_heads: int) -> torch.Tensor:
    repeats = query_heads // x.shape[1]
    return x.repeat_interleave(repeats, dim=1)


def _attention_chunk(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor | None, scale: float,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    t0 = time.perf_counter_ns()
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores.mul_(scale)
    if mask is not None:
        scores.add_(mask)
    t1 = time.perf_counter_ns()
    row_max = scores.amax(dim=-1, keepdim=True)
    probabilities = torch.exp(scores - row_max)
    probabilities.div_(probabilities.sum(dim=-1, keepdim=True))
    t2 = time.perf_counter_ns()
    output = torch.matmul(probabilities, v)
    t3 = time.perf_counter_ns()
    elements = scores.numel()
    memory = {
        "algorithm": "dense_materialized",
        "full_score_materialized": True,
        "full_probability_materialized": True,
        "score_bytes": elements * scores.element_size(),
        "probability_bytes": elements * probabilities.element_size(),
        "temporary_bytes": 2 * elements * scores.element_size(),
        "largest_temporary_allocation_bytes": elements * scores.element_size(),
        "tracked_temporary_allocations": 2,
    }
    return (output, ((t1 - t0) / 1e6, (t2 - t1) / 1e6,
                     (t3 - t2) / 1e6), memory)


def _fused_online_attention_chunk(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor | None, scale: float, query_tile: int, key_tile: int,
) -> tuple[torch.Tensor, tuple[float, float, float], dict[str, Any]]:
    """Tiled exact attention with the stable online-softmax recurrence.

    When a new tile raises the running maximum from m_old to m_new, all prior
    exponentials were expressed relative to m_old. Both the old denominator
    and output accumulator must therefore be rescaled by
    exp(m_old - m_new) before adding the new tile.
    """
    batch, heads, query_len, dim = q.shape
    context = k.shape[2]
    output = torch.empty_like(q)
    qk_ns = softmax_ns = pv_ns = 0
    largest = 0
    allocation_count = 0
    for q_start in range(0, query_len, query_tile):
        q_end = min(q_start + query_tile, query_len)
        q_block = q[:, :, q_start:q_end]
        rows = q_end - q_start
        running_max = torch.full(
            (batch, heads, rows, 1), -torch.inf, dtype=q.dtype, device=q.device)
        denominator = torch.zeros(
            (batch, heads, rows, 1), dtype=q.dtype, device=q.device)
        accumulator = torch.zeros(
            (batch, heads, rows, dim), dtype=q.dtype, device=q.device)
        allocation_count += 3
        largest = max(largest, accumulator.numel() * accumulator.element_size())
        for k_start in range(0, context, key_tile):
            k_end = min(k_start + key_tile, context)
            t0 = time.perf_counter_ns()
            tile_scores = torch.matmul(
                q_block, k[:, :, k_start:k_end].transpose(-2, -1))
            tile_scores.mul_(scale)
            if mask is not None:
                tile_scores.add_(mask[..., q_start:q_end, k_start:k_end])
            t1 = time.perf_counter_ns()
            tile_max = tile_scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, tile_max)
            finite_new = torch.isfinite(new_max)
            alpha = torch.where(
                torch.isfinite(running_max) & finite_new,
                torch.exp(running_max - new_max), torch.zeros_like(new_max))
            shifted = torch.where(
                torch.isfinite(tile_scores) & finite_new,
                tile_scores - new_max, torch.full_like(tile_scores, -torch.inf))
            tile_probabilities = torch.exp(shifted)
            new_denominator = (
                alpha * denominator
                + tile_probabilities.sum(dim=-1, keepdim=True))
            t2 = time.perf_counter_ns()
            accumulator = (
                alpha * accumulator
                + torch.matmul(tile_probabilities, v[:, :, k_start:k_end]))
            running_max = new_max
            denominator = new_denominator
            t3 = time.perf_counter_ns()
            qk_ns += t1 - t0
            softmax_ns += t2 - t1
            pv_ns += t3 - t2
            tile_bytes = tile_scores.numel() * tile_scores.element_size()
            probability_bytes = (
                tile_probabilities.numel() * tile_probabilities.element_size())
            largest = max(largest, tile_bytes, probability_bytes)
            allocation_count += 2
        output[:, :, q_start:q_end] = (
            accumulator / denominator.clamp_min(torch.finfo(q.dtype).tiny))
    running_bytes = batch * heads * min(query_tile, query_len) * (
        2 * q.element_size() + dim * q.element_size())
    tile_elements = batch * heads * min(query_tile, query_len) * min(key_tile, context)
    tile_pair_bytes = 2 * tile_elements * q.element_size()
    memory = {
        "algorithm": "fused_tiled_online_softmax",
        "full_score_materialized": False,
        "full_probability_materialized": False,
        "score_bytes": 0,
        "probability_bytes": 0,
        "tile_score_bytes": tile_elements * q.element_size(),
        "tile_probability_bytes": tile_elements * q.element_size(),
        "running_state_bytes": running_bytes,
        "temporary_bytes": running_bytes + tile_pair_bytes,
        "largest_temporary_allocation_bytes": largest,
        "tracked_temporary_allocations": allocation_count,
    }
    return output, (qk_ns / 1e6, softmax_ns / 1e6, pv_ns / 1e6), memory


class CompilerAttentionRuntime:
    def __init__(self, plan: dict[str, Any], perturbation: float = 0.0):
        validate_attention_plan(plan)
        self.plan = plan
        self.perturbation = perturbation
        workers = plan["worker_count"]
        cpu_plan = {
            "mesh": {"name": "cpu_mesh", "axes": [{"name": "cpu", "size": workers}]},
            "operator_sharding": {
                "strategy": "split_m", "tensor_dimension": 0, "mesh_axis": "cpu",
                "uneven_policy": "balanced_remainder", "provenance": "user_specified",
            },
            "collectives": [],
            "rank_mapping": {"policy": "pinned_logical_cpu",
                             "logical_cpu_ids": list(range(workers))},
            "fallback": {"strategy": "replicated"},
        }
        self.workers = PersistentCPUShardRuntime(cpu_plan)
        self.native_library = None
        if plan.get("implementation") in NATIVE_IMPLEMENTATIONS:
            from deployment.native_fused_attention import NativeFusedAttentionLibrary
            self.native_library = NativeFusedAttentionLibrary(
                plan["artifact_ref"], plan["artifact_sha256"], plan["abi_version"])
            if plan["implementation"] == "native_avx2" and not self.native_library.has_avx2:
                raise ShardingPlanError("selected native AVX2 artifact is unavailable on host")
        self.traces: list[AttentionTrace] = []
        self._lock = threading.Lock()

    def close(self) -> None:
        self.workers.close()

    def __enter__(self) -> "CompilerAttentionRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def attention(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attention_mask: torch.Tensor | None, scale: float,
    ) -> torch.Tensor:
        if any(t.dtype != torch.float32 for t in (query, key, value)):
            raise ShardingPlanError("compiler attention runtime supports FP32 only")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ShardingPlanError("Q/K/V must have rank 4 [B,H,S,D]")
        b, hq, qlen, dim = query.shape
        bk, hkv, context, dk = key.shape
        if (bk != b or value.shape != (b, hkv, context, dim) or dk != dim
                or dim != self.plan["head_dim"] or hq != self.plan["query_heads"]
                or hkv != self.plan["kv_heads"] or hq % hkv):
            raise ShardingPlanError("Q/K/V shape or GQA contract mismatch")
        actual_phase = "decode" if qlen == 1 else "prefill"
        if actual_phase != self.plan["phase"]:
            raise ShardingPlanError("runtime phase differs from selected plan")
        strategy, n = self.plan["selected_strategy"], self.plan["worker_count"]
        if strategy == "split_query" and qlen == 1:
            raise ShardingPlanError("split_query cannot execute decode")
        native = self.plan.get("implementation") in NATIVE_IMPLEMENTATIONS
        k = key if native else _repeat_gqa(key, hq)
        v = value if native else _repeat_gqa(value, hq)
        t0 = time.perf_counter_ns()
        futures = []
        algorithm = self.plan.get("algorithm", "dense_materialized")
        kernel = _attention_chunk if algorithm == "dense_materialized" else None
        def native_call(q_part, k_part, v_part, _mask, head_offset=0):
            begin = time.perf_counter_ns()
            out, memory = self.native_library.run(
                self.plan["implementation"], q_part, k_part, v_part, scale,
                self.plan["query_tile"], self.plan["key_tile"],
                query_head_offset=head_offset, total_query_heads=hq)
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            return out, (elapsed, 0.0, 0.0), memory
        def submit_args(q_part, k_part, v_part, mask_part):
            if native:
                return (q_part, k_part, v_part, mask_part)
            if kernel is not None:
                return (q_part, k_part, v_part, mask_part, scale)
            return (q_part, k_part, v_part, mask_part, scale,
                    self.plan["query_tile"], self.plan["key_tile"])
        implementation = native_call if native else (kernel or _fused_online_attention_chunk)
        if strategy == "serial":
            out, stages, memory = implementation(
                *submit_args(query, k, v, attention_mask))
            dispatched = t0
            parts = [(0, hq, out, stages, memory)]
        elif strategy == "split_head":
            for start, end in uneven_ranges(hq, n):
                m = attention_mask
                if m is not None and m.shape[1] != 1:
                    m = m[:, start:end]
                if native:
                    future = self.workers.pool.submit(
                        implementation, query[:, start:end], k, v, m, start)
                else:
                    future = self.workers.pool.submit(
                        implementation, *submit_args(
                            query[:, start:end], k[:, start:end],
                            v[:, start:end], m))
                futures.append((start, end, future))
            dispatched = time.perf_counter_ns()
            parts = [(s, e, *f.result()) for s, e, f in futures]
        else:
            for start, end in uneven_ranges(qlen, n):
                m = attention_mask[..., start:end, :] if attention_mask is not None else None
                futures.append((start, end, self.workers.pool.submit(
                    implementation, *submit_args(
                        query[:, :, start:end], k, v, m))))
            dispatched = time.perf_counter_ns()
            parts = [(s, e, *f.result()) for s, e, f in futures]
        a0 = time.perf_counter_ns()
        if strategy == "split_head":
            output = torch.cat([p[2] for p in parts], dim=1)
        elif strategy == "split_query":
            output = torch.cat([p[2] for p in parts], dim=2)
        else:
            output = parts[0][2]
        if self.perturbation:
            # Test-only causal-dependency injection, applied before o_proj.
            output = output + self.perturbation
        a1 = time.perf_counter_ns()
        stage_times = [p[3] for p in parts]
        memory_rows = [p[4] for p in parts]
        memory = dict(memory_rows[0])
        if len(memory_rows) > 1:
            for name in ("score_bytes", "probability_bytes", "temporary_bytes",
                         "tracked_temporary_allocations"):
                memory[name] = sum(row.get(name, 0) for row in memory_rows)
            memory["largest_temporary_allocation_bytes"] = max(
                row["largest_temporary_allocation_bytes"] for row in memory_rows)
        memory["gqa_expansion_bytes"] = 0 if native else (
            k.numel() * k.element_size() + v.numel() * v.element_size())
        memory["total_temporary_bytes_including_gqa"] = (
            memory["temporary_bytes"] + memory["gqa_expansion_bytes"])
        timing = AttentionTiming(
            total_ms=(a1 - t0) / 1e6,
            dispatch_ms=(dispatched - t0) / 1e6,
            qk_ms=max(x[0] for x in stage_times),
            softmax_ms=max(x[1] for x in stage_times),
            pv_ms=max(x[2] for x in stage_times),
            assembly_ms=(a1 - a0) / 1e6,
        )
        with self._lock:
            self.traces.append(AttentionTrace(
                selected_candidate_id=self.plan["native_kernel_id"],
                executed_candidate_id=self.plan["native_kernel_id"],
                phase=actual_phase, input_shape=tuple(query.shape),
                output_shape=tuple(output.shape),
                output_sum=float(output.double().sum()),
                invocation=len(self.traces) + 1, timing=timing, memory=memory))
        return output


class ExecutionPlanAttentionAdapter:
    """Consumes only a deserialized ExecutionPlan attention decision table."""

    def __init__(self, execution_plan: Any, *, perturbation: float = 0.0):
        table = execution_plan.global_decisions.attention_execution
        validate_attention_execution(table)
        if table.get("decision_kind") != "cpu_attention_plan_table_v1":
            raise ShardingPlanError("model adapter requires compiler attention plan table")
        self.plan_id = execution_plan.plan_id
        self.table = table
        self.perturbation = perturbation
        self.runtimes = {
            phase: CompilerAttentionRuntime(plan)
            for phase, plan in table["phase_decisions"].items()
        }
        self.provenance: list[dict[str, Any]] = []
        self.fallback_count = 0
        self.mismatch_count = 0

    def close(self) -> None:
        for runtime in self.runtimes.values():
            runtime.close()

    def attention(
        self, module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
        value: torch.Tensor, attention_mask: torch.Tensor | None, scale: float,
    ) -> torch.Tensor:
        from deployment.attention_planner import AttentionWorkload
        actual = AttentionWorkload.from_tensors(query, key, value)
        plan = self.table["phase_decisions"].get(actual.phase)
        if plan is None:
            raise ShardingPlanError(f"no compiler attention decision for {actual.phase}")
        domain = plan["workload_domain"]
        if (actual.batch != domain["batch"]
                or not domain["query_length_min"] <= actual.query_len <= domain["query_length_max"]
                or not domain["context_length_min"] <= actual.context_len <= domain["context_length_max"]
                or actual.query_heads != plan["query_heads"]
                or actual.kv_heads != plan["kv_heads"]
                or actual.head_dim != plan["head_dim"]):
            raise ShardingPlanError(
                f"actual workload {actual.signature} does not match compiler plan domain")
        runtime = self.runtimes[actual.phase]
        runtime.perturbation = self.perturbation
        output = runtime.attention(query, key, value, attention_mask, scale)
        trace = runtime.traces[-1]
        mismatch = (
            trace.selected_candidate_id != trace.executed_candidate_id
            or not trace.memory
            or trace.memory["algorithm"] != plan.get("algorithm", "dense_materialized"))
        self.mismatch_count += int(mismatch)
        if mismatch:
            raise ShardingPlanError("compiler-selected and executed attention candidates differ")
        layer = int(getattr(module, "layer_idx", -1))
        self.provenance.append({
            "plan_id": self.plan_id,
            "operator_id": f"model.layers.{layer}.self_attn",
            "layer_index": layer,
            "phase": actual.phase,
            "decode_step": None if actual.phase == "prefill" else actual.context_len - domain["context_length_min"],
            "workload_signature": actual.signature,
            "selector_version": plan["selector_version"],
            "selection_mode": plan["selection_mode"],
            "compiler_selected_candidate_id": plan["native_kernel_id"],
            "serialized_candidate_id": plan["native_kernel_id"],
            "executed_candidate_id": trace.executed_candidate_id,
            "worker_count": plan["worker_count"],
            "algorithm": plan.get("algorithm", "dense_materialized"),
            "implementation": plan.get("implementation"),
            "serialized_implementation": plan.get("implementation"),
            "executed_implementation": (
                trace.memory.get("implementation") if trace.memory else
                plan.get("implementation")),
            "native_symbol": trace.memory.get("native_symbol") if trace.memory else None,
            "query_tile": plan.get("query_tile", 0),
            "key_tile": plan.get("key_tile", 0),
            "serialized_algorithm": plan.get("algorithm", "dense_materialized"),
            "executed_algorithm": trace.memory["algorithm"] if trace.memory else None,
            "fallback": False,
            "candidate_mismatch": mismatch,
            "output_sum": trace.output_sum,
            "attention_total_ms": trace.timing.total_ms if trace.timing else None,
            "dispatch_ms": trace.timing.dispatch_ms if trace.timing else None,
            "qk_ms": trace.timing.qk_ms if trace.timing else None,
            "softmax_ms": trace.timing.softmax_ms if trace.timing else None,
            "pv_ms": trace.timing.pv_ms if trace.timing else None,
            "assembly_ms": trace.timing.assembly_ms if trace.timing else None,
            "memory": trace.memory,
        })
        return output


@dataclass
class ContiguousKVCache:
    layers: int
    sequences: int
    kv_heads: int
    capacity: int
    head_dim: int
    keys: torch.Tensor = field(init=False)
    values: torch.Tensor = field(init=False)
    lengths: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.layers, self.sequences, self.kv_heads,
                 self.capacity, self.head_dim)
        self.keys = torch.empty(shape, dtype=torch.float32)
        self.values = torch.empty(shape, dtype=torch.float32)
        self.lengths = torch.zeros((self.layers, self.sequences), dtype=torch.int64)

    def append(self, layer: int, sequence: int, key: torch.Tensor,
               value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ShardingPlanError("cache append expects [1,KVH,tokens,D]")
        tokens = key.shape[2]
        start = int(self.lengths[layer, sequence])
        end = start + tokens
        if end > self.capacity:
            raise ShardingPlanError("contiguous KV cache capacity exceeded")
        self.keys[layer, sequence, :, start:end].copy_(key[0])
        self.values[layer, sequence, :, start:end].copy_(value[0])
        self.lengths[layer, sequence] = end
        return (self.keys[layer, sequence:sequence + 1, :, :end],
                self.values[layer, sequence:sequence + 1, :, :end])


_ACTIVE_RUNTIME: CompilerAttentionRuntime | None = None
_ACTIVE_PLAN_ADAPTER: ExecutionPlanAttentionAdapter | None = None


def set_active_attention_runtime(runtime: CompilerAttentionRuntime | None) -> None:
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = runtime


def set_active_attention_plan_adapter(
    adapter: ExecutionPlanAttentionAdapter | None,
) -> None:
    global _ACTIVE_PLAN_ADAPTER
    _ACTIVE_PLAN_ADAPTER = adapter


def compiler_attention_interface(
    module: torch.nn.Module, query: torch.Tensor, key: torch.Tensor,
    value: torch.Tensor, attention_mask: torch.Tensor | None,
    scaling: float, dropout: float = 0.0, **_: Any,
) -> tuple[torch.Tensor, None]:
    if dropout:
        raise ShardingPlanError("inference attention requires dropout=0")
    if _ACTIVE_PLAN_ADAPTER is not None:
        output = _ACTIVE_PLAN_ADAPTER.attention(
            module, query, key, value, attention_mask, scaling)
    elif _ACTIVE_RUNTIME is not None:
        output = _ACTIVE_RUNTIME.attention(query, key, value, attention_mask, scaling)
    else:
        raise ShardingPlanError("compiler attention interface has no active runtime")
    returned = output.transpose(1, 2).contiguous()
    if _ACTIVE_PLAN_ADAPTER is not None:
        _ACTIVE_PLAN_ADAPTER.provenance[-1].update({
            "returned_tensor_data_ptr": returned.data_ptr(),
            "returned_tensor_sum": float(returned.double().sum()),
        })
    return returned, None


def register_transformers_attention_interface() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        eager_mask,
    )
    ALL_ATTENTION_FUNCTIONS.register(
        "compiler_cpu_attention", compiler_attention_interface)
    # Custom attention implementations are otherwise assumed to implement
    # causality internally and may receive no additive mask. This unfused
    # backend consumes the same explicit mask as Transformers eager attention.
    ALL_MASK_ATTENTION_FUNCTIONS.register("compiler_cpu_attention", eager_mask)
