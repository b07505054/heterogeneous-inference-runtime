#!/usr/bin/env python3
"""Raspberry Pi benchmark: native contiguous KV vs compiler-directed paged KV."""
from __future__ import annotations

import argparse
from array import array
import ctypes
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.execution_plan.contiguous_kv_cache import ContiguousKVAttentionSession
from deployment.execution_plan.kv_page_manager import KVPageManager
from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.paged_kv_runtime import (
    PagedKVRuntimeContext,
    build_paged_kv_runtime,
    paged_kv_contracts,
)
from deployment.serving_scheduler import (
    KV_ADMITTED,
    KV_DEFERRED_PRESSURE,
    KV_REJECTED_PROMPT_TOO_LARGE,
    ReplicaSchedulerState,
    RequestExecutionState,
    SchedulerCompiler,
    SchedulerProfile,
)


TOKEN_COUNTS = (1, 7, 8, 9, 16, 32, 64)
OCCUPANCY_COUNTS = (1, 4, 8, 9, 15, 64)
NATIVE_FLAGS = ("-O3", "-std=c++17", "-fPIC", "-shared")
FP32_ABS_TOL = 1e-5
MICRO_SAMPLES = 50
MICRO_WARMUPS = 10
LIFECYCLE_SAMPLES = 10
LIFECYCLE_WARMUPS = 3
DECODE_INNER = 50
ROUND_ORDERS = (
    ("contiguous", "paged"),
    ("paged", "contiguous"),
    ("contiguous", "paged"),
    ("paged", "contiguous"),
    ("contiguous", "paged"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_data(n: int, seed: int) -> array:
    return array("f", (((i * 1103515245 + seed * 12345) & 0xFFFF) / 32768 - 1 for i in range(n)))


def stat(samples: list[float], *, warmups: int) -> dict[str, float | int | bool]:
    if not samples:
        raise ValueError("insufficient samples")
    values = sorted(samples)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(x - median) for x in values])
    p95 = values[math.ceil(0.95 * len(values)) - 1]
    cv = sd / mean if mean else 0.0
    return {
        "minimum_ms": values[0],
        "median_ms": median,
        "mean_ms": mean,
        "p95_ms": p95,
        "mad_ms": mad,
        "stddev_ms": sd,
        "coefficient_of_variation": cv,
        "samples": len(values),
        "warmups": warmups,
        "unstable": bool(cv > 0.25 or p95 > median * 2.0),
        "stability_rule": "stable iff coefficient_of_variation <= 0.25 and p95 <= 2.0 * median",
    }


def timer_overhead(samples: int = 1000) -> dict[str, Any]:
    values = []
    for _ in range(samples):
        start = time.perf_counter()
        end = time.perf_counter()
        values.append((end - start) * 1000.0)
    return {"summary": stat(values, warmups=0), "raw_ms": values}


def timed(fn: Callable[[], Any], *, samples: int, warmups: int, inner: int = 1) -> dict[str, Any]:
    measured: list[float] = []
    for i in range(warmups + samples):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        elapsed = (time.perf_counter() - start) * 1000.0 / inner
        if i >= warmups:
            measured.append(elapsed)
    return {"summary": stat(measured, warmups=warmups), "raw_ms": measured, "inner_iterations": inner}


def timed_with_setup(
    setup: Callable[[], Any],
    op: Callable[[Any], Any],
    cleanup: Callable[[Any], None] | None = None,
    *,
    samples: int,
    warmups: int,
) -> dict[str, Any]:
    measured: list[float] = []
    for i in range(warmups + samples):
        obj = setup()
        start = time.perf_counter()
        op(obj)
        elapsed = (time.perf_counter() - start) * 1000.0
        if cleanup is not None:
            cleanup(obj)
        if i >= warmups:
            measured.append(elapsed)
    return {"summary": stat(measured, warmups=warmups), "raw_ms": measured, "inner_iterations": 1}


def compare(got: array | list[float], ref: list[float]) -> dict[str, float | bool]:
    errors = [abs(float(x) - y) for x, y in zip(got, ref)]
    l2_ref = math.sqrt(sum(x * x for x in ref))
    l2_err = math.sqrt(sum(x * x for x in errors))
    max_abs = max(errors) if errors else math.inf
    return {
        "max_abs_error": max_abs,
        "relative_l2_error": l2_err / max(l2_ref, 1e-30),
        "passed": max_abs <= FP32_ABS_TOL,
    }


def reference_decode(q: array, k: list[float], v: list[float], heads: int, tokens: int, dim: int) -> list[float]:
    out: list[float] = []
    scale = 1.0 / math.sqrt(dim)
    for hi in range(heads):
        scores = [
            sum(q[hi * dim + x] * k[(hi * tokens + t) * dim + x] for x in range(dim)) * scale
            for t in range(tokens)
        ]
        peak = max(scores)
        weights = [math.exp(x - peak) for x in scores]
        denom = sum(weights)
        out.extend(
            sum(weights[t] / denom * v[(hi * tokens + t) * dim + x] for t in range(tokens))
            for x in range(dim)
        )
    return out


def paged_logical(session) -> tuple[list[float], list[float]]:
    c = session.c
    heads, dim, page_tokens = c["num_kv_heads"], c["head_dim"], c["page_tokens"]
    pages = session.page_manager.block_table(session.request_id)
    k: list[float] = []
    v: list[float] = []
    for hi in range(heads):
        for token in range(session.valid_tokens):
            page = pages[token // page_tokens]
            base = ((page * heads + hi) * page_tokens + token % page_tokens) * dim
            k.extend(session.k[base : base + dim])
            v.extend(session.v[base : base + dim])
    return k, v


def rss_kib() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def memory_payload(*, tokens: int, bytes_per_token: int, capacity_tokens: int, bytes_per_page: int, page_tokens: int) -> dict[str, Any]:
    useful = tokens * bytes_per_token
    paged_pages = (tokens + page_tokens - 1) // page_tokens
    paged_alloc = paged_pages * bytes_per_page
    contiguous_alloc = capacity_tokens * bytes_per_token
    return {
        "useful_kv_bytes": useful,
        "paged": {
            "allocated_pages": paged_pages,
            "allocated_kv_bytes": paged_alloc,
            "fragmentation_bytes": paged_alloc - useful,
            "utilization_percent": useful / paged_alloc * 100 if paged_alloc else 0.0,
            "block_table_bytes": paged_pages * 4,
        },
        "contiguous": {
            "allocated_kv_bytes": contiguous_alloc,
            "fragmentation_bytes": contiguous_alloc - useful,
            "utilization_percent": useful / contiguous_alloc * 100 if contiguous_alloc else 0.0,
            "block_table_bytes": 0,
        },
    }


def label_ratio(baseline: float, candidate: float) -> dict[str, float | str]:
    if baseline <= 0 or candidate <= 0:
        raise ValueError("latencies must be positive")
    speedup = baseline / candidate
    overhead = (candidate / baseline - 1.0) * 100.0
    if abs(overhead) < 5.0:
        label = "statistically_indistinguishable"
    elif overhead > 0:
        label = "overhead"
    else:
        label = "speedup"
    return {
        "label": label,
        "speedup": speedup,
        "overhead_percent": overhead,
        "baseline_ms": baseline,
        "candidate_ms": candidate,
    }


def stage_breakdown(total_ms: float, stage_ms: dict[str, float]) -> dict[str, Any]:
    if total_ms <= 0:
        raise ValueError("total_ms must be positive")
    if any(value < 0 for value in stage_ms.values()):
        raise ValueError("stage timings must be non-negative")
    measured = sum(stage_ms.values())
    remainder = max(0.0, total_ms - measured)
    rows = {
        name: {"ms": value, "percent_of_total": value / total_ms * 100.0}
        for name, value in sorted(stage_ms.items())
    }
    rows["unclassified_remainder"] = {
        "ms": remainder,
        "percent_of_total": remainder / total_ms * 100.0,
    }
    return {
        "total_ms": total_ms,
        "stages": rows,
        "measured_stage_ms": measured,
        "unclassified_remainder_ms": remainder,
        "percent_sum": sum(item["percent_of_total"] for item in rows.values()),
    }


def aggregate_unstable_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("timing", {}).get("summary", {}).get("unstable") is True)


def contiguous_contract(contract, artifact_ref: str, artifact_sha: str, *, prompt: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    heads, dim, capacity = contract.num_kv_heads, contract.head_dim, contract.maximum_logical_tokens
    strides = [heads * capacity * dim, capacity * dim, dim, 1]
    kv = {
        "kv_execution_unit": "portable_cpu_contiguous_kv",
        "kv_candidate_id": "cpu_contiguous_kv_fp32_v1",
        "kv_cache_id": "benchmark_contiguous",
        "kv_artifact_ref": artifact_ref,
        "kv_artifact_sha256": artifact_sha,
        "kv_artifact_version": "hir.contiguous_kv.v1",
        "kv_dtype": "fp32",
        "kv_layout": "bhcd_contiguous",
        "batch": 1,
        "num_kv_heads": heads,
        "head_dim": dim,
        "capacity_tokens": capacity,
        "initial_valid_tokens": 0,
        "bytes_per_token": contract.bytes_per_token,
        "k_cache_bytes": heads * capacity * dim * 4,
        "v_cache_bytes": heads * capacity * dim * 4,
        "total_cache_bytes": capacity * contract.bytes_per_token,
        "alignment_bytes": contract.alignment_bytes,
        "k_strides": strides,
        "v_strides": strides,
        "create_entry_point": "hir_contiguous_kv_initialize",
        "prefill_write_entry_point": "hir_contiguous_kv_prefill_write",
        "decode_append_entry_point": "hir_contiguous_kv_append",
        "view_binding": "direct_contiguous_pointer_valid_prefix",
        "reset_entry_point": "hir_contiguous_kv_reset",
        "compatible_prefill_kernel_id": "cpu_attention_prefill_fp32",
        "compatible_decode_kernel_id": "cpu_attention_decode_fp32",
        "attention_entry_point": "hir_cpu_attention_decode_contiguous_kv_fp32",
        "implementation_strategy": "dimension_major_strided_v_accumulation",
        "measurement_provenance": "same_native_artifact_control_baseline",
        "runtime_no_layout_redecision": True,
    }
    base = {"dtype": "fp32", "input_layout": "bhsd_contiguous", "runtime_no_redecision": True}
    attention = {
        "prefill": {**base, "kernel_id": "cpu_attention_prefill_fp32", "entry_point": "hir_cpu_attention_prefill_fp32", "implementation_strategy": "prefill_contiguous", "query_length": prompt},
        "decode": {**base, "kernel_id": "cpu_attention_decode_fp32", "entry_point": "hir_cpu_attention_decode_contiguous_kv_fp32", "implementation_strategy": "dimension_major_strided_v_accumulation", "query_length": 1},
    }
    return kv, attention


def make_contiguous(contract, artifact_root: Path, artifact_ref: str, artifact_sha: str, *, tokens: int) -> ContiguousKVAttentionSession:
    kv, att = contiguous_contract(contract, artifact_ref, artifact_sha, prompt=tokens)
    return ContiguousKVAttentionSession(kv, att, artifact_root=artifact_root)


def prepare_contiguous(contract, artifact_root: Path, artifact_ref: str, artifact_sha: str, tokens: int, seed: int):
    session = make_contiguous(contract, artifact_root, artifact_ref, artifact_sha, tokens=tokens)
    h, d = contract.num_kv_heads, contract.head_dim
    session.prefill_write(deterministic_data(h * tokens * d, seed), deterministic_data(h * tokens * d, seed + 1))
    return session


def prepare_paged(context: PagedKVRuntimeContext, request_id: str, tokens: int, seed: int):
    session = context.create_session(request_id)
    h, d = context.contract.num_kv_heads, context.contract.head_dim
    session.prefill(deterministic_data(h * tokens * d, seed), deterministic_data(h * tokens * d, seed + 1), tokens)
    return session


def correctness_case(contract, artifact_root: Path, artifact_ref: str, artifact_sha: str, context: PagedKVRuntimeContext, tokens: int, seed: int) -> dict[str, Any]:
    h, d = contract.num_kv_heads, contract.head_dim
    q = deterministic_data(h * d, seed + 20)
    contig = prepare_contiguous(contract, artifact_root, artifact_ref, artifact_sha, tokens, seed)
    paged = prepare_paged(context, f"correct-{tokens}-{seed}", tokens, seed)
    ck, cv = contig.view()
    ref = reference_decode(q, list(ck), list(cv), h, tokens, d)
    cout = contig.decode(q)
    pout = paged.decode(q)
    pk, pv = paged_logical(paged)
    same_inputs = list(ck) == pk and list(cv) == pv
    direct = compare(pout, list(cout))
    result = {
        "tokens": tokens,
        "page_tokens": contract.page_tokens,
        "block_table": list(context.page_manager.block_table(paged.request_id)),
        "same_inputs": same_inputs,
        "contiguous_vs_reference": compare(cout, ref),
        "paged_vs_reference": compare(pout, ref),
        "paged_vs_contiguous": direct,
    }
    contig.release()
    paged.release()
    return result


def benchmark_token_count(contract, artifact_root: Path, artifact_ref: str, artifact_sha: str, plan, tokens: int, round_index: int, path_order: tuple[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h, d = contract.num_kv_heads, contract.head_dim
    q = deterministic_data(h * d, 1000 + tokens)
    kt = deterministic_data(h * d, 2000 + tokens)
    vt = deterministic_data(h * d, 3000 + tokens)
    for path in path_order:
        if path == "contiguous":
            decode_session = prepare_contiguous(contract, artifact_root, artifact_ref, artifact_sha, tokens, 20 + tokens)
            append_inside_tokens = max(1, min(contract.page_tokens - 1, contract.maximum_logical_tokens - 1))
            boundary_tokens = min(contract.page_tokens, contract.maximum_logical_tokens - 1)
            memory = memory_payload(tokens=tokens, bytes_per_token=contract.bytes_per_token, capacity_tokens=contract.maximum_logical_tokens, bytes_per_page=contract.bytes_per_combined_page, page_tokens=contract.page_tokens)["contiguous"]
            rows.extend([
                row(
                    path,
                    "prefill_write",
                    tokens,
                    round_index,
                    timed_with_setup(
                        lambda: make_contiguous(contract, artifact_root, artifact_ref, artifact_sha, tokens=tokens),
                        lambda s: s.prefill_write(
                            deterministic_data(h * tokens * d, 10 + tokens),
                            deterministic_data(h * tokens * d, 11 + tokens),
                        ),
                        lambda s: s.release(),
                        samples=MICRO_SAMPLES,
                        warmups=MICRO_WARMUPS,
                    ),
                    memory,
                ),
                row(path, "decode_attention", tokens, round_index, timed(lambda: decode_session.decode(q), samples=MICRO_SAMPLES, warmups=MICRO_WARMUPS, inner=DECODE_INNER), memory),
                row(
                    path,
                    "append_inside",
                    append_inside_tokens,
                    round_index,
                    timed_with_setup(
                        lambda: prepare_contiguous(contract, artifact_root, artifact_ref, artifact_sha, append_inside_tokens, 30 + tokens),
                        lambda s: s.append(kt, vt),
                        lambda s: s.release(),
                        samples=MICRO_SAMPLES,
                        warmups=MICRO_WARMUPS,
                    ),
                    memory,
                ),
                row(
                    path,
                    "append_boundary",
                    boundary_tokens,
                    round_index,
                    timed_with_setup(
                        lambda: prepare_contiguous(contract, artifact_root, artifact_ref, artifact_sha, boundary_tokens, 40 + tokens),
                        lambda s: s.append(kt, vt),
                        lambda s: s.release(),
                        samples=MICRO_SAMPLES,
                        warmups=MICRO_WARMUPS,
                    ),
                    memory,
                ),
            ])
        else:
            prefill_context = build_paged_kv_runtime(plan, artifact_root)
            prefill_counter = {"i": 0}
            def paged_prefill_setup():
                rid = f"prefill-{tokens}-{round_index}-{prefill_counter['i']}"
                prefill_counter["i"] += 1
                return prefill_context.create_session(rid)
            decode_context = build_paged_kv_runtime(plan, artifact_root)
            inside_context = build_paged_kv_runtime(plan, artifact_root)
            boundary_context = build_paged_kv_runtime(plan, artifact_root)
            decode_session = prepare_paged(decode_context, f"decode-{tokens}-{round_index}", tokens, 20 + tokens)
            inside_tokens = max(1, min(contract.page_tokens - 1, contract.maximum_logical_tokens - 1))
            boundary_tokens = min(contract.page_tokens, contract.maximum_logical_tokens - 1)
            memory = memory_payload(tokens=tokens, bytes_per_token=contract.bytes_per_token, capacity_tokens=contract.maximum_logical_tokens, bytes_per_page=contract.bytes_per_combined_page, page_tokens=contract.page_tokens)["paged"]
            rows.extend([
                row(
                    path,
                    "prefill_write",
                    tokens,
                    round_index,
                    timed_with_setup(
                        paged_prefill_setup,
                        lambda s: s.prefill(
                            deterministic_data(h * tokens * d, 10 + tokens),
                            deterministic_data(h * tokens * d, 11 + tokens),
                            tokens,
                        ),
                        lambda s: s.release(),
                        samples=MICRO_SAMPLES,
                        warmups=MICRO_WARMUPS,
                    ),
                    memory,
                    prefill_context,
                ),
                row(path, "decode_attention", tokens, round_index, timed(lambda: decode_session.decode(q), samples=MICRO_SAMPLES, warmups=MICRO_WARMUPS, inner=DECODE_INNER), memory, decode_context),
                row(
                    path,
                    "append_inside",
                    inside_tokens,
                    round_index,
                    timed_with_setup(
                        lambda: prepare_paged(inside_context, f"inside-{tokens}-{round_index}-{time.perf_counter_ns()}", inside_tokens, 30 + tokens),
                        lambda s: s.append(kt, vt),
                        lambda s: s.release(),
                        samples=MICRO_SAMPLES,
                        warmups=MICRO_WARMUPS,
                    ),
                    memory,
                    inside_context,
                ),
                row(
                    path,
                    "append_boundary",
                    boundary_tokens,
                    round_index,
                    timed_with_setup(
                        lambda: prepare_paged(boundary_context, f"boundary-{tokens}-{round_index}-{time.perf_counter_ns()}", boundary_tokens, 40 + tokens),
                        lambda s: s.append(kt, vt),
                        lambda s: s.release(),
                        samples=MICRO_SAMPLES,
                        warmups=MICRO_WARMUPS,
                    ),
                    memory,
                    boundary_context,
                ),
            ])
    return rows


def row(path: str, op: str, tokens: int, round_index: int, timing: dict[str, Any], memory: dict[str, Any], context: PagedKVRuntimeContext | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "operation": op,
        "token_count": tokens,
        "round": round_index,
        "timing": timing,
        "memory": memory,
        "allocator_telemetry": None if context is None else {
            "free_pages": context.page_manager.num_free_pages(),
            "allocated_pages": context.page_manager.num_allocated_pages(),
        },
    }


def lifecycle_benchmark(contract, artifact_root: Path, artifact_ref: str, artifact_sha: str, plan, active_requests: int) -> list[dict[str, Any]]:
    h, d, prompt, final_tokens = contract.num_kv_heads, contract.head_dim, min(7, contract.page_tokens - 1), min(16, contract.maximum_logical_tokens)
    rows: list[dict[str, Any]] = []
    for path in ("contiguous", "paged"):
        samples: list[float] = []
        warmups = LIFECYCLE_WARMUPS
        rss_before = rss_kib()
        for i in range(warmups + LIFECYCLE_SAMPLES):
            start = time.perf_counter()
            if path == "contiguous":
                sessions = [make_contiguous(contract, artifact_root, artifact_ref, artifact_sha, tokens=prompt) for _ in range(active_requests)]
                for ridx, session in enumerate(sessions):
                    session.prefill_write(deterministic_data(h * prompt * d, 400 + ridx), deterministic_data(h * prompt * d, 500 + ridx))
                while sessions[0].valid_tokens < final_tokens:
                    for ridx, session in enumerate(sessions):
                        session.append(deterministic_data(h * d, 600 + ridx), deterministic_data(h * d, 700 + ridx))
                        session.decode(deterministic_data(h * d, 800 + ridx))
                for session in sessions:
                    session.release()
            else:
                context = build_paged_kv_runtime(plan, artifact_root)
                sessions = [context.create_session(f"life-{active_requests}-{ridx}") for ridx in range(active_requests)]
                state = context.scheduler_state("replica-life", SchedulerProfile(max_num_seqs=max(1, active_requests), max_num_batched_tokens=128))
                sched = SchedulerCompiler()
                sched_start = time.perf_counter()
                for ridx, session in enumerate(sessions):
                    state.ingest(RequestExecutionState(session.request_id, "serving", "replica-life", 0, prompt, 0, final_tokens - prompt))
                    session.prefill(deterministic_data(h * prompt * d, 400 + ridx), deterministic_data(h * prompt * d, 500 + ridx), prompt)
                sched.compile(state, forced_policy="decode_first")
                scheduler_ms = (time.perf_counter() - sched_start) * 1000.0
                while sessions[0].valid_tokens < final_tokens:
                    for ridx, session in enumerate(sessions):
                        session.append(deterministic_data(h * d, 600 + ridx), deterministic_data(h * d, 700 + ridx))
                        session.decode(deterministic_data(h * d, 800 + ridx))
                telemetry_before_release = state.kv_telemetry()
                for session in sessions:
                    session.release()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i >= warmups:
                samples.append(elapsed)
        rss_after = rss_kib()
        bytes_per_request = (
            contract.maximum_logical_tokens * contract.bytes_per_token
            if path == "contiguous"
            else math.ceil(final_tokens / contract.page_tokens) * contract.bytes_per_combined_page
        )
        rows.append({
            "path": path,
            "operation": "full_lifecycle",
            "active_requests": active_requests,
            "prompt_tokens": prompt,
            "final_tokens": final_tokens,
            "timing": {"summary": stat(samples, warmups=warmups), "raw_ms": samples},
            "requests_per_second": active_requests / (statistics.median(samples) / 1000.0),
            "tokens_decoded_per_second": active_requests * (final_tokens - prompt) / (statistics.median(samples) / 1000.0),
            "native_kv_bytes_per_request": bytes_per_request,
            "rss_before_kib": rss_before,
            "rss_after_kib": rss_after,
            "scheduler_compile_ms_single_observed": scheduler_ms if path == "paged" else 0.0,
            "telemetry_before_release": telemetry_before_release if path == "paged" else None,
        })
    return rows


def overhead_benchmarks(contract, plan, artifact_root: Path) -> dict[str, Any]:
    samples = {}
    samples["manager_reserve"] = timed(lambda: KVPageManager(total_pages=contract.num_physical_pages, tokens_per_page=contract.page_tokens).reserve_prefill("r", 7), samples=MICRO_SAMPLES, warmups=MICRO_WARMUPS)
    manager = KVPageManager(total_pages=contract.num_physical_pages, tokens_per_page=contract.page_tokens)
    manager.reserve_prefill("append", contract.page_tokens)
    samples["manager_begin_commit_boundary_append"] = timed(lambda: _append_txn(manager), samples=MICRO_SAMPLES, warmups=MICRO_WARMUPS)
    context = build_paged_kv_runtime(plan, artifact_root)
    state = context.scheduler_state("replica-overhead", SchedulerProfile(max_num_seqs=4, max_num_batched_tokens=64, max_prefill_chunk_tokens=64))
    for idx in range(4):
        state.ingest(RequestExecutionState(f"r{idx}", "serving", "replica-overhead", 0, 7, 0, 1))
    compiler = SchedulerCompiler()
    samples["scheduler_compile_select"] = timed(lambda: compiler.compile(state, forced_policy="prefill_first"), samples=MICRO_SAMPLES, warmups=MICRO_WARMUPS)
    session = context.create_session("bt")
    session.prefill(deterministic_data(contract.num_kv_heads * 7 * contract.head_dim, 1), deterministic_data(contract.num_kv_heads * 7 * contract.head_dim, 2), 7)
    samples["block_table_materialization"] = timed(lambda: session._block_table_array(), samples=MICRO_SAMPLES, warmups=MICRO_WARMUPS)
    session.release()
    return samples


def _append_txn(manager: KVPageManager) -> None:
    rid = f"txn-{time.perf_counter_ns()}"
    manager.reserve_prefill(rid, manager.tokens_per_page)
    reservation = manager.begin_append_token(rid, manager.tokens_per_page)
    manager.commit_append_token(reservation)
    manager.release(rid)


def environment(native_artifact: Path) -> dict[str, Any]:
    def read_first(paths: list[str]) -> dict[str, str]:
        out = {}
        for path in paths:
            try:
                out[path] = Path(path).read_text().strip()
            except OSError:
                out[path] = "unavailable"
        return out
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "gxx_version": subprocess.run(["g++", "--version"], text=True, capture_output=True, check=True).stdout.splitlines()[0],
        "native_compile_flags": list(NATIVE_FLAGS),
        "native_artifact": str(native_artifact),
        "native_artifact_sha256": sha256(native_artifact),
        "governors": read_first([f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor" for i in range(os.cpu_count() or 0)]),
        "frequencies": read_first([f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq" for i in range(os.cpu_count() or 0)]),
        "temperature_before": read_first(["/sys/class/thermal/thermal_zone0/temp"]),
        "throttling": subprocess.run(["vcgencmd", "get_throttled"], text=True, capture_output=True).stdout.strip() if Path("/usr/bin/vcgencmd").exists() else "unavailable",
    }


def set_affinity(core: int | None) -> list[int]:
    if core is None or not hasattr(os, "sched_setaffinity"):
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    os.sched_setaffinity(0, {core})
    return sorted(os.sched_getaffinity(0))


def round_observation(label: str) -> dict[str, Any]:
    cores = os.cpu_count() or 0
    def read(path: str) -> str:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return "unavailable"
    return {
        "label": label,
        "monotonic_time": time.monotonic(),
        "frequencies_khz": {
            f"cpu{i}": read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
            for i in range(cores)
        },
        "temperature_millicelsius": read("/sys/class/thermal/thermal_zone0/temp"),
        "throttling": subprocess.run(["vcgencmd", "get_throttled"], text=True, capture_output=True).stdout.strip() if Path("/usr/bin/vcgencmd").exists() else "unavailable",
    }


def summarize(results: dict[str, Any]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    rows = results["operation_rows"]
    for op in ("prefill_write", "append_inside", "append_boundary", "decode_attention"):
        for tokens in sorted({r["token_count"] for r in rows if r["operation"] == op}):
            contig = [r for r in rows if r["operation"] == op and r["token_count"] == tokens and r["path"] == "contiguous"]
            paged = [r for r in rows if r["operation"] == op and r["token_count"] == tokens and r["path"] == "paged"]
            if not contig or not paged:
                continue
            cmed = statistics.median(r["timing"]["summary"]["median_ms"] for r in contig)
            pmed = statistics.median(r["timing"]["summary"]["median_ms"] for r in paged)
            comparisons.append({"operation": op, "token_count": tokens, **label_ratio(cmed, pmed)})
    results["derived_comparisons"] = comparisons
    return results


def write_report(results: dict[str, Any], path: Path) -> None:
    lines = ["# Paged KV vs Contiguous KV on Raspberry Pi 5", ""]
    lines.append(f"- Plan hash: `{results['plan_sha256']}`")
    lines.append(f"- Native artifact hash: `{results['native_artifact_sha256']}`")
    lines.append(f"- Shape: heads={results['config']['num_kv_heads']}, head_dim={results['config']['head_dim']}, page_tokens={results['config']['page_tokens']}, capacity={results['config']['maximum_logical_tokens']}")
    lines.append("")
    lines.append("## Decode Median Comparison")
    lines.append("| Tokens | Contiguous ms | Paged ms | Result | Percent |")
    lines.append("| ---: | ---: | ---: | --- | ---: |")
    for cmp in results["derived_comparisons"]:
        if cmp["operation"] == "decode_attention":
            lines.append(f"| {cmp['token_count']} | {cmp['baseline_ms']:.6f} | {cmp['candidate_ms']:.6f} | {cmp['label']} | {cmp['overhead_percent']:.2f}% |")
    lines.append("")
    lines.append("## Lifecycle")
    lines.append("| Active requests | Path | Median ms | Requests/s | Decode tokens/s |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for row in results["lifecycle_rows"]:
        lines.append(f"| {row['active_requests']} | {row['path']} | {row['timing']['summary']['median_ms']:.6f} | {row['requests_per_second']:.2f} | {row['tokens_decoded_per_second']:.2f} |")
    lines.append("")
    lines.append("## Memory Definitions")
    lines.append("Native KV payload bytes are computed from compiler contract bytes/token and actual allocation policy. RSS is `/proc/self/status` VmRSS in KiB; peak RSS is `resource.getrusage(...).ru_maxrss`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-prefix", default="paged_vs_contiguous")
    parser.add_argument("--pin-core", type=int, default=3)
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_affinity = set_affinity(args.pin_core)
    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    env = environment(args.native_artifact)
    env["selected_affinity"] = selected_affinity
    env["selected_core"] = args.pin_core
    rss_before = rss_kib()
    payload = json.loads(args.plan.read_text())
    plan = parse_execution_plan(payload)
    contract = paged_kv_contracts(plan)[0]
    context = build_paged_kv_runtime(plan, args.artifact_root)
    rss_after_context = rss_kib()
    artifact_ref = str(args.native_artifact.relative_to(args.artifact_root))
    artifact_sha = sha256(args.native_artifact)
    correctness = [correctness_case(contract, args.artifact_root, artifact_ref, artifact_sha, context, tokens, idx * 100) for idx, tokens in enumerate(TOKEN_COUNTS)]
    if not all(x["same_inputs"] and x["contiguous_vs_reference"]["passed"] and x["paged_vs_reference"]["passed"] and x["paged_vs_contiguous"]["passed"] for x in correctness):
        raise RuntimeError("correctness gate failed")
    rows: list[dict[str, Any]] = []
    round_observations: list[dict[str, Any]] = []
    for round_index, order in enumerate(ROUND_ORDERS, start=1):
        round_observations.append(round_observation(f"round_{round_index}_before_{'_then_'.join(order)}"))
        for tokens in TOKEN_COUNTS:
            rows.extend(benchmark_token_count(contract, args.artifact_root, artifact_ref, artifact_sha, plan, tokens, round_index, order))
        round_observations.append(round_observation(f"round_{round_index}_after_{'_then_'.join(order)}"))
    lifecycle: list[dict[str, Any]] = []
    for active in (1, 2, 4):
        lifecycle.extend(lifecycle_benchmark(contract, args.artifact_root, artifact_ref, artifact_sha, plan, active))
    result = summarize({
        "environment": env,
        "plan_path": str(args.plan),
        "plan_sha256": sha256(args.plan),
        "native_artifact_sha256": artifact_sha,
        "config": {
            "num_kv_heads": contract.num_kv_heads,
            "head_dim": contract.head_dim,
            "page_tokens": contract.page_tokens,
            "maximum_logical_tokens": contract.maximum_logical_tokens,
            "bytes_per_token": contract.bytes_per_token,
            "bytes_per_combined_page": contract.bytes_per_combined_page,
            "total_pool_bytes": contract.total_pool_bytes,
        },
        "rss": {
            "before_runtime_construction_kib": rss_before,
            "after_runtime_construction_kib": rss_after_context,
            "after_benchmark_kib": rss_kib(),
            "peak_kib": peak_rss_kib(),
        },
        "timer_overhead": timer_overhead(),
        "round_observations": round_observations,
        "round_orders": [list(x) for x in ROUND_ORDERS],
        "correctness": correctness,
        "operation_rows": rows,
        "lifecycle_rows": lifecycle,
        "occupancy_memory": [
            {"token_count": t, **memory_payload(tokens=t, bytes_per_token=contract.bytes_per_token, capacity_tokens=contract.maximum_logical_tokens, bytes_per_page=contract.bytes_per_combined_page, page_tokens=contract.page_tokens)}
            for t in OCCUPANCY_COUNTS
        ],
        "overheads": overhead_benchmarks(contract, plan, args.artifact_root),
        "scheduler_telemetry": context.scheduler_state("replica-summary", SchedulerProfile()).kv_telemetry(),
    })
    env["temperature_after"] = environment(args.native_artifact)["temperature_before"]
    results_path = args.output_dir / f"{args.result_prefix}_results.json"
    env_path = args.output_dir / f"{args.result_prefix}_environment.json"
    report_path = args.output_dir / f"{args.result_prefix}_report.md"
    results_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_path.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(result, report_path)
    print(json.dumps({"status": "passed", "results": str(results_path), "report": str(report_path), "environment": str(env_path), "plan_sha256": result["plan_sha256"], "native_artifact_sha256": artifact_sha}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
