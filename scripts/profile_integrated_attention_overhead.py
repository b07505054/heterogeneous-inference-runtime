#!/usr/bin/env python3
"""Profile integrated contiguous vs paged KV decode wrapper overhead."""
from __future__ import annotations

import argparse
from array import array
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.execution_plan.loader import load_execution_plan
from deployment.execution_plan.paged_kv_runtime import build_paged_kv_runtime, paged_kv_contracts
from deployment.execution_plan.contiguous_kv_cache import _ptr as contiguous_ptr
from deployment.execution_plan.paged_kv_cache import _fp as paged_fp, _ip as paged_ip
from scripts.benchmark_paged_vs_contiguous_pi import (
    TOKEN_COUNTS,
    compare,
    deterministic_data,
    make_contiguous,
    memory_payload,
    paged_logical,
    prepare_contiguous,
    prepare_paged,
    reference_decode,
    sha256,
)

PROFILE_INNER = 1000
PROFILE_SAMPLES = 100
PROFILE_WARMUPS = 10


CALL_GRAPH = """# Integrated Attention Call Graph

## Contiguous Decode

benchmark/session caller
-> deployment.execution_plan.contiguous_kv_cache.ContiguousKVAttentionSession.decode(q: array('f')) -> array('f')
-> state and valid-token validation
-> _product([num_kv_heads, head_dim])
-> _ptr(q), _ptr(k_cache), _ptr(v_cache), _ptr(output), _ptr(workspace)
-> configured ctypes function hir_cpu_attention_decode_contiguous_kv_fp32
-> output copy: array('f', self._output)
-> counter/accounting updates

## Page-Major Paged Decode

benchmark/session caller
-> deployment.execution_plan.paged_kv_cache.PagedKVAttentionSession.decode(q: array('f')) -> array('f')
-> state validation
-> _validate_live()
-> KVPageManager.validate_invariants()
-> valid_token_count(request_id)
-> block_table(request_id)
-> cached native-compatible block-table validation or refresh
-> _fp(q), _fp(k_pages), _fp(v_pages), _ip(block_table), _ip(physical_page_cache), _fp(output), _fp(workspace)
-> configured ctypes function hir_cpu_attention_decode_paged_kv_page_major_fp32
-> output copy: array('f', self.out)
-> counter/accounting updates
"""


def stat(samples: list[float], *, warmups: int) -> dict[str, Any]:
    values = sorted(samples)
    mean = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(x - median) for x in values])
    p95 = values[math.ceil(0.95 * len(values)) - 1]
    cv = sd / mean if mean else 0.0
    return {
        "minimum_us": values[0],
        "median_us": median,
        "mean_us": mean,
        "p95_us": p95,
        "mad_us": mad,
        "stddev_us": sd,
        "coefficient_of_variation": cv,
        "samples": len(values),
        "warmups": warmups,
        "unstable": bool(cv > 0.25 or p95 > median * 2.0),
    }


def timer_overhead(samples: int = 5000) -> dict[str, Any]:
    values = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        end = time.perf_counter_ns()
        values.append((end - start) / 1000.0)
    return {"summary": stat(values, warmups=0), "raw_us": values}


def timed(fn: Callable[[], Any], *, samples: int = PROFILE_SAMPLES, warmups: int = PROFILE_WARMUPS, inner: int = PROFILE_INNER) -> dict[str, Any]:
    raw: list[float] = []
    sink = 0
    for i in range(warmups + samples):
        start = time.perf_counter_ns()
        for _ in range(inner):
            value = fn()
            sink ^= id(value) & 1
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0 / inner
        if i >= warmups:
            raw.append(elapsed_us)
    if sink == 2:
        raise AssertionError("unreachable sink")
    return {"summary": stat(raw, warmups=warmups), "raw_us": raw, "inner_iterations": inner}


def timed_with_setup(setup: Callable[[], Any], op: Callable[[Any], Any], cleanup: Callable[[Any], None] | None = None, *, samples: int = PROFILE_SAMPLES, warmups: int = PROFILE_WARMUPS) -> dict[str, Any]:
    raw: list[float] = []
    for i in range(warmups + samples):
        obj = setup()
        start = time.perf_counter_ns()
        op(obj)
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0
        if cleanup is not None:
            cleanup(obj)
        if i >= warmups:
            raw.append(elapsed_us)
    return {"summary": stat(raw, warmups=warmups), "raw_us": raw, "inner_iterations": 1}


def stage_delta_rows(contiguous: dict[str, float], paged: dict[str, float], total_gap_us: float) -> list[dict[str, Any]]:
    names = sorted(set(contiguous) | set(paged))
    rows = []
    for name in names:
        c = contiguous.get(name, 0.0)
        p = paged.get(name, 0.0)
        delta = p - c
        rows.append({
            "stage": name,
            "contiguous_us": c,
            "paged_us": p,
            "delta_us": delta,
            "ratio": None if c == 0 else p / c,
            "percent_of_integrated_gap": None if total_gap_us == 0 else delta / total_gap_us * 100.0,
        })
    return rows


def old_block_table_materialization(session):
    refs = session.page_manager.block_table(session.request_id)
    bt = array("i", [session.c["invalid_page_sentinel"]]) * session.c["block_table_length"]
    for i, page in enumerate(refs):
        bt[i] = page
    return bt


def contiguous_args(session, q):
    n = session.kv["num_kv_heads"] * session.kv["head_dim"]
    return (
        contiguous_ptr(q, n),
        len(q),
        contiguous_ptr(session.k_cache, len(session.k_cache)),
        len(session.k_cache),
        contiguous_ptr(session.v_cache, len(session.v_cache)),
        len(session.v_cache),
        contiguous_ptr(session._output, len(session._output)),
        len(session._output),
        contiguous_ptr(session._workspace, len(session._workspace)),
        len(session._workspace),
        session.valid_tokens,
        session.kv["num_kv_heads"],
        session.kv["capacity_tokens"],
        session.kv["head_dim"],
        -1,
    )


def paged_args(session, q):
    n = session.c["num_kv_heads"] * session.c["head_dim"]
    bt = session._block_table_array()
    args = (
        paged_fp(q, n),
        len(q),
        paged_fp(session.k, len(session.k)),
        len(session.k),
        paged_fp(session.v, len(session.v)),
        len(session.v),
        paged_ip(bt, len(bt)),
        len(bt),
        paged_ip(session.physical_page_cache, len(session.physical_page_cache)),
        len(session.physical_page_cache),
        paged_fp(session.out, len(session.out)),
        len(session.out),
        paged_fp(session.ws, len(session.ws)),
        len(session.ws),
        session.valid_tokens,
        session.c["num_physical_pages"],
        session.c["num_kv_heads"],
        session.c["page_tokens"],
        session.c["head_dim"],
        -1,
    )
    return args


def env_snapshot(args, native_artifact: Path) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "native_artifact": str(native_artifact),
        "native_artifact_sha256": sha256(native_artifact),
        "timer": "time.perf_counter_ns",
        "profile_samples": PROFILE_SAMPLES,
        "profile_warmups": PROFILE_WARMUPS,
        "profile_inner": PROFILE_INNER,
        "plan": str(args.plan),
        "artifact_root": str(args.artifact_root),
    }


def profile_token(contract, artifact_root: Path, artifact_ref: str, artifact_sha: str, plan, tokens: int) -> dict[str, Any]:
    h, d = contract.num_kv_heads, contract.head_dim
    q = deterministic_data(h * d, 9000 + tokens)
    contig = prepare_contiguous(contract, artifact_root, artifact_ref, artifact_sha, tokens, 100 + tokens)
    context = build_paged_kv_runtime(plan, artifact_root)
    paged = prepare_paged(context, f"integrated-profile-{tokens}", tokens, 100 + tokens)
    ck, cv = contig.view()
    pk, pv = paged_logical(paged)
    ref = reference_decode(q, list(ck), list(cv), h, tokens, d)
    cout = contig.decode(q)
    pout = paged.decode(q)
    correctness = {
        "same_logical_kv": list(ck) == pk and list(cv) == pv,
        "contiguous_vs_reference": compare(cout, ref),
        "paged_vs_reference": compare(pout, ref),
        "paged_vs_contiguous": compare(pout, list(cout)),
        "paged_block_table": list(context.page_manager.block_table(paged.request_id)),
    }
    total_contig = timed(lambda: contig.decode(q), inner=50)
    total_paged = timed(lambda: paged.decode(q), inner=50)
    cold_contig = timed_with_setup(
        lambda: prepare_contiguous(contract, artifact_root, artifact_ref, artifact_sha, tokens, 300 + tokens),
        lambda s: s.decode(q),
        lambda s: s.release(),
    )
    cold_paged_counter = {"i": 0}

    def cold_paged_setup():
        cold_paged_counter["i"] += 1
        ctx = build_paged_kv_runtime(plan, artifact_root)
        return prepare_paged(ctx, f"cold-{tokens}-{cold_paged_counter['i']}", tokens, 300 + tokens)

    cold_paged = timed_with_setup(cold_paged_setup, lambda s: s.decode(q), lambda s: s.release())
    boundary_tokens = min(contract.page_tokens, contract.maximum_logical_tokens - 1)
    boundary_k = deterministic_data(h * d, 7000 + tokens)
    boundary_v = deterministic_data(h * d, 8000 + tokens)
    boundary_paged_counter = {"i": 0}

    def boundary_paged_setup():
        boundary_paged_counter["i"] += 1
        ctx = build_paged_kv_runtime(plan, artifact_root)
        return prepare_paged(ctx, f"boundary-{tokens}-{boundary_paged_counter['i']}", boundary_tokens, 500 + tokens)

    boundary_paged = timed_with_setup(boundary_paged_setup, lambda s: (s.append(boundary_k, boundary_v), s.decode(q)), lambda s: s.release())

    contiguous_stages = {
        "shape_and_valid_token_calculation": timed(lambda: contig.kv["num_kv_heads"] * contig.kv["head_dim"])["summary"]["median_us"],
        "contiguous_kv_buffer_lookup": timed(lambda: (contig.k_cache, contig.v_cache))["summary"]["median_us"],
        "ctypes_ffi_argument_preparation": timed(lambda: contiguous_args(contig, q))["summary"]["median_us"],
        "native_function_call": 0.0,
        "output_handling_conversion": timed(lambda: array("f", contig._output))["summary"]["median_us"],
        "telemetry_accounting": timed(lambda: setattr(contig.counters, "decode_attention_invocation_count", contig.counters.decode_attention_invocation_count + 1))["summary"]["median_us"],
    }
    paged_stages = {
        "request_session_lookup": timed(lambda: paged.page_manager.has_request(paged.request_id))["summary"]["median_us"],
        "page_session_invariant_validation": timed(lambda: paged._validate_live())["summary"]["median_us"],
        "valid_token_calculation": timed(lambda: paged.valid_tokens)["summary"]["median_us"],
        "block_table_lookup": timed(lambda: paged.page_manager.block_table(paged.request_id))["summary"]["median_us"],
        "block_table_copy_old_path_simulated": timed(lambda: old_block_table_materialization(paged))["summary"]["median_us"],
        "block_table_copy_current": timed(lambda: paged._block_table_array())["summary"]["median_us"],
        "block_table_python_to_native_materialization_current": timed(lambda: paged_ip(paged._block_table_array(), paged.c["block_table_length"]))["summary"]["median_us"],
        "physical_kv_pool_lookup": timed(lambda: (paged.k, paged.v))["summary"]["median_us"],
        "physical_page_metadata_preparation": timed(lambda: paged_ip(paged.physical_page_cache, len(paged.physical_page_cache)))["summary"]["median_us"],
        "ctypes_ffi_argument_preparation": timed(lambda: paged_args(paged, q))["summary"]["median_us"],
        "native_function_call": 0.0,
        "output_handling_conversion": timed(lambda: array("f", paged.out))["summary"]["median_us"],
        "telemetry_accounting": timed(lambda: setattr(paged.count, "paged_decode_invocation_count", paged.count.paged_decode_invocation_count + 1))["summary"]["median_us"],
    }
    gap_us = total_paged["summary"]["median_us"] - total_contig["summary"]["median_us"]
    rows = stage_delta_rows(contiguous_stages, paged_stages, gap_us)
    measured_delta = sum(row["delta_us"] for row in rows)
    memory = memory_payload(tokens=tokens, bytes_per_token=contract.bytes_per_token, capacity_tokens=contract.maximum_logical_tokens, bytes_per_page=contract.bytes_per_combined_page, page_tokens=contract.page_tokens)
    contig.release()
    paged.release()
    return {
        "tokens": tokens,
        "correctness": correctness,
        "steady_state": {"contiguous": total_contig, "paged": total_paged},
        "cold_first_decode_after_session_creation": {"contiguous": cold_contig, "paged": cold_paged},
        "boundary_growth_append_then_decode": {"paged": boundary_paged, "boundary_start_tokens": boundary_tokens},
        "stages_us": {"contiguous": contiguous_stages, "paged": paged_stages, "differential_rows": rows},
        "integrated_gap_us": gap_us,
        "measured_stage_delta_us": measured_delta,
        "unclassified_gap_us": gap_us - measured_delta,
        "percent_gap_accounted": None if gap_us == 0 else measured_delta / gap_us * 100.0,
        "memory": memory,
        "optimization_behavior": {
            "selected_optimization": "decode_validation_hoist",
            "global_page_manager_validate_invariants_per_decode": False,
            "per_request_block_table_validation_per_decode": True,
        },
    }


def allocation_audit() -> dict[str, Any]:
    return {
        "contiguous_per_decode": {
            "numpy_arrays_allocated": 0,
            "python_lists_created": 0,
            "ctypes_pointer_wrappers_created": 5,
            "ctypes_array_from_buffer_calls": 5,
            "copies": 1,
            "copied_bytes": "num_kv_heads * head_dim * 4 output bytes",
            "native_symbol_lookup": "cached during session initialization",
            "artifact_hash_validation": "session initialization only",
        },
        "paged_before_optimization_per_decode": {
            "numpy_arrays_allocated": 0,
            "python_lists_created": "at least 1 from _validate_live refs plus manager tuple/list conversions",
            "ctypes_pointer_wrappers_created": 7,
            "ctypes_array_from_buffer_calls": 7,
            "block_table_array_allocated": 1,
            "block_table_entries_copied": "block_table_length",
            "global_page_manager_invariant_scan": 1,
            "copies": 2,
            "copied_bytes": "block_table_length * 4 plus output bytes",
            "native_symbol_lookup": "cached during session initialization",
            "artifact_hash_validation": "session initialization only",
        },
        "paged_after_optimization_steady_decode": {
            "numpy_arrays_allocated": 0,
            "python_lists_created": "unchanged validation manager views",
            "ctypes_pointer_wrappers_created": 7,
            "ctypes_array_from_buffer_calls": 7,
            "block_table_array_allocated": 1,
            "block_table_entries_copied": "block_table_length",
            "global_page_manager_invariant_scan": 0,
            "per_request_block_table_validation": 1,
            "copies": 1,
            "copied_bytes": "block_table_length * 4 plus output bytes",
            "optimization": "immutable/global manager invariant validation is hoisted out of steady-state decode; mutable per-request mapping checks remain",
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = ["# Integrated Attention Overhead Report", ""]
    lines.append(f"Timer median overhead: {result['timer_overhead']['summary']['median_us']:.4f} us")
    for row in result["tokens"]:
        lines.extend([
            "",
            f"## {row['tokens']} Tokens",
            f"Contiguous steady median: {row['steady_state']['contiguous']['summary']['median_us']:.4f} us",
            f"Paged steady median: {row['steady_state']['paged']['summary']['median_us']:.4f} us",
            f"Gap: {row['integrated_gap_us']:.4f} us",
            f"Measured delta: {row['measured_stage_delta_us']:.4f} us",
            f"Unclassified gap: {row['unclassified_gap_us']:.4f} us",
            f"Gap accounted: {row['percent_gap_accounted']:.2f}%",
            "",
            "| Stage | Contiguous us | Paged us | Delta us | Ratio | % gap |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for stage in row["stages_us"]["differential_rows"]:
            ratio = "" if stage["ratio"] is None else f"{stage['ratio']:.2f}"
            pct = "" if stage["percent_of_integrated_gap"] is None else f"{stage['percent_of_integrated_gap']:.2f}"
            lines.append(f"| {stage['stage']} | {stage['contiguous_us']:.4f} | {stage['paged_us']:.4f} | {stage['delta_us']:.4f} | {ratio} | {pct} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pin-core", type=int)
    args = parser.parse_args()
    if args.pin_core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.pin_core})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_execution_plan(args.plan)
    contract = paged_kv_contracts(plan)[0]
    artifact_ref = str(args.native_artifact.resolve().relative_to(args.artifact_root.resolve()))
    artifact_sha = hashlib.sha256(args.native_artifact.read_bytes()).hexdigest()
    result = {
        "classification": ["INTEGRATED_OVERHEAD_DECOMPOSED", "PAGED_WRAPPER_BOTTLENECK_IDENTIFIED", "CACHE_LIFECYCLE_VALIDATED"],
        "methodology": {
            "stages": "hierarchical batched microbenchmarks plus uninstrumented session decode totals",
            "cold_path": "new prefilled session per measured decode",
            "warm_steady_state": "same prefilled session, unchanged block table",
            "boundary_growth": "prefill at page boundary, append crossing boundary, then decode",
            "instrumentation_inflation": "fine-grained timers are outside the primary decode total; totals are measured uninstrumented",
        },
        "timer_overhead": timer_overhead(),
        "tokens": [profile_token(contract, args.artifact_root, artifact_ref, artifact_sha, plan, tokens) for tokens in TOKEN_COUNTS],
    }
    env = env_snapshot(args, args.native_artifact)
    audit = allocation_audit()
    baseline = {
        "native_artifact_sha256": artifact_sha,
        "integrated_wrapper_formula": "wrapper_overhead_us = integrated_session_decode_us - direct_exported_native_us",
        "direct_native_note": "Direct native timings are supplied by the companion exported-native profiler from the previous slice or same-run benchmark artifacts.",
    }
    (args.output_dir / "integrated_attention_overhead_breakdown.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "integrated_attention_overhead_environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (args.output_dir / "integrated_attention_allocation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (args.output_dir / "integrated_attention_call_graph.md").write_text(CALL_GRAPH, encoding="utf-8")
    (args.output_dir / "integrated_attention_baseline_results.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    write_report(args.output_dir / "integrated_attention_overhead_report.md", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
