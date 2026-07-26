#!/usr/bin/env python3
"""Validate a compiler paged-KV ExecutionPlan against the live runtime path."""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from deployment.execution_plan.kv_page_manager import KVPageManager
from deployment.execution_plan.loader import parse_execution_plan
from deployment.execution_plan.paged_kv_runtime import (
    build_paged_kv_runtime,
    paged_kv_contracts,
)
from deployment.serving_scheduler import (
    KV_ADMITTED,
    KV_DEFERRED_PRESSURE,
    KV_REJECTED_PROMPT_TOO_LARGE,
    PlanOnlySchedulerRuntime,
    ReplicaSchedulerState,
    RequestExecutionState,
    SchedulerCompiler,
    SchedulerProfile,
    deserialize_schedule_plan,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data(n: int, seed: int) -> array:
    return array("f", (((i * 1103515245 + seed * 12345) & 0xFFFF) / 32768 - 1 for i in range(n)))


def _logical(session) -> tuple[list[float], list[float]]:
    c = session.c
    h, d, pt = c["num_kv_heads"], c["head_dim"], c["page_tokens"]
    pages = session.page_manager.block_table(session.request_id)
    k: list[float] = []
    v: list[float] = []
    for hi in range(h):
        for token in range(session.valid_tokens):
            page = pages[token // pt]
            base = ((page * h + hi) * pt + token % pt) * d
            k.extend(session.k[base : base + d])
            v.extend(session.v[base : base + d])
    return k, v


def _reference(q: array, k: list[float], v: list[float], h: int, tokens: int, d: int) -> list[float]:
    out: list[float] = []
    scale = 1.0 / math.sqrt(d)
    for hi in range(h):
        scores = [
            sum(q[hi * d + x] * k[(hi * tokens + t) * d + x] for x in range(d)) * scale
            for t in range(tokens)
        ]
        peak = max(scores)
        weights = [math.exp(x - peak) for x in scores]
        denom = sum(weights)
        out.extend(
            sum(weights[t] / denom * v[(hi * tokens + t) * d + x] for t in range(tokens))
            for x in range(d)
        )
    return out


def _decode_error(session, q_seed: int) -> dict[str, float]:
    q = _data(session.c["num_kv_heads"] * session.c["head_dim"], q_seed)
    got = session.decode(q)
    k, v = _logical(session)
    ref = _reference(q, k, v, session.c["num_kv_heads"], session.valid_tokens, session.c["head_dim"])
    abs_errors = [abs(float(x) - y) for x, y in zip(got, ref)]
    l2_ref = math.sqrt(sum(x * x for x in ref))
    l2_err = math.sqrt(sum(x * x for x in abs_errors))
    return {
        "max_abs_error": max(abs_errors),
        "relative_l2_error": l2_err / max(l2_ref, 1e-30),
    }


def _timed(fn, reps: int = 7, warmups: int = 2) -> tuple[Any, dict[str, float]]:
    samples: list[float] = []
    result = None
    for i in range(reps + warmups):
        start = time.perf_counter()
        result = fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        if i >= warmups:
            samples.append(elapsed)
    return result, {"median_ms": statistics.median(samples), "samples": len(samples), "warmups": warmups}


def _prefill(session, tokens: int, seed: int) -> dict[str, float]:
    h, d = session.c["num_kv_heads"], session.c["head_dim"]
    _, latency = _timed(lambda: session.prefill(_data(h * tokens * d, seed), _data(h * tokens * d, seed + 1), tokens), reps=1, warmups=0)
    return latency


def _append(session, seed: int) -> dict[str, float]:
    h, d = session.c["num_kv_heads"], session.c["head_dim"]
    _, latency = _timed(lambda: session.append(_data(h * d, seed), _data(h * d, seed + 1)), reps=1, warmups=0)
    return latency


def _scheduler(profile: SchedulerProfile, page_manager: KVPageManager) -> ReplicaSchedulerState:
    return ReplicaSchedulerState("replica-0", profile, page_manager=page_manager)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--native-artifact", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.plan.read_text())
    plan = parse_execution_plan(payload)
    contract = paged_kv_contracts(plan)[0]
    context = build_paged_kv_runtime(plan, args.artifact_root)
    h, d, pt = contract.num_kv_heads, contract.head_dim, contract.page_tokens

    result: dict[str, Any] = {
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "plan_path": str(args.plan),
        "artifact_path": str(args.native_artifact),
        "artifact_sha256": _sha256(args.native_artifact),
        "contract": {
            "plan_id": plan.plan_id,
            "kv_execution_unit": contract.kv_execution_unit,
            "kv_candidate_id": contract.kv_candidate_id,
            "kv_layout_kind": contract.kv_layout_kind,
            "dtype": contract.dtype,
            "page_tokens": contract.page_tokens,
            "num_physical_pages": contract.num_physical_pages,
            "maximum_logical_tokens": contract.maximum_logical_tokens,
            "bytes_per_token": contract.bytes_per_token,
            "bytes_per_combined_page": contract.bytes_per_combined_page,
            "total_pool_bytes": contract.total_pool_bytes,
            "paged_attention_entry_point": contract.paged_attention_entry_point,
            "runtime_no_layout_redecision": contract.runtime_no_layout_redecision,
            "runtime_no_kernel_redecision": contract.runtime_no_kernel_redecision,
        },
        "symbols": subprocess.run(
            ["nm", "-D", "--defined-only", str(args.native_artifact)],
            check=True,
            text=True,
            capture_output=True,
        ).stdout,
    }

    s1 = context.create_session("single")
    prefill_latency = _prefill(s1, pt - 1, 10)
    append_latency = _append(s1, 20)
    _, decode_latency = _timed(lambda: s1.decode(_data(h * d, 30)))
    result["single_request"] = {
        "block_table": list(context.page_manager.block_table("single")),
        "valid_tokens": context.page_manager.valid_token_count("single"),
        "errors": _decode_error(s1, 31),
        "latency": {"prefill": prefill_latency, "append": append_latency, "decode": decode_latency},
    }

    s2 = context.create_session("boundary")
    _prefill(s2, pt, 40)
    before = list(context.page_manager.block_table("boundary"))
    free_before = context.page_manager.num_free_pages()
    boundary_append_latency = _append(s2, 50)
    after = list(context.page_manager.block_table("boundary"))
    result["boundary_crossing"] = {
        "block_table_before": before,
        "block_table_after": after,
        "new_page": after[-1],
        "free_pages_before": free_before,
        "free_pages_after": context.page_manager.num_free_pages(),
        "valid_tokens": context.page_manager.valid_token_count("boundary"),
        "errors": _decode_error(s2, 51),
        "latency": {"append": boundary_append_latency},
    }

    a = context.create_session("request-a")
    b = context.create_session("request-b")
    _prefill(a, 2, 60)
    _prefill(b, 2, 70)
    pages_a = set(context.page_manager.block_table("request-a"))
    pages_b = set(context.page_manager.block_table("request-b"))
    result["two_requests"] = {
        "shared_manager": a.page_manager is b.page_manager is context.page_manager,
        "shared_storage": a.storage is b.storage is context.storage,
        "request_a_block_table": sorted(pages_a),
        "request_b_block_table": sorted(pages_b),
        "disjoint_pages": pages_a.isdisjoint(pages_b),
        "request_a_errors": _decode_error(a, 80),
        "request_b_errors": _decode_error(b, 81),
    }

    small_manager = KVPageManager(total_pages=3, tokens_per_page=pt)
    pressure = _scheduler(
        SchedulerProfile(max_num_seqs=3, max_num_batched_tokens=64, max_prefill_chunk_tokens=64),
        small_manager,
    )
    r1 = RequestExecutionState("p1", "serving", "replica-0", 0, pt, 0, 1)
    r2 = RequestExecutionState("p2", "serving", "replica-0", 0, pt, 0, 1)
    r3 = RequestExecutionState("p3", "serving", "replica-0", 0, pt, 0, 1)
    for req in (r1, r2, r3):
        pressure.ingest(req)
    boundary_req = RequestExecutionState("boundary-small", "serving", "replica-0", 0, pt, 0, 1)
    pressure.ingest(boundary_req)
    small_manager.reserve_prefill("p1", pt - 1)
    small_manager.reserve_prefill("p2", pt)
    small_manager.reserve_prefill("boundary-small", pt)
    pressure_status = pressure.kv_admission_status(r3)
    inside_decode_status = pressure.kv_decode_status(r1)
    boundary_decode_status = pressure.kv_decode_status(boundary_req)
    small_manager.release("p2")
    boundary_after_release = pressure.kv_decode_status(boundary_req)
    result["pressure_and_deferral"] = {
        "prefill_status_with_full_pool": pressure_status,
        "decode_inside_existing_page_with_zero_free": inside_decode_status,
        "boundary_decode_with_zero_free": boundary_decode_status,
        "boundary_decode_after_release": boundary_after_release,
        "telemetry": pressure.kv_telemetry(),
    }
    assert pressure_status == KV_DEFERRED_PRESSURE
    assert inside_decode_status == KV_ADMITTED
    assert boundary_decode_status == KV_DEFERRED_PRESSURE
    assert boundary_after_release == KV_ADMITTED

    reuse = build_paged_kv_runtime(plan, args.artifact_root)
    req_a = reuse.create_session("reuse-a")
    _prefill(req_a, 2, 90)
    released = req_a.page_manager.block_table("reuse-a")
    req_a.release()
    req_c = reuse.create_session("reuse-c")
    _prefill(req_c, 2, 100)
    result["release_and_reuse"] = {
        "released_pages": list(released),
        "request_c_block_table": list(reuse.page_manager.block_table("reuse-c")),
        "deterministic_reuse": tuple(released) == reuse.page_manager.block_table("reuse-c"),
        "request_c_errors": _decode_error(req_c, 101),
    }

    scheduled = build_paged_kv_runtime(plan, args.artifact_root)
    scheduled_session = scheduled.create_session("scheduled")
    scheduled_state = scheduled.scheduler_state(
        "replica-0",
        SchedulerProfile(
            max_num_seqs=1,
            max_num_batched_tokens=16,
            max_prefill_chunk_tokens=16,
            balanced_decode_reservation=1,
        ),
    )
    scheduled_request = RequestExecutionState("scheduled", "serving", "replica-0", 0, 3, 0, 1)
    scheduled_state.ingest(scheduled_request)
    scheduler_compiler = SchedulerCompiler()
    scheduler_runtime = PlanOnlySchedulerRuntime()

    def execute_item(request, item, _plan):
        if item.phase == "prefill":
            _prefill(scheduled_session, request.uncached_prompt_tokens, 120)
        else:
            _append(scheduled_session, 130)
            _decode_error(scheduled_session, 131)
        return {"kv_page_manager_committed": True}

    prefill_plan = scheduler_compiler.compile(scheduled_state, forced_policy="prefill_first")
    prefill_event = scheduler_runtime.execute(
        scheduled_state,
        deserialize_schedule_plan(prefill_plan.serialize(), scheduled_state),
        execute_item,
    )
    decode_plan = scheduler_compiler.compile(scheduled_state, forced_policy="decode_first")
    decode_event = scheduler_runtime.execute(
        scheduled_state,
        deserialize_schedule_plan(decode_plan.serialize(), scheduled_state),
        execute_item,
    )
    result["scheduler_completion_release"] = {
        "finished": scheduled_request.finished,
        "manager_has_request_after_completion": scheduled.page_manager.has_request("scheduled"),
        "prefill_event_telemetry": prefill_event["kv_telemetry"],
        "decode_event_telemetry": decode_event["kv_telemetry"],
        "final_telemetry": scheduled_state.kv_telemetry(),
    }
    assert scheduled_request.finished
    assert not scheduled.page_manager.has_request("scheduled")

    oversize = _scheduler(SchedulerProfile(), KVPageManager(total_pages=1, tokens_per_page=pt))
    oversized = RequestExecutionState("too-big", "serving", "replica-0", 0, pt + 1, 0, 1)
    oversize.ingest(oversized)
    oversize_status = oversize.kv_admission_status(oversized)
    oversize.ready()
    oversize.page_manager.validate_invariants()
    result["oom_rejection"] = {
        "status": oversize_status,
        "phase_after_ready": oversized.phase,
        "free_pages": oversize.page_manager.num_free_pages(),
        "allocated_pages": oversize.page_manager.num_allocated_pages(),
        "telemetry": oversize.kv_telemetry(),
    }
    assert oversize_status == KV_REJECTED_PROMPT_TOO_LARGE
    assert oversize.page_manager.num_allocated_pages() == 0

    result["final_manager"] = context.page_manager.validate_invariants() or context.scheduler_state(
        "replica-final", SchedulerProfile()
    ).kv_telemetry()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
