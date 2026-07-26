from array import array
import hashlib
import math
import subprocess
from pathlib import Path

import pytest

from deployment.execution_plan.attention_cpu_adapter import AttentionContractError
from deployment.execution_plan.kv_page_manager import KVPageManager
from deployment.execution_plan.paged_kv_cache import PagedKVAttentionSession, PagedKVStorage
from deployment.serving_execution import ServingPlanError
from deployment.serving_scheduler import (
    PlanOnlySchedulerRuntime,
    ReplicaSchedulerState,
    RequestExecutionState,
    SchedulerCompiler,
    SchedulerProfile,
    deserialize_schedule_plan,
)

ROOT = Path(__file__).resolve().parents[1]


class _FailureStatus:
    code = 7
    message = b"injected_native_failure"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("scheduler_kv_native")
    so = root / "libattention_fp32.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(ROOT / "native/cpu_kernels/attention_fp32.cpp"),
            "-o",
            str(so),
        ],
        check=True,
    )
    return root, so, hashlib.sha256(so.read_bytes()).hexdigest()


def cfg(artifact, pages=4, pt=4, max_tokens=16, h=2, d=4):
    root, so, sha = artifact
    blocks = (max_tokens + pt - 1) // pt
    one = h * pt * d * 4
    strides = [h * pt * d, pt * d, d, 1]
    return root, {
        "kv_candidate_id": "cpu_paged_kv_fp32_v1",
        "kv_layout_kind": "paged_phd_contiguous",
        "pool_artifact_ref": so.name,
        "pool_artifact_sha256": sha,
        "pool_artifact_version": "hir.paged_kv.v1",
        "dtype": "fp32",
        "batch": 1,
        "num_kv_heads": h,
        "head_dim": d,
        "page_tokens": pt,
        "num_physical_pages": pages,
        "maximum_logical_tokens": max_tokens,
        "maximum_logical_blocks": blocks,
        "block_table_length": blocks,
        "block_table_element_type": "int32",
        "invalid_page_sentinel": -1,
        "k_page_strides": strides,
        "v_page_strides": strides,
        "bytes_per_token": 2 * h * d * 4,
        "bytes_per_k_page": one,
        "bytes_per_v_page": one,
        "bytes_per_combined_page": 2 * one,
        "total_pool_bytes": pages * 2 * one,
        "alignment_bytes": 4,
        "pool_create_entry_point": "hir_paged_kv_initialize",
        "prefill_write_entry_point": "hir_paged_kv_prefill_write",
        "append_entry_point": "hir_paged_kv_append",
        "view_binding": "direct_int32_block_table_translation",
        "reset_entry_point": "hir_paged_kv_reset",
        "release_entry_point": "runtime_owned_pool_release",
        "paged_attention_kernel_id": "cpu_attention_decode_paged_kv_fp32",
        "contiguous_fallback_identity": "cpu_contiguous_kv_fp32_v1",
        "runtime_no_layout_redecision": True,
        "runtime_no_kernel_redecision": True,
    }


def data(n, seed=0):
    return array("f", ((i + seed) % 17 / 17 - 0.5 for i in range(n)))


def ref(q, k, v, h, tokens, d):
    out = []
    scale = 1 / math.sqrt(d)
    for hi in range(h):
        scores = [
            sum(q[hi * d + x] * k[(hi * tokens + t) * d + x] for x in range(d))
            * scale
            for t in range(tokens)
        ]
        max_score = max(scores)
        weights = [math.exp(x - max_score) for x in scores]
        denom = sum(weights)
        out.extend(
            sum(weights[t] / denom * v[(hi * tokens + t) * d + x] for t in range(tokens))
            for x in range(d)
        )
    return out


def logical(session):
    c = session.c
    h, d, pt = c["num_kv_heads"], c["head_dim"], c["page_tokens"]
    pages = session.page_manager.block_table(session.request_id)
    k = []
    v = []
    for hi in range(h):
        for token in range(session.valid_tokens):
            page = pages[token // pt]
            base = ((page * h + hi) * pt + token % pt) * d
            k.extend(session.k[base : base + d])
            v.extend(session.v[base : base + d])
    return k, v


def req(name, prompt=4, output=1, phase="WAITING"):
    request = RequestExecutionState(
        name,
        f"serving-{name}",
        "replica-0",
        0,
        prompt,
        0,
        output,
        phase=phase,
    )
    if phase == "DECODE":
        request.prefill_completed_tokens = request.uncached_prompt_tokens
    return request


def profile(seqs=4, tokens=16, chunk=16):
    return SchedulerProfile(
        max_num_seqs=seqs,
        max_num_batched_tokens=tokens,
        max_prefill_chunk_tokens=chunk,
        balanced_decode_reservation=seqs,
    )


def state(manager, requests, scheduler_profile=None):
    s = ReplicaSchedulerState(
        "replica-0",
        scheduler_profile or profile(),
        page_manager=manager,
    )
    for request in requests:
        s.ingest(request)
    return s


def compile_plan(s, policy="prefill_first"):
    return SchedulerCompiler().compile(s, forced_policy=policy)


def execute_plan(s, plan, execute_item=None):
    runtime = PlanOnlySchedulerRuntime()
    return runtime.execute(
        s,
        deserialize_schedule_plan(plan.serialize(), s),
        execute_item,
    )


def test_prefill_admitted_when_physical_pages_exist():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    s = state(manager, [req("r1", prompt=4)])

    plan = compile_plan(s)
    assert [item.request_id for item in plan.items] == ["r1"]
    execute_plan(s, plan)
    assert manager.block_table("r1") == (0,)


def test_prefill_deferred_when_insufficient_free_pages():
    manager = KVPageManager(total_pages=1, tokens_per_page=4)
    manager.reserve_prefill("busy", 4)
    s = state(manager, [req("r1", prompt=4)])

    with pytest.raises(ServingPlanError, match="NO_READY_REQUESTS"):
        compile_plan(s)
    assert not manager.has_request("r1")
    assert s.kv_telemetry()["deferred_requests_due_to_kv_page_pressure"] > 0


def test_prompt_larger_than_pool_is_rejected_deterministically():
    manager = KVPageManager(total_pages=1, tokens_per_page=4)
    request = req("r1", prompt=5)
    s = state(manager, [request])

    with pytest.raises(ServingPlanError, match="NO_READY_REQUESTS"):
        compile_plan(s)
    assert request.phase == "FAILED"
    assert s.statistics["kv_prompt_too_large_rejections"] == 1


def test_existing_max_num_seqs_still_applies():
    manager = KVPageManager(total_pages=8, tokens_per_page=4)
    s = state(
        manager,
        [req("a", 4), req("b", 4), req("c", 4)],
        profile(seqs=2, tokens=16, chunk=16),
    )

    plan = compile_plan(s)
    assert len(plan.items) == 2


def test_existing_max_num_batched_tokens_still_applies():
    manager = KVPageManager(total_pages=8, tokens_per_page=4)
    s = state(
        manager,
        [req("a", 4), req("b", 4)],
        profile(seqs=4, tokens=4, chunk=4),
    )

    plan = compile_plan(s)
    assert len(plan.items) == 1
    assert plan.scheduled_tokens == 4


def test_page_availability_is_additional_constraint_not_replacement():
    manager = KVPageManager(total_pages=1, tokens_per_page=4)
    manager.reserve_prefill("busy", 4)
    s = state(manager, [req("r1", 4)], profile(seqs=4, tokens=16, chunk=16))

    with pytest.raises(ServingPlanError, match="NO_READY_REQUESTS"):
        compile_plan(s)


def test_two_requests_fit_and_third_defers_until_release():
    manager = KVPageManager(total_pages=3, tokens_per_page=4)
    s = state(manager, [req("a", 3), req("b", 8), req("c", 4)])

    plan = compile_plan(s)
    assert [item.request_id for item in plan.items] == ["a", "b"]
    execute_plan(s, plan)
    assert manager.num_allocated_pages() == 3
    assert not manager.has_request("c")
    assert s.requests["c"].phase == "PREFILL"

    s.requests["a"].phase = "DECODE"
    s.requests["a"].decode_completed_tokens = 0
    s.requests["a"].expected_output_tokens = 1
    plan = compile_plan(s, policy="decode_first")
    assert "a" in [item.request_id for item in plan.items]
    execute_plan(s, plan)
    assert not manager.has_request("a")

    plan = compile_plan(s)
    assert any(item.request_id == "c" for item in plan.items)


def test_decode_inside_allocated_page_runnable_with_no_free_pages():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 3)
    manager.reserve_prefill("busy", 4)
    r = req("r1", prompt=3, output=1, phase="DECODE")
    s = state(manager, [r])

    plan = compile_plan(s, policy="decode_first")
    assert [item.request_id for item in plan.items] == ["r1"]


def test_decode_boundary_deferred_until_page_becomes_free():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 4)
    manager.reserve_prefill("busy", 4)
    r = req("r1", prompt=4, output=1, phase="DECODE")
    s = state(manager, [r])

    with pytest.raises(ServingPlanError, match="NO_READY_REQUESTS"):
        compile_plan(s, policy="decode_first")
    assert s.kv_telemetry()["page_boundary_decode_deferrals"] > 0

    manager.release("busy")
    plan = compile_plan(s, policy="decode_first")
    assert [item.request_id for item in plan.items] == ["r1"]


def test_native_append_failure_does_not_advance_scheduler_or_leak_page(artifact):
    root, c = cfg(artifact, pages=2, pt=4, max_tokens=8)
    storage = PagedKVStorage(
        total_pages=2,
        num_kv_heads=2,
        tokens_per_page=4,
        head_dim=4,
        workspace_tokens=8,
    )
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    session = PagedKVAttentionSession(
        c,
        artifact_root=root,
        request_id="r1",
        storage=storage,
        page_manager=manager,
    )
    session.prefill(data(2 * 4 * 4, 1), data(2 * 4 * 4, 2), 4)
    session.app = lambda *args: _FailureStatus()
    r = req("r1", prompt=4, output=1, phase="DECODE")
    s = state(manager, [r])
    plan = compile_plan(s, policy="decode_first")
    before = manager.block_table("r1"), manager.valid_token_count("r1"), manager.num_free_pages()

    def execute_item(request, item, schedule):
        assert request.request_id == session.request_id
        session.append(data(8, 3), data(8, 4))
        return {"kv_page_manager_committed": True}

    with pytest.raises(AttentionContractError, match="injected_native_failure"):
        execute_plan(s, plan, execute_item)

    assert r.decode_completed_tokens == 0
    assert manager.block_table("r1") == before[0]
    assert manager.valid_token_count("r1") == before[1]
    assert manager.num_free_pages() == before[2]


def test_completed_request_releases_pages_once():
    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    manager.reserve_prefill("r1", 3)
    r = req("r1", prompt=3, output=1, phase="DECODE")
    s = state(manager, [r])
    runtime = PlanOnlySchedulerRuntime()
    plan = compile_plan(s, policy="decode_first")

    runtime.execute(s, deserialize_schedule_plan(plan.serialize(), s))

    assert r.phase == "FINISHED"
    assert not manager.has_request("r1")
    assert s.statistics["kv_release_events"] == 1
    with pytest.raises(ServingPlanError, match="unknown request completion"):
        runtime.release_kv_request(s, "r1")


def test_scheduler_request_id_matches_session_and_manager_with_native_execution(artifact):
    root, c = cfg(artifact)
    storage = PagedKVStorage(
        total_pages=4,
        num_kv_heads=2,
        tokens_per_page=4,
        head_dim=4,
        workspace_tokens=16,
    )
    manager = KVPageManager(total_pages=4, tokens_per_page=4)
    session = PagedKVAttentionSession(
        c,
        artifact_root=root,
        request_id="r1",
        storage=storage,
        page_manager=manager,
    )
    r = req("r1", prompt=3, output=1)
    s = state(manager, [r])

    def execute_item(request, item, schedule):
        assert request.request_id == session.request_id
        if item.phase == "prefill":
            session.prefill(data(2 * 3 * 4, 1), data(2 * 3 * 4, 2), 3)
        else:
            session.append(data(8, 3), data(8, 4))
            q = data(8, 5)
            got = session.decode(q)
            k, v = logical(session)
            assert list(got) == pytest.approx(ref(q, k, v, 2, 4, 4), abs=1e-6)
        return {"kv_page_manager_committed": True}

    plan = compile_plan(s)
    execute_plan(s, plan, execute_item)
    assert manager.block_table("r1") == (0,)
    assert manager.valid_token_count("r1") == 3

    plan = compile_plan(s, policy="decode_first")
    event = execute_plan(s, plan, execute_item)
    assert r.phase == "FINISHED"
    assert event["kv_telemetry"]["free_kv_pages"] == 4


def test_telemetry_reports_live_counts_and_legacy_path_is_unchanged():
    legacy = ReplicaSchedulerState("replica-0", profile())
    legacy.ingest(req("legacy", 4))
    assert legacy.kv_telemetry() == {"kv_page_manager_enabled": False}
    assert compile_plan(legacy).items

    manager = KVPageManager(total_pages=2, tokens_per_page=4)
    s = state(manager, [req("r1", 4)])
    execute_plan(s, compile_plan(s))
    telemetry = s.kv_telemetry()
    assert telemetry["kv_page_manager_enabled"] is True
    assert telemetry["total_kv_pages"] == 2
    assert telemetry["free_kv_pages"] == 1
    assert telemetry["allocated_kv_pages"] == 1
    assert telemetry["active_kv_requests"] == 1
