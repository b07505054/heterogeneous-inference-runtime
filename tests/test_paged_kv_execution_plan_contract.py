from array import array
import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

from deployment.execution_plan.loader import ExecutionPlanError, parse_execution_plan
from deployment.execution_plan.paged_kv_runtime import (
    PagedKVRuntimeContractError,
    build_paged_kv_runtime,
    paged_kv_contracts,
)
from deployment.serving_scheduler import SchedulerProfile
from tests.test_execution_plan_loader import _plan

ROOT = Path(__file__).resolve().parents[1]
PAGED_PLAN = (
    ROOT
    / "artifacts"
    / "kv_selection_evaluation"
    / "raspberry_pi"
    / "compiler_plans"
    / "paged8.json"
)
PAGED_PAGE_MAJOR_PLAN = PAGED_PLAN.with_name("paged8_page_major.json")


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("paged_kv_plan_runtime")
    native_dir = root / "native"
    native_dir.mkdir()
    so = native_dir / "libattention_fp32.so"
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
    return root, "native/libattention_fp32.so", hashlib.sha256(so.read_bytes()).hexdigest()


def compiler_plan(native_artifact):
    root, ref, digest = native_artifact
    payload = json.loads(PAGED_PLAN.read_text())
    for contract in paged_contract_payloads(payload):
        contract["pool_artifact_ref"] = ref
        contract["pool_artifact_sha256"] = digest
    return root, payload


def paged_contract_payloads(payload):
    out = []
    for function in payload["function_plans"]:
        for op in function["per_op_decisions"]:
            if "paged_kv_execution" in op:
                out.append(op["paged_kv_execution"])
    return out


def first_contract_payload(payload):
    return paged_contract_payloads(payload)[0]


def data(n, seed=0):
    return array("f", ((i + seed) % 17 / 17 - 0.5 for i in range(n)))


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


def assert_decode_matches(session, q):
    k, v = logical(session)
    assert list(session.decode(q)) == pytest.approx(
        ref(q, k, v, session.c["num_kv_heads"], session.valid_tokens, session.c["head_dim"]),
        abs=1e-6,
    )


def test_loads_compiler_style_paged_kv_contract(native_artifact):
    _, payload = compiler_plan(native_artifact)
    plan = parse_execution_plan(payload)
    contracts = paged_kv_contracts(plan)

    assert len(contracts) == 2
    assert contracts[0].kv_layout_kind == "paged_phd_contiguous"
    assert contracts[0].page_tokens == 8
    assert plan.function_plans[0].per_op_decisions[-1].paged_kv_execution == contracts[0]


def test_checked_in_compiler_paged_fixture_contains_decode_entry_point():
    payload = json.loads(PAGED_PLAN.read_text())
    plan = parse_execution_plan(payload)
    contracts = paged_kv_contracts(plan)

    assert len(contracts) == 2
    assert {
        contract.paged_attention_entry_point for contract in contracts
    } == {"hir_cpu_attention_decode_paged_kv_fp32"}


@pytest.mark.parametrize(
    "field,error",
    [
        ("page_tokens", "missing required field: page_tokens"),
        ("num_physical_pages", "missing required field: num_physical_pages"),
        ("prefill_write_entry_point", "missing required field: prefill_write_entry_point"),
        ("paged_attention_entry_point", "missing required field: paged_attention_entry_point"),
    ],
)
def test_rejects_missing_required_paged_kv_fields(native_artifact, field, error):
    _, payload = compiler_plan(native_artifact)
    del first_contract_payload(payload)[field]

    with pytest.raises(ExecutionPlanError, match=error):
        parse_execution_plan(payload)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("page_tokens", 0, "page_tokens must be > 0"),
        ("num_physical_pages", 0, "num_physical_pages must be > 0"),
        ("bytes_per_token", 64, "bytes_per_token formula mismatch"),
        ("bytes_per_k_page", 256, "bytes_per_k_page formula mismatch"),
        ("block_table_element_type", "int64", "block_table_element_type must be int32"),
        ("prefill_write_entry_point", "", "prefill_write_entry_point must be"),
        ("contiguous_fallback_identity", "runtime_may_choose_anything", "unsupported fallback"),
    ],
)
def test_rejects_invalid_paged_kv_contract_values(native_artifact, field, value, error):
    _, payload = compiler_plan(native_artifact)
    first_contract_payload(payload)[field] = value

    with pytest.raises(ExecutionPlanError, match=error):
        parse_execution_plan(payload)


def test_rejects_unsupported_query_head_relationship(native_artifact):
    _, payload = compiler_plan(native_artifact)
    first_contract_payload(payload)["num_query_heads"] = 4

    with pytest.raises(ExecutionPlanError, match="num_query_heads must equal"):
        parse_execution_plan(payload)


def test_runtime_factory_constructs_exact_manager_storage_and_sessions(native_artifact):
    artifact_root, payload = compiler_plan(native_artifact)
    context = build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)

    assert context.page_manager.total_pages == 8
    assert context.page_manager.tokens_per_page == 8
    assert context.storage.total_pages == 8
    assert context.storage.tokens_per_page == 8
    assert context.storage.num_kv_heads == 2
    assert context.storage.head_dim == 8
    assert context.storage.dtype == "fp32"
    assert len(context.storage.k_pages) == 8 * 2 * 8 * 8
    assert len(context.storage.workspace) == 64

    a = context.create_session("a")
    b = context.create_session("b")
    assert a.request_id == "a"
    assert b.request_id == "b"
    assert a.page_manager is b.page_manager is context.page_manager
    assert a.storage is b.storage is context.storage


def test_scheduler_receives_same_manager_instance(native_artifact):
    artifact_root, payload = compiler_plan(native_artifact)
    context = build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)
    state = context.scheduler_state("replica-0", SchedulerProfile())

    assert state.page_manager is context.page_manager


def test_native_session_binds_exact_compiler_selected_entry_points(native_artifact):
    artifact_root, payload = compiler_plan(native_artifact)
    context = build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)
    session = context.create_session("r1")

    assert session.write.__name__ == context.contract.prefill_write_entry_point
    assert session.app.__name__ == context.contract.append_entry_point
    assert session.dec.__name__ == context.contract.paged_attention_entry_point


def test_page_major_compiler_fixture_binds_exact_optimized_entry_point(native_artifact):
    artifact_root, ref, digest = native_artifact
    payload = json.loads(PAGED_PAGE_MAJOR_PLAN.read_text())
    for contract in paged_contract_payloads(payload):
        contract["pool_artifact_ref"] = ref
        contract["pool_artifact_sha256"] = digest
    context = build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)
    session = context.create_session("page-major")

    assert context.contract.kv_candidate_id == "cpu_paged_kv_fp32_page_major_v1"
    assert context.contract.implementation_strategy == "page_major_cached_page_base"
    assert session.dec.__name__ == "hir_cpu_attention_decode_paged_kv_page_major_fp32"


def test_runtime_no_redecision_does_not_call_attention_selector(native_artifact, monkeypatch):
    import deployment.attention_runtime as attention_runtime

    def fail(*args, **kwargs):
        raise AssertionError("runtime attention selector was called")

    monkeypatch.setattr(attention_runtime, "select_attention_plan", fail)
    artifact_root, payload = compiler_plan(native_artifact)
    context = build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)

    assert context.contract.runtime_no_layout_redecision is True
    assert context.contract.runtime_no_kernel_redecision is True


def test_loaded_runtime_executes_native_decode_and_crosses_boundary(native_artifact):
    artifact_root, payload = compiler_plan(native_artifact)
    context = build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)
    session = context.create_session("r1")
    h, d = session.c["num_kv_heads"], session.c["head_dim"]

    session.prefill(data(h * 7 * d, 1), data(h * 7 * d, 2), 7)
    assert context.page_manager.block_table("r1") == (0,)
    assert_decode_matches(session, data(h * d, 3))

    session.append(data(h * d, 4), data(h * d, 5))
    assert context.page_manager.block_table("r1") == (0,)
    assert_decode_matches(session, data(h * d, 6))

    session.append(data(h * d, 7), data(h * d, 8))
    assert context.page_manager.block_table("r1") == (0, 1)
    assert_decode_matches(session, data(h * d, 9))


def test_scheduler_completion_releases_pages_from_loaded_runtime(native_artifact):
    from deployment.serving_scheduler import (
        PlanOnlySchedulerRuntime,
        RequestExecutionState,
        SchedulerCompiler,
        deserialize_schedule_plan,
    )

    artifact_root, payload = compiler_plan(native_artifact)
    context = build_paged_kv_runtime(parse_execution_plan(payload), artifact_root)
    session = context.create_session("r1")
    h, d = session.c["num_kv_heads"], session.c["head_dim"]
    state = context.scheduler_state("replica-0", SchedulerProfile())
    request = RequestExecutionState("r1", "serving-r1", "replica-0", 0, 3, 0, 1)
    state.ingest(request)
    runtime = PlanOnlySchedulerRuntime()

    def execute_item(req, item, plan):
        if item.phase == "prefill":
            session.prefill(data(h * 3 * d, 1), data(h * 3 * d, 2), 3)
        else:
            session.append(data(h * d, 3), data(h * d, 4))
            assert_decode_matches(session, data(h * d, 5))
        return {"kv_page_manager_committed": True}

    plan = SchedulerCompiler().compile(state, forced_policy="prefill_first")
    runtime.execute(state, deserialize_schedule_plan(plan.serialize(), state), execute_item)
    assert context.page_manager.has_request("r1")

    plan = SchedulerCompiler().compile(state, forced_policy="decode_first")
    runtime.execute(state, deserialize_schedule_plan(plan.serialize(), state), execute_item)
    assert request.finished
    assert not context.page_manager.has_request("r1")
    assert state.kv_telemetry()["free_kv_pages"] == context.contract.num_physical_pages


def test_runtime_factory_rejects_manager_storage_mismatch(native_artifact):
    artifact_root, payload = compiler_plan(native_artifact)
    plan = parse_execution_plan(payload)
    context = build_paged_kv_runtime(plan, artifact_root)
    context.storage.total_pages = 7

    with pytest.raises(AssertionError):
        assert context.page_manager.total_pages == context.storage.total_pages


def test_preserves_legacy_plan_without_paged_kv_contract():
    plan = parse_execution_plan(_plan())

    assert paged_kv_contracts(plan) == ()
    with pytest.raises(PagedKVRuntimeContractError, match="missing_paged_kv"):
        build_paged_kv_runtime(plan, ROOT)


def test_rejects_missing_artifact(native_artifact):
    artifact_root, payload = compiler_plan(native_artifact)
    for contract in paged_contract_payloads(payload):
        contract["pool_artifact_ref"] = "native/missing.so"
    plan = parse_execution_plan(payload)

    with pytest.raises(PagedKVRuntimeContractError, match="artifact_not_found"):
        build_paged_kv_runtime(plan, artifact_root)
