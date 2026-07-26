from array import array
import hashlib
import math
import subprocess
from pathlib import Path

import pytest

from deployment.execution_plan.attention_cpu_adapter import AttentionContractError
from deployment.execution_plan.kv_page_manager import KVPageManager
from deployment.execution_plan.paged_kv_cache import PagedKVAttentionSession, PagedKVStorage

ROOT = Path(__file__).resolve().parents[1]


class _FailureStatus:
    code = 7
    message = b"injected_native_failure"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("paged_shared_native")
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
        "kv_candidate_id": "cpu_paged_kv_fp32_page_major_v1",
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
        "paged_attention_kernel_id": "cpu_attention_decode_paged_kv_page_major_fp32",
        "implementation_strategy": "page_major_cached_page_base",
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


def make_shared(c):
    storage = PagedKVStorage(
        total_pages=c["num_physical_pages"],
        num_kv_heads=c["num_kv_heads"],
        tokens_per_page=c["page_tokens"],
        head_dim=c["head_dim"],
        dtype=c["dtype"],
        workspace_tokens=c["maximum_logical_tokens"],
    )
    manager = KVPageManager(
        total_pages=c["num_physical_pages"],
        tokens_per_page=c["page_tokens"],
    )
    return storage, manager


def make_session(root, c, request_id, storage, manager):
    return PagedKVAttentionSession(
        c,
        artifact_root=root,
        request_id=request_id,
        storage=storage,
        page_manager=manager,
    )


def assert_decode_matches(session, q):
    k, v = logical(session)
    got = session.decode(q)
    expected = ref(
        q,
        k,
        v,
        session.c["num_kv_heads"],
        session.valid_tokens,
        session.c["head_dim"],
    )
    assert list(got) == pytest.approx(expected, abs=1e-6)


def test_one_request_uses_manager_pages_and_crosses_boundary(artifact):
    root, c = cfg(artifact)
    storage, manager = make_shared(c)
    session = make_session(root, c, "r1", storage, manager)

    session.prefill(data(2 * 3 * 4, 1), data(2 * 3 * 4, 2), 3)
    assert manager.block_table("r1") == (0,)
    assert list(session.bt[:1]) == list(manager.block_table("r1"))
    assert_decode_matches(session, data(8, 3))

    session.append(data(8, 4), data(8, 5))
    assert manager.block_table("r1") == (0,)
    assert_decode_matches(session, data(8, 6))

    session.append(data(8, 7), data(8, 8))
    assert manager.block_table("r1") == (0, 1)
    assert_decode_matches(session, data(8, 9))
    assert list(session.physical_page_cache[:2]) == list(manager.block_table("r1"))


def test_two_requests_share_storage_and_later_request_reuses_released_page(artifact):
    root, c = cfg(artifact)
    storage, manager = make_shared(c)
    request_a = make_session(root, c, "a", storage, manager)
    request_b = make_session(root, c, "b", storage, manager)

    request_a.prefill(data(2 * 3 * 4, 10), data(2 * 3 * 4, 20), 3)
    request_b.prefill(data(2 * 5 * 4, 30), data(2 * 5 * 4, 40), 5)
    assert set(manager.block_table("a")).isdisjoint(manager.block_table("b"))
    assert manager.num_allocated_pages() == 3

    assert_decode_matches(request_a, data(8, 50))
    assert_decode_matches(request_b, data(8, 60))

    released = request_a.release()
    assert released is None
    assert not manager.has_request("a")
    assert manager.num_free_pages() == 2

    request_c = make_session(root, c, "c", storage, manager)
    request_c.prefill(data(2 * 1 * 4, 70), data(2 * 1 * 4, 80), 1)
    assert manager.block_table("c") == (0,)
    assert_decode_matches(request_c, data(8, 90))


def test_oom_before_prefill_leaves_manager_and_storage_unchanged(artifact):
    root, c = cfg(artifact, pages=1, pt=4, max_tokens=8)
    storage, manager = make_shared(c)
    request_a = make_session(root, c, "a", storage, manager)
    request_b = make_session(root, c, "b", storage, manager)
    request_a.prefill(data(2 * 4 * 4, 1), data(2 * 4 * 4, 2), 4)
    before = array("f", storage.k_pages), array("f", storage.v_pages), manager.block_table("a")

    with pytest.raises(AttentionContractError, match="out_of_physical_pages"):
        request_b.prefill(data(2 * 5 * 4, 3), data(2 * 5 * 4, 4), 5)

    assert not manager.has_request("b")
    assert manager.block_table("a") == before[2]
    assert storage.k_pages == before[0]
    assert storage.v_pages == before[1]


def test_native_prefill_failure_rolls_back_request_pages(artifact):
    root, c = cfg(artifact)
    storage, manager = make_shared(c)
    session = make_session(root, c, "r1", storage, manager)
    session.write = lambda *args: _FailureStatus()

    with pytest.raises(AttentionContractError, match="injected_native_failure"):
        session.prefill(data(2 * 3 * 4, 1), data(2 * 3 * 4, 2), 3)

    assert not manager.has_request("r1")
    assert manager.num_free_pages() == manager.total_pages
    manager.validate_invariants()


def test_native_boundary_append_failure_rolls_back_page_reservation(artifact):
    root, c = cfg(artifact, pages=2, pt=4, max_tokens=8)
    storage, manager = make_shared(c)
    session = make_session(root, c, "r1", storage, manager)
    session.prefill(data(2 * 4 * 4, 1), data(2 * 4 * 4, 2), 4)
    before_table = manager.block_table("r1")
    before_tokens = manager.valid_token_count("r1")
    before_free = manager.num_free_pages()
    session.app = lambda *args: _FailureStatus()

    with pytest.raises(AttentionContractError, match="injected_native_failure"):
        session.append(data(8, 3), data(8, 4))

    assert manager.block_table("r1") == before_table
    assert manager.valid_token_count("r1") == before_tokens
    assert manager.num_free_pages() == before_free
    manager.validate_invariants()


def test_session_execution_after_release_is_rejected(artifact):
    root, c = cfg(artifact)
    storage, manager = make_shared(c)
    session = make_session(root, c, "r1", storage, manager)
    session.prefill(data(2 * 2 * 4, 1), data(2 * 2 * 4, 2), 2)
    session.release()

    with pytest.raises(AttentionContractError):
        session.decode(data(8, 3))


def test_storage_manager_shape_mismatches_are_rejected(artifact):
    root, c = cfg(artifact)
    storage = PagedKVStorage(
        total_pages=3,
        num_kv_heads=2,
        tokens_per_page=4,
        head_dim=4,
        workspace_tokens=16,
    )
    manager = KVPageManager(total_pages=4, tokens_per_page=4)

    with pytest.raises(AttentionContractError, match="storage_manager_page_count_mismatch"):
        make_session(root, c, "r1", storage, manager)

    storage, _ = make_shared(c)
    manager = KVPageManager(total_pages=4, tokens_per_page=2)
    with pytest.raises(
        AttentionContractError,
        match="storage_manager_tokens_per_page_mismatch",
    ):
        make_session(root, c, "r1", storage, manager)
