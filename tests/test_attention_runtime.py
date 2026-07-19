import math

import pytest
import torch
import torch.nn.functional as F

from deployment.attention_runtime import (
    AttentionTrace, CompilerAttentionRuntime, ContiguousKVCache,
    ExecutionPlanAttentionAdapter, legal_attention_candidates,
    make_attention_plan, select_attention_plan, validate_attention_plan,
)
from deployment.attention_planner import (
    AttentionWorkload, emit_execution_plan,
    force_test_attention_plan,
    select_attention_plan as compiler_select_attention_plan,
    widen_context_domain,
)
from deployment.cpu_sharding import ShardingPlanError
from deployment.execution_plan.loader import (
    ExecutionPlanError, load_execution_plan, parse_execution_plan,
)
from tests.test_execution_plan_loader import _plan


torch.set_num_threads(1)


def tensors(q=7, context=7, heads=14, kv_heads=2, dim=64):
    g = torch.Generator().manual_seed(20260717 + q + context)
    return (
        torch.randn(1, heads, q, dim, generator=g),
        torch.randn(1, kv_heads, context, dim, generator=g),
        torch.randn(1, kv_heads, context, dim, generator=g),
    )


def reference(q, k, v, mask=None):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)


@pytest.mark.parametrize("strategy,workers", [
    ("serial", 1), ("split_head", 2), ("split_head", 4),
    ("split_head", 8), ("split_query", 2), ("split_query", 4),
    ("split_query", 8),
])
@pytest.mark.parametrize("qlen", [8, 11])
def test_prefill_candidates_match_complete_reference(strategy, workers, qlen):
    q, k, v = tensors(qlen, qlen)
    mask = torch.full((1, 1, qlen, qlen), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    with CompilerAttentionRuntime(make_attention_plan(
            phase="prefill", strategy=strategy, workers=workers)) as rt:
        got = rt.attention(q, k, v, mask, 1 / math.sqrt(64))
        assert rt.traces[-1].selected_candidate_id == rt.traces[-1].executed_candidate_id
    torch.testing.assert_close(got, reference(q, k, v, mask), rtol=2e-5, atol=2e-6)
    assert torch.isfinite(got).all()


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_decode_serial_and_split_head(workers):
    strategy = "serial" if workers == 1 else "split_head"
    q, k, v = tensors(1, 13)
    with CompilerAttentionRuntime(make_attention_plan(
            phase="decode", strategy=strategy, workers=workers)) as rt:
        got = rt.attention(q, k, v, None, 1 / math.sqrt(64))
    torch.testing.assert_close(got, reference(q, k, v), rtol=2e-5, atol=2e-6)


def test_legality_and_phase_selector():
    assert len(legal_attention_candidates(
        phase="prefill", batch=1, query_len=16, context_len=16,
        query_heads=14, kv_heads=2, head_dim=64)) == 7
    decode = legal_attention_candidates(
        phase="decode", batch=1, query_len=1, context_len=64,
        query_heads=14, kv_heads=2, head_dim=64)
    assert len(decode) == 4
    assert all(x["selected_strategy"] != "split_query" for x in decode)
    assert select_attention_plan(
        phase="decode", batch=1, query_len=1, context_len=64
    )["selected_strategy"] == "serial"
    bad = make_attention_plan(phase="prefill")
    bad["dtype"] = "bfloat16"
    with pytest.raises(ShardingPlanError, match="float32"):
        validate_attention_plan(bad)


def test_contiguous_cache_prefill_and_multiple_decode_steps():
    cache = ContiguousKVCache(2, 2, 2, 16, 64)
    _, k, v = tensors(4, 4)
    ck, cv = cache.append(1, 0, k, v)
    assert ck.shape == (1, 2, 4, 64)
    for step in range(5):
        _, nk, nv = tensors(1, 1)
        ck, cv = cache.append(1, 0, nk + step, nv - step)
        assert ck.shape[2] == 5 + step
    assert cache.lengths[1, 0] == 9


def test_repeated_mixed_prefill_decode_calls():
    pre = CompilerAttentionRuntime(make_attention_plan(
        phase="prefill", strategy="split_query", workers=2))
    dec = CompilerAttentionRuntime(make_attention_plan(
        phase="decode", strategy="split_head", workers=2))
    try:
        for qlen in (3, 5, 9):
            q, k, v = tensors(qlen, qlen)
            mask = torch.triu(torch.full((1, 1, qlen, qlen), float("-inf")), 1)
            torch.testing.assert_close(
                pre.attention(q, k, v, mask, 0.125),
                reference(q, k, v, mask), rtol=2e-5, atol=2e-6)
        for context in (3, 8, 17):
            q, k, v = tensors(1, context)
            torch.testing.assert_close(
                dec.attention(q, k, v, None, 0.125),
                reference(q, k, v), rtol=2e-5, atol=2e-6)
    finally:
        pre.close()
        dec.close()


def test_execution_plan_attention_round_trip_and_legacy_compatibility():
    legacy = parse_execution_plan(_plan())
    assert legacy.global_decisions.attention_execution == {}
    payload = _plan()
    decision = make_attention_plan(
        phase="prefill", strategy="split_head", workers=2,
        provenance="cost_model_selected")
    payload["global_decisions"]["attention_execution"] = decision
    parsed = parse_execution_plan(payload)
    assert parsed.global_decisions.attention_execution == decision


def compiler_plan_payload():
    pre, _ = compiler_select_attention_plan(AttentionWorkload(
        phase="prefill", batch=1, query_len=4, context_len=4,
        query_heads=14, kv_heads=2, head_dim=64))
    dec, _ = compiler_select_attention_plan(AttentionWorkload(
        phase="decode", batch=1, query_len=1, context_len=5,
        query_heads=14, kv_heads=2, head_dim=64))
    dec = widen_context_domain(dec, 5, 11)
    return emit_execution_plan(
        plan_id="attention-test", model_id="Qwen-test", prompt_tokens=4,
        generated_tokens=8, prefill=pre, decode=dec)


def test_compiler_selector_scores_all_legal_candidates_and_rejects_decode_split_query():
    _, pre = compiler_select_attention_plan(AttentionWorkload(
        phase="prefill", batch=1, query_len=128, context_len=128,
        query_heads=14, kv_heads=2, head_dim=64))
    assert pre["generated_candidate_count"] == 12
    assert pre["legal_candidate_count"] >= 6
    assert all("score" in x for x in pre["considered_candidates"] if x["legal"])
    assert pre["selected_candidate_id"] != "torch_cpu_attention_fp32_serial_w1_v1"
    _, dec = compiler_select_attention_plan(AttentionWorkload(
        phase="decode", batch=1, query_len=1, context_len=128,
        query_heads=14, kv_heads=2, head_dim=64))
    assert dec["legal_candidate_count"] >= 4
    assert any(x["implementation"] == "native_avx2" and x["legal"]
               for x in dec["considered_candidates"])


def test_compiler_attention_execution_plan_file_round_trip(tmp_path):
    payload = compiler_plan_payload()
    path = tmp_path / "attention-plan.json"
    path.write_text(__import__("json").dumps(payload))
    loaded = load_execution_plan(path)
    table = loaded.global_decisions.attention_execution
    assert table == payload["global_decisions"]["attention_execution"]
    for phase in ("prefill", "decode"):
        decision = table["phase_decisions"][phase]
        assert decision["selection_mode"] == "compiler_selected"
        assert decision["native_kernel_id"] == decision["selection_trace"]["selected_candidate_id"]


def test_model_adapter_requires_compiler_plan_table():
    with pytest.raises(ShardingPlanError, match="decision_kind mismatch"):
        ExecutionPlanAttentionAdapter(parse_execution_plan(_plan()))


def test_loader_rejects_missing_compiler_attention_candidate():
    payload = compiler_plan_payload()
    del payload["global_decisions"]["attention_execution"]["phase_decisions"]["prefill"]["native_kernel_id"]
    with pytest.raises(ExecutionPlanError, match="native_kernel_id"):
        parse_execution_plan(payload)


def test_plan_adapter_consumes_deserialized_decision_without_selector_recomputation(
        tmp_path, monkeypatch):
    payload = compiler_plan_payload()
    path = tmp_path / "attention-plan.json"
    path.write_text(__import__("json").dumps(payload))
    loaded = load_execution_plan(path)
    monkeypatch.setattr(
        "deployment.attention_planner.select_attention_plan",
        lambda *_args, **_kwargs: pytest.fail("runtime recomputed selection"))
    adapter = ExecutionPlanAttentionAdapter(loaded)
    try:
        q, k, v = tensors(4, 4)
        mask = torch.triu(torch.full((1, 1, 4, 4), float("-inf")), 1)
        class Module:
            layer_idx = 0
        got = adapter.attention(Module(), q, k, v, mask, 0.125)
        assert got.shape == q.shape
        assert adapter.provenance[-1]["selection_mode"] == "compiler_selected"
        assert adapter.provenance[-1]["candidate_mismatch"] is False
    finally:
        adapter.close()


def test_plan_adapter_rejects_candidate_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "attention-plan.json"
    path.write_text(__import__("json").dumps(compiler_plan_payload()))
    adapter = ExecutionPlanAttentionAdapter(load_execution_plan(path))
    runtime = adapter.runtimes["prefill"]
    original = runtime.attention
    def mismatching(*args, **kwargs):
        out = original(*args, **kwargs)
        runtime.traces[-1].executed_candidate_id = "wrong-candidate"
        return out
    monkeypatch.setattr(runtime, "attention", mismatching)
    try:
        q, k, v = tensors(4, 4)
        mask = torch.triu(torch.full((1, 1, 4, 4), float("-inf")), 1)
        with pytest.raises(ShardingPlanError, match="candidates differ"):
            adapter.attention(type("M", (), {"layer_idx": 0})(), q, k, v, mask, 0.125)
    finally:
        adapter.close()


@pytest.mark.parametrize("phase,qlen,context,query_tile,key_tile", [
    ("prefill", 11, 11, 1, 32),
    ("prefill", 11, 11, 4, 32),
    ("prefill", 37, 37, 8, 64),
    ("decode", 1, 17, 1, 32),
    ("decode", 1, 63, 1, 32),
])
def test_fused_online_matches_dense_complete_tensor(
        phase, qlen, context, query_tile, key_tile):
    q, k, v = tensors(qlen, context)
    mask = None if phase == "decode" else torch.triu(
        torch.full((1, 1, qlen, context), float("-inf")), 1)
    with CompilerAttentionRuntime(make_attention_plan(phase=phase)) as dense:
        expected = dense.attention(q, k, v, mask, 0.125)
    with CompilerAttentionRuntime(make_attention_plan(
            phase=phase, algorithm="fused_tiled_online_softmax",
            query_tile=query_tile, key_tile=key_tile,
            implementation="torch_tiled_online_softmax_exact_v1")) as fused:
        got = fused.attention(q, k, v, mask, 0.125)
        memory = fused.traces[-1].memory
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
    assert torch.isfinite(got).all()
    assert not memory["full_score_materialized"]
    assert not memory["full_probability_materialized"]
    assert memory["score_bytes"] == memory["probability_bytes"] == 0


def test_online_running_max_rescaling_across_key_tiles():
    q = torch.zeros(1, 14, 1, 64)
    k = torch.zeros(1, 2, 33, 64)
    v = torch.zeros(1, 2, 33, 64)
    q[..., 0] = 1
    k[:, :, :32, 0] = -10
    k[:, :, 32, 0] = 10
    v[:, :, :32] = 1
    v[:, :, 32] = 7
    with CompilerAttentionRuntime(make_attention_plan(
            phase="decode", algorithm="fused_tiled_online_softmax",
            query_tile=1, key_tile=32,
            implementation="torch_tiled_online_softmax_exact_v1")) as runtime:
        got = runtime.attention(q, k, v, None, 1.0)
    assert torch.allclose(got, torch.full_like(got, 7.0), atol=2e-6)


def test_fused_path_does_not_call_dense_softmax_or_sdpa(monkeypatch):
    monkeypatch.setattr(
        "deployment.attention_runtime._attention_chunk",
        lambda *_a, **_k: pytest.fail("fused delegated to dense helper"))
    monkeypatch.setattr(
        torch, "softmax", lambda *_a, **_k: pytest.fail("fused called torch.softmax"))
    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention",
        lambda *_a, **_k: pytest.fail("fused called SDPA"))
    q, k, v = tensors(1, 17)
    with CompilerAttentionRuntime(make_attention_plan(
            phase="decode", algorithm="fused_tiled_online_softmax",
            query_tile=1, key_tile=32,
            implementation="torch_tiled_online_softmax_exact_v1")) as runtime:
        assert torch.isfinite(runtime.attention(q, k, v, None, 0.125)).all()


def test_fused_repeated_1000_invocations():
    q, k, v = tensors(1, 5)
    with CompilerAttentionRuntime(make_attention_plan(
            phase="decode", algorithm="fused_tiled_online_softmax",
            query_tile=1, key_tile=32,
            implementation="torch_tiled_online_softmax_exact_v1")) as runtime:
        for _ in range(1000):
            got = runtime.attention(q, k, v, None, 0.125)
    assert len(runtime.traces) == 1000
    assert torch.isfinite(got).all()


def test_fused_candidate_legality_rejects_bad_tiles_and_split_query():
    with pytest.raises(ShardingPlanError, match="query_tile"):
        make_attention_plan(
            phase="prefill", algorithm="fused_tiled_online_softmax",
            query_tile=0, key_tile=32)
    with pytest.raises(ShardingPlanError, match="split_query"):
        make_attention_plan(
            phase="prefill", strategy="split_query", workers=2,
            algorithm="fused_tiled_online_softmax", query_tile=4, key_tile=32)


def test_fused_large_prefill_reduces_tracked_temporary_bytes():
    q, k, v = tensors(128, 128)
    mask = torch.triu(torch.full((1, 1, 128, 128), float("-inf")), 1)
    with CompilerAttentionRuntime(make_attention_plan(phase="prefill")) as dense:
        dense.attention(q, k, v, mask, 0.125)
        dense_bytes = dense.traces[-1].memory["temporary_bytes"]
    with CompilerAttentionRuntime(make_attention_plan(
            phase="prefill", algorithm="fused_tiled_online_softmax",
            query_tile=8, key_tile=64,
            implementation="torch_tiled_online_softmax_exact_v1")) as fused:
        fused.attention(q, k, v, mask, 0.125)
        fused_bytes = fused.traces[-1].memory["temporary_bytes"]
    assert fused_bytes < dense_bytes / 10


def test_fused_execution_plan_round_trip_and_exact_dispatch(tmp_path):
    pre_workload = AttentionWorkload(
        phase="prefill", batch=1, query_len=4, context_len=4,
        query_heads=14, kv_heads=2, head_dim=64)
    dec_workload = AttentionWorkload(
        phase="decode", batch=1, query_len=1, context_len=5,
        query_heads=14, kv_heads=2, head_dim=64)
    pre, _ = force_test_attention_plan(
        pre_workload, algorithm="fused_tiled_online_softmax",
        strategy="serial", workers=1, query_tile=4, key_tile=32)
    dec, _ = force_test_attention_plan(
        dec_workload, algorithm="fused_tiled_online_softmax",
        strategy="serial", workers=1, query_tile=1, key_tile=32)
    dec = widen_context_domain(dec, 5, 11)
    payload = emit_execution_plan(
        plan_id="forced-fused-test", model_id="Qwen-test", prompt_tokens=4,
        generated_tokens=8, prefill=pre, decode=dec)
    path = tmp_path / "fused-plan.json"
    path.write_text(__import__("json").dumps(payload))
    loaded = load_execution_plan(path)
    adapter = ExecutionPlanAttentionAdapter(loaded)
    try:
        q, k, v = tensors(4, 4)
        mask = torch.triu(torch.full((1, 1, 4, 4), float("-inf")), 1)
        adapter.attention(type("M", (), {"layer_idx": 0})(), q, k, v, mask, 0.125)
        trace = adapter.provenance[-1]
        assert trace["selection_mode"] == "forced_test_override"
        assert trace["algorithm"] == trace["serialized_algorithm"]
        assert trace["serialized_algorithm"] == trace["executed_algorithm"]
        assert trace["executed_algorithm"] == "fused_tiled_online_softmax"
        assert trace["candidate_mismatch"] is False
    finally:
        adapter.close()
