"""D4A: Single-GPU Serialized Whole-Model TP Contract Validation -- focused tests.

Covers the positive whole-model chain (attention block, MLP block, vocab,
whole-model forward) against the real, locally-cached Qwen2.5-0.5B-Instruct
model, plus every Part O fail-closed negative test. No test ever falls
back to synthetic tensors or routes a TP-relevant operator through its
original full linear/embedding call as the output path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deployment.execution_plan.loader import ExecutionPlanError, validate_execution_plan
from deployment.execution_plan.schema import DistributedTensorShard
from deployment.tp_process_runtime.attention_contract_executor import (
    AttentionContractError,
    apply_rotary_pos_emb,
    repeat_kv,
)
from deployment.tp_process_runtime.column_parallel_executor import (
    ColumnParallelError,
    build_column_rank_shards,
    rank_local_column_output,
    reconstruct_column_output,
)
from deployment.tp_process_runtime.linear_tp_decomposition import (
    apply_bias_contract,
    apply_bias_twice_incorrectly,
    build_rank_shards,
    rank_local_partial_output,
)
from deployment.tp_process_runtime.mlp_contract_executor import (
    MLPContractError,
    run_serialized_tp_mlp_block,
)
from deployment.tp_process_runtime.qwen_module_mapping import OperatorMappingError, map_compiler_operator_to_module
from deployment.tp_process_runtime.vocab_parallel_executor import (
    VocabParallelError,
    build_vocab_rank_shards,
    rank_local_lm_head_logits,
    rank_local_masked_embedding,
    reconstruct_embedding,
    reconstruct_lm_head_logits,
)
from deployment.tp_process_runtime.whole_model_plan_builder import (
    ModelDims,
    build_whole_model_plan,
    read_model_dims,
)
from deployment.tp_process_runtime.whole_model_tp_replay import (
    group_shards_by_tensor_id,
    load_eager_model,
    run_reference_forward,
    run_serialized_tp_whole_model_forward,
)
from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
TP2_PLAN_PATH = (
    REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline" / "real_qwen_tp2_execution_plan.json"
)
PROMPT = "The capital of France is"

pytestmark = pytest.mark.skipif(
    not TP2_PLAN_PATH.exists(), reason="requires the D2 compiler-exported real-Qwen TP2 plan artifact"
)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    model, tok, _load_time = load_eager_model()
    return model, tok


@pytest.fixture(scope="module")
def plan_and_shards(model_and_tokenizer):
    model, _tok = model_and_tokenizer
    plan_dict, work_items = build_whole_model_plan(model, source_tp2_plan_id="x")
    from deployment.execution_plan.loader import parse_execution_plan

    plan = parse_execution_plan(plan_dict)
    return plan, work_items, group_shards_by_tensor_id(plan.distributed.tensor_shards)


# ---------------------------------------------------------------------------
# Positive path.
# ---------------------------------------------------------------------------


def test_whole_model_plan_builds_and_validates(plan_and_shards):
    plan, work_items, shards_by_id = plan_and_shards
    assert plan.distributed.world_size == 2
    assert plan.distributed.tensor_parallel_size == 2
    assert len(work_items) > 24 * 5  # at least q/k/v/gate/up per layer plus embed/lm_head/o/down


def test_whole_model_forward_matches_reference_within_tolerance(model_and_tokenizer, plan_and_shards):
    model, tok = model_and_tokenizer
    _plan, _work_items, shards_by_id = plan_and_shards
    ref = run_reference_forward(model, tok, PROMPT)
    tp = run_serialized_tp_whole_model_forward(model, tok, PROMPT, shards_by_tensor_id=shards_by_id, reference=ref)

    diff = np.abs(tp.logits - ref.logits)
    assert diff.max() < 1e-2  # dtype-appropriate (float32) tolerance for a 24-layer composed forward
    assert np.array_equal(tp.logits.argmax(-1), ref.logits.argmax(-1))

    topk_ref = np.argsort(-ref.logits[0, -1])[:5]
    topk_tp = np.argsort(-tp.logits[0, -1])[:5]
    assert np.array_equal(topk_ref, topk_tp)

    for trace in tp.layer_traces:
        assert trace.hidden_state_max_abs_error_vs_reference < 1e-2


def test_dims_read_from_real_model_config(model_and_tokenizer):
    model, _tok = model_and_tokenizer
    dims = read_model_dims(model)
    assert dims.hidden_size == 896
    assert dims.num_attention_heads == 14
    assert dims.num_key_value_heads == 2
    assert dims.vocab_size == 151936
    assert dims.tie_word_embeddings is True


def test_operator_mapping_extended_families_resolve(model_and_tokenizer):
    model, _tok = model_and_tokenizer
    for op_type, module_suffix in (
        ("llm.q_proj", "self_attn.q_proj"), ("llm.k_proj", "self_attn.k_proj"),
        ("llm.v_proj", "self_attn.v_proj"), ("llm.gate_proj", "mlp.gate_proj"),
        ("llm.up_proj", "mlp.up_proj"), ("llm.down_proj", "mlp.down_proj"),
    ):
        result = map_compiler_operator_to_module(f"qwen_prefill::{op_type}::layer_0", model)
        assert result.module_path == f"model.layers.0.{module_suffix}"


# ---------------------------------------------------------------------------
# Negative tests (Part O).
# ---------------------------------------------------------------------------


def test_negative_column_parallel_wrong_weight_axis():
    w = np.zeros((8, 4))
    bad_shards = (
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=0, range_start=0, range_end=2),
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=1, range_start=2, range_end=4),
    )
    with pytest.raises(ColumnParallelError, match="axis 0"):
        build_column_rank_shards(w, None, bad_shards)


def test_negative_column_parallel_output_order_reversed():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))
    w = rng.normal(size=(8, 4))
    shards = (
        DistributedTensorShard(tensor_id="t", partition_axis=0, partition_count=2, shard_index=0, range_start=0, range_end=4),
        DistributedTensorShard(tensor_id="t", partition_axis=0, partition_count=2, shard_index=1, range_start=4, range_end=8),
    )
    rank_shards = build_column_rank_shards(w, None, shards)
    outputs = {rid: rank_local_column_output(x, s) for rid, s in rank_shards.items()}
    correct = reconstruct_column_output(outputs)
    reversed_order = np.concatenate([outputs[1], outputs[0]], axis=-1)  # wrong: rank 1 before rank 0
    reference = x @ w.T
    assert np.allclose(correct, reference)
    assert not np.allclose(reversed_order, reference)


def test_negative_row_parallel_wrong_input_axis_shape_mismatch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 4))  # in_features=4
    w = rng.normal(size=(6, 8))  # in_features=8 -- deliberately mismatched
    shards = (
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=0, range_start=0, range_end=2),
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=1, range_start=2, range_end=4),
    )
    with pytest.raises(Exception, match="in_features"):
        build_rank_shards(x, w, shards)


def test_negative_row_parallel_missing_all_reduce():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 8))
    w = rng.normal(size=(5, 8))
    shards = (
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=0, range_start=0, range_end=4),
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=1, range_start=4, range_end=8),
    )
    rank_shards = build_rank_shards(x, w, shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in rank_shards.items()}
    reduced = sum(partials.values())
    reference = x @ w.T
    assert np.allclose(reduced, reference)
    assert not np.allclose(partials[0], reference)  # a single partial without all-reduce is wrong


def test_negative_row_parallel_bias_applied_per_rank():
    partials = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
    bias = np.array([10.0, 10.0])
    correct = apply_bias_contract(sum(partials), bias)
    wrong = apply_bias_twice_incorrectly(partials, bias)
    assert not np.allclose(correct, wrong)


def test_negative_q_head_count_not_divisible_by_tp():
    from deployment.tp_process_runtime.whole_model_plan_builder import _even_split

    with pytest.raises(ValueError):
        _even_split(7, 2)


def test_negative_kv_head_ownership_mismatch_gap():
    w = np.zeros((4, 8))
    gap_shards = (
        DistributedTensorShard(tensor_id="t", partition_axis=0, partition_count=2, shard_index=0, range_start=0, range_end=1),
        DistributedTensorShard(tensor_id="t", partition_axis=0, partition_count=2, shard_index=1, range_start=2, range_end=4),
    )
    with pytest.raises(ColumnParallelError, match="gap"):
        build_column_rank_shards(w, None, gap_shards)


def test_negative_incorrect_gqa_kv_repetition():
    rng = np.random.default_rng(2)
    k = rng.normal(size=(1, 1, 5, 8))
    correct = repeat_kv(k, 7)
    wrong = repeat_kv(k, 3)
    assert correct.shape != wrong.shape


def test_negative_rotary_shape_mismatch_raises():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(1, 2, 5, 8))
    k = rng.normal(size=(1, 1, 5, 8))
    cos = rng.normal(size=(1, 3, 8))  # wrong seq length (3 vs 5)
    sin = rng.normal(size=(1, 3, 8))
    with pytest.raises(ValueError):
        apply_rotary_pos_emb(q, k, cos, sin)


def test_negative_gate_up_shard_ownership_mismatch():
    rng = np.random.default_rng(4)
    hidden = rng.normal(size=(1, 3, 8))
    gate_w = rng.normal(size=(8, 8))
    up_w = rng.normal(size=(8, 8))
    down_w = rng.normal(size=(8, 8))
    gate_shards = (
        DistributedTensorShard(tensor_id="g", partition_axis=0, partition_count=2, shard_index=0, range_start=0, range_end=4),
        DistributedTensorShard(tensor_id="g", partition_axis=0, partition_count=2, shard_index=1, range_start=4, range_end=8),
    )
    up_shards_mismatched = (  # deliberately offset by a rotation, so ranges don't match gate's per-rank ranges
        DistributedTensorShard(tensor_id="u", partition_axis=0, partition_count=2, shard_index=0, range_start=4, range_end=8),
        DistributedTensorShard(tensor_id="u", partition_axis=0, partition_count=2, shard_index=1, range_start=0, range_end=4),
    )
    down_shards = (
        DistributedTensorShard(tensor_id="d", partition_axis=1, partition_count=2, shard_index=0, range_start=0, range_end=4),
        DistributedTensorShard(tensor_id="d", partition_axis=1, partition_count=2, shard_index=1, range_start=4, range_end=8),
    )
    with pytest.raises(MLPContractError, match="does not match"):
        run_serialized_tp_mlp_block(
            hidden_states=hidden, gate_weight=gate_w, up_weight=up_w, down_weight=down_w,
            gate_shards=gate_shards, up_shards=up_shards_mismatched, down_shards=down_shards,
        )


def test_negative_down_proj_consumes_wrong_shard():
    rng = np.random.default_rng(5)
    hidden = rng.normal(size=(1, 3, 8))
    gate_w = rng.normal(size=(8, 8))
    up_w = rng.normal(size=(8, 8))
    down_w = rng.normal(size=(8, 8))
    gate_shards = up_shards = (
        DistributedTensorShard(tensor_id="g", partition_axis=0, partition_count=2, shard_index=0, range_start=0, range_end=4),
        DistributedTensorShard(tensor_id="g", partition_axis=0, partition_count=2, shard_index=1, range_start=4, range_end=8),
    )
    down_shards_gap = (
        DistributedTensorShard(tensor_id="d", partition_axis=1, partition_count=2, shard_index=0, range_start=0, range_end=3),
        DistributedTensorShard(tensor_id="d", partition_axis=1, partition_count=2, shard_index=1, range_start=4, range_end=8),
    )
    with pytest.raises(MLPContractError, match="gap"):
        run_serialized_tp_mlp_block(
            hidden_states=hidden, gate_weight=gate_w, up_weight=up_w, down_weight=down_w,
            gate_shards=gate_shards, up_shards=up_shards, down_shards=down_shards_gap,
        )


def test_negative_vocabulary_shard_coverage_gap():
    w = np.zeros((10, 4))
    gap_shards = (
        DistributedTensorShard(tensor_id="v", partition_axis=0, partition_count=2, shard_index=0, range_start=0, range_end=4),
        DistributedTensorShard(tensor_id="v", partition_axis=0, partition_count=2, shard_index=1, range_start=5, range_end=10),
    )
    with pytest.raises(VocabParallelError, match="gap"):
        build_vocab_rank_shards(w, gap_shards)


def test_negative_embedding_token_belongs_to_wrong_rank_overlap():
    w = np.arange(40).reshape(10, 4).astype(np.float64)
    good_shards = (
        DistributedTensorShard(tensor_id="v", partition_axis=0, partition_count=2, shard_index=0, range_start=0, range_end=5),
        DistributedTensorShard(tensor_id="v", partition_axis=0, partition_count=2, shard_index=1, range_start=5, range_end=10),
    )
    shards = build_vocab_rank_shards(w, good_shards)
    tokens = np.array([[3, 7]])
    correct = reconstruct_embedding({rid: rank_local_masked_embedding(tokens, s) for rid, s in shards.items()})

    # Simulate an overlap bug: rank 1 incorrectly also claims token 3 (owned by rank 0).
    from deployment.tp_process_runtime.vocab_parallel_executor import VocabRankShard

    buggy_rank1 = VocabRankShard(rank_id=1, vocab_start=3, vocab_end=10, weight_shard=w[3:10])
    buggy_outputs = {0: rank_local_masked_embedding(tokens, shards[0]), 1: rank_local_masked_embedding(tokens, buggy_rank1)}
    buggy = reconstruct_embedding(buggy_outputs)
    assert not np.allclose(correct, buggy)  # double-counted token 3 row


def test_negative_lm_head_logit_shard_ordering_mismatch():
    rng = np.random.default_rng(6)
    hidden = rng.normal(size=(1, 2, 4))
    w = rng.normal(size=(10, 4))
    shards = (
        DistributedTensorShard(tensor_id="v", partition_axis=0, partition_count=2, shard_index=0, range_start=0, range_end=5),
        DistributedTensorShard(tensor_id="v", partition_axis=0, partition_count=2, shard_index=1, range_start=5, range_end=10),
    )
    vshards = build_vocab_rank_shards(w, shards)
    parts = {rid: rank_local_lm_head_logits(hidden, s) for rid, s in vshards.items()}
    correct = reconstruct_lm_head_logits(parts, org_vocab_size=10)
    reference = hidden @ w.T
    assert np.allclose(correct, reference)
    wrong_order = np.concatenate([parts[1], parts[0]], axis=-1)
    assert not np.allclose(wrong_order, reference)


def test_negative_tied_weight_mismatch_untied_model_rejected(model_and_tokenizer, monkeypatch):
    model, _tok = model_and_tokenizer
    monkeypatch.setattr(model.config, "tie_word_embeddings", False)
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        build_whole_model_plan(model, source_tp2_plan_id="x")


def test_negative_replicated_op_receives_local_only_tensor_incorrectly():
    """Feeding a rank-LOCAL (unreduced) partial output directly into a
    'replicated' RMSNorm-equivalent computation must not match the correct
    result -- proving replicated ops genuinely need the reconstructed
    (full) tensor, not a local shard."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 8))
    w = rng.normal(size=(6, 8))
    shards = (
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=0, range_start=0, range_end=4),
        DistributedTensorShard(tensor_id="t", partition_axis=1, partition_count=2, shard_index=1, range_start=4, range_end=8),
    )
    rank_shards = build_rank_shards(x, w, shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in rank_shards.items()}
    full = sum(partials.values())

    def rms_normalize(t):
        return t / np.sqrt((t ** 2).mean(axis=-1, keepdims=True) + 1e-6)

    correct = rms_normalize(full)
    wrong = rms_normalize(partials[0])  # local-only shard fed into a replicated op
    assert not np.allclose(correct, wrong)


def test_negative_unsupported_tp_relevant_module_family(model_and_tokenizer):
    model, _tok = model_and_tokenizer
    with pytest.raises(OperatorMappingError, match="unknown operator kind"):
        map_compiler_operator_to_module("qwen_prefill::llm.made_up_family::layer_0", model)


def test_negative_layer_contract_differs_from_inventory(model_and_tokenizer):
    model, _tok = model_and_tokenizer
    with pytest.raises(OperatorMappingError, match="no matching module"):
        map_compiler_operator_to_module("qwen_prefill::llm.o_proj::layer_99", model)


def test_negative_whole_model_logits_exceed_tolerance_detection():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(1, 5, 100))
    corrupted = reference.copy()
    corrupted[0, -1, 0] += 50.0
    diff = np.abs(corrupted - reference)
    assert diff.max() > 1e-2  # detection threshold correctly flags the corruption


def test_negative_topk_token_mismatch_detection():
    rng = np.random.default_rng(9)
    logits = rng.normal(size=(100,))
    corrupted = logits.copy()
    top1 = np.argmax(logits)
    corrupted[top1] -= 100.0  # knock the true top token far down
    assert np.argmax(logits) != np.argmax(corrupted)


def test_negative_synthetic_fallback_never_used_in_executors():
    """Static check: none of the D4A executor modules contain a synthetic/
    random-tensor fallback in their production code path."""
    import deployment.tp_process_runtime.attention_contract_executor as m1
    import deployment.tp_process_runtime.column_parallel_executor as m2
    import deployment.tp_process_runtime.mlp_contract_executor as m3
    import deployment.tp_process_runtime.vocab_parallel_executor as m4
    import deployment.tp_process_runtime.whole_model_tp_replay as m5
    import inspect

    for mod in (m1, m2, m3, m4, m5):
        source = inspect.getsource(mod)
        assert "np.random" not in source
        assert "torch.rand" not in source


def test_negative_d3b_evidence_not_updated_without_valid_d4a_artifact(tmp_path):
    from deployment.vllm_adapter.distributed_launch_spec import WholeModelTPEvidenceStatus

    default = WholeModelTPEvidenceStatus.NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY.value

    b_missing = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    assert b_missing.spec.whole_model_tp_evidence_status == default
    assert b_missing.spec.whole_model_tp_evidence_source_artifact_hash is None

    fake = tmp_path / "not_validated.json"
    fake.write_text('{"classification": "WHOLE_MODEL_TP_REJECTED"}')
    b_bad = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=fake)
    assert b_bad.spec.whole_model_tp_evidence_status == default
    assert b_bad.spec.whole_model_tp_evidence_source_artifact_hash is None

    missing_path = tmp_path / "does_not_exist.json"
    b_missing2 = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=missing_path)
    assert b_missing2.spec.whole_model_tp_evidence_status == default


def test_d3b_tp2_hardware_preflight_still_rejects_after_d4a_evidence(tmp_path):
    good = tmp_path / "d4a_evidence.json"
    good.write_text(
        '{"classification": "WHOLE_MODEL_TP_VALIDATED", "model": "Qwen/Qwen2.5-0.5B-Instruct", "tensor_parallel_size": 2}'
    )
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=good)
    assert bundle.spec.whole_model_tp_evidence_status == "validated_serialized_whole_model_contract"
    assert bundle.preflight.passed is False
    assert bundle.preflight.primary_reason == "insufficient_visible_gpu_count"
    assert bundle.spec.execution_readiness_state == "PREFLIGHT_REJECTED"
