from perf_model import compute_model
from perf_model.schema import ModelFeatures


def _tiny_model(layer_count: int = 1) -> ModelFeatures:
    # Deliberately tiny, round-number synthetic transformer so every FLOP term
    # below can be hand-verified (see module docstring in compute_model.py).
    return ModelFeatures(
        model_id="synthetic-tiny", architecture="synthetic", parameter_count=1,
        layer_count=layer_count, hidden_size=4, intermediate_size=8,
        attention_head_count=2, kv_head_count=2, head_dimension=2, vocabulary_size=10,
        dtype="float16", quantization="none", maximum_model_length=64,
        estimated_weight_bytes=0, estimated_weight_bytes_source="analytical_flop_bandwidth",
    )


def test_prefill_op_counts_hand_verified_single_layer():
    model = _tiny_model(layer_count=1)
    ops = compute_model.prefill_op_counts(model, prompt_tokens=3)
    assert ops.qkv_proj_flops == 288       # 2*3*4*4 (Q) + 2*3*4*4 (K) + 2*3*4*4 (V)
    assert ops.attention_score_flops == 36  # 2*heads(2)*T*T*head_dim(2)*causal(0.5) = 2*2*9*2*0.5
    assert ops.attention_value_flops == 36
    assert ops.output_proj_flops == 96      # 2*T*H*H = 2*3*4*4
    assert ops.mlp_flops == 576             # gate(192)+up(192)+down(192)
    assert ops.vocab_proj_flops == 240      # 2*T*H*V = 2*3*4*10
    assert ops.total_flops == 1272


def test_decode_step_op_counts_hand_verified_single_layer():
    model = _tiny_model(layer_count=1)
    ops = compute_model.decode_step_op_counts(model, kv_context_tokens=5)
    assert ops.qkv_proj_flops == 96          # 2*1*4*4 * 3 (Q,K,V all width 4 here)
    assert ops.attention_score_flops == 40   # 2*heads(2)*kv_context(5)*head_dim(2), no causal factor
    assert ops.attention_value_flops == 40
    assert ops.output_proj_flops == 32       # 2*1*4*4
    assert ops.mlp_flops == 192              # gate(64)+up(64)+down(64)
    assert ops.vocab_proj_flops == 80        # 2*1*4*10
    assert ops.total_flops == 480


def test_op_counts_scale_linearly_with_layer_count():
    single = compute_model.prefill_op_counts(_tiny_model(1), prompt_tokens=3)
    triple = compute_model.prefill_op_counts(_tiny_model(3), prompt_tokens=3)
    # vocab_proj is a single final-layer op and must NOT scale with layer_count;
    # every other term is per-transformer-layer and must scale exactly linearly.
    assert triple.vocab_proj_flops == single.vocab_proj_flops
    per_layer_terms = ("qkv_proj_flops", "attention_score_flops", "attention_value_flops",
                        "output_proj_flops", "mlp_flops")
    for term in per_layer_terms:
        assert getattr(triple, term) == 3 * getattr(single, term)
    assert triple.total_flops == 3 * (single.total_flops - single.vocab_proj_flops) + single.vocab_proj_flops


def test_prefill_attention_terms_scale_quadratically_with_prompt_length():
    model = _tiny_model(1)
    short = compute_model.prefill_op_counts(model, prompt_tokens=4)
    long_ = compute_model.prefill_op_counts(model, prompt_tokens=8)
    # attention score/value scale with T^2; doubling T should ~4x those two terms
    assert long_.attention_score_flops == 4 * short.attention_score_flops
    assert long_.attention_value_flops == 4 * short.attention_value_flops
    # projections scale linearly with T
    assert long_.qkv_proj_flops == 2 * short.qkv_proj_flops


def test_decode_attention_terms_scale_linearly_with_kv_context():
    model = _tiny_model(1)
    short = compute_model.decode_step_op_counts(model, kv_context_tokens=10)
    long_ = compute_model.decode_step_op_counts(model, kv_context_tokens=20)
    assert long_.attention_score_flops == 2 * short.attention_score_flops
    # non-attention terms are independent of context length for a single new token
    assert long_.qkv_proj_flops == short.qkv_proj_flops


def test_decode_memory_traffic_grows_with_context_length():
    model = _tiny_model(1)
    small = compute_model.decode_step_memory_traffic_bytes(
        model, weight_bytes=1000, kv_bytes_per_token=32, kv_context_tokens=10
    )
    large = compute_model.decode_step_memory_traffic_bytes(
        model, weight_bytes=1000, kv_bytes_per_token=32, kv_context_tokens=100
    )
    assert small == 1000 + 32 * 10
    assert large == 1000 + 32 * 100
    assert large > small


def test_prefill_memory_traffic_is_single_weight_read():
    assert compute_model.prefill_memory_traffic_bytes(weight_bytes=12345) == 12345
