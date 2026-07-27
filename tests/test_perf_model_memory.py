import pytest

from perf_model.memory_model import (
    bytes_per_element, weight_memory_bytes, kv_bytes_per_token, kv_peak_bytes,
    total_predicted_memory_bytes, predict_oom,
)
from perf_model.schema import HardwareFeature, HardwareFeatures, ModelFeatures


def _qwen() -> ModelFeatures:
    return ModelFeatures(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", architecture="qwen2", parameter_count=494_032_768,
        layer_count=24, hidden_size=896, intermediate_size=4864, attention_head_count=14,
        kv_head_count=2, head_dimension=64, vocabulary_size=151936, dtype="float16",
        quantization="none", maximum_model_length=2048, estimated_weight_bytes=0,
        estimated_weight_bytes_source="analytical_flop_bandwidth",
    )


def test_bytes_per_element_known_dtypes():
    assert bytes_per_element("float16", "none") == 2
    assert bytes_per_element("bfloat16", "auto") == 2
    assert bytes_per_element("float32", "") == 4


def test_bytes_per_element_rejects_unmodeled_quantization():
    with pytest.raises(ValueError):
        bytes_per_element("float16", "awq")


def test_weight_memory_bytes_matches_param_count_times_width():
    model = _qwen()
    assert weight_memory_bytes(model) == model.parameter_count * 2


def test_kv_bytes_per_token_qwen2_shape():
    model = _qwen()
    # layers(24) * 2(K,V) * kv_heads(2) * head_dim(64) * 2 bytes(fp16) = 12288 bytes/token
    assert kv_bytes_per_token(model, kv_cache_dtype_bytes=2) == 24 * 2 * 2 * 64 * 2 == 12288


def test_kv_peak_bytes_theoretical_matches_sum_of_sequence_lengths():
    model = _qwen()
    estimate = kv_peak_bytes(model, kv_cache_dtype_bytes=2, per_sequence_token_counts=[100, 50])
    assert estimate.total_live_tokens == 150
    assert estimate.theoretical_bytes == 12288 * 150
    assert estimate.block_rounded_bytes is None


def test_kv_peak_bytes_block_rounding_rounds_up_per_sequence_not_globally():
    model = _qwen()
    # 100 -> rounds to 112 (7 blocks of 16), 50 -> rounds to 64 (4 blocks of 16)
    estimate = kv_peak_bytes(model, kv_cache_dtype_bytes=2, per_sequence_token_counts=[100, 50], block_size=16)
    assert estimate.block_rounded_bytes == 12288 * (112 + 64)
    assert estimate.block_rounded_bytes > estimate.theoretical_bytes


def test_total_predicted_memory_prefers_block_rounded_when_available():
    model = _qwen()
    est = kv_peak_bytes(model, kv_cache_dtype_bytes=2, per_sequence_token_counts=[100], block_size=16)
    total = total_predicted_memory_bytes(
        weight_bytes=1000, kv_peak_estimate=est, runtime_overhead_bytes=10, safety_margin_bytes=5,
    )
    assert total == 1000 + est.block_rounded_bytes + 10 + 5


def test_predict_oom_true_when_predicted_exceeds_gpu_memory():
    hw = HardwareFeatures(
        gpu_name="GTX 1650", gpu_memory_bytes=HardwareFeature(4 * 1024**3, "device_reported"),
        gpu_count=1, cuda_version="13.0", compute_capability=HardwareFeature("7.5", "device_reported"),
        memory_bandwidth_bytes_per_s=HardwareFeature(128e9, "vendor_spec"),
        compute_throughput_flops=HardwareFeature(2.9e12, "vendor_spec"),
    )
    result = predict_oom(5 * 1024**3, hw)
    assert result.value is True
    assert result.method == "analytical"


def test_predict_oom_false_when_within_budget():
    hw = HardwareFeatures(
        gpu_name="GTX 1650", gpu_memory_bytes=HardwareFeature(4 * 1024**3, "device_reported"),
        gpu_count=1, cuda_version="13.0", compute_capability=HardwareFeature("7.5", "device_reported"),
        memory_bandwidth_bytes_per_s=HardwareFeature(128e9, "vendor_spec"),
        compute_throughput_flops=HardwareFeature(2.9e12, "vendor_spec"),
    )
    result = predict_oom(1 * 1024**3, hw)
    assert result.value is False
