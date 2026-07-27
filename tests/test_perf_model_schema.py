import pytest

from perf_model.schema import (
    HardwareFeature, HardwareFeatures, ModelFeatures, PredictionResult, RuntimeConfiguration,
    WorkloadFeatures, MetricEstimate, stable_hash,
)


def _qwen_model_features() -> ModelFeatures:
    return ModelFeatures(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", architecture="qwen2", parameter_count=494_032_768,
        layer_count=24, hidden_size=896, intermediate_size=4864, attention_head_count=14,
        kv_head_count=2, head_dimension=64, vocabulary_size=151936, dtype="float16",
        quantization="none", maximum_model_length=2048, estimated_weight_bytes=988_065_536,
        estimated_weight_bytes_source="analytical_flop_bandwidth", tie_word_embeddings=True,
    )


def test_model_features_hash_stable_and_order_independent():
    m1 = _qwen_model_features()
    m2 = _qwen_model_features()
    assert m1.features_hash() == m2.features_hash()


def test_model_features_hash_changes_with_content():
    m1 = _qwen_model_features()
    m2 = ModelFeatures(**{**m1.to_dict(), "hidden_size": 1024})
    assert m1.features_hash() != m2.features_hash()


def test_model_features_rejects_bad_estimate_source():
    with pytest.raises(ValueError):
        ModelFeatures(
            model_id="x", architecture="qwen2", parameter_count=1, layer_count=1, hidden_size=1,
            intermediate_size=1, attention_head_count=1, kv_head_count=1, head_dimension=1,
            vocabulary_size=1, dtype="float16", quantization="none", maximum_model_length=1,
            estimated_weight_bytes=1, estimated_weight_bytes_source="measured_ish_guess",
        )


def test_hardware_features_class_validation():
    with pytest.raises(ValueError):
        HardwareFeature(value=1.0, source_class="totally_measured")
    hf = HardwareFeature(value=4 * 1024**3, source_class="device_reported")
    assert hf.to_dict()["source_class"] == "device_reported"


def test_hardware_features_hash_and_dict_shape():
    hw = HardwareFeatures(
        gpu_name="NVIDIA GeForce GTX 1650 with Max-Q Design",
        gpu_memory_bytes=HardwareFeature(4 * 1024**3, "device_reported"),
        gpu_count=1, cuda_version="13.0",
        compute_capability=HardwareFeature("7.5", "device_reported"),
        memory_bandwidth_bytes_per_s=HardwareFeature(128e9, "vendor_spec"),
        compute_throughput_flops=HardwareFeature(2.9e12, "vendor_spec"),
    )
    d = hw.to_dict()
    assert d["gpu_memory_bytes"]["source_class"] == "device_reported"
    assert d["memory_bandwidth_bytes_per_s"]["source_class"] == "vendor_spec"
    assert hw.features_hash() == HardwareFeatures(**hw.__dict__).features_hash()


def test_workload_features_hash():
    wf = WorkloadFeatures(
        workload_id="A", request_count=16, concurrency=1,
        prompt_token_distribution={"target": 32}, output_token_distribution={"target": 64},
        arrival_process="closed_loop_immediate_submission", prefix_sharing=False,
        warmup_count=3, repetition_count=2, streaming_mode=True,
        sampling_settings={"temperature": 0.0},
    )
    assert isinstance(wf.features_hash(), str) and len(wf.features_hash()) == 64


def test_runtime_configuration_records_not_owned_fields_without_owning_them():
    rc = RuntimeConfiguration(
        max_num_seqs=4, max_num_batched_tokens=2048, max_model_len=2048,
        gpu_memory_utilization=0.75, tensor_parallel_size=1, dtype="float16", quantization="none",
        recorded_not_owned={"block_size": 16, "enable_prefix_caching": False},
    )
    d = rc.to_dict()
    assert d["recorded_not_owned"]["block_size"] == 16
    assert "block_size" not in {k for k in d if k != "recorded_not_owned"}


def test_metric_estimate_rejects_unknown_method_and_boundary():
    with pytest.raises(ValueError):
        MetricEstimate(1.0, "guessed", "analytical_no_measurement")
    with pytest.raises(ValueError):
        MetricEstimate(1.0, "analytical", "trust_me")
    ok = MetricEstimate(1.0, "analytical", "analytical_no_measurement")
    assert ok.to_dict()["method"] == "analytical"


def test_stable_hash_is_deterministic_across_key_order():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_prediction_result_full_construction_and_serialization():
    unsupported = MetricEstimate(None, "unsupported", "unsupported_no_estimate", "not yet calibrated")
    supported = MetricEstimate(42.0, "analytical", "analytical_with_phase_derived_constant")
    result = PredictionResult(
        model_features_hash="m" * 64, hardware_features_hash="h" * 64,
        workload_features_hash="w" * 64, runtime_configuration_hash="r" * 64,
        predicted_weight_memory_bytes=supported, predicted_kv_memory_bytes=supported,
        predicted_total_memory_bytes=supported, predicted_oom=MetricEstimate(False, "analytical", "analytical_no_measurement"),
        predicted_prefill_ms=supported, predicted_decode_token_ms=supported,
        predicted_ttft_ms=supported, predicted_tpot_ms=supported, predicted_e2e_ms=supported,
        predicted_output_tokens_per_second=supported, predicted_total_tokens_per_second=unsupported,
        component_breakdown={"prefill": {}}, assumptions={"fixed_overhead_ms": 0}, unsupported_terms=["total_tokens_per_second"],
        confidence_class="low",
    )
    d = result.to_dict()
    assert d["prediction_schema_version"] == "perf_model.prediction.v1"
    assert d["predicted_total_tokens_per_second"]["method"] == "unsupported"
    assert d["predicted_ttft_ms"]["value"] == 42.0
    assert "total_tokens_per_second" in d["unsupported_terms"]
