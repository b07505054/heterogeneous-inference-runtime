from perf_model import phase_model
from perf_model.schema import ModelFeatures


def _tiny_model() -> ModelFeatures:
    return ModelFeatures(
        model_id="synthetic-tiny", architecture="synthetic", parameter_count=1000,
        layer_count=2, hidden_size=8, intermediate_size=16, attention_head_count=2,
        kv_head_count=2, head_dimension=4, vocabulary_size=20, dtype="float16",
        quantization="none", maximum_model_length=64, estimated_weight_bytes=2000,
        estimated_weight_bytes_source="analytical_flop_bandwidth",
    )


def test_uncalibrated_throughput_yields_unsupported_not_fabricated_ms():
    model = _tiny_model()
    estimate, breakdown = phase_model.predict_prefill_ms(
        model, prompt_tokens=32, weight_bytes=2000, throughput=phase_model.UNCALIBRATED
    )
    assert estimate.value is None
    assert estimate.method == "unsupported"
    assert estimate.truth_boundary == "unsupported_no_estimate"
    assert "op_counts" in breakdown  # op counts and traffic are still reported


def test_calibration_produces_positive_finite_constants():
    throughput = phase_model.calibrate(
        prefill_flops=1_000_000, measured_prefill_ms=10.0,
        decode_memory_bytes_batch1=50_000, measured_decode_token_ms=5.0,
        calibrated_from={"experiment_id": "calib-1"},
    )
    assert throughput.source == "derived_from_phase_measurement"
    assert throughput.flops_per_second == 1_000_000 / (10.0 / 1000.0)
    assert throughput.bandwidth_bytes_per_second == 50_000 / (5.0 / 1000.0)


def test_predicted_prefill_ms_uses_calibrated_constant():
    model = _tiny_model()
    throughput = phase_model.calibrate(
        prefill_flops=10_000, measured_prefill_ms=1.0,
        decode_memory_bytes_batch1=10_000, measured_decode_token_ms=1.0,
        calibrated_from={},
    )
    estimate, _ = phase_model.predict_prefill_ms(model, prompt_tokens=8, weight_bytes=2000, throughput=throughput)
    assert estimate.value is not None and estimate.value > 0
    assert estimate.method == "analytical"


def test_single_request_vs_concurrent_decode_are_not_interchangeable():
    model = _tiny_model()
    throughput = phase_model.calibrate(
        prefill_flops=10_000, measured_prefill_ms=1.0,
        decode_memory_bytes_batch1=10_000, measured_decode_token_ms=1.0,
        calibrated_from={},
    )
    single, _ = phase_model.predict_decode_token_ms(
        model, kv_context_tokens=50, weight_bytes=2000, kv_bytes_per_token=64, batch_size=1, throughput=throughput
    )
    batched, _ = phase_model.predict_decode_token_ms(
        model, kv_context_tokens=50, weight_bytes=2000, kv_bytes_per_token=64, batch_size=8, throughput=throughput
    )
    # a batched step costs more wall-clock than a single-sequence step ...
    assert batched.value > single.value
    # ... but concurrent steady-state throughput must still exceed the single-request rate,
    # since 8 sequences advance per batched step instead of 1.
    single_tps = phase_model.predict_output_tokens_per_second_single(single)
    batched_tps = phase_model.predict_output_tokens_per_second_concurrent(8, batched)
    assert batched_tps.value > single_tps.value


def test_queue_ms_zero_when_admitted_concurrency_covers_all_requests():
    assert phase_model.predict_queue_ms_positional(request_index=3, admitted_concurrency=8, avg_service_ms=100) == 0.0


def test_queue_ms_grows_with_position_beyond_admitted_concurrency():
    q0 = phase_model.predict_queue_ms_positional(request_index=0, admitted_concurrency=2, avg_service_ms=50)
    q2 = phase_model.predict_queue_ms_positional(request_index=2, admitted_concurrency=2, avg_service_ms=50)
    q4 = phase_model.predict_queue_ms_positional(request_index=4, admitted_concurrency=2, avg_service_ms=50)
    assert q0 == 0.0
    assert q2 == 50.0
    assert q4 == 100.0


def test_e2e_combines_ttft_and_tpot_correctly():
    from perf_model.schema import MetricEstimate
    ttft = MetricEstimate(100.0, "analytical", "analytical_with_phase_derived_constant")
    tpot = MetricEstimate(10.0, "analytical", "analytical_with_phase_derived_constant")
    e2e = phase_model.predict_e2e_ms(ttft, output_tokens=6, tpot_estimate=tpot)
    assert e2e.value == 100.0 + 5 * 10.0


def test_e2e_unsupported_when_ttft_missing():
    from perf_model.schema import MetricEstimate
    missing_ttft = MetricEstimate(None, "unsupported", "unsupported_no_estimate")
    tpot = MetricEstimate(10.0, "analytical", "analytical_with_phase_derived_constant")
    e2e = phase_model.predict_e2e_ms(missing_ttft, output_tokens=6, tpot_estimate=tpot)
    assert e2e.value is None
    assert e2e.method == "unsupported"
