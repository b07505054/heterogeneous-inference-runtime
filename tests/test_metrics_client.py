from pathlib import Path

from deployment.vllm_adapter import metrics_client as mc

FIXTURE_TEXT = (Path(__file__).parent / "fixtures" / "metrics_sample.txt").read_text()


def test_parse_prometheus_text_preserves_bucket_data():
    parsed = mc.parse_prometheus_text(FIXTURE_TEXT)
    samples = parsed["histograms"]["vllm:time_to_first_token_seconds"]
    assert len(samples) == 1
    assert samples[0].buckets["0.5"] == 1.0
    assert samples[0].buckets["+Inf"] == 1.0
    assert parsed["raw_text"] == FIXTURE_TEXT


def test_histogram_mean_ms_matches_real_captured_values():
    parsed = mc.parse_prometheus_text(FIXTURE_TEXT)
    ttft_mean = mc.histogram_mean_ms(parsed, "vllm:time_to_first_token_seconds")
    assert ttft_mean is not None and abs(ttft_mean - 305.82571029663086) < 1e-6
    prefill_mean = mc.histogram_mean_ms(parsed, "vllm:request_prefill_time_seconds")
    assert abs(prefill_mean - 289.8652059957385) < 1e-6
    decode_tpot_mean = mc.histogram_mean_ms(parsed, "vllm:request_time_per_output_token_seconds")
    assert abs(decode_tpot_mean - 10.553341327855984) < 1e-6


def test_histogram_mean_ms_none_when_metric_absent():
    parsed = mc.parse_prometheus_text(FIXTURE_TEXT)
    assert mc.histogram_mean_ms(parsed, "vllm:does_not_exist") is None


def test_gauge_value_reads_scalar_metric():
    parsed = mc.parse_prometheus_text(FIXTURE_TEXT)
    running = mc.gauge_value(parsed, "vllm:num_requests_running")
    assert running == 0.0


def test_snapshot_summary_includes_all_required_histograms_and_gauges():
    parsed = mc.parse_prometheus_text(FIXTURE_TEXT)
    summary = mc.snapshot_summary(parsed)
    for metric in mc.HISTOGRAM_METRICS:
        assert metric in summary["histograms"]
        assert summary["histograms"][metric]["buckets"] is not None
    for metric in mc.GAUGE_OR_COUNTER_METRICS:
        assert metric in summary["gauges"]


def test_parser_does_not_naively_sum_histogram_buckets_like_old_diagnostic_parser():
    # Regression guard: the pre-existing diagnostic script's _parse_prometheus
    # sums every matched line's value, which would be wrong for histograms
    # (bucket cumulative counts, not independent samples). Confirm our parser
    # reports the real mean, not a bucket-sum artifact.
    parsed = mc.parse_prometheus_text(FIXTURE_TEXT)
    mean_ms = mc.histogram_mean_ms(parsed, "vllm:request_queue_time_seconds")
    naive_bucket_sum = sum(parsed["histograms"]["vllm:request_queue_time_seconds"][0].buckets.values())
    assert mean_ms != naive_bucket_sum
