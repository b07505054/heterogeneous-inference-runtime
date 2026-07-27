from perf_model.steady_state_window import (
    validate_running_count_window, extract_common_window_gaps, per_request_tpot_ms, batch_step_latency_ms,
    aggregate_throughput_tokens_per_s, scaling_efficiency, latency_inflation, coefficient_of_variation,
)


# --- 4. verification that num_requests_running remains N ---
def test_validate_running_count_window_accepts_stable_n():
    samples = [{"time": t, "num_requests_running": 2} for t in (1.0, 1.2, 1.4, 1.6)]
    result = validate_running_count_window(samples, 1.0, 1.6, expected_n=2)
    assert result.valid is True


# --- 5. early-completion rejection ---
def test_validate_running_count_window_rejects_deviation():
    samples = [{"time": 1.0, "num_requests_running": 2}, {"time": 1.2, "num_requests_running": 1},
               {"time": 1.4, "num_requests_running": 2}]
    result = validate_running_count_window(samples, 1.0, 1.4, expected_n=2)
    assert result.valid is False
    assert "deviated" in result.reason


def test_validate_running_count_window_rejects_no_samples():
    result = validate_running_count_window([], 1.0, 2.0, expected_n=2)
    assert result.valid is False
    assert result.reason == "no_running_count_samples_in_window"


# --- 3. common steady-state measurement-window extraction ---
def test_extract_common_window_gaps_uses_index_range():
    arrivals = [i * 0.01 for i in range(50)]  # 50 tokens, 10ms apart
    gaps = extract_common_window_gaps(arrivals, window_start_token=9, window_end_token=40)
    assert len(gaps) == 31
    assert all(abs(g - 10.0) < 1e-6 for g in gaps)


def test_extract_common_window_gaps_empty_when_too_short():
    arrivals = [0.0, 0.01, 0.02]
    assert extract_common_window_gaps(arrivals, 9, 40) == []


# --- 6. per-request TPOT calculation ---
def test_per_request_tpot_ms_is_median():
    assert per_request_tpot_ms([10.0, 12.0, 11.0]) == 11.0
    assert per_request_tpot_ms([]) is None


def test_batch_step_latency_ms_from_per_request_medians():
    assert batch_step_latency_ms([167.0, 168.0, 169.0]) == 168.0
    assert batch_step_latency_ms([]) is None


# --- 7. aggregate throughput calculation ---
def test_aggregate_throughput():
    assert aggregate_throughput_tokens_per_s(n_requests=4, window_wall_time_s=2.0) == 2.0
    assert aggregate_throughput_tokens_per_s(4, 0.0) is None


# --- 8. scaling-efficiency calculation ---
def test_scaling_efficiency_perfect_scaling_is_one():
    assert scaling_efficiency(throughput_n=8.0, throughput_1=2.0, n=4) == 1.0


def test_scaling_efficiency_handles_missing_inputs():
    assert scaling_efficiency(None, 2.0, 4) is None
    assert scaling_efficiency(8.0, 0.0, 4) is None


def test_latency_inflation():
    assert latency_inflation(tpot_n=168.0, tpot_1=12.0) == 14.0
    assert latency_inflation(None, 12.0) is None


def test_coefficient_of_variation():
    cv = coefficient_of_variation([10.0, 10.0, 10.0])
    assert cv == 0.0
    assert coefficient_of_variation([5.0]) is None
