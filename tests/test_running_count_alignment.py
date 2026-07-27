from perf_model.running_count_alignment import align_gaps_to_running_count, median_or_none, peak_running_waiting


def test_align_gaps_classifies_solo_vs_concurrent_by_nearest_sample():
    arrivals = [0.0, 0.10, 0.20, 0.30, 0.40]  # 4 gaps of 100ms
    samples = [
        {"time": 0.09, "num_requests_running": 1}, {"time": 0.19, "num_requests_running": 1},
        {"time": 0.29, "num_requests_running": 2}, {"time": 0.39, "num_requests_running": 2},
    ]
    aligned = align_gaps_to_running_count(arrivals, samples, tolerance_s=0.06)
    assert len(aligned.solo_gaps_ms) == 2
    assert len(aligned.concurrent_gaps_ms) == 2


def test_align_gaps_marks_unmatched_when_no_sample_within_tolerance():
    arrivals = [0.0, 0.10]
    samples = [{"time": 5.0, "num_requests_running": 1}]  # far away
    aligned = align_gaps_to_running_count(arrivals, samples, tolerance_s=0.1)
    assert aligned.unmatched_gaps_ms == [100.0]
    assert aligned.solo_gaps_ms == []


def test_align_gaps_handles_no_samples():
    aligned = align_gaps_to_running_count([0.0, 0.1, 0.2], [], tolerance_s=0.1)
    assert aligned.solo_gaps_ms == []
    assert aligned.concurrent_gaps_ms == []
    assert len(aligned.unmatched_gaps_ms) == 2


def test_median_or_none():
    assert median_or_none([1.0, 2.0, 3.0]) == 2.0
    assert median_or_none([]) is None


def test_peak_running_waiting_reports_max_and_sample_count():
    samples = [
        {"time": 0.0, "num_requests_running": 1, "num_requests_waiting": 0},
        {"time": 0.1, "num_requests_running": 3, "num_requests_waiting": 2},
        {"time": 0.2, "num_requests_running": 2, "num_requests_waiting": 0},
    ]
    result = peak_running_waiting(samples)
    assert result["peak_requests_running"] == 3
    assert result["peak_requests_waiting"] == 2
    assert result["n_samples"] == 3


def test_peak_running_waiting_handles_empty_samples():
    result = peak_running_waiting([])
    assert result["peak_requests_running"] is None
    assert result["n_samples"] == 0
