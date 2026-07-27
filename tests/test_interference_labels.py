from perf_model.interference_labels import (
    split_timeline, baseline_gap_ms, peak_stall_ms, total_stall_area_ms, recovery_time_ms,
    sustained_slowdown_ratio, post_admission_percentiles, interference_e2e_ms, affected_token_count,
)


def _arrivals(gaps_s, start=0.0):
    times = [start]
    for g in gaps_s:
        times.append(times[-1] + g)
    return times


def test_split_timeline_separates_pre_and_post_admission_by_real_timestamp():
    # 4 baseline gaps of 10ms, admission occurs right after token 4, then 3 stalled gaps
    arrivals = _arrivals([0.010, 0.010, 0.010, 0.010, 0.400, 0.170, 0.170])
    admission_time = arrivals[4] + 0.001  # admission observed just after the 4th gap ends
    split = split_timeline(arrivals, admission_time)
    assert len(split.pre_gaps_ms) == 4
    assert len(split.post_gaps_ms) == 3
    assert abs(split.pre_gaps_ms[0] - 10.0) < 1e-6


def test_baseline_and_peak_stall():
    arrivals = _arrivals([0.010, 0.010, 0.400, 0.170])
    split = split_timeline(arrivals, arrivals[2] + 0.0005)
    baseline = baseline_gap_ms(split)
    assert abs(baseline - 10.0) < 1e-6
    peak = peak_stall_ms(split, baseline)
    assert abs(peak - (400.0 - 10.0)) < 1e-3


def test_total_stall_area_only_counts_excess_over_baseline():
    arrivals = _arrivals([0.010, 0.010, 0.400, 0.170, 0.011])
    split = split_timeline(arrivals, arrivals[2] + 0.0005)
    baseline = baseline_gap_ms(split)  # 10.0ms
    area = total_stall_area_ms(split, baseline)
    # post gaps: 400, 170, 11 -> excess: 390, 160, ~1 (11-10=1, positive)
    assert area > 549 and area < 552


def test_affected_token_count_uses_threshold_ratio():
    arrivals = _arrivals([0.010, 0.010, 0.400, 0.011, 0.009])
    split = split_timeline(arrivals, arrivals[2] + 0.0005)
    baseline = baseline_gap_ms(split)
    count = affected_token_count(split, baseline, threshold_ratio=1.2)
    assert count == 1  # only the 400ms gap exceeds 10*1.2=12ms


def test_recovery_time_finds_first_run_of_consecutive_low_gaps():
    # stall, then 2 elevated, then 3 consecutive back at baseline -> recovery should trigger at the 3rd
    arrivals = _arrivals([0.010, 0.010, 0.400, 0.020, 0.020, 0.011, 0.010, 0.010])
    admission_time = arrivals[2] + 0.0005
    split = split_timeline(arrivals, admission_time)
    baseline = baseline_gap_ms(split)
    recovery = recovery_time_ms(split, baseline, tolerance=1.5, consecutive=3)
    assert recovery is not None
    assert recovery > 0


def test_recovery_time_none_when_never_recovers():
    arrivals = _arrivals([0.010, 0.010, 0.400, 0.300, 0.300, 0.300])
    split = split_timeline(arrivals, arrivals[2] + 0.0005)
    baseline = baseline_gap_ms(split)
    recovery = recovery_time_ms(split, baseline, tolerance=1.5, consecutive=3)
    assert recovery is None


def test_sustained_slowdown_ratio():
    arrivals = _arrivals([0.010, 0.010, 0.020, 0.020, 0.020])
    split = split_timeline(arrivals, arrivals[2] + 0.0005)
    baseline = baseline_gap_ms(split)  # 10.0
    ratio = sustained_slowdown_ratio(split, baseline)
    assert abs(ratio - 2.0) < 1e-6


def test_post_admission_percentiles_empty_when_no_post_gaps():
    arrivals = _arrivals([0.010, 0.010])
    split = split_timeline(arrivals, arrivals[-1] + 1.0)  # admission after everything
    stats = post_admission_percentiles(split)
    assert stats["p50"] is None


def test_interference_e2e_ms_positive_when_slower_than_isolated_estimate():
    val = interference_e2e_ms(measured_anchor_e2e_ms=2000.0, ttft_ms=150.0, baseline_gap=10.0, output_tokens=64)
    isolated_estimate = 150.0 + 63 * 10.0  # 780
    assert abs(val - (2000.0 - isolated_estimate)) < 1e-6
    assert val > 0


def test_interference_e2e_ms_none_when_inputs_missing():
    assert interference_e2e_ms(None, 150.0, 10.0, 64) is None
    assert interference_e2e_ms(2000.0, 150.0, None, 64) is None
