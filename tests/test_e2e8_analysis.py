from scripts.analyze_perf_model_results_e2e8 import (
    summarize_reps, baseline_vs_optimized, realization_ratio, model_y, model_z, evaluate_y_z,
    evaluate_hypotheses, E2E7_LM_HEAD_ISOLATED_MS,
)


def _row(tpot, engine_tpot, throughput, window_valid=True, ref_match=True):
    return {"client_tpot_ms": tpot, "engine_tpot_ms": engine_tpot, "aggregate_throughput_tokens_per_s": throughput,
            "gpu_util_mean_percent": 99.0, "cpu_percent": 25.0, "gpu_memory_mean_mib": 3000.0,
            "window_valid": window_valid, "reference_match": ref_match}


def test_summarize_reps_computes_median_min_max():
    rows = [_row(168.0, 169.0, 12.0), _row(170.0, 171.0, 11.8), _row(169.0, 170.0, 11.9)]
    summary = summarize_reps(rows)
    assert summary["client_tpot_ms"]["median"] == 169.0
    assert summary["client_tpot_ms"]["min"] == 168.0
    assert summary["client_tpot_ms"]["max"] == 170.0
    assert summary["n_reps"] == 3
    assert summary["window_valid_all"] is True


def test_summarize_reps_flags_window_invalid():
    rows = [_row(168.0, 169.0, 12.0, window_valid=False)]
    summary = summarize_reps(rows)
    assert summary["window_valid_all"] is False


def test_summarize_reps_empty_rows_returns_none_stats():
    summary = summarize_reps([])
    assert summary["client_tpot_ms"]["median"] is None
    assert summary["n_reps"] == 0


def _grouped_fixture():
    def raw(batch, tpot, engine_tpot, throughput, optimized):
        return {
            "batch_size": batch, "prompt_length": 128, "window_valid": True,
            "timelines": [], "gpu_samples": [], "cpu_info": {"cpu_percent": 25.0},
            "adherence": None, "server_info_raw": None, "reference_match": True,
            "oom_detected_in_log": False, "process_cleanup_status": "graceful_sigterm",
            "classification": "VALID", "tiny_m_enable": optimized,
            "pre_metrics_text": None, "final_metrics_text": None,
        }
    return raw


# --- baseline_vs_optimized / realization_ratio use build_row internally, which
# needs real timelines to compute client_tpot_ms; test the pure downstream
# functions (model_y/z, evaluate_y_z, evaluate_hypotheses, realization_ratio)
# directly against synthetic `comparison` dicts instead of re-deriving timelines.

def _synthetic_comparison():
    return {
        "1": {"baseline": {"client_tpot_ms": {"median": 11.5}}, "optimized": {"client_tpot_ms": {"median": 11.6}},
              "tpot_percent_change": 0.87, "measured_saved_ms": -0.1},
        "2": {"baseline": {"client_tpot_ms": {"median": 168.9}}, "optimized": {"client_tpot_ms": {"median": 130.5}},
              "tpot_percent_change": -22.7, "measured_saved_ms": 38.4},
        "3": {"baseline": {"client_tpot_ms": {"median": 169.2}}, "optimized": {"client_tpot_ms": {"median": 135.0}},
              "tpot_percent_change": -20.2, "measured_saved_ms": 34.2},
        "4": {"baseline": {"client_tpot_ms": {"median": 169.4}}, "optimized": {"client_tpot_ms": {"median": 140.0}},
              "tpot_percent_change": -17.4, "measured_saved_ms": 29.4},
        "6": {"baseline": {"client_tpot_ms": {"median": 170.1}}, "optimized": {"client_tpot_ms": {"median": 150.0}},
              "tpot_percent_change": -11.8, "measured_saved_ms": 20.1},
        "8": {"baseline": {"client_tpot_ms": {"median": 170.5}}, "optimized": {"client_tpot_ms": {"median": 155.0}},
              "tpot_percent_change": -9.1, "measured_saved_ms": 15.5},
    }


def test_model_y_binary():
    assert model_y(1) == 11.5
    assert model_y(2) == 169.0
    assert model_y(8) == 169.0


def test_model_z_uses_measured_baseline_minus_measured_saving():
    comparison = _synthetic_comparison()
    val = model_z(comparison, 2)
    assert val == 168.9 - 38.4


def test_model_z_none_for_missing_batch_size():
    comparison = _synthetic_comparison()
    assert model_z(comparison, 99) is None


def test_evaluate_y_z_splits_calibration_and_held_out():
    comparison = _synthetic_comparison()
    result = evaluate_y_z(comparison)
    assert result["Y"]["calibration_mae"] is not None
    assert result["Y"]["held_out_mae"] is not None
    # Model Z, built from measured data at the exact same batch size, should
    # have near-zero error since it isn't extrapolating in this synthetic fixture.
    assert result["Z"]["calibration_mae"] < 1.0
    assert result["Z"]["held_out_mae"] < 1.0


def test_realization_ratio_matches_synthetic_prediction():
    comparison = _synthetic_comparison()
    ratios = realization_ratio(comparison)
    assert "2" in ratios
    predicted = E2E7_LM_HEAD_ISOLATED_MS[2]["default"] - E2E7_LM_HEAD_ISOLATED_MS[2]["gemv"]
    assert abs(ratios["2"]["predicted_lm_head_saved_ms"] - predicted) < 1e-6
    assert ratios["2"]["measured_saved_ms"] == 38.4
    assert ratios["2"]["realization_ratio"] == 38.4 / predicted


def test_evaluate_hypotheses_batch1_not_regressed():
    comparison = _synthetic_comparison()
    ratios = realization_ratio(comparison)
    h = evaluate_hypotheses(comparison, ratios)
    assert h["H4_batch1_not_regressed"]["result"] == "SUPPORTED"


def test_evaluate_hypotheses_flags_regression_over_3_percent():
    comparison = _synthetic_comparison()
    comparison["1"]["tpot_percent_change"] = 5.0  # simulate a real regression
    ratios = realization_ratio(comparison)
    h = evaluate_hypotheses(comparison, ratios)
    assert h["H4_batch1_not_regressed"]["result"] == "REJECTED"


def test_evaluate_hypotheses_batch2_gain_supported_above_15_percent():
    comparison = _synthetic_comparison()
    ratios = realization_ratio(comparison)
    h = evaluate_hypotheses(comparison, ratios)
    assert h["H3_isolated_gain_survives"]["result"] == "SUPPORTED"
    assert h["H3_isolated_gain_survives"]["evidence"]["batch2_tpot_improvement_percent"] > 15.0


def test_evaluate_hypotheses_batch2_gain_rejected_below_15_percent():
    comparison = _synthetic_comparison()
    comparison["2"]["tpot_percent_change"] = -5.0  # only 5% improvement
    ratios = realization_ratio(comparison)
    h = evaluate_hypotheses(comparison, ratios)
    assert h["H3_isolated_gain_survives"]["result"] == "REJECTED"
