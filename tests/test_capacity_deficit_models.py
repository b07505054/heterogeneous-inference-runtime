from perf_model.capacity_deficit_models import (
    requested_deficit, admission_deficit, positive_deficit, capacity_utilization, observed_running_deficit,
    fit_model_g, predict_model_g, fit_model_h, predict_model_h, fit_model_i, predict_model_i,
    fit_model_j, predict_model_j, fit_model_k, predict_model_k, fit_model_l, predict_model_l,
    predict_model_m_peak_stall, fit_model_m_transient_term, evaluate, classification_metrics, MODELS,
)


# --- 5. positive-deficit calculation ---
def test_positive_deficit_clamps_negative_to_zero():
    assert positive_deficit(-3) == 0
    assert positive_deficit(0) == 0
    assert positive_deficit(2) == 2


# --- 3. capacity-deficit feature calculation ---
def test_admission_deficit_matches_scheduler_semantics():
    # 1 active anchor + 2 newly admitted, capacity=2 -> 3-2=1 (running would exceed cap by 1)
    assert admission_deficit(active_decode_requests=1, newly_admitted_requests=2, max_num_seqs=2) == 1
    assert admission_deficit(active_decode_requests=2, newly_admitted_requests=1, max_num_seqs=8) == -5


def test_requested_deficit():
    assert requested_deficit(workload_concurrency=4, max_num_seqs=2) == 2


# --- 6. utilization calculation ---
def test_capacity_utilization():
    assert capacity_utilization(active_or_scheduled_sequences=3, max_num_seqs=2) == 1.5
    assert capacity_utilization(active_or_scheduled_sequences=1, max_num_seqs=4) == 0.25


def test_observed_running_deficit_handles_missing_sample():
    assert observed_running_deficit(None, 2) is None
    assert observed_running_deficit(5, 2) == 3
    assert observed_running_deficit(1, 2) == 0


def _rows():
    # deficit<=0 -> 0 interference; deficit==1 -> ~150ms; deficit>=2 -> similar plateau (not linear growth)
    return [
        {"max_num_seqs": 4, "admission_deficit": -2, "capacity_utilization": 0.5, "interference_ms": 0.5,
         "observed_running_deficit": 0, "iteration_tokens_mean": 5.0},
        {"max_num_seqs": 2, "admission_deficit": 0, "capacity_utilization": 1.0, "interference_ms": 0.4,
         "observed_running_deficit": 0, "iteration_tokens_mean": 8.0},
        {"max_num_seqs": 2, "admission_deficit": 1, "capacity_utilization": 1.5, "interference_ms": 155.0,
         "observed_running_deficit": 1, "iteration_tokens_mean": 130.0},
        {"max_num_seqs": 2, "admission_deficit": 2, "capacity_utilization": 2.0, "interference_ms": 158.0,
         "observed_running_deficit": 2, "iteration_tokens_mean": 132.0},
        {"max_num_seqs": 8, "admission_deficit": -4, "capacity_utilization": 0.5, "interference_ms": 0.3,
         "observed_running_deficit": 0, "iteration_tokens_mean": 6.0},
    ]


def test_model_g_binary_deficit_uses_median_of_positive_rows():
    fitted = fit_model_g(_rows())
    assert abs(fitted.params["C_deficit"] - 156.5) < 1.0  # median(155,158)
    assert predict_model_g(fitted.params, admission_deficit=0) == 0.0
    assert predict_model_g(fitted.params, admission_deficit=1) == fitted.params["C_deficit"]


def test_model_h_linear_deficit_fits_through_origin():
    fitted = fit_model_h(_rows())
    pred_at_1 = predict_model_h(fitted.params, admission_deficit=1)
    pred_at_2 = predict_model_h(fitted.params, admission_deficit=2)
    assert pred_at_2 > pred_at_1  # linear model necessarily grows with deficit -- this is exactly the assumption E2E-5 tests


def test_model_i_utilization_threshold_zero_below_threshold():
    fitted = fit_model_i(_rows(), threshold=1.0)
    assert predict_model_i(fitted.params, capacity_utilization=0.8) == 0.0
    assert predict_model_i(fitted.params, capacity_utilization=1.5) > 0.0


def test_model_j_piecewise_distinguishes_deficit_one_from_more():
    fitted = fit_model_j(_rows())
    assert predict_model_j(fitted.params, admission_deficit=0) == 0.0
    val1 = predict_model_j(fitted.params, admission_deficit=1)
    val2 = predict_model_j(fitted.params, admission_deficit=2)
    assert abs(val1 - 155.0) < 1.0
    # with near-plateau data (155 -> 158), C_more should be small, unlike model H's forced-through-origin slope
    assert abs(val2 - val1) < 10.0


def test_model_k_requires_observed_running_deficit():
    fitted = fit_model_k(_rows())
    assert predict_model_k(fitted.params, observed_running_deficit=None) == 0.0
    assert predict_model_k(fitted.params, observed_running_deficit=2) >= 0.0


def test_model_l_scheduler_token_regime_gate():
    fitted = fit_model_l(_rows(), token_threshold=40.0)
    assert predict_model_l(fitted.params, iteration_tokens_mean=10.0) == 0.0
    assert predict_model_l(fitted.params, iteration_tokens_mean=130.0) > 0.0


# --- 8. separate transient and sustained labels (Model M) ---
def test_model_m_peak_stall_adds_transient_term_to_sustained():
    val = predict_model_m_peak_stall(sustained_ms=155.0, measured_prefill_ms=100.0, c_transient=2.0)
    assert val == 155.0 + 100.0 * 2.0


def test_model_m_peak_stall_none_when_prefill_missing():
    assert predict_model_m_peak_stall(155.0, None, 2.0) is None


def test_fit_model_m_transient_term_isolates_residual_above_sustained():
    rows = [
        {"peak_stall_ms": 500.0, "interference_ms": 155.0, "measured_new_request_prefill_ms": 100.0, "max_num_seqs": 2},
        {"peak_stall_ms": 900.0, "interference_ms": 158.0, "measured_new_request_prefill_ms": 300.0, "max_num_seqs": 2},
    ]
    fitted = fit_model_m_transient_term(rows)
    assert fitted.params["C_transient"] > 0


# --- 11. positive-row and negative-row error reporting ---
def test_evaluate_separates_positive_and_negative_row_errors():
    fitted = fit_model_g(_rows())
    result = evaluate(predict_model_g, fitted, _rows())
    assert result["n_positive_rows"] == 2
    assert result["n_negative_rows"] == 3
    assert result["positive_row_mae"] is not None
    assert result["negative_row_mae"] is not None


# --- 10. interference-regime classification metrics ---
def test_classification_metrics_reports_confusion_matrix():
    fitted = fit_model_g(_rows())
    result = classification_metrics(predict_model_g, fitted, _rows())
    assert result["tp"] == 2
    assert result["tn"] == 3
    assert result["accuracy"] == 1.0


def test_all_models_registered():
    assert set(MODELS) == {"G_binary_deficit", "H_linear_deficit", "I_utilization_threshold",
                            "J_piecewise_deficit", "K_observed_running_deficit", "L_scheduler_token_regime"}


# --- 16. stable serialization ---
def test_fitted_model_serializes_stably():
    import json
    fitted = fit_model_g(_rows())
    blob1 = json.dumps(fitted.to_dict(), sort_keys=True)
    blob2 = json.dumps(fitted.to_dict(), sort_keys=True)
    assert blob1 == blob2
