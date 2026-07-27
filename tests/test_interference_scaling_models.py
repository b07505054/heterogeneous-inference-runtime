from perf_model.interference_scaling_models import (
    fit_model_a, predict_model_a, fit_model_b, predict_model_b, fit_model_c, predict_model_c,
    fit_model_d, predict_model_d, fit_model_e, predict_model_e, fit_model_f, predict_model_f,
    evaluate, MODELS, FIXED_151MS_BASELINE,
)


def _rows():
    # a small, exactly-linear-in-tokens synthetic dataset: interference = 2.0 * tokens
    return [
        {"max_num_seqs": 2, "admitted_prompt_tokens": 64, "admitted_request_count": 1,
         "admitted_prefill_flops": 64_000.0, "measured_new_request_prefill_ms": 32.0, "interference_ms": 128.0},
        {"max_num_seqs": 2, "admitted_prompt_tokens": 128, "admitted_request_count": 1,
         "admitted_prefill_flops": 128_000.0, "measured_new_request_prefill_ms": 64.0, "interference_ms": 256.0},
        {"max_num_seqs": 8, "admitted_prompt_tokens": 512, "admitted_request_count": 2,
         "admitted_prefill_flops": 512_000.0, "measured_new_request_prefill_ms": 256.0, "interference_ms": 1024.0},
        {"max_num_seqs": 1, "admitted_prompt_tokens": 64, "admitted_request_count": 1,
         "admitted_prefill_flops": 64_000.0, "measured_new_request_prefill_ms": 32.0, "interference_ms": 0.0},
    ]


def test_model_b_recovers_true_linear_relationship():
    fitted = fit_model_b(_rows())
    assert abs(fitted.params["C_token"] - 2.0) < 0.05
    pred = predict_model_b(fitted.params, admitted_prompt_tokens=256)
    assert abs(pred - 512.0) < 20


def test_model_a_uses_median_of_regime_rows_only():
    fitted = fit_model_a(_rows())
    # only the 3 max_num_seqs>1 rows count: 128, 256, 1024 -> median 256
    assert abs(fitted.params["C_fixed"] - 256.0) < 1e-6
    assert predict_model_a(fitted.params, max_num_seqs=1) == 0.0
    assert predict_model_a(fitted.params, max_num_seqs=2) == fitted.params["C_fixed"]


def test_model_c_two_parameter_fit_runs_and_predicts():
    fitted = fit_model_c(_rows())
    assert set(fitted.params) == {"C_request", "C_token"}
    pred = predict_model_c(fitted.params, admitted_request_count=1, admitted_prompt_tokens=128)
    assert isinstance(pred, float)


def test_model_d_flop_scaled_matches_token_scaled_when_flops_proportional_to_tokens():
    fitted = fit_model_d(_rows())
    pred = predict_model_d(fitted.params, admitted_prefill_flops=256_000.0)
    assert abs(pred - 512.0) < 20  # flops = 1000*tokens in this synthetic set, so shape matches model B


def test_model_e_uses_measured_prefill_time():
    fitted = fit_model_e(_rows())
    pred = predict_model_e(fitted.params, measured_new_request_prefill_ms=128.0)
    assert abs(pred - 512.0) < 40  # prefill_ms = 0.5*tokens in this synthetic set


def test_model_f_piecewise_zero_at_max_num_seqs_one():
    fitted = fit_model_f(_rows())
    assert predict_model_f(fitted.params, max_num_seqs=1, admitted_prompt_tokens=999, admitted_request_count=99) == 0.0
    val = predict_model_f(fitted.params, max_num_seqs=2, admitted_prompt_tokens=128, admitted_request_count=1)
    assert val != 0.0


def test_evaluate_reports_near_zero_error_on_the_regime_gt_1_rows():
    # Model B has no regime gate, so it necessarily has residual on the
    # max_num_seqs=1 row (interference=0 despite nonzero tokens) -- evaluate
    # only the rows the linear relationship was actually built from.
    regime_rows = [r for r in _rows() if r["max_num_seqs"] > 1]
    fitted = fit_model_b(regime_rows)
    result = evaluate(predict_model_b, fitted, regime_rows)
    assert result["mae"] < 5.0
    assert result["n"] == 3


def test_evaluate_reports_nonzero_error_when_a_regime_zero_row_is_included():
    fitted = fit_model_b(_rows())
    result = evaluate(predict_model_b, fitted, _rows())
    assert result["mae"] > 5.0  # the max_num_seqs=1 row is a real, expected residual for this model
    assert result["n"] == 4


def test_all_six_models_registered():
    assert set(MODELS) == {"A_fixed_regime", "B_prompt_token_linear", "C_request_plus_token",
                            "D_prefill_flop_scaled", "E_measured_prefill_scaled", "F_piecewise_regime"}


def test_fitted_model_serializes_to_dict():
    fitted = fit_model_b(_rows())
    d = fitted.to_dict()
    assert d["name"] == "B_prompt_token_linear"
    assert "C_token" in d["params"]


def test_fixed_151ms_baseline_available_for_comparison():
    assert FIXED_151MS_BASELINE.params["C_fixed"] == 151.0


def test_models_fit_gracefully_on_empty_rows():
    for fit_fn, predict_fn in MODELS.values():
        fitted = fit_fn([])
        assert fitted.n_calibration_rows == 0
