from perf_model.batch_shape_models import (
    fit_model_n, predict_model_n, fit_model_o, predict_model_o, fit_model_p, predict_model_p,
    fit_model_q, predict_model_q, build_model_r, predict_model_r, fit_model_s, predict_model_s,
    evaluate, MODELS,
)


def _rows():
    # matches the real E2E-5/E2E-6 shape: batch=1 fast, batch>=2 flat-elevated
    return [
        {"batch_size": 1, "batch_step_ms": 11.7, "decode_flops": 1e9, "decode_bytes": 1e6},
        {"batch_size": 2, "batch_step_ms": 168.0, "decode_flops": 2e9, "decode_bytes": 2e6},
        {"batch_size": 4, "batch_step_ms": 169.0, "decode_flops": 4e9, "decode_bytes": 4e6},
        {"batch_size": 8, "batch_step_ms": 170.0, "decode_flops": 8e9, "decode_bytes": 8e6},
    ]


# --- 10. Models N-S ---
def test_model_n_binary_transition():
    fitted = fit_model_n(_rows())
    assert abs(fitted.params["C1"] - 11.7) < 0.01
    assert predict_model_n(fitted.params, batch_size=1) == fitted.params["C1"]
    assert predict_model_n(fitted.params, batch_size=4) == fitted.params["C_multi"]


def test_model_o_linear_scaling_fits_reasonably():
    fitted = fit_model_o(_rows())
    pred_at_2 = predict_model_o(fitted.params, batch_size=2)
    pred_at_8 = predict_model_o(fitted.params, batch_size=8)
    assert pred_at_8 >= pred_at_2  # linear model necessarily monotonic in batch_size


def test_model_p_graph_bucket_pads_to_nearest_capture_size():
    fitted = fit_model_p(_rows(), capture_sizes=[1, 2, 4])
    # batch=3 has no captured size, pads up to bucket 4
    pred_3 = predict_model_p(fitted.params, batch_size=3)
    pred_4 = predict_model_p(fitted.params, batch_size=4)
    assert pred_3 == pred_4  # both fall in the same bucket


def test_model_p_bucket_beyond_max_capture_size_uses_largest_bucket():
    fitted = fit_model_p(_rows(), capture_sizes=[1, 2, 4])
    pred_8 = predict_model_p(fitted.params, batch_size=8)  # exceeds max capture size 4
    pred_4 = predict_model_p(fitted.params, batch_size=4)
    assert pred_8 == pred_4


def test_model_q_roofline_calibrates_from_batch1_and_scales():
    fitted = fit_model_q(_rows())
    assert fitted.params["source"] == "calibrated_this_slice_batch1"
    pred_1 = predict_model_q(fitted.params, decode_flops=1e9, decode_bytes=1e6)
    assert abs(pred_1 - 11.7) < 0.5  # should reproduce the calibration point closely


def test_model_q_roofline_unavailable_without_batch1_row():
    fitted = fit_model_q([{"batch_size": 2, "batch_step_ms": 168.0}])
    assert fitted.params["source"] == "unavailable"
    assert predict_model_q(fitted.params, decode_flops=1e9, decode_bytes=1e6) is None


def test_model_r_unsupported_without_profiler_evidence():
    fitted = build_model_r(None)
    assert fitted.params["status"] == "unsupported_no_profiler_evidence"
    assert predict_model_r(fitted.params) is None


def test_model_r_sums_profiler_derived_components():
    fitted = build_model_r({"attention_ms": 50.0, "mlp_ms": 30.0, "logits_sampling_ms": 20.0})
    assert predict_model_r(fitted.params) == 100.0


def test_model_s_special_batch1_path_differs_from_general_path():
    fitted = fit_model_s(_rows())
    val_1 = predict_model_s(fitted.params, batch_size=1)
    val_2 = predict_model_s(fitted.params, batch_size=2)
    assert abs(val_1 - 11.7) < 0.01
    assert val_2 > val_1 * 5  # the special path must be materially different, not a smooth continuation


def test_evaluate_reports_mae_and_max_error():
    fitted = fit_model_n(_rows())
    result = evaluate(predict_model_n, fitted, _rows())
    assert result["n"] == 4
    assert result["mae"] is not None


def test_models_registered():
    assert set(MODELS) == {"N_binary_transition", "O_linear_scaling", "S_special_batch1_path"}
