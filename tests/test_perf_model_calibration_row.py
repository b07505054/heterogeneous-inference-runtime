from perf_model import calibration_row as cr


def test_compute_error_basic():
    err = cr.compute_error(100.0, 120.0)
    assert err["prediction_available"] is True
    assert err["measurement_available"] is True
    assert abs(err["absolute_error"] - 20.0) < 1e-9
    assert abs(err["relative_error"] - (20.0 / 120.0)) < 1e-9


def test_compute_error_unsupported_prediction():
    err = cr.compute_error(None, 120.0)
    assert err["prediction_available"] is False
    assert err["absolute_error"] is None


def test_compute_error_missing_measurement():
    err = cr.compute_error(100.0, None)
    assert err["measurement_available"] is False


def test_distribution_summary_shape():
    summary = cr.distribution_summary([10.0, 12.0, 11.0, 50.0])
    assert summary["count"] == 4
    assert summary["min"] == 10.0
    assert summary["max"] == 50.0
    assert summary["relative_mad"] is not None


def test_attribute_error_unsupported_term_short_circuits():
    err = {"prediction_available": False, "measurement_available": True}
    findings = cr.attribute_error(
        metric_name="predicted_tpot_ms", error=err, measured_distribution=None,
        adherence_mismatches=[], concurrency=1, resolved_max_num_seqs=4, warmup_count=3,
    )
    assert findings == [{"category": "UNSUPPORTED_PREDICTION_TERM",
                          "explanation": findings[0]["explanation"]}]


def test_attribute_error_small_error_produces_no_findings():
    err = cr.compute_error(100.0, 101.0)
    findings = cr.attribute_error(
        metric_name="predicted_ttft_ms", error=err, measured_distribution=None,
        adherence_mismatches=[], concurrency=1, resolved_max_num_seqs=4, warmup_count=3,
    )
    assert findings == []


def test_attribute_error_flags_config_mismatch():
    err = cr.compute_error(100.0, 200.0)
    findings = cr.attribute_error(
        metric_name="predicted_ttft_ms", error=err, measured_distribution=None,
        adherence_mismatches=["block_size"], concurrency=1, resolved_max_num_seqs=4, warmup_count=3,
    )
    categories = {f["category"] for f in findings}
    assert "VLLM_DERIVED_CONFIG_DIFFERENCE" in categories


def test_attribute_error_flags_batching_and_queue_when_concurrency_exceeds_admission():
    err = cr.compute_error(50.0, 300.0)
    findings = cr.attribute_error(
        metric_name="predicted_ttft_ms", error=err, measured_distribution=None,
        adherence_mismatches=[], concurrency=8, resolved_max_num_seqs=2, warmup_count=3,
    )
    categories = {f["category"] for f in findings}
    assert "BATCHING_INTERACTION_ERROR" in categories
    assert "QUEUE_MODEL_ERROR" in categories


def test_attribute_error_flags_measurement_instability():
    err = cr.compute_error(100.0, 200.0)
    unstable = cr.distribution_summary([50.0, 400.0, 60.0, 350.0])
    findings = cr.attribute_error(
        metric_name="predicted_e2e_ms", error=err, measured_distribution=unstable,
        adherence_mismatches=[], concurrency=1, resolved_max_num_seqs=4, warmup_count=3,
    )
    categories = {f["category"] for f in findings}
    assert "MEASUREMENT_INSTABILITY" in categories


def test_attribute_error_defaults_to_runtime_overhead_when_nothing_else_applies():
    err = cr.compute_error(100.0, 150.0)  # relative error 0.333, above the 0.30 attribution threshold
    stable = cr.distribution_summary([149.0, 150.0, 151.0, 150.5])
    findings = cr.attribute_error(
        metric_name="predicted_e2e_ms", error=err, measured_distribution=stable,
        adherence_mismatches=[], concurrency=1, resolved_max_num_seqs=4, warmup_count=3,
    )
    categories = {f["category"] for f in findings}
    assert "RUNTIME_OVERHEAD_ERROR" in categories


def test_build_calibration_row_shape():
    row = cr.build_calibration_row(
        identity={"experiment_id": "e1"}, configuration={"requested": {}, "resolved": {}},
        predictions={"predicted_ttft_ms": 100.0}, measurements={"ttft_ms": {"median": 110.0}},
        errors={"predicted_ttft_ms": {"absolute_error": 10.0}},
    )
    assert row["identity"]["experiment_id"] == "e1"
    assert row["calibration_row_schema_version"] == "perf_model.calibration_row.v1"
