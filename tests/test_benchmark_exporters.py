from benchmark.exporters import REQUIRED_MEASURED_KEYS, measured_envelope


def test_measured_envelope_contains_required_schema():
    payload = measured_envelope(
        artifact_type="unit_test",
        benchmark_target={"kind": "test"},
        metrics={"x": 1},
        notes=["note"],
        command=["cmd"],
        software={"pkg": "1"},
        hardware={"cpu": "test"},
    )
    assert REQUIRED_MEASURED_KEYS <= payload.keys()
    assert payload["evidence_type"] == "measured"
    assert payload["benchmark_target"]["kind"] == "test"
    assert payload["metrics"] == {"x": 1}

