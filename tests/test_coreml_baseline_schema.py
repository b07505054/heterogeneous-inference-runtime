from benchmark.backends import coreml
from benchmark.backends.coreml import CoreMLMobileNetV2Backend
from benchmark.exporters import REQUIRED_MEASURED_KEYS, measured_envelope


def test_coreml_backend_unavailable_schema(monkeypatch):
    monkeypatch.setattr(coreml, "coreml_available", lambda: False)
    result = CoreMLMobileNetV2Backend("missing.mlpackage").setup()
    assert result["status"] == "unavailable"
    assert result["reason"] == "coremltools_not_installed"
    assert result["metrics"] == {}


def test_coreml_measured_envelope_allows_partial_unavailable_backend():
    payload = measured_envelope(
        artifact_type="coreml_mobilenetv2_baseline",
        benchmark_target={"kind": "native_coreml_cv"},
        metrics={"coreml": {"status": "unavailable", "metrics": {}}},
        notes=[],
        command=["script"],
        status="partial",
        hardware={},
        software={},
    )
    assert REQUIRED_MEASURED_KEYS <= payload.keys()
    assert payload["status"] == "partial"
