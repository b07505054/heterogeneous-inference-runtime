from types import SimpleNamespace

import numpy as np

from benchmark.backends import coreml
from benchmark.backends.coreml import CoreMLMobileNetV2Backend, resolve_compute_unit
from benchmark.exporters import measured_envelope
from scripts import benchmark_coreml_cv_baseline as coreml_script


def test_coreml_compute_unit_cli_parsing():
    parser = coreml_script.build_parser()
    assert parser.parse_args([]).compute_unit == "all"
    assert parser.parse_args(["--compute-unit", "cpu"]).compute_unit == "cpu"
    assert parser.parse_args(["--compute-unit", "cpu_gpu"]).compute_unit == "cpu_gpu"
    assert parser.parse_args(["--compute-unit", "all"]).compute_unit == "all"


def test_compute_unit_mapping_with_fake_coremltools():
    fake_ct = SimpleNamespace(
        ComputeUnit=SimpleNamespace(
            CPU_ONLY="cpu_only_enum",
            CPU_AND_GPU="cpu_and_gpu_enum",
            ALL="all_enum",
        )
    )
    assert resolve_compute_unit("cpu", fake_ct) == "cpu_only_enum"
    assert resolve_compute_unit("cpu_gpu", fake_ct) == "cpu_and_gpu_enum"
    assert resolve_compute_unit("all", fake_ct) == "all_enum"


def test_coreml_schema_includes_backend_and_execution_compute_unit():
    payload = measured_envelope(
        artifact_type="coreml_mobilenetv2_baseline",
        benchmark_target={
            "kind": "native_coreml_cv",
            "backend": "coreml",
        },
        metrics={"coreml": {"status": "unavailable", "metrics": {}}},
        notes=[],
        command=["script", "--compute-unit", "cpu_gpu"],
        status="partial",
        hardware={},
        software={},
        extra={"execution": {"compute_unit": "cpu_gpu"}},
    )
    assert payload["benchmark_target"]["backend"] == "coreml"
    assert payload["execution"]["compute_unit"] == "cpu_gpu"


def test_coreml_unavailable_preserves_requested_compute_unit(monkeypatch):
    monkeypatch.setattr(coreml, "coreml_available", lambda: False)
    backend = CoreMLMobileNetV2Backend("missing.mlpackage", compute_unit="cpu")
    result = backend.setup(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert backend.compute_unit == "cpu"
    assert result["status"] == "unavailable"
    assert result["reason"] == "coremltools_not_installed"
