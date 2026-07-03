from types import SimpleNamespace

import pytest

from scripts import benchmark_coreml_cv_baseline as benchmark_script
from scripts import export_coreml_mobilenetv2 as export_script


def test_export_coreml_cli_parses_precision_and_compression_defaults():
    args = export_script.build_parser().parse_args([])
    assert args.precision == "fp16"
    assert args.compression == "none"
    assert args.output is None


def test_export_coreml_cli_parses_palettized_output():
    args = export_script.build_parser().parse_args(
        [
            "--precision",
            "fp16",
            "--compression",
            "palettize",
            "--output",
            "models/coreml/mobilenet_v2_fp16_palettized.mlpackage",
        ]
    )
    assert args.precision == "fp16"
    assert args.compression == "palettize"
    assert args.output.endswith("palettized.mlpackage")


def test_benchmark_coreml_cli_parses_model_metadata_options():
    args = benchmark_script.build_parser().parse_args(
        ["--model-precision", "fp16", "--model-compression", "palettize"]
    )
    assert args.model_precision == "fp16"
    assert args.model_compression == "palettize"


def test_benchmark_model_metadata_includes_precision_compression_and_size(tmp_path):
    package = tmp_path / "model.mlpackage"
    package.mkdir()
    (package / "weights.bin").write_bytes(b"1234")
    metadata = {
        "name": "MobileNetV2",
        "precision": "fp16",
        "compression": "palettize",
        "package_size_mb": benchmark_script.package_size_mb(package),
    }
    assert metadata["precision"] == "fp16"
    assert metadata["compression"] == "palettize"
    assert metadata["package_size_mb"] is not None


def test_palettize_missing_coremltools_optimize_fails_cleanly(capsys):
    fake_ct = SimpleNamespace()
    with pytest.raises(SystemExit) as exc:
        export_script.palettize_model(fake_ct, object())
    assert exc.value.code == 2
    assert "palettization unavailable" in capsys.readouterr().err


def test_missing_coremltools_fp16_precision_fails_cleanly(capsys):
    fake_ct = SimpleNamespace(precision=SimpleNamespace())
    with pytest.raises(SystemExit) as exc:
        export_script._coreml_precision(fake_ct, "fp16")
    assert exc.value.code == 2
    assert "FP16 export unavailable" in capsys.readouterr().err
