import pytest

from scripts import benchmark_coreml_cv_baseline as benchmark_script
from scripts import export_coreml_mobilenetv2 as export_script


def test_export_coreml_cli_parses_input_size():
    args = export_script.build_parser().parse_args(["--input-size", "256"])
    assert args.input_size == 256
    assert export_script.default_output(args.precision, args.compression, args.input_size).endswith(
        "mobilenet_v2_fp16_256.mlpackage"
    )


def test_benchmark_coreml_cli_parses_input_size():
    args = benchmark_script.build_parser().parse_args(["--input-size", "384"])
    assert args.input_size == 384


def test_benchmark_model_metadata_includes_input_size(tmp_path):
    package = tmp_path / "model.mlpackage"
    package.mkdir()
    (package / "weights.bin").write_bytes(b"1234")

    metadata = benchmark_script.model_metadata(
        package,
        precision="fp16",
        compression="none",
        input_size=384,
    )

    assert metadata["input_size"] == 384
    assert metadata["package_size_mb"] is not None


@pytest.mark.parametrize("script", [export_script, benchmark_script])
def test_invalid_input_size_rejected(script):
    with pytest.raises(SystemExit):
        script.build_parser().parse_args(["--input-size", "0"])
