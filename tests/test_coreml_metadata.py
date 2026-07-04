import json
from pathlib import Path

import pytest

from deployment.model_adapter.coreml_metadata import (
    CoreMLMetadataError,
    load_coreml_compiler_metadata,
    parse_coreml_compiler_metadata,
)
from deployment.model_adapter.neutral_runtime_graph import NeutralRuntimeGraph


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "coreml_mock_package"
PACKAGE_PATH = FIXTURE_DIR / "MockModel.mlpackage"


def test_valid_coreml_metadata_parse():
    metadata = load_coreml_compiler_metadata(PACKAGE_PATH)

    assert metadata.validate() == []
    assert metadata.model_family == "cv"
    assert metadata.preferred_backend == "coreml"
    assert metadata.compiler_version == "fixture"
    assert metadata.source_artifact_kind == "mock"
    assert [stage.stage_id for stage in metadata.stages] == [
        "preprocess",
        "inference",
    ]
    assert metadata.input_tensors[0].name == "input_tensor"
    assert metadata.output_tensors[0].name == "output_tensor"
    assert metadata.kv_cache_requirements.required is False


def test_missing_metadata_error(tmp_path):
    package = tmp_path / "MissingMetadata.mlpackage"
    package.mkdir()

    with pytest.raises(CoreMLMetadataError, match="compiler_metadata.json not found"):
        load_coreml_compiler_metadata(package)


def test_invalid_metadata_error(tmp_path):
    package = tmp_path / "InvalidMetadata.mlpackage"
    package.mkdir()
    (tmp_path / "compiler_metadata.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(CoreMLMetadataError, match="invalid compiler_metadata.json"):
        load_coreml_compiler_metadata(package)


def test_metadata_can_be_converted_toward_neutral_runtime_graph_fields():
    metadata = load_coreml_compiler_metadata(PACKAGE_PATH)
    graph = NeutralRuntimeGraph(
        graph_id="from_coreml_metadata",
        model_family=metadata.model_family,
        stages=metadata.stages,
        tensors=metadata.tensors,
        memory_requirements=metadata.memory_requirements,
        kv_cache_requirements=metadata.kv_cache_requirements,
        backend_target=metadata.backend_target,
        constraints=metadata.constraints,
        metadata=metadata.metadata,
    )

    assert graph.validate() == []
    assert graph.backend_target.preferred_backend == "coreml"
    assert graph.tensors[0].name == "input_tensor"


def test_metadata_does_not_require_execution_plan():
    payload = json.loads((FIXTURE_DIR / "compiler_metadata.json").read_text())

    assert "execution_plan" not in payload
    assert "function_plans" not in payload
    assert "compiler_ir" not in payload
    assert parse_coreml_compiler_metadata(payload).validate() == []


def test_metadata_rejects_execution_plan_fields():
    payload = json.loads((FIXTURE_DIR / "compiler_metadata.json").read_text())
    payload["execution_plan"] = {"artifact_type": "execution_plan"}

    with pytest.raises(CoreMLMetadataError, match="compiler IR fields"):
        parse_coreml_compiler_metadata(payload)


def test_missing_package_path_error(tmp_path):
    package = tmp_path / "DoesNotExist.mlpackage"

    with pytest.raises(CoreMLMetadataError, match="does not exist"):
        load_coreml_compiler_metadata(package)


def test_required_field_validation_error(tmp_path):
    package = tmp_path / "BadMetadata.mlpackage"
    package.mkdir()
    payload = json.loads((FIXTURE_DIR / "compiler_metadata.json").read_text())
    payload.pop("model_family")
    (tmp_path / "compiler_metadata.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CoreMLMetadataError, match="missing required metadata field"):
        load_coreml_compiler_metadata(package)
