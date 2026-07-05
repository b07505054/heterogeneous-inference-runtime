import sys
from dataclasses import fields

import pytest

from deployment.model_adapter.artifact import ModelArtifact
from deployment.model_adapter.base import ModelAdapter
from deployment.model_adapter.mock_adapter import MockModelAdapter
from deployment.model_adapter.registry import (
    create_adapter,
    list_adapters,
    register_adapter,
)


def test_mock_adapter_registered_by_default():
    assert list_adapters() == ["mock"]


def test_create_mock_adapter():
    adapter = create_adapter("mock", graph_kind="cv", graph_id="registered_mock")

    assert isinstance(adapter, MockModelAdapter)
    graph = adapter.load()
    assert graph.graph_id == "registered_mock"
    assert graph.model_family == "cv"


def test_unknown_adapter_kind_raises_clear_error():
    with pytest.raises(ValueError, match="unknown model adapter kind 'onnx'"):
        create_adapter("onnx")


def test_registry_does_not_import_source_format_libraries():
    forbidden = {
        "onnx",
        "onnxruntime",
        "tensorrt",
        "torch",
        "vllm",
    }

    assert forbidden.isdisjoint(sys.modules)


def test_model_artifact_is_neutral_and_does_not_require_compiler_ir():
    artifact = ModelArtifact(
        kind="mock",
        uri=None,
        metadata={"purpose": "test"},
    )

    assert artifact.validate() == []
    field_names = {field.name.lower() for field in fields(ModelArtifact)}
    assert field_names == {"kind", "uri", "metadata"}
    assert "compiler_ir" not in field_names
    assert "execution_plan" not in field_names
    assert "model_name" not in field_names


def test_model_artifact_allows_future_kinds_without_loading_adapters():
    artifact = ModelArtifact(kind="vllm_endpoint", uri="http://example.invalid")

    assert artifact.validate() == []
    assert artifact.kind == "vllm_endpoint"


def test_register_adapter_rejects_non_adapter_class():
    class NotAnAdapter:
        pass

    with pytest.raises(TypeError, match="ModelAdapter subclass"):
        register_adapter("bad", NotAnAdapter)


def test_register_adapter_accepts_adapter_subclass():
    class LocalAdapter(ModelAdapter):
        def load(self):
            return MockModelAdapter(graph_kind="cv").load()

    register_adapter("local_test_adapter", LocalAdapter)
    try:
        assert "local_test_adapter" in list_adapters()
        assert isinstance(create_adapter("local_test_adapter"), LocalAdapter)
    finally:
        # The production registry currently has no unregister API. Keep this
        # test-local registration from affecting later assertions.
        from deployment.model_adapter import registry

        registry._ADAPTERS.pop("local_test_adapter", None)
