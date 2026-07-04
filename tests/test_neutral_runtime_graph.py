from dataclasses import fields

from deployment.model_adapter import (
    MockModelAdapter,
    NeutralBackendTarget,
    NeutralKVCacheRequirement,
    NeutralRuntimeGraph,
    NeutralStage,
    NeutralTensor,
)


def test_neutral_runtime_graph_construction():
    graph = NeutralRuntimeGraph(
        graph_id="graph",
        model_family="generic",
        stages=(
            NeutralStage(
                stage_id="inference",
                stage_type="inference",
                inputs=("input",),
                outputs=("output",),
            ),
        ),
        tensors=(
            NeutralTensor(
                name="input",
                role="input",
                shape=("batch", "features"),
                dtype="fp32",
                dynamic=True,
            ),
            NeutralTensor(
                name="output",
                role="output",
                shape=("batch", "features"),
                dtype="fp32",
                dynamic=True,
            ),
        ),
        backend_target=NeutralBackendTarget(
            preferred_backend="generic",
            allowed_backends=("generic",),
        ),
    )

    assert graph.validate() == []
    assert graph.model_family == "generic"
    assert graph.stages[0].stage_type == "inference"
    assert graph.tensors[0].shape == ("batch", "features")


def test_mock_model_adapter_cv_graph():
    graph = MockModelAdapter(graph_kind="cv", graph_id="mock_cv").load()

    assert graph.validate() == []
    assert graph.graph_id == "mock_cv"
    assert graph.model_family == "cv"
    assert graph.kv_cache_requirements.required is False
    assert [stage.stage_id for stage in graph.stages] == [
        "preprocess",
        "inference",
        "postprocess",
    ]


def test_mock_model_adapter_llm_graph_with_kv_cache_requirement():
    graph = MockModelAdapter(graph_kind="llm", graph_id="mock_llm").load()

    assert graph.validate() == []
    assert graph.graph_id == "mock_llm"
    assert graph.model_family == "llm"
    assert graph.kv_cache_requirements.required is True
    assert graph.kv_cache_requirements.max_context_tokens == 4096
    assert graph.kv_cache_requirements.bytes_per_token == 1024
    assert {stage.stage_id for stage in graph.stages} == {"prefill", "decode"}


def test_schema_does_not_require_compiler_ir():
    graph = MockModelAdapter(graph_kind="cv").load()

    assert graph.validate() == []
    assert "compiler_ir" not in _field_names(NeutralRuntimeGraph)
    assert "execution_plan" not in _field_names(NeutralRuntimeGraph)
    assert "mlir" not in graph.metadata


def test_schema_does_not_require_model_specific_names():
    graph = NeutralRuntimeGraph(
        graph_id="anonymous",
        model_family="vision",
        stages=(NeutralStage(stage_id="run", stage_type="inference"),),
        tensors=(NeutralTensor(name="x", role="input"),),
        kv_cache_requirements=NeutralKVCacheRequirement(required=False),
    )

    assert graph.validate() == []
    schema_names = set(_field_names(NeutralRuntimeGraph))
    assert "model_name" not in schema_names
    assert "architecture_name" not in schema_names
    assert "qwen" not in schema_names
    assert "llama" not in schema_names
    assert "mobilenet" not in schema_names


def test_model_adapter_validate_returns_errors_for_unknown_mock_kind():
    errors = MockModelAdapter(graph_kind="unknown").validate()

    assert errors == ["adapter_load_failed:ValueError"]


def _field_names(cls):
    return [field.name.lower() for field in fields(cls)]
