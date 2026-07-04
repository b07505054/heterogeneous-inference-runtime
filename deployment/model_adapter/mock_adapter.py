"""Test-only neutral graph adapter.

MockModelAdapter does not load models or contact external runtimes. It exists
to exercise scheduler/policy-facing schema code without optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from deployment.model_adapter.base import ModelAdapter
from deployment.model_adapter.neutral_runtime_graph import (
    NeutralBackendTarget,
    NeutralConstraint,
    NeutralKVCacheRequirement,
    NeutralMemoryRequirement,
    NeutralRuntimeGraph,
    NeutralStage,
    NeutralTensor,
)


@dataclass(frozen=True)
class MockModelAdapter(ModelAdapter):
    graph_kind: str = "cv"
    graph_id: str = "mock_graph"
    metadata: dict[str, Any] = field(default_factory=dict)

    def load(self) -> NeutralRuntimeGraph:
        if self.graph_kind == "cv":
            return self._cv_graph()
        if self.graph_kind == "llm":
            return self._llm_graph()
        raise ValueError(f"unsupported mock graph kind: {self.graph_kind}")

    def _cv_graph(self) -> NeutralRuntimeGraph:
        return NeutralRuntimeGraph(
            graph_id=self.graph_id,
            model_family="cv",
            stages=(
                NeutralStage(
                    stage_id="preprocess",
                    stage_type="preprocess",
                    inputs=("input_frame",),
                    outputs=("input_tensor",),
                ),
                NeutralStage(
                    stage_id="inference",
                    stage_type="inference",
                    inputs=("input_tensor",),
                    outputs=("output_tensor",),
                ),
                NeutralStage(
                    stage_id="postprocess",
                    stage_type="postprocess",
                    inputs=("output_tensor",),
                    outputs=("prediction",),
                    required=False,
                ),
            ),
            tensors=(
                NeutralTensor(
                    name="input_frame",
                    role="input",
                    shape=("height", "width", "channels"),
                    dtype="uint8",
                    layout="hwc",
                    dynamic=True,
                ),
                NeutralTensor(
                    name="input_tensor",
                    role="intermediate",
                    shape=("batch", "channels", "height", "width"),
                    dtype="fp32",
                    layout="nchw",
                    dynamic=True,
                ),
                NeutralTensor(
                    name="output_tensor",
                    role="output",
                    shape=("batch", "features"),
                    dtype="fp32",
                    layout="feature",
                    dynamic=True,
                ),
            ),
            memory_requirements=NeutralMemoryRequirement(
                estimated_static_mb=None,
                estimated_peak_mb=None,
                requires_unified_memory=None,
            ),
            kv_cache_requirements=NeutralKVCacheRequirement(required=False),
            backend_target=NeutralBackendTarget(
                preferred_backend="mock",
                allowed_backends=("mock",),
            ),
            constraints=(
                NeutralConstraint(name="latency_ms", value=100.0, operator="max"),
            ),
            metadata={"adapter": "mock", **dict(self.metadata)},
        )

    def _llm_graph(self) -> NeutralRuntimeGraph:
        return NeutralRuntimeGraph(
            graph_id=self.graph_id,
            model_family="llm",
            stages=(
                NeutralStage(
                    stage_id="prefill",
                    stage_type="prefill",
                    inputs=("prompt_tokens",),
                    outputs=("kv_cache",),
                ),
                NeutralStage(
                    stage_id="decode",
                    stage_type="decode",
                    inputs=("decode_token", "kv_cache"),
                    outputs=("next_token", "kv_cache"),
                ),
            ),
            tensors=(
                NeutralTensor(
                    name="prompt_tokens",
                    role="input",
                    shape=("batch", "sequence"),
                    dtype="token",
                    layout="sequence",
                    dynamic=True,
                ),
                NeutralTensor(
                    name="decode_token",
                    role="input",
                    shape=("batch", 1),
                    dtype="token",
                    layout="sequence",
                    dynamic=True,
                ),
                NeutralTensor(
                    name="kv_cache",
                    role="state",
                    shape=("layers", "tokens", "heads", "head_dim"),
                    dtype="fp16",
                    layout="cache",
                    dynamic=True,
                ),
                NeutralTensor(
                    name="next_token",
                    role="output",
                    shape=("batch", 1),
                    dtype="token",
                    layout="sequence",
                    dynamic=True,
                ),
            ),
            memory_requirements=NeutralMemoryRequirement(
                estimated_static_mb=None,
                estimated_peak_mb=None,
                requires_unified_memory=None,
            ),
            kv_cache_requirements=NeutralKVCacheRequirement(
                required=True,
                max_context_tokens=4096,
                bytes_per_token=1024,
                cache_layout="abstract",
            ),
            backend_target=NeutralBackendTarget(
                preferred_backend="mock",
                allowed_backends=("mock",),
            ),
            constraints=(
                NeutralConstraint(name="ttft_ms", value=500.0, operator="max"),
                NeutralConstraint(name="tpot_ms", value=50.0, operator="max"),
            ),
            metadata={"adapter": "mock", **dict(self.metadata)},
        )
