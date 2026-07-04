"""Neutral model adapter interfaces.

This package is intentionally not wired into runtime execution yet. It defines
the schema boundary that future source-specific adapters will target.
"""

from deployment.model_adapter.base import ModelAdapter
from deployment.model_adapter.artifact import ModelArtifact
from deployment.model_adapter.mock_adapter import MockModelAdapter
from deployment.model_adapter.neutral_runtime_graph import (
    NeutralBackendTarget,
    NeutralConstraint,
    NeutralKVCacheRequirement,
    NeutralMemoryRequirement,
    NeutralRuntimeGraph,
    NeutralStage,
    NeutralTensor,
)
from deployment.model_adapter.registry import (
    create_adapter,
    list_adapters,
    register_adapter,
)

__all__ = [
    "ModelAdapter",
    "ModelArtifact",
    "MockModelAdapter",
    "NeutralBackendTarget",
    "NeutralConstraint",
    "NeutralKVCacheRequirement",
    "NeutralMemoryRequirement",
    "NeutralRuntimeGraph",
    "NeutralStage",
    "NeutralTensor",
    "create_adapter",
    "list_adapters",
    "register_adapter",
]
