"""Backend adapter protocol for materializing ExecutionPath objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from deployment.execution_plan.capability_view import CapabilityValidationView
from deployment.execution_plan.schema import ExecutionPath


@dataclass(frozen=True)
class BackendMaterialization:
    backend: str
    method: str
    config: dict[str, Any]
    command: tuple[str, ...] | None
    benchmark_command: tuple[str, ...] | None
    expected_output_artifact: str
    truth_boundary: str


class BackendAdapter(Protocol):
    backend_id: str

    def supports(self, path: ExecutionPath, capabilities: CapabilityValidationView) -> bool:
        ...

    def validate(self, path: ExecutionPath, capabilities: CapabilityValidationView) -> list[str]:
        ...

    def materialize(self, path: ExecutionPath) -> BackendMaterialization:
        ...
