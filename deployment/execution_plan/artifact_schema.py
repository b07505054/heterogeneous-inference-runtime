"""Runtime measured artifact schema helpers for ExecutionPath outputs."""

from __future__ import annotations

from typing import Any

from deployment.execution_plan.schema import ExecutionPath


def planned_runtime_path_artifact(path: ExecutionPath) -> dict[str, Any]:
    """Return an unmeasured artifact shell for a materialized execution path."""
    return {
        "artifact_type": "runtime_path_measurement",
        "schema_version": "1.0.0",
        "evidence_type": "planned",
        "status": "planned_not_executed",
        "execution_path": {
            "path_id": path.path_id,
            "kind": path.path_kind.value,
            "backend": path.selected_backend,
            "method": path.execution_method.value,
            "stage_id": path.stage_id,
        },
        "metrics": {},
        "truth_boundary": path.truth_boundary,
    }
