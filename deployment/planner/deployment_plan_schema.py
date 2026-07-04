from __future__ import annotations

from typing import Any


PLANNER_VERSION = "v1"
TRUTH_BOUNDARY = "Deployment recommendation only. No runtime optimization or backend modification."


def build_deployment_plan(
    *,
    status: str,
    selected_candidate: dict[str, Any] | None,
    constraints: dict[str, Any],
    objective: str,
    decision_reason: list[str],
    source_artifacts: list[str],
) -> dict[str, Any]:
    selected_runtime = selected_candidate.get("runtime") if selected_candidate else None
    selected_policy = selected_candidate.get("policy") if selected_candidate else None
    selected_config = selected_candidate.get("config") if selected_candidate else None

    return {
        "artifact_type": "deployment_plan",
        "planner_version": PLANNER_VERSION,
        "status": status,
        "selected_runtime": selected_runtime,
        "selected_policy": selected_policy,
        "selected_candidate": selected_config,
        "constraints": constraints,
        "objective": objective,
        "decision_reason": decision_reason,
        "source_artifacts": source_artifacts,
        "truth_boundary": TRUTH_BOUNDARY,
    }
