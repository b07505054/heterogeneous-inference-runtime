from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from capabilities.profile_loader import load_profile
from deployment.planner.constraint_solver import filter_candidates
from deployment.planner.deployment_plan_schema import build_deployment_plan
from deployment.planner.objective import validate_objective
from deployment.planner.recommendation_engine import recommend_candidate


class DeploymentPlanner:
    """Architecture layer that recommends a deployment plan from evidence."""

    def plan(
        self,
        *,
        profile_paths: Iterable[str | Path] = (),
        artifact_paths: Iterable[str | Path] = (),
        runtime: str | None,
        constraints: dict[str, Any],
        objective: str,
    ) -> dict[str, Any]:
        objective = validate_objective(objective)
        profiles = _load_profiles(profile_paths)
        candidates = _load_candidates(artifact_paths)
        if runtime:
            candidates = [candidate for candidate in candidates if candidate["runtime"] == runtime]

        constrained = filter_candidates(candidates, constraints)
        recommendation = recommend_candidate(constrained, objective)
        source_artifacts = sorted({candidate["source_artifact"] for candidate in constrained})
        decision_reason = [
            f"Loaded {len(profiles)} capability profiles.",
            f"Loaded {len(candidates)} deployment candidates for runtime={runtime or 'any'}.",
            *recommendation["decision_reason"],
        ]
        return build_deployment_plan(
            status=recommendation["status"],
            selected_candidate=recommendation["selected_candidate"],
            constraints=constraints,
            objective=objective,
            decision_reason=decision_reason,
            source_artifacts=source_artifacts,
        )


def plan_deployment(
    *,
    profile_paths: Iterable[str | Path] = (),
    artifact_paths: Iterable[str | Path] = (),
    runtime: str | None = None,
    constraints: dict[str, Any] | None = None,
    objective: str = "latency",
) -> dict[str, Any]:
    return DeploymentPlanner().plan(
        profile_paths=profile_paths,
        artifact_paths=artifact_paths,
        runtime=runtime,
        constraints=constraints or {},
        objective=objective,
    )


def write_plan(path: str | Path, plan: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def _load_profiles(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    profiles = []
    for path in paths:
        profiles.append(load_profile(path))
    return profiles


def _load_candidates(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    candidates = []
    for path in paths:
        candidate_payloads = _candidates_from_artifact(Path(path))
        candidates.extend(candidate_payloads)
    return candidates


def _candidates_from_artifact(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if payload.get("artifact_type") == "optimization_policy":
        return _candidates_from_policy(payload, path)
    if payload.get("artifact_type") == "measured_baseline":
        candidate = _candidate_from_measured_baseline(payload, path)
        return [candidate] if candidate else []
    return []


def _candidates_from_policy(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    policy_name = payload.get("policy_name")
    candidates = []
    for index, raw_candidate in enumerate(payload.get("candidates", [])):
        if not isinstance(raw_candidate, dict):
            continue
        candidate = _candidate_from_policy_candidate(policy_name, raw_candidate, path, index)
        if candidate:
            candidates.append(candidate)
    return candidates


def _candidate_from_policy_candidate(
    policy_name: str | None,
    raw: dict[str, Any],
    path: Path,
    index: int,
) -> dict[str, Any] | None:
    if policy_name == "coreml_edge_policy":
        return {
            "runtime": "coreml",
            "policy": "coreml_edge_policy",
            "config": _compact_config(raw, ["compute_unit", "compression", "input_size"]),
            "metrics": {
                "latency_ms": raw.get("p95_ms"),
                "package_mb": raw.get("package_mb"),
                "memory_mb": raw.get("rss_mb"),
                "drift": raw.get("drift"),
            },
            "source_artifact": raw.get("artifact") or f"{path}#candidate{index}",
            "reasons": list(raw.get("reasons", [])),
        }
    if policy_name == "server_runtime_policy":
        return {
            "runtime": "server",
            "policy": "server_runtime_policy",
            "config": _compact_config(raw, ["concurrency", "model", "max_model_len", "max_tokens"]),
            "metrics": {
                "latency_ms": raw.get("e2e_p95_ms"),
                "ttft_p95_ms": raw.get("ttft_p95_ms"),
                "tpot_p95_ms": raw.get("tpot_p95_ms"),
                "throughput": raw.get("tokens_per_second"),
            },
            "source_artifact": raw.get("artifact") or f"{path}#candidate{index}",
            "reasons": list(raw.get("reasons", [])),
        }
    return None


def _candidate_from_measured_baseline(payload: dict[str, Any], path: Path) -> dict[str, Any] | None:
    target = _dict_at(payload, "benchmark_target")
    if target.get("kind") == "native_coreml_cv" and target.get("backend") == "coreml":
        return _coreml_candidate(payload, path)
    if target.get("kind") == "openai_compatible_server":
        return _server_candidate(payload, path)
    return None


def _coreml_candidate(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    target = _dict_at(payload, "benchmark_target")
    metrics = _dict_at(payload, "metrics")
    model_metrics = _dict_at(metrics, "model")
    coreml_metrics = _dict_at(_dict_at(metrics, "coreml"), "metrics")
    latency = _number_at(_dict_at(coreml_metrics, "steady_state_latency_ms"), "p95")
    package = _number_at(coreml_metrics, "package_size_mb")
    if package is None:
        package = _number_at(model_metrics, "package_size_mb")
    input_size = _number_at(target, "input_size")
    if input_size is None:
        input_size = _number_at(model_metrics, "input_size")
    compression = target.get("model_compression") or model_metrics.get("compression")
    return {
        "runtime": "coreml",
        "policy": "coreml_edge_policy",
        "config": {
            "compute_unit": _dict_at(payload, "execution").get("compute_unit"),
            "compression": compression,
            "input_size": int(input_size) if input_size is not None else None,
        },
        "metrics": {
            "latency_ms": latency,
            "package_mb": package,
            "memory_mb": _number_at(coreml_metrics, "rss_delta_mb"),
            "drift": _drift_value(coreml_metrics.get("numerical_drift")),
        },
        "source_artifact": str(path),
        "reasons": [],
    }


def _server_candidate(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    target = _dict_at(payload, "benchmark_target")
    metrics = _dict_at(payload, "metrics")
    return {
        "runtime": "server",
        "policy": "server_runtime_policy",
        "config": _compact_config(target, ["concurrency", "model", "max_model_len", "max_tokens"]),
        "metrics": {
            "latency_ms": _number_at(_dict_at(metrics, "e2e_latency_ms"), "p95"),
            "ttft_p95_ms": _number_at(_dict_at(metrics, "ttft_ms"), "p95"),
            "tpot_p95_ms": _number_at(_dict_at(metrics, "tpot_ms"), "p95"),
            "throughput": _number_at(metrics, "tokens_per_second"),
        },
        "source_artifact": str(path),
        "reasons": ["error_count_nonzero"] if (_number_at(metrics, "error_count") or 0.0) > 0 else [],
    }


def _compact_config(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if payload.get(key) is not None}


def _dict_at(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get(key), dict):
        return value[key]
    return {}


def _number_at(value: Any, key: str) -> float | None:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return float(candidate)
    return None


def _drift_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        return _number_at(value, "max_abs")
    return None
