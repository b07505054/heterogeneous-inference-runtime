from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


POLICY_NAME = "coreml_edge_policy"
TRUTH_BOUNDARY = (
    "Policy selection from measured CoreML artifacts; no custom CoreML kernel "
    "or ANE scheduling claim."
)
VALID_PREFERENCES = {"latency", "size", "memory"}


def generate_coreml_edge_policy(
    baseline_paths: Iterable[str | Path],
    *,
    max_p95_ms: float,
    max_package_mb: float,
    max_drift: float,
    prefer: str,
    capability_profile: Any | None = None,
) -> dict[str, Any]:
    """Select a CoreML deployment option from existing measured artifacts."""

    constraints = {
        "max_p95_ms": float(max_p95_ms),
        "max_package_mb": float(max_package_mb),
        "max_drift": float(max_drift),
        "prefer": _validate_preference(prefer),
    }
    measured_support = _capability_measured_support(capability_profile)
    candidates = [
        _candidate_from_path(Path(path), constraints, measured_support)
        for path in baseline_paths
    ]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]

    selected = None
    status = "no_eligible_candidate"
    decision_reason = _no_eligible_reason(candidates)
    if eligible:
        selected_candidate = min(eligible, key=_selection_key(constraints["prefer"]))
        status = "selected"
        selected = {
            "input_size": selected_candidate["input_size"],
            "compression": selected_candidate["compression"],
            "compute_unit": selected_candidate["compute_unit"],
            "source_artifact": selected_candidate["artifact"],
        }
        decision_reason = _selected_reason(selected_candidate, constraints["prefer"])

    return {
        "artifact_type": "optimization_policy",
        "policy_name": POLICY_NAME,
        "evidence_type": "policy_from_measured_baselines",
        "status": status,
        "selected": selected,
        "constraints": constraints,
        "candidates": candidates,
        "decision_reason": decision_reason,
        "truth_boundary": TRUTH_BOUNDARY,
    }


def load_capability_profile(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_policy(path: str | Path, policy: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def _candidate_from_path(
    path: Path,
    constraints: dict[str, Any],
    measured_support: set[str] | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    payload: dict[str, Any] = {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"artifact_unreadable: {exc}")

    if payload:
        reasons.extend(_artifact_rejection_reasons(payload))

    target = _dict_at(payload, "benchmark_target")
    metrics = _dict_at(payload, "metrics")
    model_metrics = _dict_at(metrics, "model")
    coreml = _dict_at(metrics, "coreml")
    coreml_metrics = _dict_at(coreml, "metrics")

    input_size = _number_at(target, "input_size")
    if input_size is None:
        input_size = _number_at(model_metrics, "input_size")

    compression = _string_at(target, "model_compression")
    if compression is None:
        compression = _string_at(model_metrics, "compression")

    compute_unit = _string_at(_dict_at(payload, "execution"), "compute_unit")
    p95_ms = _number_at(_dict_at(coreml_metrics, "steady_state_latency_ms"), "p95")
    package_mb = _number_at(coreml_metrics, "package_size_mb")
    if package_mb is None:
        package_mb = _number_at(model_metrics, "package_size_mb")
    rss_mb = _number_at(coreml_metrics, "rss_delta_mb")
    drift = _drift_value(coreml_metrics.get("numerical_drift") if isinstance(coreml_metrics, dict) else None)

    required = {
        "input_size": input_size,
        "compression": compression,
        "compute_unit": compute_unit,
        "p95_ms": p95_ms,
        "package_mb": package_mb,
        "rss_mb": rss_mb,
        "drift": drift,
    }
    for name, value in required.items():
        if value is None:
            reasons.append(f"missing_{name}")

    if _string_at(coreml, "status") != "ok":
        reasons.append("coreml_status_not_ok")

    if measured_support is not None and str(path) not in measured_support:
        reasons.append("not_in_capability_measured_support")

    if p95_ms is not None and p95_ms > constraints["max_p95_ms"]:
        reasons.append("p95_exceeds_max")
    if package_mb is not None and package_mb > constraints["max_package_mb"]:
        reasons.append("package_exceeds_max")
    if drift is not None and drift > constraints["max_drift"]:
        reasons.append("drift_exceeds_max")

    return {
        "artifact": str(path),
        "input_size": int(input_size) if input_size is not None else None,
        "compression": compression,
        "compute_unit": compute_unit,
        "p95_ms": p95_ms,
        "package_mb": package_mb,
        "rss_mb": rss_mb,
        "drift": drift,
        "eligible": not reasons,
        "reasons": reasons,
    }


def _artifact_rejection_reasons(payload: dict[str, Any]) -> list[str]:
    reasons = []
    if payload.get("artifact_type") != "measured_baseline":
        reasons.append("artifact_type_not_measured_baseline")
    if payload.get("evidence_type") != "measured":
        reasons.append("evidence_type_not_measured")
    target = _dict_at(payload, "benchmark_target")
    if target.get("kind") != "native_coreml_cv":
        reasons.append("benchmark_target_not_native_coreml_cv")
    if target.get("backend") != "coreml":
        reasons.append("benchmark_target_not_coreml")
    if payload.get("status") not in {"ok", "partial"}:
        reasons.append("artifact_status_not_measured_success")
    return reasons


def _capability_measured_support(profile: Any | None) -> set[str] | None:
    if profile is None:
        return None
    if is_dataclass(profile):
        profile = asdict(profile)
    support = profile.get("measured_support", []) if isinstance(profile, dict) else []
    artifact_paths = set()
    for item in support:
        if is_dataclass(item):
            item = asdict(item)
        if not isinstance(item, dict):
            continue
        if item.get("evidence", "measured") != "measured":
            continue
        if item.get("status") != "ok":
            continue
        artifact_path = item.get("measured_artifact_path")
        if isinstance(artifact_path, str):
            artifact_paths.add(artifact_path)
    return artifact_paths


def _selection_key(prefer: str):
    if prefer == "latency":
        return lambda candidate: (
            candidate["p95_ms"],
            candidate["package_mb"],
            candidate["rss_mb"],
            candidate["artifact"],
        )
    if prefer == "size":
        return lambda candidate: (
            candidate["package_mb"],
            candidate["p95_ms"],
            candidate["rss_mb"],
            candidate["artifact"],
        )
    return lambda candidate: (
        candidate["rss_mb"],
        candidate["p95_ms"],
        candidate["package_mb"],
        candidate["artifact"],
    )


def _selected_reason(candidate: dict[str, Any], prefer: str) -> str:
    metric = {
        "latency": f"lowest eligible p95 latency ({candidate['p95_ms']} ms)",
        "size": f"lowest eligible package size ({candidate['package_mb']} MB)",
        "memory": f"lowest eligible RSS delta ({candidate['rss_mb']} MB)",
    }[prefer]
    return f"Selected {candidate['artifact']} because prefer={prefer} chose the {metric}."


def _no_eligible_reason(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "No baseline artifacts were provided."
    reason_counts: dict[str, int] = {}
    for candidate in candidates:
        for reason in candidate["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if not reason_counts:
        return "No eligible candidates were found."
    details = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
    return f"No candidate satisfied the measured-artifact and constraint checks: {details}."


def _validate_preference(prefer: str) -> str:
    if prefer not in VALID_PREFERENCES:
        raise ValueError(f"prefer must be one of {sorted(VALID_PREFERENCES)}")
    return prefer


def _dict_at(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, dict) and isinstance(value.get(key), dict):
        return value[key]
    return {}


def _string_at(value: Any, key: str) -> str | None:
    if isinstance(value, dict) and isinstance(value.get(key), str):
        return value[key]
    return None


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
