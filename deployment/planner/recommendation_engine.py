from __future__ import annotations

from typing import Any

from deployment.planner.objective import balanced_score, objective_sort_key, validate_objective


def recommend_candidate(candidates: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    objective = validate_objective(objective)
    eligible = [candidate for candidate in candidates if candidate.get("eligible")]
    if not eligible:
        return {
            "status": "no_eligible_candidate",
            "selected_candidate": None,
            "decision_reason": [_no_eligible_reason(candidates)],
        }

    selected = min(eligible, key=lambda candidate: objective_sort_key(candidate, eligible, objective))
    return {
        "status": "selected",
        "selected_candidate": selected,
        "decision_reason": _decision_reason(selected, eligible, objective),
    }


def _decision_reason(selected: dict[str, Any], eligible: list[dict[str, Any]], objective: str) -> list[str]:
    metrics = selected.get("metrics", {})
    reasons = [
        f"Evaluated {len(eligible)} eligible deployment candidates.",
        f"Selected {selected['source_artifact']} for objective={objective}.",
    ]
    if objective == "latency":
        reasons.append(f"Latency objective chose lowest eligible latency_ms={metrics.get('latency_ms')}.")
    elif objective == "throughput":
        reasons.append(f"Throughput objective chose highest eligible throughput={metrics.get('throughput')}.")
    elif objective == "memory":
        reasons.append(f"Memory objective chose lowest eligible memory_mb={metrics.get('memory_mb')}.")
    elif objective in {"package_size", "size"}:
        reasons.append(f"Package-size objective chose lowest eligible package_mb={metrics.get('package_mb')}.")
    else:
        score = balanced_score(selected, eligible)
        reasons.append(
            "Balanced objective used weights latency=0.4, throughput=0.3, memory=0.2, package=0.1."
        )
        reasons.append(f"Selected candidate balanced_score={score}.")
    return reasons


def _no_eligible_reason(candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        return "No deployment candidates were available."
    reason_counts: dict[str, int] = {}
    for candidate in candidates:
        for reason in candidate.get("reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if not reason_counts:
        return "No eligible deployment candidates were available after filtering."
    details = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
    return f"No candidate satisfied deployment constraints: {details}."
