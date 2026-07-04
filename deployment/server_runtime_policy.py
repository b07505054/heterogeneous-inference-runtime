from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


POLICY_NAME = "server_runtime_policy"
TRUTH_BOUNDARY = (
    "Policy selection from measured OpenAI-compatible/vLLM artifacts; no vLLM "
    "scheduler or kernel modification."
)
VALID_PREFERENCES = {"latency", "throughput"}


def generate_server_runtime_policy(
    baseline_paths: Iterable[str | Path],
    *,
    max_ttft_p95_ms: float,
    max_tpot_p95_ms: float,
    max_e2e_p95_ms: float,
    prefer: str,
    allow_errors: bool = False,
) -> dict[str, Any]:
    """Select server runtime settings from existing measured artifacts."""

    constraints = {
        "max_ttft_p95_ms": float(max_ttft_p95_ms),
        "max_tpot_p95_ms": float(max_tpot_p95_ms),
        "max_e2e_p95_ms": float(max_e2e_p95_ms),
        "prefer": _validate_preference(prefer),
        "allow_errors": bool(allow_errors),
    }
    candidates = [_candidate_from_path(Path(path), constraints) for path in baseline_paths]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]

    selected = None
    status = "no_eligible_candidate"
    decision_reason = _no_eligible_reason(candidates)
    if eligible:
        selected_candidate = min(eligible, key=_selection_key(constraints["prefer"]))
        status = "selected"
        selected = {
            "concurrency": selected_candidate["concurrency"],
            "model": selected_candidate["model"],
            "source_artifact": selected_candidate["artifact"],
        }
        if selected_candidate["max_model_len"] is not None:
            selected["max_model_len"] = selected_candidate["max_model_len"]
        if selected_candidate["max_tokens"] is not None:
            selected["max_tokens"] = selected_candidate["max_tokens"]
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


def write_policy(path: str | Path, policy: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")


def _candidate_from_path(path: Path, constraints: dict[str, Any]) -> dict[str, Any]:
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

    concurrency = _number_at(target, "concurrency")
    model = _string_at(target, "model")
    max_model_len = _number_at(target, "max_model_len")
    max_tokens = _number_at(target, "max_tokens")
    ttft_p95_ms = _number_at(_dict_at(metrics, "ttft_ms"), "p95")
    tpot_p95_ms = _number_at(_dict_at(metrics, "tpot_ms"), "p95")
    e2e_p95_ms = _number_at(_dict_at(metrics, "e2e_latency_ms"), "p95")
    tokens_per_second = _number_at(metrics, "tokens_per_second")
    success_count = _number_at(metrics, "success_count")
    error_count = _number_at(metrics, "error_count")

    required = {
        "concurrency": concurrency,
        "model": model,
        "ttft_p95_ms": ttft_p95_ms,
        "tpot_p95_ms": tpot_p95_ms,
        "e2e_p95_ms": e2e_p95_ms,
        "tokens_per_second": tokens_per_second,
        "success_count": success_count,
        "error_count": error_count,
    }
    for name, value in required.items():
        if value is None:
            reasons.append(f"missing_{name}")

    if error_count is not None and error_count > 0 and not constraints["allow_errors"]:
        reasons.append("error_count_nonzero")
    if ttft_p95_ms is not None and ttft_p95_ms > constraints["max_ttft_p95_ms"]:
        reasons.append("ttft_p95_exceeds_max")
    if tpot_p95_ms is not None and tpot_p95_ms > constraints["max_tpot_p95_ms"]:
        reasons.append("tpot_p95_exceeds_max")
    if e2e_p95_ms is not None and e2e_p95_ms > constraints["max_e2e_p95_ms"]:
        reasons.append("e2e_p95_exceeds_max")

    return {
        "artifact": str(path),
        "concurrency": int(concurrency) if concurrency is not None else None,
        "model": model,
        "max_model_len": int(max_model_len) if max_model_len is not None else None,
        "max_tokens": int(max_tokens) if max_tokens is not None else None,
        "ttft_p95_ms": ttft_p95_ms,
        "tpot_p95_ms": tpot_p95_ms,
        "e2e_p95_ms": e2e_p95_ms,
        "tokens_per_second": tokens_per_second,
        "success_count": int(success_count) if success_count is not None else None,
        "error_count": int(error_count) if error_count is not None else None,
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
    if target.get("kind") != "openai_compatible_server":
        reasons.append("benchmark_target_not_openai_compatible_server")
    if payload.get("status") not in {"ok", "partial"}:
        reasons.append("artifact_status_not_measured_success")
    return reasons


def _selection_key(prefer: str):
    if prefer == "latency":
        return lambda candidate: (
            candidate["tpot_p95_ms"],
            candidate["e2e_p95_ms"],
            candidate["ttft_p95_ms"],
            candidate["artifact"],
        )
    return lambda candidate: (
        -candidate["tokens_per_second"],
        candidate["tpot_p95_ms"],
        candidate["e2e_p95_ms"],
        candidate["artifact"],
    )


def _selected_reason(candidate: dict[str, Any], prefer: str) -> str:
    if prefer == "latency":
        metric = (
            f"lowest eligible TPOT p95 ({candidate['tpot_p95_ms']} ms), "
            f"then E2E p95 ({candidate['e2e_p95_ms']} ms)"
        )
    else:
        metric = f"highest eligible tokens/sec ({candidate['tokens_per_second']})"
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
