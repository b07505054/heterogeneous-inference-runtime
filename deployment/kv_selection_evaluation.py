"""Exact-profile measured KV selection and regret calculations.

This is deliberately not a predictive cost model. Evidence is legal only for
an exact target/workload identity and every objective weight is explicit.
"""
from __future__ import annotations

from typing import Any

OBJECTIVES = {
    "latency": {"latency_weight": 1.0, "memory_weight": 0.0, "fragmentation_weight": 0.0,
                "max_latency_regression": 0.0},
    "memory_efficiency": {"latency_weight": 0.0, "memory_weight": 1.0, "fragmentation_weight": 1.0,
                          "max_latency_regression": 0.25},
    "balanced": {"latency_weight": 0.6, "memory_weight": 0.3, "fragmentation_weight": 0.1,
                 "max_latency_regression": 1.0},
}

def _identity(row: dict[str, Any]) -> tuple[str, str, str, int | None]:
    return (row["target_id"], row["workload_id"], row["candidate_id"], row.get("page_tokens"))

def select_measured(evidence: list[dict[str, Any]], *, target_id: str,
                    workload_id: str, objective: str,
                    fallback_candidate_id: str = "cpu_contiguous_kv_fp32_v1") -> dict[str, Any]:
    weights = OBJECTIVES[objective]
    rows = [r for r in evidence if r.get("target_id") == target_id and
            r.get("workload_id") == workload_id and r.get("correctness_passed") is True]
    if not rows:
        return {"candidate_id": fallback_candidate_id, "page_tokens": None,
                "selection_reason": "unseen_exact_profile_declared_contiguous_fallback",
                "evidence_identity": None, "objective": objective, "weights": weights}
    min_latency = min(float(r["append_decode_p95_ms"]) for r in rows)
    max_memory = max(int(r["request_owned_bytes"]) for r in rows) or 1
    eligible = rows
    if objective == "memory_efficiency":
        eligible = [r for r in rows if float(r["append_decode_p95_ms"]) <=
                    min_latency * (1 + weights["max_latency_regression"])]
    for r in eligible:
        r["objective_score"] = (weights["latency_weight"] * float(r["append_decode_p95_ms"]) / min_latency +
                                weights["memory_weight"] * int(r["request_owned_bytes"]) / max_memory +
                                weights["fragmentation_weight"] * float(r["internal_fragmentation_ratio"]))
    selected = min(eligible, key=lambda r: (r["objective_score"], _identity(r)))
    return {"candidate_id": selected["candidate_id"], "page_tokens": selected.get("page_tokens"),
            "selection_reason": "exact_target_workload_measured_profile_minimum_objective_score",
            "evidence_identity": list(_identity(selected)), "objective": objective,
            "weights": weights, "objective_score": selected["objective_score"]}

def regret(selected: dict[str, Any], oracle: dict[str, Any]) -> dict[str, float]:
    sl, ol = float(selected["append_decode_p95_ms"]), float(oracle["append_decode_p95_ms"])
    sm, om = int(selected["request_owned_bytes"]), int(oracle["request_owned_bytes"])
    sf, of = float(selected["internal_fragmentation_ratio"]), float(oracle["internal_fragmentation_ratio"])
    return {"absolute_latency_regret_ms": sl-ol, "relative_latency_regret": sl/ol-1,
            "memory_regret_bytes": sm-om, "fragmentation_regret": sf-of,
            "objective_score_regret": float(selected["objective_score"])-float(oracle["objective_score"])}

def admission_count(budget_bytes: int, lengths: list[tuple[int, float]], *, bytes_per_token: int,
                    contiguous_capacity: int, page_tokens: int) -> dict[str, int]:
    contiguous = budget_bytes // (contiguous_capacity * bytes_per_token)
    expected_paged = sum((((tokens + page_tokens - 1)//page_tokens)*page_tokens*bytes_per_token)*weight
                         for tokens, weight in lengths)
    return {"contiguous": contiguous, "paged_formula": int(budget_bytes // expected_paged)}
