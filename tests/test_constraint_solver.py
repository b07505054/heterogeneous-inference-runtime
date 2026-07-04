from deployment.planner.constraint_solver import evaluate_constraints, filter_candidates


def test_constraint_filtering_accepts_candidate_within_limits():
    candidate = {
        "metrics": {
            "latency_ms": 4.0,
            "package_mb": 6.0,
            "memory_mb": 20.0,
            "drift": 0.0,
            "throughput": 80.0,
        },
        "reasons": [],
    }

    result = evaluate_constraints(
        candidate,
        {
            "max_p95_ms": 5.0,
            "max_package_mb": 10.0,
            "max_memory_mb": 32.0,
            "max_drift": 0.01,
            "min_throughput": 50.0,
        },
    )

    assert result["eligible"] is True
    assert result["reasons"] == []


def test_constraint_filtering_rejects_exceeding_metrics():
    candidate = {
        "metrics": {
            "latency_ms": 8.0,
            "package_mb": 12.0,
            "memory_mb": 40.0,
            "drift": 0.2,
            "throughput": 20.0,
        },
        "reasons": [],
    }

    result = evaluate_constraints(
        candidate,
        {
            "max_latency_ms": 5.0,
            "max_package_mb": 10.0,
            "max_memory_mb": 32.0,
            "max_drift": 0.01,
            "min_tokens_per_second": 50.0,
        },
    )

    assert result["eligible"] is False
    assert result["reasons"] == [
        "latency_exceeds_max",
        "package_exceeds_max",
        "memory_exceeds_max",
        "drift_exceeds_max",
        "throughput_below_min",
    ]


def test_constraint_filtering_missing_metric_is_not_good():
    candidate = {"metrics": {}, "reasons": []}

    result = evaluate_constraints(candidate, {"max_p95_ms": 5.0, "min_throughput": 50.0})

    assert result["eligible"] is False
    assert "missing_latency_ms" in result["reasons"]
    assert "missing_throughput" in result["reasons"]


def test_filter_candidates_preserves_existing_reasons():
    candidates = [{"metrics": {"latency_ms": 1.0}, "reasons": ["source_rejected"]}]

    filtered = filter_candidates(candidates, {"max_p95_ms": 5.0})

    assert filtered[0]["eligible"] is False
    assert filtered[0]["reasons"] == ["source_rejected"]
