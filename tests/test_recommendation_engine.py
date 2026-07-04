from deployment.planner.objective import balanced_score
from deployment.planner.recommendation_engine import recommend_candidate


def test_latency_objective_selects_lowest_latency():
    candidates = [
        _candidate("slow.json", latency=10.0, throughput=100.0),
        _candidate("fast.json", latency=4.0, throughput=50.0),
    ]

    result = recommend_candidate(candidates, "latency")

    assert result["status"] == "selected"
    assert result["selected_candidate"]["source_artifact"] == "fast.json"
    assert "Latency objective" in result["decision_reason"][2]


def test_throughput_objective_selects_highest_throughput():
    candidates = [
        _candidate("low.json", latency=4.0, throughput=50.0),
        _candidate("high.json", latency=8.0, throughput=150.0),
    ]

    result = recommend_candidate(candidates, "throughput")

    assert result["status"] == "selected"
    assert result["selected_candidate"]["source_artifact"] == "high.json"
    assert "Throughput objective" in result["decision_reason"][2]


def test_balanced_objective_uses_weighted_score():
    candidates = [
        _candidate("balanced.json", latency=10.0, throughput=100.0, memory=10.0, package=10.0),
        _candidate("throughput_only.json", latency=20.0, throughput=200.0, memory=10.0, package=10.0),
    ]

    result = recommend_candidate(candidates, "balanced")

    assert result["status"] == "selected"
    assert result["selected_candidate"]["source_artifact"] == "balanced.json"
    assert balanced_score(result["selected_candidate"], candidates) < balanced_score(candidates[1], candidates)
    assert "latency=0.4" in result["decision_reason"][2]


def test_no_eligible_candidate_path():
    candidates = [
        {
            **_candidate("rejected.json", latency=10.0, throughput=10.0),
            "eligible": False,
            "reasons": ["latency_exceeds_max"],
        }
    ]

    result = recommend_candidate(candidates, "latency")

    assert result["status"] == "no_eligible_candidate"
    assert result["selected_candidate"] is None
    assert "latency_exceeds_max" in result["decision_reason"][0]


def _candidate(
    artifact: str,
    *,
    latency: float,
    throughput: float,
    memory: float = 10.0,
    package: float = 10.0,
):
    return {
        "runtime": "test",
        "policy": "test_policy",
        "config": {},
        "metrics": {
            "latency_ms": latency,
            "throughput": throughput,
            "memory_mb": memory,
            "package_mb": package,
        },
        "source_artifact": artifact,
        "eligible": True,
        "reasons": [],
    }
