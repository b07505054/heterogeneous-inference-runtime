import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
D9_DIR = REPO_ROOT / "results/runtime_paths/distributed_d9_break_even_tp_selection"


def test_d9_boundary_validation_recovers_both_empirical_regimes():
    validation = json.loads((D9_DIR / "measured_boundary_validation.json").read_text())
    rows = validation["rows"]
    assert validation["answer"] is True
    assert len(rows) == 6
    assert {r["model_key"] for r in rows} == {"qwen2.5-0.5b", "qwen2.5-7b"}
    for row in rows:
        assert row["predicted_winner"] == row["measured_winner"]
        evidence = row["candidate_evidence"]
        assert evidence["policy_id"] == "d9_break_even_tp_selector_v1"
        assert evidence["break_even"]["overlap_assumption"] == "zero"
        assert evidence["communication"]["estimated_collective_call_count"] > 0
        assert evidence["communication"]["bytes_per_collective_call"] > 0


def test_d9_comparison_improves_boundary_accuracy_and_regret():
    comparison = json.loads((D9_DIR / "comparison_against_d6_d7.json").read_text())
    assert comparison["d9"]["accuracy"] == 1.0
    assert comparison["d9"]["mean_regret_us"] == 0.0
    assert comparison["d9"]["max_regret_us"] == 0.0
    assert comparison["d9"]["accuracy"] >= comparison["d6"]["accuracy"]
    assert comparison["d9"]["accuracy"] >= comparison["d7"]["accuracy"]


def test_d9_decision_flips_are_not_harmful_against_phase4d_labels():
    flips = json.loads((D9_DIR / "decision_flips.json").read_text())
    assert flips["harmful_flips"] == 0
    assert flips["corrective_flips"] >= 1
