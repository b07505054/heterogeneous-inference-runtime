import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from deployment.llm_runtime_decision import CostModel, MemoryPlanner, Request, RuntimeScheduler, summarize_policy
from generate_llm_runtime_artifacts import (
    inflight_gate,
    run_scenario_comparison,
    workload_scenarios,
)


def run_inflight(requests, *, total_blocks=128, seed=7):
    scheduler = RuntimeScheduler(
        policy="inflight_paged_kv_continuous_batching",
        cost_model=CostModel(),
        memory=MemoryPlanner(
            total_blocks=total_blocks,
            block_size_tokens=16,
            kv_mb_per_block=3.125,
        ),
        rng=random.Random(seed),
        max_decode_batch_size=8,
    )
    return scheduler.run(requests)


def test_short_prompt_reaches_ttft_before_long_prompt_finishes_prefill():
    result = run_inflight(
        [
            Request("long", prompt_tokens=1024, output_tokens=8, arrival_ms=0.0),
            Request("short", prompt_tokens=64, output_tokens=8, arrival_ms=1.0),
        ],
        total_blocks=128,
    )

    short_first_token = next(
        event for event in result.serving_events
        if event.get("event") == "first_token" and event.get("request_id") == "short"
    )
    long_prefill_done = next(
        event for event in result.serving_events
        if (
            event.get("event") == "prefill_chunk"
            and event.get("request_id") == "long"
            and event.get("prompt_tokens_prefilled") == 1024
        )
    )

    assert short_first_token["time_ms"] < long_prefill_done["time_ms"]
    assert any(step["selected_action"] == "mixed_step" for step in result.scheduler_steps)


def test_high_pressure_reduces_batch_cap_and_keeps_page_lifecycle_balanced():
    requests = [
        Request(f"req-{idx}", prompt_tokens=1024, output_tokens=12, arrival_ms=float(idx))
        for idx in range(6)
    ]
    result = run_inflight(requests, total_blocks=80)
    summary = summarize_policy(result)

    high_pressure_ticks = [
        step for step in result.scheduler_steps
        if step.get("pressure_level") in {"high", "critical"}
    ]
    assert high_pressure_ticks
    assert any(step["effective_batch_cap"] <= 2 for step in high_pressure_ticks)
    assert summary["pages_allocated"] == summary["pages_freed"]
    assert summary["invariants"]["page_leak_count"] == 0


def test_finished_and_rejected_requests_do_not_own_resident_pages():
    result = run_inflight(
        [
            Request("fits", prompt_tokens=128, output_tokens=4, arrival_ms=0.0),
            Request("too-large", prompt_tokens=4096, output_tokens=512, arrival_ms=0.0),
        ],
        total_blocks=64,
    )
    invariants = result.lifecycle["invariants"]

    assert invariants["finished_request_releases_pages"]
    assert invariants["rejected_request_owns_no_pages"]
    assert invariants["every_admitted_request_reaches_terminal_state"]


def test_hard_limit_forbids_new_prefill_admission():
    requests = [
        Request(f"req-{idx}", prompt_tokens=1024, output_tokens=32, arrival_ms=0.0)
        for idx in range(4)
    ]
    result = run_inflight(requests, total_blocks=72)

    assert result.lifecycle["invariants"]["hard_limit_forbids_new_prefill"]
    assert any(
        step["memory_pressure_before"] >= result.lifecycle["config"]["memory_pressure_hard_limit"]
        for step in result.scheduler_steps
    )


def test_inflight_policy_gate_can_select_or_retain_old_policy():
    incumbent = {
        "policy": "cost_aware_memory_pressure_page_prefetch",
        "tpot_p95_ms": 2.0,
        "tokens_per_second": 100.0,
        "reject_oom_count": 0,
        "rejected_requests": 0,
        "ttft_p95_ms": 100.0,
        "invariants": {},
    }
    winning_candidate = {
        **incumbent,
        "policy": "inflight_paged_kv_continuous_batching",
        "tpot_p95_ms": 2.02,
        "tokens_per_second": 120.0,
        "ttft_p95_ms": 105.0,
        "invariants": {"page_leak_count": 0, "finished_request_releases_pages": True},
    }
    losing_candidate = {
        **winning_candidate,
        "tokens_per_second": 90.0,
    }

    assert inflight_gate(winning_candidate, incumbent)["passes_gate"]
    assert not inflight_gate(losing_candidate, incumbent)["passes_gate"]


def test_workload_aware_scenarios_exercise_both_policy_outcomes():
    default_requests = [
        Request("mixed-001", prompt_tokens=1024, output_tokens=128, arrival_ms=0.0),
        Request("mixed-002", prompt_tokens=512, output_tokens=64, arrival_ms=4.0),
        Request("mixed-003", prompt_tokens=256, output_tokens=96, arrival_ms=8.0),
        Request("mixed-004", prompt_tokens=1024, output_tokens=32, arrival_ms=12.0),
    ]
    results = [
        run_scenario_comparison(
            scenario,
            seed=42 + idx * 1000,
            total_blocks=512,
            block_size_tokens=16,
            kv_mb_per_block=3.125,
        )
        for idx, scenario in enumerate(workload_scenarios(default_requests))
    ]
    by_name = {row["scenario"]["name"]: row["scenario_result"] for row in results}

    assert by_name["default_mixed_pressure"]["decision"]["selected_policy"] != (
        "inflight_paged_kv_continuous_batching"
    )
    assert by_name["decode_interleave_heavy"]["decision"]["selected_policy"] == (
        "inflight_paged_kv_continuous_batching"
    )

    for row in by_name.values():
        lifecycle = row["page_lifecycle"]["inflight_candidate"]
        assert lifecycle["allocated_pages"] == lifecycle["freed_pages"]
        assert lifecycle["page_leak_count"] == 0

    decode_heavy = by_name["decode_interleave_heavy"]
    assert decode_heavy["checks"]["oom_count_not_regressed"]
    assert decode_heavy["checks"]["reject_count_not_regressed"]
