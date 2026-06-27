"""Tests for scripts/generate_iphone_runtime_trace.py.

Covers:
  - helper functions (_simulated_queue_depth, _simulated_memory_mb, _cpu_only_overlay)
  - fixture fallback behavior (plan path missing)
  - compiler artifact behavior (fake plan file provided via tmp_path)
  - fixture provenance fields in generated trace
  - compiler artifact provenance fields in generated trace

No subprocess or file-system side effects except via tmp_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add repo root to sys.path so the script module is importable.
_REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import the script module directly (not __main__ — importable because
# _generate_trace and helpers are module-level functions).
import scripts.generate_iphone_runtime_trace as gen


# ---------------------------------------------------------------------------
# _simulated_queue_depth
# ---------------------------------------------------------------------------

def test_queue_depth_baseline_starts_high():
    assert gen._simulated_queue_depth(0, "baseline") >= 8


def test_queue_depth_baseline_drains_over_requests():
    depths = [gen._simulated_queue_depth(i, "baseline") for i in range(32)]
    assert depths[0] > depths[-1]


def test_queue_depth_baseline_minimum_is_two():
    for i in range(64):
        assert gen._simulated_queue_depth(i, "baseline") >= 2


def test_queue_depth_optimized_starts_lower_than_baseline():
    assert gen._simulated_queue_depth(0, "optimized") < gen._simulated_queue_depth(0, "baseline")


def test_queue_depth_optimized_minimum_is_one():
    for i in range(64):
        assert gen._simulated_queue_depth(i, "optimized") >= 1


# ---------------------------------------------------------------------------
# _simulated_memory_mb
# ---------------------------------------------------------------------------

def test_memory_baseline_is_higher_than_optimized():
    for i in range(32):
        assert gen._simulated_memory_mb(i, "baseline") > gen._simulated_memory_mb(i, "optimized")


def test_memory_baseline_stays_within_expected_range():
    for i in range(32):
        mem = gen._simulated_memory_mb(i, "baseline")
        assert 480.0 <= mem <= 620.0


def test_memory_optimized_stays_within_expected_range():
    for i in range(32):
        mem = gen._simulated_memory_mb(i, "optimized")
        assert 130.0 <= mem <= 200.0


# ---------------------------------------------------------------------------
# _cpu_only_overlay
# ---------------------------------------------------------------------------

def test_cpu_only_overlay_sets_primary_backend_cpu():
    plan = {
        "function_plans": [
            {
                "function_name": "prefill",
                "backend_execution_plan": {"primary_backend": "coreml", "fallback_chain": ["metal"]},
                "cost_summary": {"colocated_total_ms": 10.0},
                "kv_plan": {},
                "replay_plan": {},
            }
        ]
    }
    result = gen._cpu_only_overlay(plan)
    bep = result["function_plans"][0]["backend_execution_plan"]
    assert bep["primary_backend"] == "cpu"
    assert bep["fallback_chain"] == ["cpu"]


def test_cpu_only_overlay_sets_contiguous_kv():
    plan = {
        "function_plans": [
            {
                "function_name": "decode",
                "backend_execution_plan": {"primary_backend": "coreml"},
                "cost_summary": {"colocated_total_ms": 5.0},
                "kv_plan": {"layout": "paged"},
                "replay_plan": {},
            }
        ]
    }
    result = gen._cpu_only_overlay(plan)
    assert result["function_plans"][0]["kv_plan"]["layout"] == "contiguous"


def test_cpu_only_overlay_disables_replay():
    plan = {
        "function_plans": [
            {
                "function_name": "decode",
                "backend_execution_plan": {"primary_backend": "coreml"},
                "cost_summary": {"colocated_total_ms": 5.0},
                "kv_plan": {},
                "replay_plan": {"replay_eligible": True, "cuda_graph_bucket": "decode_static"},
            }
        ]
    }
    result = gen._cpu_only_overlay(plan)
    rp = result["function_plans"][0]["replay_plan"]
    assert rp["replay_eligible"] is False
    assert rp["cuda_graph_bucket"] == ""


def test_cpu_only_overlay_scales_up_cost():
    plan = {
        "function_plans": [
            {
                "function_name": "prefill",
                "backend_execution_plan": {"primary_backend": "coreml"},
                "cost_summary": {"colocated_total_ms": 10.0},
                "kv_plan": {},
                "replay_plan": {},
            }
        ]
    }
    result = gen._cpu_only_overlay(plan)
    scaled = result["function_plans"][0]["cost_summary"]["colocated_total_ms"]
    assert scaled == pytest.approx(55.0)  # 10.0 * 5.5


def test_cpu_only_overlay_does_not_mutate_original():
    plan = {
        "function_plans": [
            {
                "function_name": "prefill",
                "backend_execution_plan": {"primary_backend": "coreml"},
                "cost_summary": {"colocated_total_ms": 10.0},
                "kv_plan": {},
                "replay_plan": {},
            }
        ]
    }
    gen._cpu_only_overlay(plan)
    assert plan["function_plans"][0]["backend_execution_plan"]["primary_backend"] == "coreml"


# ---------------------------------------------------------------------------
# _generate_trace — fixture fallback (no compiler plan on disk)
# ---------------------------------------------------------------------------

def test_fixture_fallback_when_plan_missing(tmp_path):
    """A path that does not exist triggers the built-in fixture."""
    missing = tmp_path / "nonexistent_plan.json"
    trace = gen._generate_trace(missing, num_requests=4)

    assert trace.compiler_plan_source == "built_in_fixture"
    assert trace.do_not_use_for_demo is True


def test_fixture_json_has_do_not_use_for_demo_true(tmp_path):
    missing = tmp_path / "nonexistent_plan.json"
    trace = gen._generate_trace(missing, num_requests=4)
    d = trace.to_dict()
    assert d["do_not_use_for_demo"] is True


def test_fixture_headline_contains_fixture_prefix(tmp_path):
    missing = tmp_path / "nonexistent_plan.json"
    trace = gen._generate_trace(missing, num_requests=4)
    assert "[FIXTURE]" in trace.comparison_summary.headline


def test_fixture_provenance_notes_contains_compiler_not_in_pipeline(tmp_path):
    missing = tmp_path / "nonexistent_plan.json"
    trace = gen._generate_trace(missing, num_requests=4)
    assert "compiler_not_in_pipeline" in trace.provenance_notes


def test_fixture_plan_ref_is_fixture_built_in(tmp_path):
    missing = tmp_path / "nonexistent_plan.json"
    trace = gen._generate_trace(missing, num_requests=4)
    assert trace.compiler_plan_ref == "fixture:built_in"


def test_fixture_trace_has_both_variants(tmp_path):
    missing = tmp_path / "nonexistent_plan.json"
    trace = gen._generate_trace(missing, num_requests=4)
    assert "baseline" in trace.variants
    assert "optimized" in trace.variants


def test_fixture_optimized_p95_lower_than_baseline(tmp_path):
    missing = tmp_path / "nonexistent_plan.json"
    trace = gen._generate_trace(missing, num_requests=8)
    b_p95 = trace.variants["baseline"].summary.p95_latency_ms
    o_p95 = trace.variants["optimized"].summary.p95_latency_ms
    assert o_p95 < b_p95


# ---------------------------------------------------------------------------
# _generate_trace — compiler artifact mode (fake plan file in tmp_path)
# ---------------------------------------------------------------------------

def _write_fake_compiler_plan(path: Path) -> None:
    """Write a minimal valid ServingExecutionPlan to path."""
    plan = {
        "target_profile_id": "test-profile",
        "model_name": "test-model",
        "function_plans": [
            {
                "function_name": "prefill",
                "execution_mode": "colocated",
                "cost_summary": {
                    "colocated_total_ms": 6.0,
                    "confidence": "low",
                    "policy": "colocated",
                    "cost_source": "formula_synthetic",
                },
                "kv_plan": {
                    "layout": "paged",
                    "kv_byte_estimate_mb": 8.0,
                    "layout_reason": "",
                    "truth_boundary": "static_formula_estimate_not_measured_memory",
                },
                "replay_plan": {
                    "replay_eligible": False,
                    "cuda_graph_bucket": "",
                    "override_reason": "",
                    "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
                },
                "backend_execution_plan": {
                    "primary_backend": "coreml",
                    "fallback_chain": ["cpu"],
                    "decision_source": "target_preferred",
                    "required_precision": "fp16",
                    "required_kv_layout": "paged",
                    "requires_replay": False,
                },
                "provenance": {
                    "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
                    "cost_source": "formula_synthetic",
                },
                "source_passes": ["serving-phase-analysis"],
            }
        ],
    }
    path.write_text(json.dumps(plan))


def test_compiler_artifact_mode_uses_compiler_artifact_source(tmp_path):
    plan_path = tmp_path / "serving_execution_plan_iphone.json"
    _write_fake_compiler_plan(plan_path)
    trace = gen._generate_trace(plan_path, num_requests=4)
    assert trace.compiler_plan_source == "compiler_artifact"


def test_compiler_artifact_headline_does_not_contain_fixture_prefix(tmp_path):
    plan_path = tmp_path / "serving_execution_plan_iphone.json"
    _write_fake_compiler_plan(plan_path)
    trace = gen._generate_trace(plan_path, num_requests=4)
    assert "[FIXTURE]" not in trace.comparison_summary.headline


def test_compiler_artifact_do_not_use_for_demo_is_false(tmp_path):
    plan_path = tmp_path / "serving_execution_plan_iphone.json"
    _write_fake_compiler_plan(plan_path)
    trace = gen._generate_trace(plan_path, num_requests=4)
    assert trace.do_not_use_for_demo is False


def test_compiler_artifact_provenance_notes_no_compiler_not_in_pipeline(tmp_path):
    plan_path = tmp_path / "serving_execution_plan_iphone.json"
    _write_fake_compiler_plan(plan_path)
    trace = gen._generate_trace(plan_path, num_requests=4)
    assert "compiler_not_in_pipeline" not in trace.provenance_notes


def test_compiler_artifact_plan_path_in_trace(tmp_path):
    plan_path = tmp_path / "serving_execution_plan_iphone.json"
    _write_fake_compiler_plan(plan_path)
    trace = gen._generate_trace(plan_path, num_requests=4)
    assert str(plan_path) in trace.compiler_plan_path


def test_compiler_artifact_model_name_from_plan(tmp_path):
    plan_path = tmp_path / "serving_execution_plan_iphone.json"
    _write_fake_compiler_plan(plan_path)
    trace = gen._generate_trace(plan_path, num_requests=4)
    assert trace.model_name == "test-model"


def test_compiler_artifact_target_profile_id_from_plan(tmp_path):
    plan_path = tmp_path / "serving_execution_plan_iphone.json"
    _write_fake_compiler_plan(plan_path)
    trace = gen._generate_trace(plan_path, num_requests=4)
    assert trace.target_profile_id == "test-profile"
