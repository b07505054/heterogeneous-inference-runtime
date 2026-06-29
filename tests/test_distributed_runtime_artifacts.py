"""Tests for distributed_runtime_artifacts: export of plan/result/trace as stable JSON.

All tests are deterministic. No wall clock. No random.
Uses the same _PREFILL_DICT/_DECODE_DICT/_make_plan fixture pattern as the
other distributed runtime tests in this directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.distributed_runtime_artifacts import (
    build_plan_artifact,
    build_result_artifact,
    build_trace_artifact,
    export_distributed_runtime_artifacts,
)
from deployment.distributed_runtime_plan import PDSplitPlanner
from deployment.distributed_execution_engine import DistributedExecutionEngine
from deployment.prefix_cache_simulator import PrefixCacheResult
from deployment.runtime_execution_plan import RuntimeExecutionPlanAdapter

# ---------------------------------------------------------------------------
# Shared plan fixtures (same as test_distributed_runtime_trace.py)
# ---------------------------------------------------------------------------

_PREFILL_DICT = {
    "function_name": "prefill",
    "execution_mode": "pd_split",
    "target_profile_id": "test-profile",
    "cost_summary": {
        "pd_split_total_ms": 31.2,
        "colocated_total_ms": 31.2,
        "confidence": "high",
        "policy": "pd_split",
    },
    "kv_plan": {
        "layout": "paged",
        "kv_byte_estimate_mb": 4.0,
        "layout_reason": "paged_kv_preferred",
        "truth_boundary": "static_kv_layout_plan_not_runtime_allocation",
    },
    "replay_plan": {
        "replay_eligible": False,
        "cuda_graph_bucket": "",
        "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
    },
    "backend_execution_plan": {
        "primary_backend": "cuda",
        "fallback_chain": ["cpu"],
        "decision_source": "default_policy",
        "required_precision": "fp16",
        "required_kv_layout": "paged",
        "requires_replay": False,
    },
}

_DECODE_DICT = {
    "function_name": "decode",
    "execution_mode": "pd_split",
    "target_profile_id": "test-profile",
    "cost_summary": {
        "pd_split_total_ms": 8.5,
        "colocated_total_ms": 8.5,
        "confidence": "high",
        "policy": "pd_split",
    },
    "kv_plan": {
        "layout": "paged",
        "kv_byte_estimate_mb": 0.5,
        "layout_reason": "paged_kv_preferred",
        "truth_boundary": "static_kv_layout_plan_not_runtime_allocation",
    },
    "replay_plan": {
        "replay_eligible": True,
        "cuda_graph_bucket": "decode_static",
        "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
    },
    "backend_execution_plan": {
        "primary_backend": "cuda",
        "fallback_chain": ["cpu"],
        "decision_source": "default_policy",
        "required_precision": "fp16",
        "required_kv_layout": "paged",
        "requires_replay": True,
    },
}


def _make_plan(**kwargs):
    p = RuntimeExecutionPlanAdapter.from_dict(_PREFILL_DICT)
    d = RuntimeExecutionPlanAdapter.from_dict(_DECODE_DICT)
    return PDSplitPlanner.plan([p, d], **kwargs)


def _local_hit_result(hit_tokens: int = 50, saved_ms: float = 10.0) -> PrefixCacheResult:
    return PrefixCacheResult(
        cache_key="test_key",
        hit_type="local_hit",
        hit_tokens=hit_tokens,
        miss_tokens=50,
        hit_ratio=hit_tokens / (hit_tokens + 50),
        saved_prefill_ms=saved_ms,
        remote_transfer_bytes=0.0,
        evicted_cache_keys=(),
        truth_boundary="prefix_cache_simulated_not_real_kv_cache",
    )


def _remote_hit_result(
    hit_tokens: int = 50,
    remote_bytes: float = 2.0 * 1024 * 1024,
    saved_ms: float = 10.0,
) -> PrefixCacheResult:
    return PrefixCacheResult(
        cache_key="test_key",
        hit_type="remote_hit",
        hit_tokens=hit_tokens,
        miss_tokens=50,
        hit_ratio=hit_tokens / (hit_tokens + 50),
        saved_prefill_ms=saved_ms,
        remote_transfer_bytes=float(remote_bytes),
        evicted_cache_keys=(),
        truth_boundary="prefix_cache_simulated_not_real_kv_cache",
    )


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------

def test_exports_plan_result_and_trace_json(tmp_path):
    plan = _make_plan()
    plan_path, result_path, trace_path = export_distributed_runtime_artifacts(plan, tmp_path)

    assert plan_path.exists()
    assert result_path.exists()
    assert trace_path.exists()
    assert plan_path.name == "distributed_runtime_plan.json"
    assert result_path.name == "distributed_runtime_result.json"
    assert trace_path.name == "distributed_runtime_trace.json"

    # Each file is valid JSON.
    json.loads(plan_path.read_text())
    json.loads(result_path.read_text())
    json.loads(trace_path.read_text())


def test_artifact_contains_prefix_cache_fields(tmp_path):
    cache_result = _local_hit_result(hit_tokens=50, saved_ms=10.0)
    plan = _make_plan(prefix_cache_result=cache_result)
    plan_path, _, _ = export_distributed_runtime_artifacts(
        plan, tmp_path, model_name="llama3_8b"
    )
    data = json.loads(plan_path.read_text())

    assert data["prefix_cache_hit_type"] == "local_hit"
    assert data["prefix_cache_hit_tokens"] == 50
    assert data["prefix_cache_saved_prefill_ms"] == pytest.approx(10.0)
    assert data["prefix_cache_remote_transfer_bytes"] == pytest.approx(0.0)
    assert "prefix_cache_adjustment" in data
    assert data["prefix_cache_adjustment"] is not None


def test_neutral_export_when_no_prefix_cache_adjustment(tmp_path):
    plan = _make_plan()
    plan_path, _, _ = export_distributed_runtime_artifacts(plan, tmp_path)
    data = json.loads(plan_path.read_text())

    assert data["prefix_cache_hit_type"] == "miss"
    assert data["prefix_cache_hit_tokens"] == 0
    assert data["prefix_cache_saved_prefill_ms"] == pytest.approx(0.0)
    assert data["prefix_cache_remote_transfer_bytes"] == pytest.approx(0.0)
    assert data.get("prefix_cache_adjustment") is None


def test_local_hit_export_matches_adjustment(tmp_path):
    cache_result = _local_hit_result(hit_tokens=60, saved_ms=12.0)
    plan = _make_plan(prefix_cache_result=cache_result)
    plan_path, _, _ = export_distributed_runtime_artifacts(plan, tmp_path)
    data = json.loads(plan_path.read_text())

    pca = plan.prefix_cache_adjustment
    assert pca is not None
    assert data["prefix_cache_saved_prefill_ms"] == pytest.approx(pca.saved_prefill_ms)
    assert data["prefix_cache_hit_tokens"] == pca.hit_tokens
    assert data["adjusted_prefill_service_ms"] == pytest.approx(pca.adjusted_prefill_service_ms)
    nested = data["prefix_cache_adjustment"]
    assert nested["hit_type"] == "local_hit"
    assert nested["saved_prefill_ms"] == pytest.approx(12.0)
    assert nested["remote_transfer_bytes"] == pytest.approx(0.0)
    assert nested["remote_transfer_cost_ms"] == pytest.approx(0.0)


def test_remote_hit_export_includes_transfer_bytes(tmp_path):
    remote_bytes = 2.0 * 1024 * 1024  # 2 MB
    cache_result = _remote_hit_result(remote_bytes=remote_bytes, saved_ms=10.0)
    plan = _make_plan(prefix_cache_result=cache_result, kv_bandwidth_mb_per_ms=32.0)
    plan_path, _, _ = export_distributed_runtime_artifacts(plan, tmp_path)
    data = json.loads(plan_path.read_text())

    assert data["prefix_cache_hit_type"] == "remote_hit"
    assert data["prefix_cache_remote_transfer_bytes"] == pytest.approx(remote_bytes)
    nested = data["prefix_cache_adjustment"]
    assert nested["remote_transfer_bytes"] == pytest.approx(remote_bytes)
    # remote_transfer_cost_ms = 2 MB / 32 MB/ms = 0.0625 ms
    assert nested["remote_transfer_cost_ms"] == pytest.approx(0.0625)
    assert nested["remote_transfer_cost_ms"] > 0.0


def test_baseline_ttft_reconstructs_pre_cache_value(tmp_path):
    # Local hit: no remote cost, so baseline = optimized + saved.
    cache_result = _local_hit_result(hit_tokens=50, saved_ms=10.0)
    plan = _make_plan(prefix_cache_result=cache_result)
    plan_path, _, _ = export_distributed_runtime_artifacts(plan, tmp_path)
    data = json.loads(plan_path.read_text())

    optimized = data["optimized_ttft_ms"]
    baseline = data["baseline_ttft_ms"]
    saved = data["prefix_cache_saved_prefill_ms"]
    # For local_hit: remote_transfer_cost_ms == 0, so baseline = optimized + saved.
    assert baseline == pytest.approx(optimized + saved)
    assert baseline > optimized


def test_baseline_ttft_remote_hit_subtracts_transfer_cost(tmp_path):
    # Remote hit: baseline = optimized + saved - remote_cost.
    # 2 MB at 32 MB/ms → remote_cost = 0.0625 ms.
    remote_bytes = 2.0 * 1024 * 1024
    cache_result = _remote_hit_result(remote_bytes=remote_bytes, saved_ms=10.0)
    plan = _make_plan(prefix_cache_result=cache_result, kv_bandwidth_mb_per_ms=32.0)
    plan_path, _, _ = export_distributed_runtime_artifacts(plan, tmp_path)
    data = json.loads(plan_path.read_text())

    optimized = data["optimized_ttft_ms"]
    baseline = data["baseline_ttft_ms"]
    saved = data["prefix_cache_saved_prefill_ms"]
    remote_cost = data["prefix_cache_adjustment"]["remote_transfer_cost_ms"]
    assert baseline == pytest.approx(optimized + saved - remote_cost)


def test_json_key_order_is_stable(tmp_path):
    plan = _make_plan()
    path1, _, _ = export_distributed_runtime_artifacts(plan, tmp_path / "r1")
    path2, _, _ = export_distributed_runtime_artifacts(plan, tmp_path / "r2")
    # sort_keys=True makes the raw text identical for identical inputs.
    assert path1.read_text() == path2.read_text()


def test_output_is_deterministic(tmp_path):
    cache_result = _local_hit_result()
    plan = _make_plan(prefix_cache_result=cache_result)
    j1, r1, t1 = export_distributed_runtime_artifacts(plan, tmp_path / "run1")
    j2, r2, t2 = export_distributed_runtime_artifacts(plan, tmp_path / "run2")

    assert json.loads(j1.read_text()) == json.loads(j2.read_text())
    assert json.loads(r1.read_text()) == json.loads(r2.read_text())
    assert json.loads(t1.read_text()) == json.loads(t2.read_text())


def test_does_not_mutate_plan(tmp_path):
    plan = _make_plan()
    pre_prefill_ms = plan.prefill.service_ms
    pre_kv_ms = plan.kv_transfer.transfer_cost_ms
    pre_policy = plan.decision_comparison.selected_policy
    pre_cache = plan.prefix_cache_adjustment

    export_distributed_runtime_artifacts(plan, tmp_path)

    assert plan.prefill.service_ms == pre_prefill_ms
    assert plan.kv_transfer.transfer_cost_ms == pre_kv_ms
    assert plan.decision_comparison.selected_policy == pre_policy
    assert plan.prefix_cache_adjustment is pre_cache


def test_trace_contains_stage_results(tmp_path):
    plan = _make_plan()
    _, _, trace_path = export_distributed_runtime_artifacts(plan, tmp_path)
    data = json.loads(trace_path.read_text())

    assert "stage_results" in data
    stages = data["stage_results"]
    assert isinstance(stages, list)
    # DistributedExecutionEngine produces 6 stages (no decision_summary event).
    assert len(stages) == 6
    names = {s["stage_name"] for s in stages}
    assert "prefill_compute" in names
    assert "decode_compute" in names
    assert "kv_transfer" in names
    # Each stage must carry worker_id and backend fields.
    for s in stages:
        assert "worker_id" in s
        assert "backend" in s
        assert "duration_ms" in s
        assert "truth_boundary" in s


def test_truth_boundary_mentions_simulation(tmp_path):
    # Without cache: all artifacts mention "simulated".
    plan_nc = _make_plan()
    j1, r1, t1 = export_distributed_runtime_artifacts(plan_nc, tmp_path / "nc")
    for path in (j1, r1, t1):
        data = json.loads(path.read_text())
        assert "simulated" in data["truth_boundary"].lower(), path.name

    # With local hit cache: all artifacts still mention "simulated".
    plan_wc = _make_plan(prefix_cache_result=_local_hit_result())
    j2, r2, t2 = export_distributed_runtime_artifacts(plan_wc, tmp_path / "wc")
    for path in (j2, r2, t2):
        data = json.loads(path.read_text())
        assert "simulated" in data["truth_boundary"].lower(), path.name

    # Plan and trace with cache must also mention "prefix_cache" in truth boundary.
    plan_data = json.loads(j2.read_text())
    assert "prefix_cache" in plan_data["truth_boundary"].lower()
    trace_data = json.loads(t2.read_text())
    assert "prefix_cache" in trace_data["truth_boundary"].lower()


def test_exported_artifact_matches_validation_adapter_contract(tmp_path):
    """Plan JSON must carry all keys that runtime_artifact_adapter.py reads."""
    cache_result = _local_hit_result()
    plan = _make_plan(prefix_cache_result=cache_result)
    plan_path, _, _ = export_distributed_runtime_artifacts(
        plan, tmp_path, model_name="llama3_8b"
    )
    data = json.loads(plan_path.read_text())

    # Adapter reads: artifact.get("model_name")
    assert data.get("model_name") == "llama3_8b"

    # Adapter reads: decision_comparison.selected_policy
    cmp = data.get("decision_comparison", {})
    assert "selected_policy" in cmp

    # Adapter reads: decision_comparison.pd_split.{ttft_ms, tpot_ms}
    pd = cmp.get("pd_split", {})
    assert "ttft_ms" in pd
    assert "tpot_ms" in pd

    # Adapter reads: decision_comparison.colocated.{ttft_ms, tpot_ms}
    col = cmp.get("colocated", {})
    assert "ttft_ms" in col
    assert "tpot_ms" in col

    # Adapter reads: prefix_cache_adjustment.{hit_type, hit_tokens,
    #   saved_prefill_ms, remote_transfer_bytes, remote_transfer_cost_ms, truth_boundary}
    pca = data.get("prefix_cache_adjustment")
    assert pca is not None
    for key in (
        "hit_type",
        "hit_tokens",
        "saved_prefill_ms",
        "remote_transfer_bytes",
        "remote_transfer_cost_ms",
        "truth_boundary",
    ):
        assert key in pca, f"missing prefix_cache_adjustment.{key}"


# ---------------------------------------------------------------------------
# Additional builder-level coverage
# ---------------------------------------------------------------------------

def test_build_plan_artifact_model_name_override():
    plan = _make_plan()
    art = build_plan_artifact(plan, model_name="mistral_7b")
    assert art["model_name"] == "mistral_7b"
    assert "mistral_7b" in art["artifact_name"]


def test_build_plan_artifact_schema_version():
    plan = _make_plan()
    art = build_plan_artifact(plan)
    assert art["schema_version"] == "1.0"
    assert art["artifact_type"] == "distributed_runtime_plan"


def test_build_result_artifact_includes_all_stages():
    plan = _make_plan()
    engine = DistributedExecutionEngine()
    result = engine.execute(plan)
    art = build_result_artifact(result, model_name="llama3_8b")

    assert art["artifact_type"] == "distributed_runtime_result"
    assert art["schema_version"] == "1.0"
    assert art["model_name"] == "llama3_8b"
    assert isinstance(art["stage_results"], list)
    assert len(art["stage_results"]) == 6


def test_build_trace_artifact_includes_events_and_stage_results():
    plan = _make_plan()
    engine = DistributedExecutionEngine()
    result = engine.execute(plan)
    art = build_trace_artifact(plan, result)

    assert art["artifact_type"] == "distributed_runtime_trace"
    assert "events" in art
    assert isinstance(art["events"], list)
    assert len(art["events"]) >= 1
    assert "stage_results" in art
    assert len(art["stage_results"]) == 6


def test_export_creates_output_dir_if_missing(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    assert not deep.exists()
    plan = _make_plan()
    export_distributed_runtime_artifacts(plan, deep)
    assert deep.exists()
    assert (deep / "distributed_runtime_plan.json").exists()
