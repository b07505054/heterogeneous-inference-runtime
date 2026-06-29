"""Export of DistributedRuntimePlan/Result/Trace as stable JSON artifacts.

Produces three deterministic files:
  distributed_runtime_plan.json   — plan with prefix-cache fields in adapter contract format
  distributed_runtime_result.json — simulated execution result with stage timings
  distributed_runtime_trace.json  — timeline trace augmented with cache adjustment

All exports are deterministic and stable under repeated calls with the same plan.
No wall clock. No random. Input objects are never mutated.

The plan JSON satisfies the Inference-Validation-Platform runtime_artifact_adapter
contract: it carries both top-level summary fields and the full nested structure
the adapter reads (decision_comparison, prefix_cache_adjustment).

Truth boundary is always explicit and mentions simulation. When a prefix-cache
adjustment is present the truth boundary mentions both the plan simulation and
the prefix-cache simulation.

Architecture:
  DistributedRuntimePlan
    → build_plan_artifact()         → distributed_runtime_plan.json
    → DistributedExecutionEngine
        → DistributedRuntimeResult
            → build_result_artifact()    → distributed_runtime_result.json
            → build_trace_artifact()     → distributed_runtime_trace.json
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from deployment.distributed_execution_engine import (
    DistributedExecutionEngine,
    DistributedRuntimeResult,
)
from deployment.distributed_runtime_plan import (
    DistributedRuntimePlan,
    PrefixCacheAdjustment,
)
from deployment.distributed_runtime_trace import build_distributed_runtime_trace

_SCHEMA_VERSION = "1.0"

_TB_PLAN = (
    "distributed_runtime_plan_simulated_not_real_cluster_execution"
)
_TB_PLAN_WITH_CACHE = (
    "distributed_runtime_plan_and_prefix_cache_simulated_not_real_cluster_execution_or_kv_cache"
)
_TB_RESULT = (
    "distributed_runtime_execution_simulated_not_real_cluster_execution"
)
_TB_TRACE = (
    "distributed_runtime_trace_simulated_not_real_cluster_execution"
)
_TB_TRACE_WITH_CACHE = (
    "distributed_runtime_trace_and_prefix_cache_simulated_not_real_cluster_execution_or_kv_cache"
)


# ---------------------------------------------------------------------------
# Artifact builders (pure functions; never mutate inputs)
# ---------------------------------------------------------------------------

def build_plan_artifact(
    plan: DistributedRuntimePlan,
    *,
    model_name: str = "",
) -> dict:
    """Serialize DistributedRuntimePlan to an artifact dict.

    Top-level summary fields (for human readability and validation tooling):
      artifact_name, schema_version, model_name, target_profile_id,
      selected_policy, baseline_ttft_ms, optimized_ttft_ms,
      baseline_tpot_ms, optimized_tpot_ms, prefix_cache_hit_type,
      prefix_cache_hit_tokens, prefix_cache_saved_prefill_ms,
      prefix_cache_remote_transfer_bytes, adjusted_prefill_service_ms,
      truth_boundary.

    Nested structure (for adapter contract):
      decision_comparison.{selected_policy, pd_split, colocated},
      prefix_cache_adjustment.{hit_type, hit_tokens, saved_prefill_ms,
        remote_transfer_bytes, remote_transfer_cost_ms, truth_boundary}.

    Does not mutate plan.
    """
    effective_model = model_name or plan.model_name or "unknown"
    cmp = plan.decision_comparison
    pca: PrefixCacheAdjustment | None = plan.prefix_cache_adjustment

    selected_policy = cmp.selected_policy
    if selected_policy == "pd_split":
        optimized_ttft = cmp.pd_split.ttft_ms
        optimized_tpot = cmp.pd_split.tpot_ms
    else:
        optimized_ttft = cmp.colocated.ttft_ms
        optimized_tpot = cmp.colocated.tpot_ms

    baseline_ttft = _baseline_ttft(optimized_ttft, pca)
    baseline_tpot = optimized_tpot  # prefix cache does not affect decode service time

    has_cache = pca is not None and pca.hit_type != "miss"
    tb = _TB_PLAN_WITH_CACHE if has_cache else _TB_PLAN

    if pca is not None:
        pc_hit_type = pca.hit_type
        pc_hit_tokens = pca.hit_tokens
        pc_saved_ms = pca.saved_prefill_ms
        pc_remote_bytes = pca.remote_transfer_bytes
        pc_adjusted_ms = pca.adjusted_prefill_service_ms
        pc_adj_dict: dict | None = dataclasses.asdict(pca)
    else:
        pc_hit_type = "miss"
        pc_hit_tokens = 0
        pc_saved_ms = 0.0
        pc_remote_bytes = 0.0
        pc_adjusted_ms = plan.prefill.service_ms
        pc_adj_dict = None

    return {
        "artifact_type": "distributed_runtime_plan",
        "artifact_name": f"distributed_runtime_plan_{effective_model}",
        "schema_version": _SCHEMA_VERSION,
        # Top-level summary
        "model_name": effective_model,
        "target_profile_id": plan.target_profile_id,
        "selected_policy": selected_policy,
        "truth_boundary": tb,
        "baseline_ttft_ms": baseline_ttft,
        "optimized_ttft_ms": optimized_ttft,
        "baseline_tpot_ms": baseline_tpot,
        "optimized_tpot_ms": optimized_tpot,
        "prefix_cache_hit_type": pc_hit_type,
        "prefix_cache_hit_tokens": pc_hit_tokens,
        "prefix_cache_saved_prefill_ms": pc_saved_ms,
        "prefix_cache_remote_transfer_bytes": pc_remote_bytes,
        "adjusted_prefill_service_ms": pc_adjusted_ms,
        # Full nested structure for adapter contract
        "decision_comparison": dataclasses.asdict(cmp),
        "prefix_cache_adjustment": pc_adj_dict,
        # Additional plan metadata
        "total_compiler_cost_ms": plan.total_compiler_cost_ms,
        "prefill": dataclasses.asdict(plan.prefill),
        "kv_transfer": dataclasses.asdict(plan.kv_transfer),
        "decode": dataclasses.asdict(plan.decode),
        "replay": dataclasses.asdict(plan.replay),
    }


def build_result_artifact(
    result: DistributedRuntimeResult,
    *,
    model_name: str = "",
) -> dict:
    """Serialize DistributedRuntimeResult to an artifact dict.

    Includes all six stage results with start/duration/end/worker/backend fields.
    Does not mutate result.
    """
    effective_model = model_name or result.model_name or "unknown"
    return {
        "artifact_type": "distributed_runtime_result",
        "artifact_name": f"distributed_runtime_result_{effective_model}",
        "schema_version": _SCHEMA_VERSION,
        "model_name": effective_model,
        "target_profile_id": result.target_profile_id,
        "selected_policy": result.selected_policy,
        "total_latency_ms": result.total_latency_ms,
        "ttft_ms": result.ttft_ms,
        "tpot_ms": result.tpot_ms,
        "truth_boundary": _TB_RESULT,
        "stage_results": [dataclasses.asdict(s) for s in result.stage_results],
    }


def build_trace_artifact(
    plan: DistributedRuntimePlan,
    result: DistributedRuntimeResult,
    *,
    model_name: str = "",
) -> dict:
    """Serialize the combined plan+result trace to an artifact dict.

    Includes:
    - Timeline events from build_distributed_runtime_trace (7 events with timestamps)
    - Stage results from the execution engine (6 stages, no decision_summary)
    - Prefix-cache adjustment info when present
    - Cost summary (colocated vs pd_split comparison)

    Does not mutate plan or result.
    """
    effective_model = model_name or plan.model_name or result.model_name or "unknown"
    pca: PrefixCacheAdjustment | None = plan.prefix_cache_adjustment
    has_cache = pca is not None and pca.hit_type != "miss"
    tb = _TB_TRACE_WITH_CACHE if has_cache else _TB_TRACE

    base_trace = build_distributed_runtime_trace(plan)

    return {
        "artifact_type": "distributed_runtime_trace",
        "artifact_name": f"distributed_runtime_trace_{effective_model}",
        "schema_version": _SCHEMA_VERSION,
        "model_name": effective_model,
        "target_profile_id": plan.target_profile_id,
        "selected_policy": plan.decision_comparison.selected_policy,
        "truth_boundary": tb,
        "total_compiler_cost_ms": base_trace["total_compiler_cost_ms"],
        "prefix_cache_adjustment": dataclasses.asdict(pca) if pca is not None else None,
        "stage_results": [dataclasses.asdict(s) for s in result.stage_results],
        "events": base_trace["events"],
        "cost_summary": base_trace["cost_summary"],
    }


# ---------------------------------------------------------------------------
# Main export entry point
# ---------------------------------------------------------------------------

def export_distributed_runtime_artifacts(
    plan: DistributedRuntimePlan,
    output_dir: Path | str,
    *,
    model_name: str = "",
) -> tuple[Path, Path, Path]:
    """Export plan, result, and trace as stable deterministic JSON artifacts.

    Runs DistributedExecutionEngine on the plan to produce the result, then
    serializes all three artifacts with sort_keys=True for stable ordering.

    Args:
        plan: The DistributedRuntimePlan to export. Not mutated.
        output_dir: Directory to write the three JSON files into.
        model_name: Optional model name override (plan.model_name is often "").

    Returns:
        (plan_path, result_path, trace_path) — Path of each written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = DistributedExecutionEngine()
    result = engine.execute(plan)

    plan_dict = build_plan_artifact(plan, model_name=model_name)
    result_dict = build_result_artifact(result, model_name=model_name)
    trace_dict = build_trace_artifact(plan, result, model_name=model_name)

    plan_path = output_dir / "distributed_runtime_plan.json"
    result_path = output_dir / "distributed_runtime_result.json"
    trace_path = output_dir / "distributed_runtime_trace.json"

    _write_json(plan_path, plan_dict)
    _write_json(result_path, result_dict)
    _write_json(trace_path, trace_dict)

    return plan_path, result_path, trace_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _baseline_ttft(
    optimized_ttft: float,
    pca: PrefixCacheAdjustment | None,
) -> float:
    """Reconstruct TTFT before prefix-cache adjustment.

    The planner applies:
      prefill_service_ms -= saved_prefill_ms
      kv_transfer_ms     += remote_transfer_cost_ms
    Net TTFT change = -(saved_prefill_ms - remote_transfer_cost_ms).
    So baseline = optimized + saved_prefill_ms - remote_transfer_cost_ms.
    """
    if pca is None:
        return optimized_ttft
    return optimized_ttft + pca.saved_prefill_ms - pca.remote_transfer_cost_ms


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
