#!/usr/bin/env python3
"""Generate offline iPhone runtime profile trace.

Pipeline:
  ServingExecutionPlan JSON (or built-in fixture)
    → CompilerRuntimeAdapter.from_serving_plan()
    → list[RuntimeExecutionPlan]
    → ExecutionEngine.execute(plan, recorder)   ×32 requests
    → ExecutionTraceRecorder
    → RuntimeProfileTraceBuilder.from_recorder()
    → RuntimeProfileTrace
    → iphone_a17pro_runtime_trace.json

Two variants are produced in a single artifact:
  "baseline"  — cpu-only, no compiler optimizations, fixed-batch scheduling
  "optimized" — compiler-guided backend selection, paged KV, replay eligibility

Truth boundary:
  "offline_runtime_simulation_not_iphone_execution"

The generated artifact is labelled as an offline simulation.
No iPhone hardware is involved. No CoreML, Metal, or ANE dispatch occurs.
Latency values derive from the compiler's cost estimates (formula_synthetic),
not from measured device execution.

Usage:
  PYTHONPATH="$PWD" .venv/bin/python scripts/generate_iphone_runtime_trace.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root so relative imports resolve whether the script is run from
# the repo root or from scripts/.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from deployment.compiler_runtime_adapter import CompilerRuntimeAdapter
from deployment.execution_engine import ExecutionEngine
from deployment.execution_trace_recorder import ExecutionTraceRecorder
from deployment.runtime_profile_trace import RuntimeProfileTrace, RuntimeProfileTraceBuilder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_COMPILER_REPO_ROOT = _REPO_ROOT.parent / "ml-graph-compiler-runtime"
_COMPILER_PLAN_PATH = (
    _COMPILER_REPO_ROOT / "artifacts" / "apple_demo" / "serving_execution_plan_iphone.json"
)
_OUTPUT_DIR = _REPO_ROOT / "results" / "llm_runtime_artifacts" / "runtime_profile_trace"
_OUTPUT_PATH = _OUTPUT_DIR / "iphone_a17pro_runtime_trace.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUM_REQUESTS = 32
_TRUTH_BOUNDARY = "offline_runtime_simulation_not_iphone_execution"

# Simulated inter-request gaps for each variant.
# Baseline: wide gap (cpu-only, slow scheduling).
# Optimized: narrow gap (compiler-guided, faster dispatch).
_BASELINE_GAP_MS: float = 20.0
_OPTIMIZED_GAP_MS: float = 2.0

# ---------------------------------------------------------------------------
# Built-in fixture — used when the compiler iphone plan is not yet generated
# (i.e., before Compiler Commit 1 runs).
# ---------------------------------------------------------------------------

_FIXTURE_TARGET_PROFILE_ID = "apple-a17pro-mobile"
_FIXTURE_MODEL_NAME = "tiny-gpt"

_FIXTURE_OPTIMIZED: dict = {
    "target_profile_id": _FIXTURE_TARGET_PROFILE_ID,
    "model_name": _FIXTURE_MODEL_NAME,
    "function_plans": [
        {
            "function_name": "prefill",
            "execution_mode": "colocated",
            "cost_summary": {
                "colocated_total_ms": 8.2,
                "confidence": "low",
                "policy": "colocated",
                "cost_source": "formula_synthetic",
            },
            "kv_plan": {
                "layout": "paged",
                "kv_byte_estimate_mb": 12.0,
                "layout_reason": "prefill_paged_kv",
                "truth_boundary": "static_formula_estimate_not_measured_memory",
            },
            "replay_plan": {
                "replay_eligible": False,
                "cuda_graph_bucket": "",
                "override_reason": "dynamic_shape",
                "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
            },
            "backend_execution_plan": {
                "primary_backend": "coreml",
                "fallback_chain": ["metal", "cpu"],
                "decision_source": "target_preferred",
                "required_precision": "fp16",
                "required_kv_layout": "paged",
                "requires_replay": False,
            },
            "provenance": {
                "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
                "cost_source": "formula_synthetic",
            },
            "source_passes": [
                "serving-phase-analysis",
                "kv-layout-planning",
                "replay-eligibility",
                "execution-provider-planning",
            ],
        },
        {
            "function_name": "decode",
            "execution_mode": "colocated",
            "cost_summary": {
                "colocated_total_ms": 3.6,
                "confidence": "low",
                "policy": "colocated",
                "cost_source": "formula_synthetic",
            },
            "kv_plan": {
                "layout": "paged",
                "kv_byte_estimate_mb": 6.0,
                "layout_reason": "decode_paged_kv",
                "truth_boundary": "static_formula_estimate_not_measured_memory",
            },
            "replay_plan": {
                "replay_eligible": True,
                "cuda_graph_bucket": "decode_static",
                "override_reason": "",
                "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
            },
            "backend_execution_plan": {
                "primary_backend": "coreml",
                "fallback_chain": ["metal", "cpu"],
                "decision_source": "target_preferred",
                "required_precision": "fp16",
                "required_kv_layout": "paged",
                "requires_replay": True,
            },
            "provenance": {
                "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
                "cost_source": "formula_synthetic",
            },
            "source_passes": [
                "serving-phase-analysis",
                "kv-layout-planning",
                "replay-eligibility",
                "execution-provider-planning",
            ],
        },
    ],
}

# Baseline: cpu-only, no compiler optimizations, contiguous KV, no replay.
# Uses same cost estimates as fixture optimized — the difference in latency
# comes from higher baseline costs reflecting unoptimized CPU execution.
_FIXTURE_BASELINE: dict = {
    "target_profile_id": _FIXTURE_TARGET_PROFILE_ID,
    "model_name": _FIXTURE_MODEL_NAME,
    "function_plans": [
        {
            "function_name": "prefill",
            "execution_mode": "colocated",
            "cost_summary": {
                "colocated_total_ms": 48.0,
                "confidence": "low",
                "policy": "colocated",
                "cost_source": "formula_synthetic",
            },
            "kv_plan": {
                "layout": "contiguous",
                "kv_byte_estimate_mb": 24.0,
                "layout_reason": "baseline_cpu_contiguous",
                "truth_boundary": "static_formula_estimate_not_measured_memory",
            },
            "replay_plan": {
                "replay_eligible": False,
                "cuda_graph_bucket": "",
                "override_reason": "baseline_no_replay",
                "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
            },
            "backend_execution_plan": {
                "primary_backend": "cpu",
                "fallback_chain": ["cpu"],
                "decision_source": "baseline_cpu_fixed",
                "required_precision": "fp16",
                "required_kv_layout": "contiguous",
                "requires_replay": False,
            },
            "provenance": {
                "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
                "cost_source": "formula_synthetic",
            },
            "source_passes": [],
        },
        {
            "function_name": "decode",
            "execution_mode": "colocated",
            "cost_summary": {
                "colocated_total_ms": 22.0,
                "confidence": "low",
                "policy": "colocated",
                "cost_source": "formula_synthetic",
            },
            "kv_plan": {
                "layout": "contiguous",
                "kv_byte_estimate_mb": 10.0,
                "layout_reason": "baseline_cpu_contiguous",
                "truth_boundary": "static_formula_estimate_not_measured_memory",
            },
            "replay_plan": {
                "replay_eligible": False,
                "cuda_graph_bucket": "",
                "override_reason": "baseline_no_replay",
                "truth_boundary": "static_shape_replay_eligibility_not_cuda_graph_capture",
            },
            "backend_execution_plan": {
                "primary_backend": "cpu",
                "fallback_chain": ["cpu"],
                "decision_source": "baseline_cpu_fixed",
                "required_precision": "fp16",
                "required_kv_layout": "contiguous",
                "requires_replay": False,
            },
            "provenance": {
                "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
                "cost_source": "formula_synthetic",
            },
            "source_passes": [],
        },
    ],
}


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulated_queue_depth(req_idx: int, variant: str) -> int:
    """Return a deterministic simulated queue depth for req_idx."""
    if variant == "baseline":
        # Baseline: high initial pressure, slow drain.
        return max(2, 10 - req_idx // 4)
    # Optimized: low initial pressure, fast drain.
    return max(1, 3 - req_idx // 12)


def _simulated_memory_mb(req_idx: int, variant: str) -> float:
    """Return a deterministic simulated memory footprint for req_idx."""
    if variant == "baseline":
        # Baseline: contiguous KV — higher resident footprint.
        return 480.0 + (req_idx % 8) * 12.0
    # Optimized: paged KV — lower, more stable footprint.
    return 140.0 + (req_idx % 4) * 8.0


def _run_variant(
    serving_plan_dict: dict,
    *,
    variant_id: str,
    runtime_mode: str,
    optimizer_features: list[str],
    gap_ms: float,
    num_requests: int,
) -> object:
    """Run num_requests simulated requests and return a TraceVariant."""
    plans = CompilerRuntimeAdapter.from_serving_plan(serving_plan_dict)
    engine = ExecutionEngine()
    recorder = ExecutionTraceRecorder()

    for req_idx in range(num_requests):
        t_start = recorder.current_time_ms()

        for plan in plans:
            engine.execute(plan, recorder=recorder)

        t_end = recorder.current_time_ms()
        recorder.record_request_latency(t_end - t_start)

        # Inter-request gap (simulates scheduler overhead / arrival spacing).
        recorder.advance_clock(gap_ms)

        # Snapshot after each request for the playback timeseries.
        recorder.record_snapshot(
            queue_depth=_simulated_queue_depth(req_idx, variant_id),
            memory_mb=_simulated_memory_mb(req_idx, variant_id),
            active_requests=_simulated_queue_depth(req_idx, variant_id),
        )

    return RuntimeProfileTraceBuilder.from_recorder(
        recorder,
        variant_id=variant_id,
        runtime_mode=runtime_mode,
        optimizer_features=optimizer_features,
        truth_boundary=_TRUTH_BOUNDARY,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load or fall back ────────────────────────────────────────────────
    compiler_plan_ref: str
    optimized_dict: dict
    baseline_dict: dict = _FIXTURE_BASELINE

    if _COMPILER_PLAN_PATH.exists():
        with _COMPILER_PLAN_PATH.open(encoding="utf-8") as f:
            optimized_dict = json.load(f)
        compiler_plan_ref = str(_COMPILER_PLAN_PATH.relative_to(_REPO_ROOT.parent))
        print(f"[generate] Compiler plan loaded: {_COMPILER_PLAN_PATH}")
        print(f"[generate] Source: compiler artifact")

        # Derive baseline by overlaying cpu-only settings on every function plan.
        baseline_dict = _cpu_only_overlay(optimized_dict)
    else:
        optimized_dict = _FIXTURE_OPTIMIZED
        compiler_plan_ref = "fixture:built_in"
        print(f"[generate] Compiler plan NOT found at {_COMPILER_PLAN_PATH}")
        print(f"[generate] Source: built-in fixture (serving_execution_plan_iphone.json pending)")

    target_profile_id: str = optimized_dict.get("target_profile_id", _FIXTURE_TARGET_PROFILE_ID)
    model_name: str = optimized_dict.get("model_name", _FIXTURE_MODEL_NAME)

    print(f"[generate] target_profile_id: {target_profile_id}")
    print(f"[generate] model_name: {model_name}")
    print(f"[generate] requests per variant: {_NUM_REQUESTS}")

    # ── Baseline variant ─────────────────────────────────────────────────
    print("[generate] Running baseline variant...")
    baseline = _run_variant(
        baseline_dict,
        variant_id="baseline",
        runtime_mode="fcfs_fixed_batch_cpu",
        optimizer_features=[],
        gap_ms=_BASELINE_GAP_MS,
        num_requests=_NUM_REQUESTS,
    )
    print(
        f"[generate]   baseline p95: {baseline.summary.p95_latency_ms:.1f} ms  "
        f"  peak_memory: {baseline.summary.peak_memory_mb:.1f} MB  "
        f"  total_events: {baseline.summary.total_events}"
    )

    # ── Optimized variant ────────────────────────────────────────────────
    print("[generate] Running optimized variant...")
    optimized = _run_variant(
        optimized_dict,
        variant_id="optimized",
        runtime_mode="cost_aware_paged_kv_coreml",
        optimizer_features=[
            "compiler_backend_selection",
            "paged_kv",
            "replay_eligibility",
            "cost_aware_scheduling",
        ],
        gap_ms=_OPTIMIZED_GAP_MS,
        num_requests=_NUM_REQUESTS,
    )
    print(
        f"[generate]   optimized p95: {optimized.summary.p95_latency_ms:.1f} ms  "
        f"  peak_memory: {optimized.summary.peak_memory_mb:.1f} MB  "
        f"  total_events: {optimized.summary.total_events}"
    )

    # ── Assemble trace ───────────────────────────────────────────────────
    trace = RuntimeProfileTraceBuilder.build_trace(
        target_profile_id=target_profile_id,
        model_name=model_name,
        compiler_plan_ref=compiler_plan_ref,
        baseline=baseline,
        optimized=optimized,
    )

    # ── Write ────────────────────────────────────────────────────────────
    trace.write_json(_OUTPUT_PATH)
    size_kb = _OUTPUT_PATH.stat().st_size / 1024

    print(f"[generate] Output: {_OUTPUT_PATH}")
    print(f"[generate] File size: {size_kb:.1f} KB")
    print(f"[generate] Comparison: {trace.comparison_summary.headline}")


def _cpu_only_overlay(plan_dict: dict) -> dict:
    """Return a copy of plan_dict with every function plan forced to cpu-only, contiguous KV."""
    import copy
    d = copy.deepcopy(plan_dict)
    for fp in d.get("function_plans", []):
        bep = fp.setdefault("backend_execution_plan", {})
        bep["primary_backend"] = "cpu"
        bep["fallback_chain"] = ["cpu"]
        bep["decision_source"] = "baseline_cpu_fixed"
        bep["requires_replay"] = False
        kv = fp.setdefault("kv_plan", {})
        kv["layout"] = "contiguous"
        rp = fp.setdefault("replay_plan", {})
        rp["replay_eligible"] = False
        rp["cuda_graph_bucket"] = ""
        rp["override_reason"] = "baseline_no_replay"
        # Scale up cost estimates to simulate unoptimized CPU execution:
        # multiply compiler cost by a fixed factor representing the absence of
        # hardware-accelerated dispatch.
        cs = fp.setdefault("cost_summary", {})
        base_ms = float(cs.get("colocated_total_ms", 8.0))
        cs["colocated_total_ms"] = round(base_ms * 5.5, 2)
    return d


if __name__ == "__main__":
    main()
