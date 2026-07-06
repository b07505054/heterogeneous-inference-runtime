#!/usr/bin/env python3
"""Generate offline iPhone runtime profile trace.

Pipeline:
  ServingExecutionPlan JSON (or built-in V2 fixture)
    -> list of ExecutionPlanV2 dicts (one per serving phase)
    -> ExecutionEngine.execute(plan, function_name, recorder)  x32 requests
    -> ExecutionTraceRecorder
    -> RuntimeProfileTraceBuilder.from_recorder()
    -> RuntimeProfileTrace
    -> iphone_a17pro_runtime_trace.json

Two variants are produced in a single artifact:
  "baseline"  -- cpu-only, no compiler optimizations, fixed-batch scheduling
  "optimized" -- compiler-guided backend selection, paged KV, replay eligibility

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

import argparse
import copy
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_REPO_ROOT))

from deployment.execution_engine import ExecutionEngine
from deployment.execution_plan_v2.loader import parse_execution_plan_v2
from deployment.execution_trace_recorder import ExecutionTraceRecorder
from deployment.runtime_profile_trace import (
    COMPILER_PLAN_SOURCE_ARTIFACT,
    COMPILER_PLAN_SOURCE_FIXTURE,
    RuntimeProfileTrace,
    RuntimeProfileTraceBuilder,
    TraceVariant,
)

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

_BASELINE_GAP_MS: float = 20.0
_OPTIMIZED_GAP_MS: float = 2.0

# ---------------------------------------------------------------------------
# Built-in fixture -- V2 schema, one plan dict per serving phase.
# Used when the compiler iphone plan is not yet generated.
# ---------------------------------------------------------------------------

_FIXTURE_TARGET_PROFILE_ID = "apple-a17pro-mobile"
_FIXTURE_MODEL_NAME = "tiny-gpt"

_FIXTURE_PLANS_OPTIMIZED: list[dict] = [
    {
        "schema": "execution_plan",
        "schema_version": "2.0.0",
        "plan_id": "prefill",
        "provenance": {
            "compiler_tool": "fixture",
            "model_spec_ref": "",
            "capability_bundle": {"hardware_profile_ref": ""},
            "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
        },
        "model_identity": {},
        "global_decisions": {
            "memory": {
                "kv_cache_layout": "paged",
                "estimated_kv_peak_mb": 12.0,
                "truth_boundary": "static_formula_estimate_not_measured_memory",
            },
            "serving": {
                "topology": "colocated",
                "colocated_cost_estimate_ms": 8.2,
                "replay_eligible": False,
            },
        },
        "function_plans": [{
            "function_name": "prefill",
            "serving_phase": "prefill",
            "backend": {
                "selected_backend": "coreml_ane",
                "fallback_backends": ["arm_compute", "cpu"],
                "reason": "target_preferred",
            },
            "per_op_decisions": [],
        }],
    },
    {
        "schema": "execution_plan",
        "schema_version": "2.0.0",
        "plan_id": "decode",
        "provenance": {
            "compiler_tool": "fixture",
            "model_spec_ref": "",
            "capability_bundle": {"hardware_profile_ref": ""},
            "truth_boundary": "compiler_execution_provider_plan_not_runtime_dispatch",
        },
        "model_identity": {},
        "global_decisions": {
            "memory": {
                "kv_cache_layout": "paged",
                "estimated_kv_peak_mb": 6.0,
                "truth_boundary": "static_formula_estimate_not_measured_memory",
            },
            "serving": {
                "topology": "colocated",
                "colocated_cost_estimate_ms": 3.6,
                "replay_eligible": True,
            },
        },
        "function_plans": [{
            "function_name": "decode",
            "serving_phase": "decode",
            "backend": {
                "selected_backend": "coreml_ane",
                "fallback_backends": ["arm_compute", "cpu"],
                "reason": "target_preferred",
            },
            "per_op_decisions": [],
        }],
    },
]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulated_queue_depth(req_idx: int, variant: str) -> int:
    """Return a deterministic simulated queue depth for req_idx."""
    if variant == "baseline":
        return max(2, 10 - req_idx // 4)
    return max(1, 3 - req_idx // 12)


def _simulated_memory_mb(req_idx: int, variant: str) -> float:
    """Return a deterministic simulated memory footprint for req_idx."""
    if variant == "baseline":
        return 480.0 + (req_idx % 8) * 12.0
    return 140.0 + (req_idx % 4) * 8.0


def _cpu_only_overlay(plan_dict: dict) -> dict:
    """Return a copy of a V2 plan dict with all function plans forced to cpu-only, contiguous KV.

    Scales colocated_cost_estimate_ms by 5.5 to simulate unoptimized CPU execution.
    """
    d = copy.deepcopy(plan_dict)
    gd = d.setdefault("global_decisions", {})
    mem = gd.setdefault("memory", {})
    serving = gd.setdefault("serving", {})
    mem["kv_cache_layout"] = "contiguous"
    serving["replay_eligible"] = False
    base_ms = float(serving.get("colocated_cost_estimate_ms", 8.0))
    serving["colocated_cost_estimate_ms"] = round(base_ms * 5.5, 2)
    for fp in d.get("function_plans", []):
        backend = fp.setdefault("backend", {})
        backend["selected_backend"] = "cpu"
        backend["fallback_backends"] = ["cpu"]
        backend["reason"] = "baseline_cpu_fixed"
    return d


def _v1_serving_plan_to_v2_plans(v1_dict: dict) -> list[dict]:
    """Convert a V1 ServingExecutionPlan JSON to a list of V2 execution plan dicts.

    One V2 plan dict is produced per function_plan entry in the V1 dict.
    Each carries the per-phase cost, KV layout, replay eligibility, and backend.
    """
    tb = "compiler_execution_provider_plan_not_runtime_dispatch"
    result: list[dict] = []
    for fp in v1_dict.get("function_plans", []):
        name = fp.get("function_name", "unknown")
        phase = "decode" if "decode" in name.lower() else "prefill"
        d: dict = {
            "schema": "execution_plan",
            "schema_version": "2.0.0",
            "plan_id": name,
            "provenance": {
                "compiler_tool": "v1_plan_import",
                "model_spec_ref": "",
                "capability_bundle": {"hardware_profile_ref": ""},
                "truth_boundary": fp.get("provenance", {}).get("truth_boundary", tb),
            },
            "model_identity": {},
            "global_decisions": {
                "memory": {
                    "kv_cache_layout": fp.get("kv_plan", {}).get("layout", "contiguous"),
                    "estimated_kv_peak_mb": float(
                        fp.get("kv_plan", {}).get("kv_byte_estimate_mb", 0.0)
                    ),
                    "truth_boundary": fp.get("kv_plan", {}).get("truth_boundary", ""),
                },
                "serving": {
                    "topology": fp.get("execution_mode", "colocated"),
                    "colocated_cost_estimate_ms": float(
                        fp.get("cost_summary", {}).get("colocated_total_ms", 0.0)
                    ),
                    "replay_eligible": bool(
                        fp.get("replay_plan", {}).get("replay_eligible", False)
                    ),
                },
            },
            "function_plans": [{
                "function_name": name,
                "serving_phase": phase,
                "backend": {
                    "selected_backend": fp.get("backend_execution_plan", {}).get(
                        "primary_backend", "cpu"
                    ),
                    "fallback_backends": list(
                        fp.get("backend_execution_plan", {}).get("fallback_chain", [])
                    ),
                    "reason": fp.get("backend_execution_plan", {}).get("decision_source", ""),
                },
                "per_op_decisions": [],
            }],
        }
        result.append(d)
    return result


def _run_variant(
    plan_dicts: list[dict],
    *,
    variant_id: str,
    runtime_mode: str,
    optimizer_features: list[str],
    gap_ms: float,
    num_requests: int,
) -> TraceVariant:
    """Run num_requests simulated requests and return a TraceVariant."""
    engine = ExecutionEngine()
    recorder = ExecutionTraceRecorder()

    for req_idx in range(num_requests):
        t_start = recorder.current_time_ms()

        for plan_dict in plan_dicts:
            plan_v2 = parse_execution_plan_v2(plan_dict)
            for fp in plan_v2.function_plans:
                engine.execute(plan_v2, fp.function_name, recorder=recorder)

        t_end = recorder.current_time_ms()
        recorder.record_request_latency(t_end - t_start)

        recorder.advance_clock(gap_ms)

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
# Warning / summary printers
# ---------------------------------------------------------------------------

def _print_fixture_warning(expected_path: Path) -> None:
    print("=========================================================")
    print("WARNING: COMPILER ARTIFACT MISSING")
    print()
    print("Compiler ServingExecutionPlan was NOT found.")
    print()
    print("Falling back to built-in development fixture.")
    print()
    print("This runtime trace DOES NOT originate from the compiler.")
    print()
    print("DO_NOT_USE_FOR_DEMO = true")
    print()
    print(f"Expected artifact:")
    print(f"  {expected_path}")
    print("=========================================================")


def _print_summary(
    *,
    compiler_plan_source: str,
    compiler_plan_path: str,
    do_not_use_for_demo: bool,
    output_path: Path,
    baseline_p95: float,
    optimized_p95: float,
    headline: str,
    size_kb: float,
) -> None:
    source_label = (
        "COMPILER ARTIFACT"
        if compiler_plan_source == COMPILER_PLAN_SOURCE_ARTIFACT
        else "BUILT-IN FIXTURE"
    )
    print()
    print(f"Compiler Plan Source:  {source_label}")
    print(f"  compiler_plan_path:  {compiler_plan_path}")
    print(f"  do_not_use_for_demo: {str(do_not_use_for_demo).lower()}")
    print(f"  output:              {output_path}")
    print(f"  file size:           {size_kb:.1f} KB")
    print(f"  baseline p95:        {baseline_p95:.1f} ms")
    print(f"  optimized p95:       {optimized_p95:.1f} ms")
    print(f"  headline:            {headline}")


# ---------------------------------------------------------------------------
# Core generation logic (extracted for testability)
# ---------------------------------------------------------------------------

def _generate_trace(
    compiler_plan_path: Path,
    *,
    num_requests: int = _NUM_REQUESTS,
) -> RuntimeProfileTrace:
    """Build a RuntimeProfileTrace from compiler plan or fixture.

    Does not write output or print anything. Returns the assembled trace.
    Callers (main, tests) handle I/O and printing.
    """
    if compiler_plan_path.exists():
        with compiler_plan_path.open(encoding="utf-8") as f:
            v1_dict = json.load(f)
        optimized_plan_dicts = _v1_serving_plan_to_v2_plans(v1_dict)
        target_profile_id = v1_dict.get("target_profile_id", _FIXTURE_TARGET_PROFILE_ID)
        model_name = v1_dict.get("model_name", _FIXTURE_MODEL_NAME)
        compiler_plan_ref = str(compiler_plan_path)
        compiler_plan_path_str = str(compiler_plan_path)
        compiler_plan_source = COMPILER_PLAN_SOURCE_ARTIFACT
        do_not_use_for_demo = False
        provenance_notes: list[str] = []
    else:
        optimized_plan_dicts = _FIXTURE_PLANS_OPTIMIZED
        target_profile_id = _FIXTURE_TARGET_PROFILE_ID
        model_name = _FIXTURE_MODEL_NAME
        compiler_plan_ref = "fixture:built_in"
        compiler_plan_path_str = str(compiler_plan_path)
        compiler_plan_source = COMPILER_PLAN_SOURCE_FIXTURE
        do_not_use_for_demo = True
        provenance_notes = ["compiler_not_in_pipeline"]

    baseline_plan_dicts = [_cpu_only_overlay(pd) for pd in optimized_plan_dicts]

    baseline = _run_variant(
        baseline_plan_dicts,
        variant_id="baseline",
        runtime_mode="fcfs_fixed_batch_cpu",
        optimizer_features=[],
        gap_ms=_BASELINE_GAP_MS,
        num_requests=num_requests,
    )
    optimized = _run_variant(
        optimized_plan_dicts,
        variant_id="optimized",
        runtime_mode="cost_aware_paged_kv_coreml",
        optimizer_features=[
            "compiler_backend_selection",
            "paged_kv",
            "replay_eligibility",
            "cost_aware_scheduling",
        ],
        gap_ms=_OPTIMIZED_GAP_MS,
        num_requests=num_requests,
    )

    return RuntimeProfileTraceBuilder.build_trace(
        target_profile_id=target_profile_id,
        model_name=model_name,
        compiler_plan_ref=compiler_plan_ref,
        baseline=baseline,
        optimized=optimized,
        compiler_plan_source=compiler_plan_source,
        compiler_plan_path=compiler_plan_path_str,
        do_not_use_for_demo=do_not_use_for_demo,
        provenance_notes=provenance_notes,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline iPhone runtime profile trace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--compiler-plan",
        type=Path,
        default=_COMPILER_PLAN_PATH,
        metavar="PATH",
        help="Path to ServingExecutionPlan JSON from compile-for-target",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_OUTPUT_PATH,
        metavar="PATH",
        help="Output path for the runtime profile trace JSON",
    )
    args = parser.parse_args()

    compiler_plan_path: Path = args.compiler_plan
    output_path: Path = args.out

    using_fixture = not compiler_plan_path.exists()

    if using_fixture:
        _print_fixture_warning(compiler_plan_path)
    else:
        print(f"[generate] Compiler plan loaded: {compiler_plan_path}")

    print(f"[generate] requests per variant: {_NUM_REQUESTS}")
    print("[generate] Running baseline variant...")
    print("[generate] Running optimized variant...")

    trace = _generate_trace(compiler_plan_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace.write_json(output_path)
    size_kb = output_path.stat().st_size / 1024

    _print_summary(
        compiler_plan_source=trace.compiler_plan_source,
        compiler_plan_path=trace.compiler_plan_path,
        do_not_use_for_demo=trace.do_not_use_for_demo,
        output_path=output_path,
        baseline_p95=trace.variants["baseline"].summary.p95_latency_ms,
        optimized_p95=trace.variants["optimized"].summary.p95_latency_ms,
        headline=trace.comparison_summary.headline,
        size_kb=size_kb,
    )


if __name__ == "__main__":
    main()
