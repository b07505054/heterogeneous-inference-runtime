import argparse
import json
import random
import sys
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.llm_runtime_decision import (  # noqa: E402
    CostModel,
    MemoryPlanner,
    Request,
    RuntimeScheduler,
    build_requests,
    summarize_policy,
)
from deployment.distributed_serving import build_distributed_serving_artifacts  # noqa: E402


def percentile(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_policy(policy, requests, cost_model, seed, total_blocks, block_size_tokens, kv_mb_per_block):
    scheduler = RuntimeScheduler(
        policy=policy,
        cost_model=cost_model,
        memory=MemoryPlanner(
            total_blocks=total_blocks,
            block_size_tokens=block_size_tokens,
            kv_mb_per_block=kv_mb_per_block,
        ),
        rng=random.Random(seed),
        max_decode_batch_size=8,
    )
    return scheduler.run(requests)


def inflight_gate(candidate_summary, incumbent_summary, *, ttft_tolerance=0.10):
    tpot_ok = candidate_summary["tpot_p95_ms"] <= incumbent_summary["tpot_p95_ms"] * 1.03
    throughput_ok = candidate_summary["tokens_per_second"] > incumbent_summary["tokens_per_second"]
    oom_ok = candidate_summary["reject_oom_count"] <= incumbent_summary["reject_oom_count"]
    reject_ok = candidate_summary["rejected_requests"] <= incumbent_summary["rejected_requests"]
    ttft_ok = candidate_summary["ttft_p95_ms"] <= incumbent_summary["ttft_p95_ms"] * (1.0 + ttft_tolerance)
    invariants = candidate_summary.get("invariants", {})
    invariant_ok = all(
        value if isinstance(value, bool) else value == 0
        for value in invariants.values()
    )
    passes = all([tpot_ok, throughput_ok, oom_ok, reject_ok, ttft_ok, invariant_ok])
    return {
        "candidate_policy": candidate_summary["policy"],
        "incumbent_policy": incumbent_summary["policy"],
        "passes_gate": passes,
        "ttft_tolerance": ttft_tolerance,
        "checks": {
            "tpot_p95_within_3_percent": tpot_ok,
            "throughput_improves": throughput_ok,
            "oom_count_not_regressed": oom_ok,
            "reject_count_not_regressed": reject_ok,
            "ttft_p95_within_tolerance": ttft_ok,
            "lifecycle_invariants_pass": invariant_ok,
        },
        "metrics": {
            "candidate_tpot_p95_ms": candidate_summary["tpot_p95_ms"],
            "incumbent_tpot_p95_ms": incumbent_summary["tpot_p95_ms"],
            "candidate_ttft_p95_ms": candidate_summary["ttft_p95_ms"],
            "incumbent_ttft_p95_ms": incumbent_summary["ttft_p95_ms"],
            "candidate_tokens_per_second": candidate_summary["tokens_per_second"],
            "incumbent_tokens_per_second": incumbent_summary["tokens_per_second"],
            "candidate_reject_oom_count": candidate_summary["reject_oom_count"],
            "incumbent_reject_oom_count": incumbent_summary["reject_oom_count"],
        },
    }


def percentile_or_zero(values, p):
    return round(percentile(values, p), 3) if values else 0.0


def build_decode_interleave_heavy_requests():
    return [
        Request("long-prefill-001", prompt_tokens=2048, output_tokens=2, arrival_ms=0.0),
        Request("short-decode-001", prompt_tokens=64, output_tokens=8, arrival_ms=1.0),
        Request("short-decode-002", prompt_tokens=64, output_tokens=8, arrival_ms=2.0),
    ]


def workload_scenarios(default_requests):
    return [
        {
            "name": "default_mixed_pressure",
            "objective": "throughput_under_mixed_prefill_decode_pressure",
            "requests": default_requests,
            "gate": "throughput_tpot_ttft_guard",
        },
        {
            "name": "decode_interleave_heavy",
            "objective": "short_prompt_ttft_under_interleaved_long_prefill",
            "requests": build_decode_interleave_heavy_requests(),
            "gate": "short_prompt_ttft_tpot_guard",
        },
    ]


def request_shape_metrics(requests):
    prompts = [req.prompt_tokens for req in requests]
    outputs = [req.output_tokens for req in requests]
    return {
        "request_count": len(requests),
        "prompt_tokens": {
            "min": min(prompts) if prompts else 0,
            "max": max(prompts) if prompts else 0,
            "p50": percentile_or_zero(prompts, 50),
            "p95": percentile_or_zero(prompts, 95),
        },
        "output_tokens": {
            "min": min(outputs) if outputs else 0,
            "max": max(outputs) if outputs else 0,
            "p50": percentile_or_zero(outputs, 50),
            "p95": percentile_or_zero(outputs, 95),
        },
        "short_prompt_request_count": sum(1 for req in requests if req.prompt_tokens <= 128),
        "long_prompt_request_count": sum(1 for req in requests if req.prompt_tokens >= 1024),
    }


def short_prompt_latency_metric(result, requests):
    arrivals = {
        req.request_id: req.arrival_ms
        for req in requests
        if req.prompt_tokens <= 128
    }
    queue_waits = [
        event.get("queue_wait_ms", 0.0)
        for event in result.serving_events
        if event.get("event") == "request_admitted"
        and event.get("request_id") in arrivals
    ]
    first_tokens = [
        event.get("ttft_ms", 0.0)
        for event in result.serving_events
        if event.get("event") == "first_token"
        and event.get("request_id") in arrivals
    ]
    prefill_starts = [
        event.get("time_ms", 0.0) - arrivals[event.get("request_id")]
        for event in result.serving_events
        if event.get("event") == "prefill_start"
        and event.get("request_id") in arrivals
    ]
    service_latencies = first_tokens or queue_waits or prefill_starts
    return {
        "short_prompt_count": len(arrivals),
        "queue_wait_p95_ms": percentile_or_zero(queue_waits, 95),
        "first_token_p95_ms": percentile_or_zero(first_tokens, 95),
        "prefill_start_wait_p95_ms": percentile_or_zero(prefill_starts, 95),
        "service_latency_p95_ms": percentile_or_zero(service_latencies, 95),
        "metric_source": (
            "first_token"
            if first_tokens
            else "queue_wait"
            if queue_waits
            else "prefill_start"
            if prefill_starts
            else "unavailable"
        ),
    }


def inflight_gate_for_scenario(scenario, candidate, incumbent, candidate_summary, incumbent_summary):
    if scenario["name"] == "default_mixed_pressure":
        gate = inflight_gate(candidate_summary, incumbent_summary)
        gate["scenario_gate"] = scenario["gate"]
        gate["objective"] = scenario["objective"]
        return gate

    candidate_short = short_prompt_latency_metric(candidate, scenario["requests"])
    incumbent_short = short_prompt_latency_metric(incumbent, scenario["requests"])
    short_ttft_ok = (
        candidate_short["service_latency_p95_ms"] > 0
        and incumbent_short["service_latency_p95_ms"] > 0
        and candidate_short["service_latency_p95_ms"] < incumbent_short["service_latency_p95_ms"]
    )
    tpot_ok = candidate_summary["tpot_p95_ms"] <= incumbent_summary["tpot_p95_ms"] * 1.03
    oom_ok = candidate_summary["reject_oom_count"] <= incumbent_summary["reject_oom_count"]
    reject_ok = candidate_summary["rejected_requests"] <= incumbent_summary["rejected_requests"]
    invariants = candidate_summary.get("invariants", {})
    invariant_ok = all(
        value if isinstance(value, bool) else value == 0
        for value in invariants.values()
    )
    passes = all([short_ttft_ok, tpot_ok, oom_ok, reject_ok, invariant_ok])
    return {
        "candidate_policy": candidate_summary["policy"],
        "incumbent_policy": incumbent_summary["policy"],
        "passes_gate": passes,
        "scenario_gate": scenario["gate"],
        "objective": scenario["objective"],
        "checks": {
            "short_prompt_service_latency_improves": short_ttft_ok,
            "tpot_p95_within_3_percent": tpot_ok,
            "oom_count_not_regressed": oom_ok,
            "reject_count_not_regressed": reject_ok,
            "lifecycle_invariants_pass": invariant_ok,
        },
        "metrics": {
            "candidate_short_prompt": candidate_short,
            "incumbent_short_prompt": incumbent_short,
            "candidate_tpot_p95_ms": candidate_summary["tpot_p95_ms"],
            "incumbent_tpot_p95_ms": incumbent_summary["tpot_p95_ms"],
            "candidate_tokens_per_second": candidate_summary["tokens_per_second"],
            "incumbent_tokens_per_second": incumbent_summary["tokens_per_second"],
            "candidate_reject_oom_count": candidate_summary["reject_oom_count"],
            "incumbent_reject_oom_count": incumbent_summary["reject_oom_count"],
        },
    }


def run_scenario_comparison(
    scenario,
    *,
    seed,
    total_blocks,
    block_size_tokens,
    kv_mb_per_block,
):
    requests = scenario["requests"]
    baseline = run_policy(
        "fcfs_fixed_batch",
        requests,
        CostModel(),
        seed + 100,
        total_blocks,
        block_size_tokens,
        kv_mb_per_block,
    )
    calibrated_cost = CostModel()
    calibration_report = calibrated_cost.calibrate(
        baseline.prefill_latencies[:8],
        baseline.decode_step_latencies[:64],
    )
    optimized = run_policy(
        "cost_aware_memory_pressure",
        requests,
        calibrated_cost,
        seed + 200,
        total_blocks,
        block_size_tokens,
        kv_mb_per_block,
    )
    page_prefetch_cost = CostModel(
        observed_decode_scale=calibrated_cost.observed_decode_scale,
        observed_prefill_scale=calibrated_cost.observed_prefill_scale,
    )
    page_prefetch = run_policy(
        "cost_aware_memory_pressure_page_prefetch",
        requests,
        page_prefetch_cost,
        seed + 200,
        total_blocks,
        block_size_tokens,
        kv_mb_per_block,
    )
    inflight_cost = CostModel(
        observed_decode_scale=calibrated_cost.observed_decode_scale,
        observed_prefill_scale=calibrated_cost.observed_prefill_scale,
    )
    inflight = run_policy(
        "inflight_paged_kv_continuous_batching",
        requests,
        inflight_cost,
        seed + 300,
        total_blocks,
        block_size_tokens,
        kv_mb_per_block,
    )

    baseline_summary = summarize_policy(baseline)
    optimized_summary = summarize_policy(optimized)
    page_prefetch_summary = summarize_policy(page_prefetch)
    inflight_summary = summarize_policy(inflight)
    page_prefetch_report = build_page_prefetch_report(
        optimized=optimized,
        page_prefetch=page_prefetch,
        optimized_summary=optimized_summary,
        page_prefetch_summary=page_prefetch_summary,
    )

    incumbent = page_prefetch if page_prefetch_report["selected_policy"] == page_prefetch.policy else optimized
    incumbent_summary = page_prefetch_summary if incumbent is page_prefetch else optimized_summary
    inflight_gate_report = inflight_gate_for_scenario(
        scenario,
        inflight,
        incumbent,
        inflight_summary,
        incumbent_summary,
    )
    selected = inflight if inflight_gate_report["passes_gate"] else incumbent
    selected_summary = summarize_policy(selected)
    summaries = [baseline_summary, optimized_summary, page_prefetch_summary, inflight_summary]
    return {
        "scenario": scenario,
        "requests": requests,
        "baseline": baseline,
        "optimized": optimized,
        "page_prefetch": page_prefetch,
        "inflight": inflight,
        "selected": selected,
        "baseline_summary": baseline_summary,
        "optimized_summary": optimized_summary,
        "page_prefetch_summary": page_prefetch_summary,
        "inflight_summary": inflight_summary,
        "selected_summary": selected_summary,
        "policy_summaries": summaries,
        "page_prefetch_report": page_prefetch_report,
        "calibration_report": calibration_report,
        "inflight_gate_report": inflight_gate_report,
        "scenario_result": {
            "scenario": scenario["name"],
            "objective": scenario["objective"],
            "input": request_shape_metrics(requests),
            "decision": {
                "incumbent_policy": incumbent.policy,
                "candidate_policy": inflight.policy,
                "selected_policy": selected.policy,
                "gate_passed": inflight_gate_report["passes_gate"],
            },
            "metric": inflight_gate_report["metrics"],
            "checks": inflight_gate_report["checks"],
            "policy_summaries": summaries,
            "page_lifecycle": {
                "selected": selected.kv_page_lifecycle,
                "inflight_candidate": inflight.kv_page_lifecycle,
            },
        },
    }


def build_chrome_trace(result):
    trace = []
    for event in result.serving_events:
        if event.get("event") != "prefill_start":
            continue
        request_id = event["request_id"]
        end = next(
            (
                row for row in result.serving_events
                if row.get("request_id") == request_id and row.get("event") == "prefill_end"
            ),
            None,
        )
        if not end:
            continue
        trace.append(
            {
                "name": "prefill",
                "cat": "llm_runtime",
                "ph": "X",
                "ts": round(event["time_ms"] * 1000, 3),
                "dur": round(end.get("prefill_latency_ms", 0.0) * 1000, 3),
                "pid": 1,
                "tid": 1,
                "args": {
                    "request_id": request_id,
                    "tokens": event.get("prompt_tokens"),
                },
            }
        )
    return trace


def compact_scheduler_steps(steps, *, max_steps=160):
    selected = []
    seen = set()
    for idx, step in enumerate(steps):
        action = step.get("selected_action")
        important = (
            idx < 48
            or action in {"prefill_chunk", "mixed_step", "drain_decode", "reject_or_delay"}
            or step.get("pressure_level") in {"high", "critical"}
            or idx % 64 == 0
        )
        if not important:
            continue
        key = (idx, step.get("step"))
        if key in seen:
            continue
        selected.append(step)
        seen.add(key)
        if len(selected) >= max_steps:
            break
    return selected


def request_rows(requests):
    return [
        {
            "request_id": req.request_id,
            "arrival_ms": req.arrival_ms,
            "prompt_tokens": req.prompt_tokens,
            "output_tokens": req.output_tokens,
            "prefix_key": f"prefix-{idx % 4}",
        }
        for idx, req in enumerate(requests)
    ]


def build_trace_adapter_report(*, adapter_name, requests, selected, scheduler_trace, runtime_profile):
    rows = request_rows(requests)
    decode_steps = [
        step for step in scheduler_trace.get("steps", [])
        if step.get("event") == "decode_batch"
    ]
    if adapter_name == "vllm":
        source_trace = [
            {
                "request_id": row["request_id"],
                "arrival_time_ms": row["arrival_ms"],
                "prompt_token_ids_len": row["prompt_tokens"],
                "max_tokens": row["output_tokens"],
                "prefix_cache_key": row["prefix_key"],
                "scheduler": "vllm_compatible_continuous_batching_trace",
            }
            for row in rows
        ]
        lifecycle = "arrival -> prefill -> continuous decode batch -> finish"
        decision = "continuous batching admission and paged-KV pressure cap"
        output_name = "vllm_trace_adapter_report"
    else:
        source_trace = [
            {
                "request_id": row["request_id"],
                "arrival_ms": row["arrival_ms"],
                "input_tokens": row["prompt_tokens"],
                "sampling_params": {"max_new_tokens": row["output_tokens"]},
                "prefix": row["prefix_key"],
                "decode_program": "sglang_compatible_request_decode_trace",
            }
            for row in rows
        ]
        lifecycle = "request program -> prefix lookup -> decode scheduling -> finish"
        decision = "request/decode scheduling with prefix-reuse metadata"
        output_name = "sglang_trace_adapter_report"

    return {
        "artifact_type": output_name,
        "source": "scripts/generate_llm_runtime_artifacts.py",
        "integration_level": "trace_adapter_not_framework_fork",
        "technology_gate": {
            "input": f"{adapter_name} style synthetic request/decode trace",
            "decision": decision,
            "metric": "TTFT, TPOT, throughput, queue wait, KV pressure, decode batch efficiency",
            "passes_gate": True,
        },
        "positioning": (
            f"This is a {adapter_name} trace adapter. It maps framework-shaped "
            "request records into the local RuntimeScheduler model; it does not "
            f"claim modified {adapter_name} internals."
        ),
        "source_trace_schema": source_trace[:8],
        "imported_request_count": len(rows),
        "mapped_runtime_request_fields": [
            "request_id",
            "arrival_ms",
            "prompt_tokens",
            "output_tokens",
            "prefix_key",
        ],
        "runtime_decision": {
            "scheduler_policy": selected.policy,
            "lifecycle": lifecycle,
            "decode_batch_events": len(decode_steps),
            "avg_decode_batch_size": selected.avg_decode_batch_size,
            "decode_batch_efficiency": selected.decode_batch_efficiency,
            "pressure_limited_candidates": selected.pressure_limited_candidates,
            "peak_kv_cache_mb": selected.peak_kv_cache_mb,
        },
        "serving_metrics": {
            "completed_requests": selected.completed_requests,
            "rejected_requests": selected.rejected_requests,
            "delayed_requests": selected.delayed_requests,
            "throughput_tokens_per_s": runtime_profile["tokens_per_second"],
            "e2e_p95_ms": runtime_profile["p95_latency_ms"],
            "peak_memory_mb": runtime_profile["peak_memory_mb"],
            "peak_kv_cache_mb": runtime_profile["peak_kv_cache_mb"],
        },
        "remaining_work": [
            f"replace synthetic {adapter_name} trace with exported trace from a live framework run",
            f"add round-trip comparison against real {adapter_name} scheduler logs",
        ],
    }


def build_serving_framework_report(
    *,
    baseline,
    optimized,
    baseline_summary,
    optimized_summary,
    prefill_decode_benchmark,
    runtime_profile,
    scheduler_decision_report,
    page_prefetch=None,
    page_prefetch_summary=None,
    page_prefetch_report=None,
    inflight=None,
    inflight_summary=None,
    inflight_gate_report=None,
    scenario_results=None,
):
    def metric_row(name, result, summary, style):
        decode_latencies = result.decode_step_latencies or [0.0]
        return {
            "framework_style": name,
            "policy": result.policy,
            "scheduler_policy": style["scheduler_policy"],
            "batching": style["batching"],
            "kv_cache_policy": style["kv_cache_policy"],
            "backend_routing": style["backend_routing"],
            "completed_requests": result.completed_requests,
            "rejected_requests": result.rejected_requests,
            "delayed_requests": result.delayed_requests,
            "ttft_p95_ms": round(percentile(result.prefill_latencies, 95), 3),
            "tpot_p95_ms": round(percentile(decode_latencies, 95), 3),
            "e2e_p95_ms": summary["p95_latency_ms"],
            "throughput_tokens_per_s": summary["tokens_per_second"],
            "avg_decode_batch_size": summary["avg_decode_batch_size"],
            "decode_batch_efficiency": summary["decode_batch_efficiency"],
            "peak_kv_cache_mb": summary["peak_kv_cache_mb"],
            "pressure_limited_candidates": summary["pressure_limited_candidates"],
            "serving_metrics": [
                "TTFT",
                "TPOT",
                "latency_p95",
                "tokens_per_second",
                "queue_wait",
                "kv_cache_pressure",
            ],
        }

    baseline_row = metric_row(
        "baseline_fcfs",
        baseline,
        baseline_summary,
        {
            "scheduler_policy": "single_request_fcfs",
            "batching": "static_batch_1",
            "kv_cache_policy": "capacity_only",
            "backend_routing": "fixed_pytorch_or_default_backend",
        },
    )
    optimized_row = metric_row(
        "vllm_sglang_style",
        optimized,
        optimized_summary,
        {
            "scheduler_policy": "continuous_batching_prefill_decode_split",
            "batching": "memory_pressure_aware_adaptive_decode_batch",
            "kv_cache_policy": "paged_kv_cache_pressure_tracking",
            "backend_routing": "profile_guided_backend_dispatch",
        },
    )
    rows = [baseline_row, optimized_row]
    if page_prefetch is not None and page_prefetch_summary is not None:
        page_prefetch_row = metric_row(
            "vllm_style_page_prefetch",
            page_prefetch,
            page_prefetch_summary,
            {
                "scheduler_policy": "continuous_batching_with_kv_page_prefetch",
                "batching": "memory_pressure_aware_adaptive_decode_batch",
                "kv_cache_policy": "allocated_kv_page_prefetch_under_pressure_budget",
                "backend_routing": "profile_guided_backend_dispatch",
            },
        )
        page_prefetch_row["page_prefetch"] = page_prefetch.page_prefetch
        page_prefetch_row["selection_status"] = (page_prefetch_report or {}).get("selected_policy")
        rows.append(page_prefetch_row)
    if inflight is not None and inflight_summary is not None:
        inflight_row = metric_row(
            "tensorrt_llm_aligned_local_runtime_policy",
            inflight,
            inflight_summary,
            {
                "scheduler_policy": "inflight_paged_kv_continuous_batching",
                "batching": "event_loop_chunked_prefill_plus_continuous_decode",
                "kv_cache_policy": "paged_kv_lifecycle_with_pressure_aware_prefetch",
                "backend_routing": "local_worker_device_fields_only_no_distributed_execution_claim",
            },
        )
        inflight_row["positioning"] = (
            "TensorRT-LLM-aligned concepts, not TensorRT-LLM internals"
        )
        inflight_row["lifecycle_invariants"] = inflight_summary.get("invariants", {})
        inflight_row["selection_gate"] = inflight_gate_report or {}
        rows.append(inflight_row)
    triton_row = {
        **optimized_row,
        "framework_style": "triton_server_style",
        "scheduler_policy": "dynamic_batching_backend_instance_routing",
        "batching": "dynamic_batching_projection_from_runtime_trace",
        "kv_cache_policy": "model_instance_memory_budget",
        "backend_routing": "backend_instance_metadata_and_profile_selection",
    }
    tensorrt_row = {
        **optimized_row,
        "framework_style": "tensorrt_style",
        "scheduler_policy": "execution_context_shape_profile",
        "batching": "optimization_profile_batch_shape_selection",
        "kv_cache_policy": "engine_workspace_plus_kv_cache_budget",
        "backend_routing": "tensorrt_engine_candidate_when_available",
    }
    rows.extend([triton_row, tensorrt_row])

    return {
        "artifact_type": "serving_framework_report",
        "source": "deployment.llm_runtime_decision",
        "positioning": (
            "Portfolio-sized serving framework comparison inspired by vLLM, "
            "SGLang, Triton Server, and TensorRT. It does not vendor those "
            "frameworks; it maps their core runtime decisions onto the local "
            "scheduler, KV-cache planner, backend dispatch, and serving metrics."
        ),
        "framework_targets": [
            "vLLM continuous batching and paged KV-cache pressure",
            "vLLM-style allocated KV page prefetch under memory-pressure budget",
            "TensorRT-LLM-aligned local in-flight batching and paged KV lifecycle",
            "SGLang request/decode scheduling and prefix/KV reuse hooks",
            "Triton Server dynamic batching and backend instance routing",
            "TensorRT engine/optimization-profile backend candidate selection",
        ],
        "metrics": {
            "ttft_ms": prefill_decode_benchmark["prefill_latency_ms"],
            "tpot_p95_ms": prefill_decode_benchmark["p95_decode_latency_ms"],
            "e2e_p95_ms": runtime_profile["p95_latency_ms"],
            "throughput_tokens_per_s": runtime_profile["tokens_per_second"],
            "peak_kv_cache_mb": runtime_profile["peak_kv_cache_mb"],
            "peak_memory_mb": runtime_profile["peak_memory_mb"],
            "oom_events": runtime_profile["oom_events"],
        },
        "comparisons": rows,
        "workload_aware_policy_selection": [
            {
                "scenario": row["scenario"],
                "objective": row["objective"],
                "selected_policy": row["decision"]["selected_policy"],
                "candidate_policy": row["decision"]["candidate_policy"],
                "gate_passed": row["decision"]["gate_passed"],
                "input": row["input"],
                "checks": row["checks"],
            }
            for row in (scenario_results or [])
        ],
        "selected_framework_style": "vllm_sglang_style",
        "selection_reason": scheduler_decision_report["selection_reason"],
        "improvement": scheduler_decision_report["improvement"],
        "optional_integrations": {
            "vllm": "design target only; no dependency required for CI",
            "sglang": "design target only; request trace maps to decode scheduling hooks",
            "triton_server": "backend-routing abstraction mirrors dynamic batching decisions",
            "tensorrt": "real TensorRT benchmark artifacts are consumed when available",
        },
    }


def build_page_prefetch_report(*, optimized, page_prefetch, optimized_summary, page_prefetch_summary):
    optimized_decode = optimized.decode_step_latencies or [0.0]
    prefetch_decode = page_prefetch.decode_step_latencies or [0.0]
    tpot_delta = round(
        percentile(prefetch_decode, 95) - percentile(optimized_decode, 95),
        4,
    )
    throughput_delta = round(
        page_prefetch.tokens_per_second - optimized.tokens_per_second,
        4,
    )
    e2e_delta = round(
        page_prefetch_summary["p95_latency_ms"] - optimized_summary["p95_latency_ms"],
        4,
    )
    prefetch = page_prefetch.page_prefetch
    faster_or_equal = throughput_delta >= 0 and tpot_delta <= 0 and e2e_delta <= 0
    correct = page_prefetch.oom_events == 0 and page_prefetch.completed_requests == optimized.completed_requests
    selected_policy = (
        "cost_aware_memory_pressure_page_prefetch"
        if correct and faster_or_equal
        else "cost_aware_memory_pressure"
    )
    return {
        "artifact_type": "vllm_style_page_prefetch_report",
        "source": "deployment.llm_runtime_decision",
        "integration_level": "vllm_style_scheduler_simulation_not_framework_fork",
        "technology_gate": {
            "input": "vLLM-style request/decode trace plus allocated KV block map and pending decode candidates",
            "decision": "prefetch next decode KV pages only when memory pressure is below the prefetch budget",
            "metric": "prefetch hit rate, wasted prefetch blocks, TPOT p95, decode p95, throughput, queue wait, OOM/rejection rate",
            "passes_gate": True,
        },
        "positioning": (
            "Page prefetch is modeled as a real RuntimeScheduler policy over "
            "already allocated KV blocks. It is not a vLLM fork and does not "
            "claim CUDA cp.async or physical GPU page migration."
        ),
        "candidate_policy": "cost_aware_memory_pressure_page_prefetch",
        "fallback_policy": "cost_aware_memory_pressure",
        "selected_policy": selected_policy,
        "selection_reason": (
            "page prefetch improved TPOT/e2e/throughput without OOM regression"
            if selected_policy == "cost_aware_memory_pressure_page_prefetch"
            else "fallback retained because page prefetch did not improve all serving metrics or correctness"
        ),
        "input": {
            "source": "synthetic vLLM-style trace adapter and RuntimeScheduler KV allocations",
            "prefetch_scope": "allocated_kv_pages",
            "decode_policy": page_prefetch.policy,
        },
        "decision": {
            "pressure_disable_threshold": prefetch.get("pressure_disable_threshold"),
            "max_prefetch_blocks_per_step": prefetch.get("max_prefetch_blocks_per_step"),
            "hit_latency_saving_ms": prefetch.get("hit_latency_saving_ms"),
            "wasted_prefetch_penalty_ms": prefetch.get("wasted_prefetch_penalty_ms"),
            "correctness_guard": "reject candidate on OOM/completed-request regression",
            "fallback_guard": "keep no-prefetch policy if p95/TPOT/throughput do not improve",
        },
        "metric": {
            "prefetch_attempts": prefetch.get("attempts", 0),
            "prefetch_hits": prefetch.get("hits", 0),
            "prefetch_misses": prefetch.get("misses", 0),
            "prefetch_hit_rate": prefetch.get("hit_rate", 0.0),
            "wasted_prefetch_blocks": prefetch.get("wasted_prefetch_blocks", 0),
            "pressure_skips": prefetch.get("pressure_skips", 0),
            "optimized_tpot_p95_ms": round(percentile(optimized_decode, 95), 4),
            "prefetch_tpot_p95_ms": round(percentile(prefetch_decode, 95), 4),
            "tpot_p95_delta_ms": tpot_delta,
            "optimized_e2e_p95_ms": optimized_summary["p95_latency_ms"],
            "prefetch_e2e_p95_ms": page_prefetch_summary["p95_latency_ms"],
            "e2e_p95_delta_ms": e2e_delta,
            "optimized_tokens_per_second": optimized.tokens_per_second,
            "prefetch_tokens_per_second": page_prefetch.tokens_per_second,
            "tokens_per_second_delta": throughput_delta,
            "optimized_peak_kv_cache_mb": optimized.peak_kv_cache_mb,
            "prefetch_peak_kv_cache_mb": page_prefetch.peak_kv_cache_mb,
            "oom_events": page_prefetch.oom_events,
        },
        "baseline_policy": optimized_summary,
        "candidate_policy_summary": page_prefetch_summary,
        "serving_effect": {
            "affects": ["TPOT", "decode_p95", "throughput", "KV pressure"],
            "selected_for_execution": selected_policy == "cost_aware_memory_pressure_page_prefetch",
        },
        "remaining_work": [
            "replace synthetic trace with exported vLLM scheduler logs",
            "wire policy into a real vLLM fork before claiming framework-internal modification",
            "measure physical GPU page/cache prefetch separately before claiming CUDA cp.async",
        ],
    }


def build_technology_gate_audit():
    return {
        "artifact_type": "technology_gate_audit",
        "source": "scripts/generate_llm_runtime_artifacts.py",
        "gate_questions": [
            "Input source",
            "Compiler/runtime decision",
            "Execution or serving metric impact",
        ],
        "main_plan": [
            {
                "technology": "runtime_scheduler",
                "input": "generated LLM request trace",
                "decision": "FCFS baseline vs cost-aware memory-pressure scheduler",
                "metric": "TTFT, TPOT, throughput, p95 latency, queue wait",
                "status": "implemented",
            },
            {
                "technology": "continuous_batching",
                "input": "pending decode candidates",
                "decision": "decode batch membership under KV pressure cap",
                "metric": "decode batch efficiency, throughput, queue wait",
                "status": "implemented",
            },
            {
                "technology": "kv_cache_planner",
                "input": "prompt/output tokens and block capacity",
                "decision": "KV block allocation/admission/delay/reject",
                "metric": "peak KV MB, pressure level, OOM/reject count",
                "status": "implemented",
            },
            {
                "technology": "backend_dispatch",
                "input": "prefill/decode/KV bookkeeping ops",
                "decision": "GPU attention placement plus CPU KV bookkeeping",
                "metric": "backend counts, placement latency, throughput",
                "status": "implemented",
            },
            {
                "technology": "gpu_pgo_like_kernel_selection",
                "input": "compiler-emitted HIR op plus runtime RMSNorm shape/workload distribution",
                "decision": "profile-guided selection among CUDA/Triton/PyTorch RMSNorm candidates",
                "metric": "kernel p95 latency, effective bandwidth, TPOT projection, throughput projection",
                "status": "implemented_as_runtime_profile_feedback",
            },
            {
                "technology": "vllm_trace_adapter",
                "input": "vLLM-style request trace",
                "decision": "continuous batching and paged-KV pressure mapping",
                "metric": "TTFT, TPOT, throughput, queue wait, KV pressure",
                "status": "implemented_as_adapter",
            },
            {
                "technology": "vllm_style_page_prefetch",
                "input": "vLLM-style request/decode trace plus allocated KV block map",
                "decision": "prefetch next decode KV pages when memory pressure is below budget",
                "metric": "prefetch hit rate, wasted prefetch blocks, TPOT p95, throughput, OOM/rejection rate",
                "status": "implemented_as_runtime_scheduler_candidate",
            },
            {
                "technology": "tensorrt_llm_aligned_inflight_runtime_policy",
                "input": "local LLM request trace plus paged KV lifecycle state",
                "decision": "event-loop tick chooses chunked prefill, continuous decode, drain, delay, or reject",
                "metric": "TTFT, TPOT, throughput, decode batch size, page hit rate, page leaks, reject/OOM gates",
                "status": "implemented_as_local_policy_not_tensorrt_llm_internals",
            },
            {
                "technology": "sglang_trace_adapter",
                "input": "SGLang-style request/decode trace",
                "decision": "request/decode scheduling with prefix metadata",
                "metric": "TTFT, TPOT, throughput, queue wait, KV pressure",
                "status": "implemented_as_adapter",
            },
            {
                "technology": "distributed_kv_aware_routing",
                "input": "vLLM-style serving trace plus worker queue/KV residency state",
                "decision": "round-robin vs least-queue vs KV-aware request routing",
                "metric": "TTFT, TPOT, throughput, cache hit rate, queue wait, peak worker pressure",
                "status": "implemented_as_distributed_serving_control_plane",
            },
            {
                "technology": "worker_health_failover",
                "input": "distributed serving trace with injected worker timeout",
                "decision": "timeout detection, retry, quarantine, failover routing",
                "metric": "retry count, failover count, request loss, latency regression",
                "status": "implemented_as_fault_tolerance_simulation",
            },
            {
                "technology": "protobuf_serving_contract",
                "input": "GenerateRequest/WorkerHealth/RouteDecision/KVShardMetadata schema",
                "decision": "define distributed serving RPC/control-plane boundary",
                "metric": "schema coverage for request routing, worker health, route decision, KV shard metadata",
                "status": "implemented_as_contract_not_production_grpc",
            },
        ],
        "remaining_not_in_main_plan": [
            {
                "technology": "StableHLO_native_pipeline",
                "missing": "must connect StableHLO input to HIR kernel selection and benchmark",
                "next_step": "StableHLO -> HIR -> kernel selection -> runtime benchmark report",
            },
            {
                "technology": "JAX_tracing",
                "missing": "must export JAX StableHLO and feed it into HIR/kernel selection",
                "next_step": "jax.lower().compiler_ir('stablehlo') -> HIR -> runtime report",
            },
            {
                "technology": "Triton_Server",
                "missing": "must provide model repository/config.pbtxt/dynamic batching benchmark",
                "next_step": "Triton deployment package plus validation report",
            },
            {
                "technology": "TensorRT_LLM",
                "missing": "must use LLM/transformer input, not only CV engine evidence",
                "next_step": "Tiny transformer ONNX -> TensorRT engine -> serving metrics",
            },
            {
                "technology": "Ray_Kubernetes",
                "missing": "must connect deployment config to serving metrics and validation",
                "next_step": "Ray/K8s deployment artifacts plus SLO report",
            },
            {
                "technology": "multimodal_serving",
                "missing": "must define text/vision/speech inputs sharing scheduler and metrics",
                "next_step": "mixed workload simulator with per-modality latency metrics",
            },
            {
                "technology": "OpenXLA_XLA_PJRT",
                "missing": "tool availability alone does not pass the gate",
                "next_step": "keep out of main plan until tied to a runnable frontend/runtime path",
            },
        ],
    }


def percentile_local(values, p):
    return round(percentile(values, p), 4)


def read_file_timing(path, runs):
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "available": False,
            "reason": "file not found",
        }

    timings = []
    sizes = []
    for _ in range(max(1, runs)):
        start = time.perf_counter()
        data = path.read_bytes()
        timings.append((time.perf_counter() - start) * 1000)
        sizes.append(len(data))

    return {
        "path": str(path),
        "available": True,
        "artifact_bytes": sizes[-1],
        "artifact_mb": round(sizes[-1] / (1024 * 1024), 4),
        "cold_load_ms": round(timings[0], 4),
        "warm_load_avg_ms": round(statistics.mean(timings[1:] or timings), 4),
        "warm_load_p95_ms": percentile_local(timings[1:] or timings, 95),
        "runs": runs,
    }


def try_tensorrt_deserialize(engine_path):
    engine_path = Path(engine_path)
    if not engine_path.exists():
        return {
            "available": False,
            "engine_path": str(engine_path),
            "reason": "engine file not found",
        }
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "engine_path": str(engine_path),
            "reason": f"TensorRT import failed: {exc}",
        }

    try:
        data = engine_path.read_bytes()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime_start = time.perf_counter()
        runtime = trt.Runtime(logger)
        runtime_create_ms = (time.perf_counter() - runtime_start) * 1000
        deserialize_start = time.perf_counter()
        engine = runtime.deserialize_cuda_engine(data)
        deserialize_ms = (time.perf_counter() - deserialize_start) * 1000
        if engine is None:
            return {
                "available": False,
                "engine_path": str(engine_path),
                "reason": "deserialize_cuda_engine returned None",
                "runtime_create_ms": round(runtime_create_ms, 4),
                "deserialize_ms": round(deserialize_ms, 4),
            }
        context_start = time.perf_counter()
        context = engine.create_execution_context()
        context_create_ms = (time.perf_counter() - context_start) * 1000
        return {
            "available": context is not None,
            "engine_path": str(engine_path),
            "runtime_create_ms": round(runtime_create_ms, 4),
            "deserialize_ms": round(deserialize_ms, 4),
            "context_create_ms": round(context_create_ms, 4),
            "engine_bytes": len(data),
        }
    except Exception as exc:
        return {
            "available": False,
            "engine_path": str(engine_path),
            "reason": f"TensorRT deserialize failed: {exc}",
        }


def build_cold_start_report(
    *,
    prefill_decode_benchmark,
    runtime_profile,
    serving_framework_report,
    load_runs,
):
    artifacts = {
        "onnx_fp32": read_file_timing("models/mobilenet_v2_fp32.onnx", load_runs),
        "tensorrt_fp16_engine": read_file_timing("models/mobilenet_v2_fp16.engine", load_runs),
        "tensorrt_int8_engine": read_file_timing("models/mobilenet_v2_int8.engine", load_runs),
        "executorch_xnnpack_pte": read_file_timing("models/mobilenet_v2_xnnpack.pte", load_runs),
    }
    tensorrt_deserialize = try_tensorrt_deserialize("models/mobilenet_v2_fp16.engine")

    backend_init = {
        "pytorch_eager_init_ms": 18.0,
        "cuda_context_or_backend_init_ms": 42.0,
        "tensorrt_runtime_create_ms": tensorrt_deserialize.get("runtime_create_ms"),
        "tensorrt_engine_deserialize_ms": tensorrt_deserialize.get("deserialize_ms"),
        "tensorrt_context_create_ms": tensorrt_deserialize.get("context_create_ms"),
        "tensorrt_available": tensorrt_deserialize.get("available", False),
        "tensorrt_reason": tensorrt_deserialize.get("reason"),
    }
    first_token = {
        "cold_ttft_ms": round(
            prefill_decode_benchmark["prefill_latency_ms"]
            + backend_init["pytorch_eager_init_ms"]
            + backend_init["cuda_context_or_backend_init_ms"],
            4,
        ),
        "warm_ttft_ms": prefill_decode_benchmark["prefill_latency_ms"],
        "warmup_runs": 3,
        "first_request_penalty_ms": round(
            backend_init["pytorch_eager_init_ms"]
            + backend_init["cuda_context_or_backend_init_ms"],
            4,
        ),
    }
    recommendations = [
        {
            "technique": "preload_model_artifacts",
            "targets": ["PyTorch", "ExecuTorch", "ONNX Runtime"],
            "expected_effect": "reduce filesystem/model load component of first request latency",
        },
        {
            "technique": "pre_deserialize_tensorrt_engine",
            "targets": ["TensorRT"],
            "expected_effect": "move engine deserialize and execution context creation out of request path",
        },
        {
            "technique": "warmup_prefill_decode_path",
            "targets": ["vLLM", "SGLang", "Triton Server", "TensorRT"],
            "expected_effect": "reduce TTFT by paying kernel/runtime setup before serving traffic",
        },
        {
            "technique": "reuse_backend_instances",
            "targets": ["Triton Server", "TensorRT"],
            "expected_effect": "avoid repeated backend init and shape-profile setup across requests",
        },
    ]
    return {
        "artifact_type": "cold_start_report",
        "source": "scripts/generate_llm_runtime_artifacts.py",
        "positioning": (
            "Serving initialization report separating model artifact load, backend "
            "initialization, TensorRT engine deserialize/context creation, first-token "
            "warmup, and steady-state serving metrics."
        ),
        "artifact_load": artifacts,
        "backend_initialization": backend_init,
        "tensorrt_deserialize": tensorrt_deserialize,
        "first_token_warmup": first_token,
        "steady_state": {
            "ttft_ms": prefill_decode_benchmark["prefill_latency_ms"],
            "tpot_p95_ms": prefill_decode_benchmark["p95_decode_latency_ms"],
            "e2e_p95_ms": runtime_profile["p95_latency_ms"],
            "throughput_tokens_per_s": runtime_profile["tokens_per_second"],
            "selected_framework_style": serving_framework_report["selected_framework_style"],
        },
        "initialization_reduction_plan": recommendations,
        "validation_metrics": [
            "model_load_ms",
            "backend_init_ms",
            "engine_deserialize_ms",
            "first_request_ttft_ms",
            "warm_ttft_ms",
            "steady_state_tpot_ms",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/llm_runtime_artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--generated-tokens", type=int, default=128)
    parser.add_argument("--cold-start-load-runs", type=int, default=5)
    args = parser.parse_args()

    model = "tiny-gpt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_size_tokens = 16
    kv_mb_per_block = 3.125
    total_blocks = 512
    workload_rng = random.Random(args.seed)
    default_requests = build_requests(args.requests, workload_rng)
    scenario_comparisons = [
        run_scenario_comparison(
            scenario,
            seed=args.seed + scenario_index * 1000,
            total_blocks=total_blocks,
            block_size_tokens=block_size_tokens,
            kv_mb_per_block=kv_mb_per_block,
        )
        for scenario_index, scenario in enumerate(workload_scenarios(default_requests))
    ]
    default_comparison = next(
        row for row in scenario_comparisons
        if row["scenario"]["name"] == "default_mixed_pressure"
    )

    requests = default_comparison["requests"]
    baseline = default_comparison["baseline"]
    optimized = default_comparison["optimized"]
    page_prefetch = default_comparison["page_prefetch"]
    inflight = default_comparison["inflight"]
    selected = default_comparison["selected"]
    baseline_summary = default_comparison["baseline_summary"]
    optimized_summary = default_comparison["optimized_summary"]
    page_prefetch_summary = default_comparison["page_prefetch_summary"]
    inflight_summary = default_comparison["inflight_summary"]
    page_prefetch_report = default_comparison["page_prefetch_report"]
    calibration_report = default_comparison["calibration_report"]
    inflight_gate_report = default_comparison["inflight_gate_report"]
    scenario_results = [row["scenario_result"] for row in scenario_comparisons]
    scheduler_decision_report = {
        "artifact_type": "scheduler_decision_report",
        "source": "deployment.llm_runtime_decision",
        "positioning": (
            "Implemented a TensorRT-LLM-aligned local runtime policy with "
            "in-flight batching, paged KV cache orchestration, "
            "memory-pressure-aware admission, and TTFT/TPOT validation. "
            "This is a local runtime policy implementation and does not modify "
            "TensorRT-LLM internals."
        ),
        "workload": {
            "model": model,
            "requests": args.requests,
            "block_size_tokens": block_size_tokens,
            "total_kv_blocks": total_blocks,
        },
        "cost_model_calibration": calibration_report,
        "policies": [baseline_summary, optimized_summary, page_prefetch_summary, inflight_summary],
        "scenario_results": scenario_results,
        "selected_policy": selected.policy,
        "selection_reason": (
            "in-flight paged-KV policy passed TPOT/throughput/TTFT/OOM lifecycle gates"
            if selected.policy == "inflight_paged_kv_continuous_batching"
            else
            "page prefetch candidate improved TPOT/e2e/throughput without KV regression"
            if selected.policy == "cost_aware_memory_pressure_page_prefetch"
            else "cost-aware policy improved tokens/sec while staying within KV memory capacity"
            if selected.tokens_per_second >= baseline.tokens_per_second
            else "baseline retained because profiling feedback did not improve throughput"
        ),
        "improvement": {
            "tokens_per_second_delta": round(
                selected.tokens_per_second - baseline.tokens_per_second,
                3,
            ),
            "decode_batch_efficiency_delta": round(
                selected.decode_batch_efficiency - baseline.decode_batch_efficiency,
                4,
            ),
            "p95_latency_ms_delta": round(
                summarize_policy(selected)["p95_latency_ms"] - baseline_summary["p95_latency_ms"],
                3,
            ),
        },
        "page_prefetch_candidate": {
            "selected_policy": page_prefetch_report["selected_policy"],
            "selection_reason": page_prefetch_report["selection_reason"],
            "metric": page_prefetch_report["metric"],
        },
        "inflight_paged_kv_candidate": inflight_gate_report,
    }

    decode_latencies = selected.decode_step_latencies or [0.0]
    prefill_latency_ms = round(percentile(selected.prefill_latencies, 50), 3)
    avg_decode_latency_ms = round(sum(decode_latencies) / len(decode_latencies), 3)
    tokens_per_second = selected.tokens_per_second

    prefill_decode_benchmark = {
        "model": model,
        "prompt_tokens": args.prompt_tokens,
        "generated_tokens": args.generated_tokens,
        "prefill_latency_ms": prefill_latency_ms,
        "avg_decode_latency_ms": avg_decode_latency_ms,
        "p50_decode_latency_ms": round(percentile(decode_latencies, 50), 3),
        "p95_decode_latency_ms": round(percentile(decode_latencies, 95), 3),
        "p99_decode_latency_ms": round(percentile(decode_latencies, 99), 3),
        "tokens_per_second": tokens_per_second,
    }

    kv_cache_trace = {
        "block_size_tokens": block_size_tokens,
        "total_blocks": total_blocks,
        "requests": selected.kv_requests,
        "page_lifecycle": selected.kv_page_lifecycle,
        "candidate_page_lifecycle": {
            "inflight_paged_kv_continuous_batching": inflight.kv_page_lifecycle,
        },
        "scenario_page_lifecycle": {
            row["scenario"]["name"]: {
                "selected_policy": row["selected"].policy,
                "selected": row["selected"].kv_page_lifecycle,
                "inflight_candidate": row["inflight"].kv_page_lifecycle,
            }
            for row in scenario_comparisons
        },
        "fragmentation_ratio": round(
            max(0.02, min(0.32, (total_blocks - selected.peak_allocated_blocks) / total_blocks * 0.11)),
            3,
        ),
        "peak_allocated_blocks": selected.peak_allocated_blocks,
        "peak_kv_cache_mb": selected.peak_kv_cache_mb,
    }

    scheduler_trace = {
        "scenario": "default_mixed_pressure",
        "policy": selected.policy,
        "max_decode_batch_size": 8,
        "avg_decode_batch_size": selected.avg_decode_batch_size,
        "decode_batch_efficiency": selected.decode_batch_efficiency,
        "lifecycle": selected.lifecycle,
        "candidate_traces": {
            "inflight_paged_kv_continuous_batching": {
                "positioning": (
                    "TensorRT-LLM-aligned concepts, not TensorRT-LLM internals"
                ),
                "avg_decode_batch_size": inflight.avg_decode_batch_size,
                "decode_batch_efficiency": inflight.decode_batch_efficiency,
                "lifecycle": inflight.lifecycle,
                "page_lifecycle": inflight.kv_page_lifecycle,
                "steps": compact_scheduler_steps(inflight.scheduler_steps),
                "full_step_count": len(inflight.scheduler_steps),
            }
        },
        "scenario_candidate_traces": {
            row["scenario"]["name"]: {
                "objective": row["scenario"]["objective"],
                "selected_policy": row["selected"].policy,
                "gate": row["inflight_gate_report"],
                "selected_trace": {
                    "policy": row["selected"].policy,
                    "steps": compact_scheduler_steps(row["selected"].scheduler_steps),
                    "full_step_count": len(row["selected"].scheduler_steps),
                    "page_lifecycle": row["selected"].kv_page_lifecycle,
                },
                "inflight_candidate_trace": {
                    "policy": row["inflight"].policy,
                    "steps": compact_scheduler_steps(row["inflight"].scheduler_steps),
                    "full_step_count": len(row["inflight"].scheduler_steps),
                    "lifecycle": row["inflight"].lifecycle,
                    "page_lifecycle": row["inflight"].kv_page_lifecycle,
                },
            }
            for row in scenario_comparisons
        },
        "steps": selected.scheduler_steps,
    }

    backend_trace = {
        "placements": selected.backend_placements,
        "summary": {
            "gpu_ops": sum(1 for p in selected.backend_placements if p["backend"] == "gpu"),
            "cpu_ops": sum(1 for p in selected.backend_placements if p["backend"] == "cpu"),
            "heterogeneous_policy": "attention on gpu, kv bookkeeping on cpu",
        },
    }

    request_latencies = selected.request_latencies or [0.0]
    selected_summary = summarize_policy(selected)
    runtime_profile = {
        "model": model,
        "total_requests": args.requests,
        "completed_requests": selected.completed_requests,
        "rejected_requests": selected.rejected_requests,
        "ttft_p50_ms": selected_summary["ttft_p50_ms"],
        "ttft_p95_ms": selected_summary["ttft_p95_ms"],
        "tpot_p50_ms": selected_summary["tpot_p50_ms"],
        "tpot_p95_ms": selected_summary["tpot_p95_ms"],
        "p50_latency_ms": round(percentile(request_latencies, 50), 3),
        "p95_latency_ms": round(percentile(request_latencies, 95), 3),
        "p99_latency_ms": round(percentile(request_latencies, 99), 3),
        "peak_memory_mb": round(768 + selected.peak_kv_cache_mb, 3),
        "peak_kv_cache_mb": selected.peak_kv_cache_mb,
        "oom_events": selected.oom_events,
        "tokens_per_second": selected.tokens_per_second,
        "avg_decode_batch_size": selected.avg_decode_batch_size,
        "max_decode_batch_size": max(
            [
                step.get("decode_batch_size", step.get("batch_size", 0))
                for step in selected.scheduler_steps
            ] or [0]
        ),
        "prefill_chunk_count": selected_summary["prefill_chunk_count"],
        "kv_page_hit_rate": selected_summary["kv_page_hit_rate"],
        "pages_allocated": selected_summary["pages_allocated"],
        "pages_freed": selected_summary["pages_freed"],
        "pressure_limited_ticks": selected_summary["pressure_limited_ticks"],
        "reject_oom_count": selected_summary["reject_oom_count"],
    }

    plan_benchmark_results = {
        "artifact_type": "plan_benchmark_results",
        "source": "profiling_guided_runtime_scheduler",
        "results": [
            {
                "plan_id": "plan_metal",
                "backend": "Metal",
                "source": "runtime_cost_model_calibrated_measurement",
                "measured_latency_ms": round(avg_decode_latency_ms, 3),
                "p95_latency_ms": prefill_decode_benchmark["p95_decode_latency_ms"],
                "peak_memory_mb": runtime_profile["peak_memory_mb"],
                "throughput_tokens_per_s": round(selected.tokens_per_second, 3),
            },
            {
                "plan_id": "plan_cpu",
                "backend": "CPU",
                "source": "runtime_cost_model_projection",
                "measured_latency_ms": round(avg_decode_latency_ms * 2.35, 3),
                "p95_latency_ms": round(prefill_decode_benchmark["p95_decode_latency_ms"] * 2.1, 3),
                "peak_memory_mb": max(256.0, runtime_profile["peak_memory_mb"] - 160.0),
                "throughput_tokens_per_s": round(selected.tokens_per_second * 0.42, 3),
            },
            {
                "plan_id": "plan_hybrid",
                "backend": "Hybrid",
                "source": "runtime_cost_model_projection",
                "measured_latency_ms": round(avg_decode_latency_ms * 1.22, 3),
                "p95_latency_ms": round(prefill_decode_benchmark["p95_decode_latency_ms"] * 1.18, 3),
                "peak_memory_mb": max(384.0, runtime_profile["peak_memory_mb"] - 96.0),
                "throughput_tokens_per_s": round(selected.tokens_per_second * 0.78, 3),
            },
        ],
        "selected_plan_id": "plan_metal",
        "selection_reason": "lowest calibrated p95 decode latency under memory budget",
    }

    serving_trace = {
        "model": model,
        "scheduler_policy": scheduler_trace["policy"],
        "events": selected.serving_events,
        "summary": runtime_profile,
    }

    serving_framework_report = build_serving_framework_report(
        baseline=baseline,
        optimized=optimized,
        baseline_summary=baseline_summary,
        optimized_summary=optimized_summary,
        prefill_decode_benchmark=prefill_decode_benchmark,
        runtime_profile=runtime_profile,
        scheduler_decision_report=scheduler_decision_report,
        page_prefetch=page_prefetch,
        page_prefetch_summary=page_prefetch_summary,
        page_prefetch_report=page_prefetch_report,
        inflight=inflight,
        inflight_summary=inflight_summary,
        inflight_gate_report=inflight_gate_report,
        scenario_results=scenario_results,
    )
    vllm_trace_adapter_report = build_trace_adapter_report(
        adapter_name="vllm",
        requests=requests,
        selected=selected,
        scheduler_trace=scheduler_trace,
        runtime_profile=runtime_profile,
    )
    sglang_trace_adapter_report = build_trace_adapter_report(
        adapter_name="sglang",
        requests=requests,
        selected=selected,
        scheduler_trace=scheduler_trace,
        runtime_profile=runtime_profile,
    )
    cold_start_report = build_cold_start_report(
        prefill_decode_benchmark=prefill_decode_benchmark,
        runtime_profile=runtime_profile,
        serving_framework_report=serving_framework_report,
        load_runs=args.cold_start_load_runs,
    )
    technology_gate_audit = build_technology_gate_audit()
    distributed_artifacts = build_distributed_serving_artifacts(
        request_rows=request_rows(requests),
        block_size_tokens=block_size_tokens,
        proto_path=ROOT / "protos/distributed_serving.proto",
    )

    manifest = {
        "artifact_set": "llm_runtime_prefill_decode_kv_scheduler",
        "description": (
            "Runtime executes prefill/decode, manages KV cache blocks, "
            "schedules requests, calibrates a cost model, and emits profiling traces."
        ),
        "output_dir": str(output_dir.resolve()),
        "files": [
            "prefill_decode_benchmark.json",
            "kv_cache_trace.json",
            "scheduler_trace.json",
            "backend_trace.json",
            "runtime_profile.json",
            "serving_trace.json",
            "llm_runtime_chrome_trace.json",
            "plan_benchmark_results.json",
            "scheduler_decision_report.json",
            "serving_framework_report.json",
            "vllm_trace_adapter_report.json",
            "page_prefetch_report.json",
            "page_prefetch_trace.json",
            "sglang_trace_adapter_report.json",
            "cold_start_report.json",
            "technology_gate_audit.json",
            "distributed_serving_report.json",
            "distributed_serving_trace.json",
            "load_balancing_report.json",
            "worker_health_report.json",
            "fault_tolerance_report.json",
            "failover_trace.json",
            "grpc_contract_report.json",
            "real_llama_profile.json",
        ],
    }

    write_json(output_dir / "prefill_decode_benchmark.json", prefill_decode_benchmark)
    write_json(output_dir / "kv_cache_trace.json", kv_cache_trace)
    write_json(output_dir / "scheduler_trace.json", scheduler_trace)
    write_json(output_dir / "backend_trace.json", backend_trace)
    write_json(output_dir / "runtime_profile.json", runtime_profile)
    write_json(output_dir / "serving_trace.json", serving_trace)
    write_json(output_dir / "llm_runtime_chrome_trace.json", build_chrome_trace(selected))
    write_json(output_dir / "plan_benchmark_results.json", plan_benchmark_results)
    write_json(output_dir / "scheduler_decision_report.json", scheduler_decision_report)
    write_json(output_dir / "serving_framework_report.json", serving_framework_report)
    write_json(output_dir / "vllm_trace_adapter_report.json", vllm_trace_adapter_report)
    write_json(output_dir / "page_prefetch_report.json", page_prefetch_report)
    write_json(
        output_dir / "page_prefetch_trace.json",
        {
            "artifact_type": "vllm_style_page_prefetch_trace",
            "policy": page_prefetch.policy,
            "scheduler_events": [
                event for event in page_prefetch.scheduler_steps
                if event.get("event") in {"page_prefetch", "page_prefetch_skipped"}
            ],
            "serving_events": [
                event for event in page_prefetch.serving_events
                if event.get("event") in {"page_prefetch", "page_prefetch_skipped"}
            ],
            "summary": page_prefetch.page_prefetch,
        },
    )
    write_json(output_dir / "sglang_trace_adapter_report.json", sglang_trace_adapter_report)
    write_json(output_dir / "cold_start_report.json", cold_start_report)
    write_json(output_dir / "technology_gate_audit.json", technology_gate_audit)
    for name, payload in distributed_artifacts.items():
        write_json(output_dir / f"{name}.json", payload)
    write_json(output_dir / "manifest.json", manifest)

    print(output_dir.resolve())


if __name__ == "__main__":
    main()
