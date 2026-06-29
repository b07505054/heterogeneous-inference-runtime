"""Trace generation for DistributedRuntimePlan.

build_distributed_runtime_trace converts a DistributedRuntimePlan into an
ordered event dict suitable for offline analysis and artifact export.

Events are sequenced along a simulated timeline beginning at t=0:
  prefill_queue_wait → prefill_compute → kv_transfer → handoff
  → decode_queue_wait → decode_compute → decision_summary

Timestamps reflect compiler cost estimates and queue model. All timings are
static plan estimates; they are not live execution traces.

Queue wait events are always included in the event list. When a queue wait
duration is 0.0, the event is present with duration_ms=0.0 and the next event
starts at the same timestamp.

This is a read-only operation. The input plan is never mutated.

Truth boundary on all produced artifacts:
  "pd_split_schedule_static_plan_not_live_cluster_execution"
"""

from __future__ import annotations

from deployment.distributed_runtime_plan import DistributedRuntimePlan


def build_distributed_runtime_trace(plan: DistributedRuntimePlan) -> dict:
    """Build an ordered trace dict from a DistributedRuntimePlan.

    Returns a dict with keys:
      artifact_type, model_name, target_profile_id, truth_boundary,
      total_compiler_cost_ms, events, cost_summary.

    Does not mutate plan.
    """
    q_prefill_ms = plan.prefill.queue_wait_ms
    prefill_service_ms = plan.prefill.service_ms
    kv_ms = plan.kv_transfer.transfer_cost_ms
    handoff_ms = plan.decision_comparison.pd_split.handoff_overhead_ms
    q_decode_ms = plan.decode.queue_wait_ms
    decode_service_ms = plan.decode.service_ms

    t0 = 0.0
    t_prefill_compute = t0 + q_prefill_ms
    t_kv = t_prefill_compute + prefill_service_ms
    t_handoff = t_kv + kv_ms
    t_decode_queue = t_handoff + handoff_ms
    t_decode_compute = t_decode_queue + q_decode_ms
    t_end = t_decode_compute + decode_service_ms

    events = [
        {
            "name": "prefill_queue_wait",
            "category": "prefill",
            "start_ms": t0,
            "duration_ms": q_prefill_ms,
            "worker_id": plan.prefill.worker_id,
            "backend": None,
            "truth_boundary": plan.prefill.truth_boundary,
        },
        {
            "name": "prefill_compute",
            "category": "prefill",
            "start_ms": t_prefill_compute,
            "duration_ms": prefill_service_ms,
            "worker_id": plan.prefill.worker_id,
            "backend": plan.prefill.backend,
            "truth_boundary": plan.prefill.truth_boundary,
        },
        {
            "name": "kv_transfer",
            "category": "kv_transfer",
            "start_ms": t_kv,
            "duration_ms": kv_ms,
            "worker_id": None,
            "backend": None,
            "truth_boundary": plan.kv_transfer.truth_boundary,
        },
        {
            "name": "handoff",
            "category": "handoff",
            "start_ms": t_handoff,
            "duration_ms": handoff_ms,
            "worker_id": None,
            "backend": None,
            "truth_boundary": plan.truth_boundary,
        },
        {
            "name": "decode_queue_wait",
            "category": "decode",
            "start_ms": t_decode_queue,
            "duration_ms": q_decode_ms,
            "worker_id": plan.decode.worker_id,
            "backend": None,
            "truth_boundary": plan.decode.truth_boundary,
        },
        {
            "name": "decode_compute",
            "category": "decode",
            "start_ms": t_decode_compute,
            "duration_ms": decode_service_ms,
            "worker_id": plan.decode.worker_id,
            "backend": plan.decode.backend,
            "truth_boundary": plan.decode.truth_boundary,
        },
        {
            "name": "decision_summary",
            "category": "decision",
            "start_ms": t_end,
            "duration_ms": 0.0,
            "worker_id": None,
            "backend": None,
            "truth_boundary": plan.decision_comparison.truth_boundary,
        },
    ]

    cmp = plan.decision_comparison
    cost_summary = {
        "colocated": {
            "total_ms": cmp.colocated.total_ms,
            "ttft_ms": cmp.colocated.ttft_ms,
            "tpot_ms": cmp.colocated.tpot_ms,
            "truth_boundary": cmp.colocated.truth_boundary,
        },
        "pd_split": {
            "total_ms": cmp.pd_split.total_ms,
            "ttft_ms": cmp.pd_split.ttft_ms,
            "tpot_ms": cmp.pd_split.tpot_ms,
            "truth_boundary": cmp.pd_split.truth_boundary,
        },
        "selected_policy": cmp.selected_policy,
        "decision_reason": cmp.decision_reason,
        "goodput_proxy": cmp.goodput_proxy,
        "truth_boundary": cmp.truth_boundary,
    }

    return {
        "artifact_type": "distributed_runtime_trace",
        "model_name": plan.model_name,
        "target_profile_id": plan.target_profile_id,
        "truth_boundary": plan.truth_boundary,
        "total_compiler_cost_ms": plan.total_compiler_cost_ms,
        "events": events,
        "cost_summary": cost_summary,
    }
