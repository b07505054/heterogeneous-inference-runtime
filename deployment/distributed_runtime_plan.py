"""Distributed runtime planning for prefill/decode disaggregation.

PDSplitPlanner converts a list[RuntimeExecutionPlan] (one prefill + one decode)
into a DistributedRuntimePlan that models both colocated and PD-split execution,
compares them against SLO targets, and selects a policy.

This is a planning/simulation object only. It does not dispatch to hardware,
communicate over RPC, or allocate real KV memory.

Truth boundaries used throughout:
  "pd_split_schedule_static_plan_not_live_cluster_execution"
  "kv_transfer_cost_model_not_measured_network"
  "goodput_proxy_not_cluster_throughput"
  "prefill_decode_cost_model_based_on_compiler_estimates"
  "queue_wait_derived_from_worker_availability_not_live_scheduler"
  "service_time_model_static_estimate_not_live_measurement"
"""

from __future__ import annotations

from dataclasses import dataclass

from deployment.runtime_execution_plan import RuntimeExecutionPlan

_TB_PLAN = "pd_split_schedule_static_plan_not_live_cluster_execution"
_TB_KV = "kv_transfer_cost_model_not_measured_network"
_TB_GOODPUT = "goodput_proxy_not_cluster_throughput"
_TB_COST = "prefill_decode_cost_model_based_on_compiler_estimates"
_TB_QUEUE = "queue_wait_derived_from_worker_availability_not_live_scheduler"
_TB_SERVICE = "service_time_model_static_estimate_not_live_measurement"

# Sources whose service time already embeds interference/batching overhead.
# Adding service adjustments on top would double-count that cost.
_MEASURED_SOURCES: frozenset[str] = frozenset({
    "measured_continuous_batching_curve",
    "production_trace_service_time",
})


# ---------------------------------------------------------------------------
# Queue and service time models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueueState:
    """Worker availability at request arrival time.

    Queue waits are derived from when a worker can next accept work relative to
    when the request arrives — not passed in as fixed constants.

    colocated_worker_available_at_ms models the case where a colocated worker
    finishes a full prefill+decode cycle before accepting the next request, making
    it available later than a pd_split prefill worker (which is free after prefill
    only). When None, falls back to prefill_worker_available_at_ms.

    truth_boundary = "queue_wait_derived_from_worker_availability_not_live_scheduler"
    """

    arrival_time_ms: float
    prefill_worker_available_at_ms: float
    decode_worker_available_at_ms: float
    colocated_worker_available_at_ms: float | None = None
    truth_boundary: str = _TB_QUEUE


@dataclass(frozen=True)
class ServiceTimeModel:
    """Service time adjustment and source tracking for the planner.

    Adjustments represent unmodeled overhead layered on top of the compiler cost.
    When source is "measured_continuous_batching_curve" or
    "production_trace_service_time", the measured service time already embeds
    batching interference; adding non-zero adjustments would double-count it.

    Allowed sources:
      "compiler_estimate"
      "isolated_single_request_measurement"
      "measured_continuous_batching_curve"
      "production_trace_service_time"

    truth_boundary = "service_time_model_static_estimate_not_live_measurement"
    """

    source: str
    prefill_service_adjustment_ms: float = 0.0
    decode_service_adjustment_ms: float = 0.0
    truth_boundary: str = _TB_SERVICE

    def __post_init__(self) -> None:
        if self.source in _MEASURED_SOURCES:
            if self.prefill_service_adjustment_ms != 0.0 or self.decode_service_adjustment_ms != 0.0:
                raise ValueError(
                    f"ServiceTimeModel.source='{self.source}' already embeds interference in "
                    "service time; non-zero service adjustments would double-count it. "
                    "Set prefill_service_adjustment_ms=0.0 and decode_service_adjustment_ms=0.0."
                )


# ---------------------------------------------------------------------------
# Cost breakdown records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColocatedCostBreakdown:
    """Estimated cost for colocated prefill+decode on one worker.

    queue_wait_ms is derived from worker availability at arrival time.
    prefill_service_ms and decode_service_ms include any service_time_model adjustments.
    All values are model-based estimates, not measured latency.

    total_ms = queue_wait_ms + prefill_service_ms + decode_service_ms
    ttft_ms  = queue_wait_ms + prefill_service_ms
    tpot_ms  = decode_service_ms

    truth_boundary = "prefill_decode_cost_model_based_on_compiler_estimates"
    """

    queue_wait_ms: float
    prefill_service_ms: float
    decode_service_ms: float
    total_ms: float
    ttft_ms: float
    tpot_ms: float
    service_time_model_source: str
    truth_boundary: str


@dataclass(frozen=True)
class PDSplitCostBreakdown:
    """Estimated cost for disaggregated prefill/decode on separate workers.

    queue_wait_prefill_ms and queue_wait_decode_ms are derived from worker
    availability, not fixed caller constants.
    KV transfer cost is a bandwidth model, not a measured network measurement.

    total_ms = queue_wait_prefill_ms + prefill_service_ms + kv_transfer_ms
               + handoff_overhead_ms + queue_wait_decode_ms + decode_service_ms
    ttft_ms  = queue_wait_prefill_ms + prefill_service_ms + kv_transfer_ms
               + handoff_overhead_ms + queue_wait_decode_ms
    tpot_ms  = decode_service_ms

    truth_boundary = "kv_transfer_cost_model_not_measured_network"
    """

    queue_wait_prefill_ms: float
    prefill_service_ms: float
    kv_transfer_ms: float
    kv_transfer_bytes: int
    handoff_overhead_ms: float
    queue_wait_decode_ms: float
    decode_service_ms: float
    total_ms: float
    ttft_ms: float
    tpot_ms: float
    service_time_model_source: str
    truth_boundary: str


@dataclass(frozen=True)
class PDSplitDecisionComparison:
    """Side-by-side comparison of colocated vs PD-split with a selected policy.

    goodput_proxy is 1 / total_ms for the selected policy. It is a proxy only;
    it does not model batch size, cluster utilization, or real scheduling.
    truth_boundary = "goodput_proxy_not_cluster_throughput"
    """

    colocated: ColocatedCostBreakdown
    pd_split: PDSplitCostBreakdown
    selected_policy: str       # "colocated" | "pd_split"
    decision_reason: str
    slo_ttft_ms: float
    slo_tpot_ms: float
    goodput_proxy: float
    truth_boundary: str


# ---------------------------------------------------------------------------
# Stage records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefillStage:
    """Static prefill stage descriptor for a disaggregated plan.

    compiler_cost_ms is the raw compiler estimate; service_ms adds any
    service_time_model adjustment on top of it.
    queue_wait_ms is derived from worker availability at arrival time.
    """

    worker_id: str
    backend: str
    compiler_cost_ms: float
    service_ms: float
    queue_wait_ms: float
    truth_boundary: str


@dataclass(frozen=True)
class KVTransferStage:
    """KV transfer stage between prefill and decode workers.

    transfer_cost_ms = (transfer_bytes / 1024^2) / bandwidth_mb_per_ms.
    This is a bandwidth-model estimate, not a measured network transfer.
    truth_boundary = "kv_transfer_cost_model_not_measured_network"
    """

    transfer_bytes: int
    transfer_cost_ms: float
    bandwidth_mb_per_ms: float
    truth_boundary: str


@dataclass(frozen=True)
class DecodeStage:
    """Static decode stage descriptor for a disaggregated plan.

    compiler_cost_ms is the raw compiler estimate; service_ms adds any
    service_time_model adjustment on top of it.
    queue_wait_ms is derived from worker availability at decode_ready time.
    """

    worker_id: str
    backend: str
    compiler_cost_ms: float
    service_ms: float
    queue_wait_ms: float
    truth_boundary: str


@dataclass(frozen=True)
class OptionalReplayStage:
    """CUDA graph replay eligibility for the decode stage."""

    eligible: bool
    bucket: str
    truth_boundary: str


# ---------------------------------------------------------------------------
# Top-level plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DistributedRuntimePlan:
    """Immutable distributed runtime plan for prefill/decode disaggregation.

    total_compiler_cost_ms = prefill_service_ms + kv_transfer_ms
                             + handoff_overhead_ms + decode_service_ms.
    Queue waits are not included in total_compiler_cost_ms; they appear in
    decision_comparison.pd_split.total_ms and decision_comparison.colocated.total_ms.

    truth_boundary = "pd_split_schedule_static_plan_not_live_cluster_execution"
    """

    model_name: str
    target_profile_id: str
    prefill: PrefillStage
    kv_transfer: KVTransferStage
    decode: DecodeStage
    replay: OptionalReplayStage
    decision_comparison: PDSplitDecisionComparison
    total_compiler_cost_ms: float
    truth_boundary: str


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class PDSplitPlanner:
    """Converts a prefill + decode RuntimeExecutionPlan pair into a DistributedRuntimePlan.

    Compares colocated vs PD-split execution and selects a policy based on
    total cost and SLO constraints for TTFT and TPOT.

    Queue waits are derived from QueueState (worker availability at arrival time),
    not passed as fixed constants. Service time adjustments are validated against
    the source to prevent double-counting interference already embedded in
    measured service times.

    Does not mutate the input plans. All cost values are compiler estimates.
    """

    @staticmethod
    def plan(
        plans: list[RuntimeExecutionPlan],
        *,
        slo_ttft_ms: float = 200.0,
        slo_tpot_ms: float = 20.0,
        kv_bandwidth_mb_per_ms: float = 24.0,
        handoff_overhead_ms: float = 0.2,
        queue_state: QueueState | None = None,
        service_time_model: ServiceTimeModel | None = None,
    ) -> DistributedRuntimePlan:
        """Build a DistributedRuntimePlan from a prefill + decode plan pair.

        Requires exactly one plan with function_name == "prefill" and one with
        function_name == "decode". Raises ValueError otherwise.
        """
        if queue_state is None:
            queue_state = QueueState(
                arrival_time_ms=0.0,
                prefill_worker_available_at_ms=0.0,
                decode_worker_available_at_ms=0.0,
            )
        if service_time_model is None:
            service_time_model = ServiceTimeModel(source="compiler_estimate")

        prefill_plan = _find_plan(plans, "prefill")
        decode_plan = _find_plan(plans, "decode")

        prefill_compiler_cost_ms = prefill_plan.scheduling_policy.compiler_cost_ms
        decode_compiler_cost_ms = decode_plan.scheduling_policy.compiler_cost_ms

        prefill_service_ms = (
            prefill_compiler_cost_ms + service_time_model.prefill_service_adjustment_ms
        )
        decode_service_ms = (
            decode_compiler_cost_ms + service_time_model.decode_service_adjustment_ms
        )

        # KV transfer: bytes from prefill memory policy; cost from bandwidth model.
        kv_mb = prefill_plan.memory_policy.kv_byte_estimate_mb
        kv_bytes = int(kv_mb * 1024 * 1024)
        kv_transfer_ms = (
            kv_mb / kv_bandwidth_mb_per_ms if kv_bandwidth_mb_per_ms > 0.0 else 0.0
        )

        # ── Queue wait derivation ─────────────────────────────────────────────
        arrival = queue_state.arrival_time_ms

        # Colocated: use colocated_worker_available_at_ms when set; otherwise fall
        # back to prefill_worker_available_at_ms. A colocated worker that just
        # finished a full prefill+decode cycle is available later than a pd_split
        # prefill-only worker, which is the primary reason pd_split can be cheaper.
        colocated_avail = (
            queue_state.colocated_worker_available_at_ms
            if queue_state.colocated_worker_available_at_ms is not None
            else queue_state.prefill_worker_available_at_ms
        )
        colocated_start_ms = max(arrival, colocated_avail)
        queue_wait_colocated_ms = colocated_start_ms - arrival

        # PD-split prefill queue wait.
        prefill_start_ms = max(arrival, queue_state.prefill_worker_available_at_ms)
        queue_wait_prefill_ms = prefill_start_ms - arrival

        # PD-split decode queue wait: decode may only start after prefill + KV + handoff.
        decode_ready_ms = (
            prefill_start_ms + prefill_service_ms + kv_transfer_ms + handoff_overhead_ms
        )
        decode_start_ms = max(decode_ready_ms, queue_state.decode_worker_available_at_ms)
        queue_wait_decode_ms = decode_start_ms - decode_ready_ms

        # ── Colocated cost ───────────────────────────────────────────────────
        col_total = queue_wait_colocated_ms + prefill_service_ms + decode_service_ms
        col_ttft = queue_wait_colocated_ms + prefill_service_ms
        col_tpot = decode_service_ms

        colocated = ColocatedCostBreakdown(
            queue_wait_ms=queue_wait_colocated_ms,
            prefill_service_ms=prefill_service_ms,
            decode_service_ms=decode_service_ms,
            total_ms=col_total,
            ttft_ms=col_ttft,
            tpot_ms=col_tpot,
            service_time_model_source=service_time_model.source,
            truth_boundary=_TB_COST,
        )

        # ── PD-split cost ────────────────────────────────────────────────────
        pd_total = (
            queue_wait_prefill_ms
            + prefill_service_ms
            + kv_transfer_ms
            + handoff_overhead_ms
            + queue_wait_decode_ms
            + decode_service_ms
        )
        pd_ttft = (
            queue_wait_prefill_ms
            + prefill_service_ms
            + kv_transfer_ms
            + handoff_overhead_ms
            + queue_wait_decode_ms
        )
        pd_tpot = decode_service_ms

        pd_split = PDSplitCostBreakdown(
            queue_wait_prefill_ms=queue_wait_prefill_ms,
            prefill_service_ms=prefill_service_ms,
            kv_transfer_ms=kv_transfer_ms,
            kv_transfer_bytes=kv_bytes,
            handoff_overhead_ms=handoff_overhead_ms,
            queue_wait_decode_ms=queue_wait_decode_ms,
            decode_service_ms=decode_service_ms,
            total_ms=pd_total,
            ttft_ms=pd_ttft,
            tpot_ms=pd_tpot,
            service_time_model_source=service_time_model.source,
            truth_boundary=_TB_KV,
        )

        # ── Policy selection ─────────────────────────────────────────────────
        pd_total_lower = pd_split.total_ms < colocated.total_ms
        pd_ttft_ok = pd_split.ttft_ms <= slo_ttft_ms
        pd_tpot_ok = pd_split.tpot_ms <= slo_tpot_ms

        if pd_total_lower and pd_ttft_ok and pd_tpot_ok:
            selected_policy = "pd_split"
            goodput_proxy = 1.0 / max(pd_split.total_ms, 1e-6)
            decision_reason = (
                "pd_split selected: lower total cost and within SLO bounds for TTFT and TPOT"
            )
        else:
            selected_policy = "colocated"
            goodput_proxy = 1.0 / max(colocated.total_ms, 1e-6)
            reason_parts: list[str] = []
            if not pd_total_lower:
                reason_parts.append("pd_split total cost not lower than colocated")
            if not pd_ttft_ok:
                reason_parts.append(
                    f"pd_split TTFT {pd_split.ttft_ms:.3f}ms exceeds SLO {slo_ttft_ms}ms"
                )
            if not pd_tpot_ok:
                reason_parts.append(
                    f"pd_split TPOT {pd_split.tpot_ms:.3f}ms exceeds SLO {slo_tpot_ms}ms"
                )
            decision_reason = "colocated selected: " + "; ".join(reason_parts)

        comparison = PDSplitDecisionComparison(
            colocated=colocated,
            pd_split=pd_split,
            selected_policy=selected_policy,
            decision_reason=decision_reason,
            slo_ttft_ms=slo_ttft_ms,
            slo_tpot_ms=slo_tpot_ms,
            goodput_proxy=goodput_proxy,
            truth_boundary=_TB_GOODPUT,
        )

        # ── Stage objects ────────────────────────────────────────────────────
        prefill_stage = PrefillStage(
            worker_id="prefill-worker-0",
            backend=prefill_plan.backend_policy.primary_backend,
            compiler_cost_ms=prefill_compiler_cost_ms,
            service_ms=prefill_service_ms,
            queue_wait_ms=queue_wait_prefill_ms,
            truth_boundary=_TB_PLAN,
        )
        kv_stage = KVTransferStage(
            transfer_bytes=kv_bytes,
            transfer_cost_ms=kv_transfer_ms,
            bandwidth_mb_per_ms=kv_bandwidth_mb_per_ms,
            truth_boundary=_TB_KV,
        )
        decode_stage = DecodeStage(
            worker_id="decode-worker-0",
            backend=decode_plan.backend_policy.primary_backend,
            compiler_cost_ms=decode_compiler_cost_ms,
            service_ms=decode_service_ms,
            queue_wait_ms=queue_wait_decode_ms,
            truth_boundary=_TB_PLAN,
        )
        replay_stage = OptionalReplayStage(
            eligible=decode_plan.replay_policy.eligible,
            bucket=decode_plan.replay_policy.cuda_graph_bucket,
            truth_boundary=decode_plan.replay_policy.truth_boundary,
        )

        total_compiler_cost_ms = (
            prefill_service_ms + kv_transfer_ms + handoff_overhead_ms + decode_service_ms
        )
        target_profile_id = prefill_plan.target_profile_id or decode_plan.target_profile_id

        return DistributedRuntimePlan(
            model_name="",
            target_profile_id=target_profile_id,
            prefill=prefill_stage,
            kv_transfer=kv_stage,
            decode=decode_stage,
            replay=replay_stage,
            decision_comparison=comparison,
            total_compiler_cost_ms=total_compiler_cost_ms,
            truth_boundary=_TB_PLAN,
        )


def _find_plan(
    plans: list[RuntimeExecutionPlan], function_name: str
) -> RuntimeExecutionPlan:
    matches = [p for p in plans if p.function_name == function_name]
    if not matches:
        raise ValueError(
            f"PDSplitPlanner requires a plan with function_name='{function_name}'; "
            f"got function names: {[p.function_name for p in plans]}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"PDSplitPlanner requires exactly one '{function_name}' plan; "
            f"found {len(matches)}"
        )
    return matches[0]
