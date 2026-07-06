"""Distributed runtime planning for prefill/decode disaggregation.

PDSplitPlanner converts a list[PhaseTimingSpec] (one prefill + one decode)
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
from enum import Enum

from deployment.prefix_cache_simulator import PrefixCacheResult
from deployment.speculative_decoding import (
    SpeculativeDecodingConfig,
    SpeculativeDecodingDecision,
    SpeculativeDecodingEvaluator,
    SpeculativeRuntimeContext,
)

_TB_PLAN = "pd_split_schedule_static_plan_not_live_cluster_execution"
_TB_KV = "kv_transfer_cost_model_not_measured_network"
_TB_GOODPUT = "goodput_proxy_not_cluster_throughput"
_TB_COST = "prefill_decode_cost_model_based_on_compiler_estimates"
_TB_QUEUE = "queue_wait_derived_from_worker_availability_not_live_scheduler"
_TB_SERVICE = "service_time_model_static_estimate_not_live_measurement"
_TB_PREFIX = "prefix_cache_simulated_adjusted_plan_not_real_kv_cache_or_network_measurement"

# Sources whose service time already embeds interference/batching overhead.
# Adding service adjustments on top would double-count that cost.
_MEASURED_SOURCES: frozenset[str] = frozenset({
    "measured_continuous_batching_curve",
    "production_trace_service_time",
})


# ---------------------------------------------------------------------------
# Hardware model
# ---------------------------------------------------------------------------

class LinkType(Enum):
    """GPU interconnect preset used for KV handoff bandwidth modeling."""

    PCIE_GEN4_X16 = "pcie_gen4_x16"
    PCIE_GEN5_X16 = "pcie_gen5_x16"
    NVLINK = "nvlink"
    CUSTOM = "custom"


_LINK_BANDWIDTH_GB_PER_S: dict[LinkType, float] = {
    LinkType.PCIE_GEN4_X16: 32.0,
    LinkType.PCIE_GEN5_X16: 64.0,
    LinkType.NVLINK: 900.0,
}


def link_bandwidth_gb_per_s(link_type: LinkType) -> float:
    """Return nominal one-way link bandwidth in GB/s for built-in presets."""
    if link_type is LinkType.CUSTOM:
        raise ValueError("CUSTOM link type requires bandwidth_override_gb_per_s")
    return _LINK_BANDWIDTH_GB_PER_S[link_type]


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware assumptions for the distributed runtime simulator.

    gpu_count=2 models the current PD-split setup: one prefill GPU and one
    decode GPU. gpu_count=1 means colocated execution; no KV crosses GPUs.

    bandwidth_override_gb_per_s is for measured or experimental links. When it
    is set, link_type is treated as descriptive metadata and the override is
    used for the transfer-cost calculation.
    """

    gpu_count: int = 2
    link_type: LinkType = LinkType.PCIE_GEN4_X16
    gpu_memory_gb: float | None = None
    bandwidth_override_gb_per_s: float | None = None

    def __post_init__(self) -> None:
        if self.gpu_count < 1:
            raise ValueError("HardwareConfig.gpu_count must be >= 1")
        if (
            self.bandwidth_override_gb_per_s is not None
            and self.bandwidth_override_gb_per_s <= 0.0
        ):
            raise ValueError("bandwidth_override_gb_per_s must be > 0")
        if self.link_type is LinkType.CUSTOM and self.bandwidth_override_gb_per_s is None:
            raise ValueError("LinkType.CUSTOM requires bandwidth_override_gb_per_s")

    @property
    def effective_bandwidth_gb_per_s(self) -> float:
        if self.bandwidth_override_gb_per_s is not None:
            return self.bandwidth_override_gb_per_s
        return link_bandwidth_gb_per_s(self.link_type)


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
    bandwidth_mb_per_ms is numerically equivalent to GB/s in this model.
    This is a bandwidth-model estimate, not a measured network transfer.
    truth_boundary = "kv_transfer_cost_model_not_measured_network"
    """

    transfer_bytes: int
    transfer_cost_ms: float
    bandwidth_mb_per_ms: float
    link_type: str
    gpu_count: int
    bandwidth_source: str
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
    speculative_decoding: SpeculativeDecodingDecision | None = None


@dataclass(frozen=True)
class OptionalReplayStage:
    """CUDA graph replay eligibility for the decode stage."""

    eligible: bool
    bucket: str
    truth_boundary: str


# ---------------------------------------------------------------------------
# Prefix cache adjustment record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrefixCacheAdjustment:
    """Effect of a prefix-cache lookup on the distributed runtime plan.

    Captures both the token savings (reduced prefill service time) and any
    remote transfer overhead for remote_hit (additional network cost).

    adjusted_prefill_service_ms is max(0.0, original_prefill_service_ms - saved_prefill_ms).
    remote_transfer_cost_ms is the bandwidth-model cost of moving the prefix KV
    from the remote cache worker; 0.0 for local_hit and miss.

    All values are derived from PrefixCacheResult; the planner does not call
    PrefixCacheSimulator directly. No wall clock. No real KV memory.

    truth_boundary = "prefix_cache_simulated_adjusted_plan_not_real_kv_cache_or_network_measurement"
    """

    hit_type: str
    hit_tokens: int
    saved_prefill_ms: float
    remote_transfer_bytes: float
    remote_transfer_cost_ms: float
    adjusted_prefill_service_ms: float
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
    prefix_cache_adjustment: PrefixCacheAdjustment | None = None


# ---------------------------------------------------------------------------
# Planner input type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseTimingSpec:
    """Timing and routing spec for one serving phase (prefill or decode).

    This is the planner's input type — it carries the fields PDSplitPlanner
    needs from each phase. It is not a compiler/runtime contract; it is an
    internal planning object.
    """

    function_name: str
    service_ms: float
    kv_byte_estimate_mb: float
    backend: str
    replay_eligible: bool
    cuda_graph_bucket: str
    replay_truth_boundary: str
    target_profile_id: str = ""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class PDSplitPlanner:
    """Converts a prefill + decode PhaseTimingSpec pair into a DistributedRuntimePlan.

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
        plans: list[PhaseTimingSpec],
        *,
        slo_ttft_ms: float = 200.0,
        slo_tpot_ms: float = 20.0,
        hardware_config: HardwareConfig | None = None,
        kv_bandwidth_mb_per_ms: float | None = None,
        handoff_overhead_ms: float = 0.2,
        queue_state: QueueState | None = None,
        service_time_model: ServiceTimeModel | None = None,
        prefix_cache_result: PrefixCacheResult | None = None,
        speculative_decoding_config: SpeculativeDecodingConfig | None = None,
        speculative_runtime_context: SpeculativeRuntimeContext | None = None,
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
        if hardware_config is None:
            hardware_config = HardwareConfig()
        if kv_bandwidth_mb_per_ms is not None:
            if kv_bandwidth_mb_per_ms <= 0.0:
                raise ValueError("kv_bandwidth_mb_per_ms must be > 0")
            hardware_config = HardwareConfig(
                gpu_count=hardware_config.gpu_count,
                link_type=LinkType.CUSTOM,
                gpu_memory_gb=hardware_config.gpu_memory_gb,
                bandwidth_override_gb_per_s=kv_bandwidth_mb_per_ms,
            )

        prefill_plan = _find_plan(plans, "prefill")
        decode_plan = _find_plan(plans, "decode")

        prefill_compiler_cost_ms = prefill_plan.service_ms
        decode_compiler_cost_ms = decode_plan.service_ms

        prefill_service_ms = (
            prefill_compiler_cost_ms + service_time_model.prefill_service_adjustment_ms
        )
        decode_service_ms = (
            decode_compiler_cost_ms + service_time_model.decode_service_adjustment_ms
        )
        speculative_decision: SpeculativeDecodingDecision | None = None
        if speculative_decoding_config is not None:
            if speculative_runtime_context is None:
                raise ValueError(
                    "speculative_runtime_context is required when "
                    "speculative_decoding_config is provided"
                )
            speculative_decision = SpeculativeDecodingEvaluator.evaluate(
                speculative_decoding_config,
                baseline_decode_service_ms=decode_service_ms,
                runtime_context=speculative_runtime_context,
            )

        # KV transfer: bytes from prefill memory policy; cost from bandwidth model.
        kv_mb = prefill_plan.kv_byte_estimate_mb
        kv_bytes = int(kv_mb * 1024 * 1024)
        effective_bandwidth_mb_per_ms = hardware_config.effective_bandwidth_gb_per_s
        kv_transfer_ms = (
            kv_mb / effective_bandwidth_mb_per_ms if hardware_config.gpu_count > 1 else 0.0
        )
        effective_handoff_overhead_ms = (
            handoff_overhead_ms if hardware_config.gpu_count > 1 else 0.0
        )

        # ── Prefix cache adjustment ───────────────────────────────────────────
        # Applied after KV transfer is computed (remote transfer cost shares the
        # same bandwidth model) and before queue wait derivation (so decode_ready_ms
        # reflects the adjusted prefill service time).
        prefix_cache_adj: PrefixCacheAdjustment | None = None
        if prefix_cache_result is not None:
            _saved_ms = prefix_cache_result.saved_prefill_ms
            _adj_prefill_ms = max(0.0, prefill_service_ms - _saved_ms)

            if (
                prefix_cache_result.hit_type == "remote_hit"
                and effective_bandwidth_mb_per_ms > 0.0
            ):
                _remote_mb = prefix_cache_result.remote_transfer_bytes / (1024.0 * 1024.0)
                _remote_transfer_cost_ms = _remote_mb / effective_bandwidth_mb_per_ms
            else:
                _remote_transfer_cost_ms = 0.0

            prefix_cache_adj = PrefixCacheAdjustment(
                hit_type=prefix_cache_result.hit_type,
                hit_tokens=prefix_cache_result.hit_tokens,
                saved_prefill_ms=_saved_ms,
                remote_transfer_bytes=prefix_cache_result.remote_transfer_bytes,
                remote_transfer_cost_ms=_remote_transfer_cost_ms,
                adjusted_prefill_service_ms=_adj_prefill_ms,
                truth_boundary=_TB_PREFIX,
            )

            prefill_service_ms = _adj_prefill_ms
            kv_transfer_ms = kv_transfer_ms + _remote_transfer_cost_ms

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
            prefill_start_ms
            + prefill_service_ms
            + kv_transfer_ms
            + effective_handoff_overhead_ms
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
            + effective_handoff_overhead_ms
            + queue_wait_decode_ms
            + decode_service_ms
        )
        pd_ttft = (
            queue_wait_prefill_ms
            + prefill_service_ms
            + kv_transfer_ms
            + effective_handoff_overhead_ms
            + queue_wait_decode_ms
        )
        pd_tpot = decode_service_ms

        pd_split = PDSplitCostBreakdown(
            queue_wait_prefill_ms=queue_wait_prefill_ms,
            prefill_service_ms=prefill_service_ms,
            kv_transfer_ms=kv_transfer_ms,
            kv_transfer_bytes=kv_bytes,
            handoff_overhead_ms=effective_handoff_overhead_ms,
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

        if hardware_config.gpu_count < 2:
            selected_policy = "colocated"
            goodput_proxy = 1.0 / max(colocated.total_ms, 1e-6)
            decision_reason = (
                "colocated selected: HardwareConfig.gpu_count < 2, so PD-split "
                "prefill/decode handoff is not available"
            )
        elif pd_total_lower and pd_ttft_ok and pd_tpot_ok:
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
            backend=prefill_plan.backend,
            compiler_cost_ms=prefill_compiler_cost_ms,
            service_ms=prefill_service_ms,
            queue_wait_ms=queue_wait_prefill_ms,
            truth_boundary=_TB_PLAN,
        )
        kv_stage = KVTransferStage(
            transfer_bytes=kv_bytes,
            transfer_cost_ms=kv_transfer_ms,
            bandwidth_mb_per_ms=effective_bandwidth_mb_per_ms,
            link_type=hardware_config.link_type.value,
            gpu_count=hardware_config.gpu_count,
            bandwidth_source=(
                "override" if hardware_config.bandwidth_override_gb_per_s is not None
                else "link_type_preset"
            ),
            truth_boundary=_TB_KV,
        )
        decode_stage = DecodeStage(
            worker_id="decode-worker-0",
            backend=decode_plan.backend,
            compiler_cost_ms=decode_compiler_cost_ms,
            service_ms=decode_service_ms,
            queue_wait_ms=queue_wait_decode_ms,
            truth_boundary=_TB_PLAN,
            speculative_decoding=speculative_decision,
        )
        replay_stage = OptionalReplayStage(
            eligible=decode_plan.replay_eligible,
            bucket=decode_plan.cuda_graph_bucket,
            truth_boundary=decode_plan.replay_truth_boundary,
        )

        total_compiler_cost_ms = (
            prefill_service_ms
            + kv_transfer_ms
            + effective_handoff_overhead_ms
            + decode_service_ms
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
            prefix_cache_adjustment=prefix_cache_adj,
        )


def _find_plan(
    plans: list[PhaseTimingSpec], function_name: str
) -> PhaseTimingSpec:
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
