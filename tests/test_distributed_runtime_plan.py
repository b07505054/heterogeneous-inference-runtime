"""Tests for DistributedRuntimePlan and PDSplitPlanner."""

import pytest

from deployment.distributed_runtime_plan import (
    DecodeStage,
    DistributedRuntimePlan,
    HardwareConfig,
    KVTransferStage,
    LinkType,
    OptionalReplayStage,
    PDSplitDecisionComparison,
    PDSplitPlanner,
    PhaseTimingSpec,
    PrefillStage,
    QueueState,
    ServiceTimeModel,
)
from deployment.speculative_decoding import (
    BASELINE_DECODE_STAGE_SERVICE_MS,
    SpeculativeDecodingConfig,
    SpeculativeRuntimeContext,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
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


def _dict_to_spec(d: dict) -> PhaseTimingSpec:
    cs = d["cost_summary"]
    return PhaseTimingSpec(
        function_name=d["function_name"],
        service_ms=float(cs.get("colocated_total_ms", cs.get("pd_split_total_ms", 0.0))),
        kv_byte_estimate_mb=float(d["kv_plan"]["kv_byte_estimate_mb"]),
        backend=d["backend_execution_plan"]["primary_backend"],
        replay_eligible=bool(d["replay_plan"]["replay_eligible"]),
        cuda_graph_bucket=d["replay_plan"].get("cuda_graph_bucket", ""),
        replay_truth_boundary=d["replay_plan"].get(
            "truth_boundary",
            "static_shape_replay_eligibility_not_cuda_graph_capture",
        ),
        target_profile_id=d.get("target_profile_id", ""),
    )


def _make_plans(prefill_dict=None, decode_dict=None):
    return [
        _dict_to_spec(prefill_dict or _PREFILL_DICT),
        _dict_to_spec(decode_dict or _DECODE_DICT),
    ]


def _speculative_config(**overrides):
    fields = {
        "enabled": True,
        "draft_model_name": "tiny-draft",
        "target_model_name": "target-llm",
        "draft_length": 4,
        "draft_model_decode_ms_per_token": 0.2,
        "target_verify_ms_per_draft_token": 0.2,
        "expected_acceptance_rate": 0.75,
        "draft_kv_overhead_ms": 0.1,
        "target_kv_commit_ms": 0.1,
        "acceptance_check_ms": 0.05,
        "correction_token_ms": 0.5,
        "rollback_cost_ms": 0.1,
        "output_commit_ms": 0.05,
        "coordinator_overhead_ms": 0.1,
    }
    fields.update(overrides)
    return SpeculativeDecodingConfig(**fields)


def _speculative_context(**overrides):
    fields = {
        "request_id": "req-pd",
        "decode_iteration": 1,
        "current_sequence_length": 128,
        "generated_tokens_so_far": 16,
        "remaining_token_budget": 32,
        "target_kv_ready": True,
        "draft_kv_ready": True,
        "backend": "cuda",
    }
    fields.update(overrides)
    return SpeculativeRuntimeContext(**fields)


# ---------------------------------------------------------------------------
# Stage construction
# ---------------------------------------------------------------------------

def test_plan_from_prefill_decode_pair_creates_all_stages():
    result = PDSplitPlanner.plan(_make_plans())
    assert isinstance(result, DistributedRuntimePlan)
    assert isinstance(result.prefill, PrefillStage)
    assert isinstance(result.kv_transfer, KVTransferStage)
    assert isinstance(result.decode, DecodeStage)
    assert isinstance(result.replay, OptionalReplayStage)
    assert isinstance(result.decision_comparison, PDSplitDecisionComparison)


def test_prefill_stage_fields():
    result = PDSplitPlanner.plan(_make_plans())
    assert result.prefill.backend == "cuda"
    assert result.prefill.compiler_cost_ms == pytest.approx(31.2)
    assert result.prefill.worker_id == "prefill-worker-0"


def test_decode_stage_fields():
    result = PDSplitPlanner.plan(_make_plans())
    assert result.decode.backend == "cuda"
    assert result.decode.compiler_cost_ms == pytest.approx(8.5)
    assert result.decode.worker_id == "decode-worker-0"
    assert result.decode.speculative_decoding is None


def test_no_speculative_config_leaves_decode_decision_unset():
    result = PDSplitPlanner.plan(
        _make_plans(),
        speculative_runtime_context=_speculative_context(),
    )

    assert result.decode.speculative_decoding is None


def test_speculative_config_attaches_decision_to_decode_stage():
    result = PDSplitPlanner.plan(
        _make_plans(),
        speculative_decoding_config=_speculative_config(),
        speculative_runtime_context=_speculative_context(),
    )

    decision = result.decode.speculative_decoding
    assert decision is not None
    assert decision.selected is True
    assert decision.plan is not None
    assert decision.plan.runtime_context.request_id == "req-pd"


def test_speculative_decision_baseline_equals_decode_service_ms():
    result = PDSplitPlanner.plan(
        _make_plans(),
        speculative_decoding_config=_speculative_config(),
        speculative_runtime_context=_speculative_context(),
    )

    assert result.decode.speculative_decoding is not None
    plan = result.decode.speculative_decoding.plan
    assert plan is not None
    assert plan.baseline_decode_service_ms == pytest.approx(result.decode.service_ms)
    assert plan.baseline_source == BASELINE_DECODE_STAGE_SERVICE_MS


def test_speculative_requires_runtime_context_when_config_is_provided():
    with pytest.raises(ValueError, match="speculative_runtime_context"):
        PDSplitPlanner.plan(
            _make_plans(),
            speculative_decoding_config=_speculative_config(),
        )


def test_service_time_adjustment_affects_speculative_baseline():
    stm = ServiceTimeModel(
        source="compiler_estimate",
        decode_service_adjustment_ms=2.0,
    )
    result = PDSplitPlanner.plan(
        _make_plans(),
        service_time_model=stm,
        speculative_decoding_config=_speculative_config(),
        speculative_runtime_context=_speculative_context(),
    )

    assert result.decode.service_ms == pytest.approx(10.5)
    assert result.decode.speculative_decoding is not None
    plan = result.decode.speculative_decoding.plan
    assert plan is not None
    assert plan.baseline_decode_service_ms == pytest.approx(10.5)


def test_decode_queue_wait_does_not_affect_speculative_baseline():
    kv_ms = 4.0 / 24.0
    decode_ready = 31.2 + kv_ms + 0.2
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=24.0,
        queue_state=QueueState(
            arrival_time_ms=0.0,
            prefill_worker_available_at_ms=0.0,
            decode_worker_available_at_ms=decode_ready + 9.0,
        ),
        speculative_decoding_config=_speculative_config(),
        speculative_runtime_context=_speculative_context(),
    )

    assert result.decode.queue_wait_ms == pytest.approx(9.0)
    assert result.decode.speculative_decoding is not None
    plan = result.decode.speculative_decoding.plan
    assert plan is not None
    assert plan.baseline_decode_service_ms == pytest.approx(result.decode.service_ms)
    assert plan.baseline_decode_service_ms == pytest.approx(8.5)


def test_kv_transfer_and_handoff_do_not_affect_speculative_baseline():
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=0.5,
        handoff_overhead_ms=7.0,
        speculative_decoding_config=_speculative_config(),
        speculative_runtime_context=_speculative_context(),
    )

    assert result.kv_transfer.transfer_cost_ms == pytest.approx(8.0)
    assert result.decision_comparison.pd_split.handoff_overhead_ms == pytest.approx(7.0)
    assert result.decode.speculative_decoding is not None
    plan = result.decode.speculative_decoding.plan
    assert plan is not None
    assert plan.baseline_decode_service_ms == pytest.approx(8.5)


def test_speculative_attachment_does_not_change_decode_service_tpot_or_total_cost():
    base = PDSplitPlanner.plan(_make_plans())
    with_spec = PDSplitPlanner.plan(
        _make_plans(),
        speculative_decoding_config=_speculative_config(),
        speculative_runtime_context=_speculative_context(),
    )

    assert with_spec.decode.service_ms == pytest.approx(base.decode.service_ms)
    assert with_spec.decision_comparison.pd_split.tpot_ms == pytest.approx(
        with_spec.decode.service_ms
    )
    assert with_spec.decision_comparison.colocated.tpot_ms == pytest.approx(
        with_spec.decode.service_ms
    )
    assert with_spec.total_compiler_cost_ms == pytest.approx(
        base.total_compiler_cost_ms
    )


def test_replay_stage_from_decode_plan():
    result = PDSplitPlanner.plan(_make_plans())
    assert result.replay.eligible is True
    assert result.replay.bucket == "decode_static"


def test_target_profile_id_propagated():
    result = PDSplitPlanner.plan(_make_plans())
    assert result.target_profile_id == "test-profile"


# ---------------------------------------------------------------------------
# KV transfer cost
# ---------------------------------------------------------------------------

def test_kv_transfer_cost_computed_from_bytes_and_bandwidth():
    # kv_byte_estimate_mb = 4.0 from prefill plan.
    result = PDSplitPlanner.plan(_make_plans(), kv_bandwidth_mb_per_ms=24.0)
    expected_ms = 4.0 / 24.0
    assert result.kv_transfer.transfer_cost_ms == pytest.approx(expected_ms)
    assert result.kv_transfer.transfer_bytes == 4 * 1024 * 1024
    assert result.kv_transfer.bandwidth_mb_per_ms == pytest.approx(24.0)
    assert result.kv_transfer.link_type == "custom"
    assert result.kv_transfer.gpu_count == 2
    assert result.kv_transfer.bandwidth_source == "override"


def test_default_hardware_config_models_two_gpu_pcie_gen4_pd_split():
    result = PDSplitPlanner.plan(_make_plans())
    assert result.kv_transfer.gpu_count == 2
    assert result.kv_transfer.link_type == LinkType.PCIE_GEN4_X16.value
    assert result.kv_transfer.bandwidth_mb_per_ms == pytest.approx(32.0)
    assert result.kv_transfer.transfer_cost_ms == pytest.approx(4.0 / 32.0)
    assert result.kv_transfer.bandwidth_source == "link_type_preset"


def test_nvlink_hardware_config_uses_nvlink_bandwidth():
    result = PDSplitPlanner.plan(
        _make_plans(),
        hardware_config=HardwareConfig(gpu_count=2, link_type=LinkType.NVLINK),
    )
    assert result.kv_transfer.link_type == LinkType.NVLINK.value
    assert result.kv_transfer.bandwidth_mb_per_ms == pytest.approx(900.0)
    assert result.kv_transfer.transfer_cost_ms == pytest.approx(4.0 / 900.0)


def test_single_gpu_hardware_config_has_no_cross_gpu_kv_transfer_or_handoff():
    result = PDSplitPlanner.plan(
        _make_plans(),
        hardware_config=HardwareConfig(gpu_count=1, link_type=LinkType.PCIE_GEN4_X16),
        handoff_overhead_ms=0.2,
    )
    assert result.kv_transfer.gpu_count == 1
    assert result.kv_transfer.transfer_cost_ms == pytest.approx(0.0)
    assert result.decision_comparison.pd_split.handoff_overhead_ms == pytest.approx(0.0)
    assert result.decision_comparison.pd_split.ttft_ms == pytest.approx(31.2)
    assert result.decision_comparison.pd_split.total_ms == pytest.approx(31.2 + 8.5)
    assert result.total_compiler_cost_ms == pytest.approx(31.2 + 8.5)
    assert result.decision_comparison.selected_policy == "colocated"
    assert "gpu_count < 2" in result.decision_comparison.decision_reason


def test_kv_transfer_zero_when_no_kv_estimate():
    no_kv = {**_PREFILL_DICT, "kv_plan": {
        "layout": "unknown",
        "kv_byte_estimate_mb": 0.0,
        "layout_reason": "",
        "truth_boundary": "",
    }}
    result = PDSplitPlanner.plan(_make_plans(prefill_dict=no_kv))
    assert result.kv_transfer.transfer_cost_ms == pytest.approx(0.0)
    assert result.kv_transfer.transfer_bytes == 0


# ---------------------------------------------------------------------------
# Queue wait derivation
# ---------------------------------------------------------------------------

def test_queue_wait_prefill_derived_from_worker_availability():
    # prefill worker available 5ms after arrival → queue_wait_prefill = 5ms
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=5.0,
        decode_worker_available_at_ms=0.0,
    )
    result = PDSplitPlanner.plan(_make_plans(), queue_state=qs)
    assert result.decision_comparison.pd_split.queue_wait_prefill_ms == pytest.approx(5.0)
    assert result.prefill.queue_wait_ms == pytest.approx(5.0)


def test_queue_wait_decode_derived_from_decode_worker_availability():
    # decode worker available long after decode_ready → extra queue wait
    # decode_ready ≈ 0 + 31.2 + 4/24 + 0.2 ≈ 31.567ms; set available 10ms later
    kv_ms = 4.0 / 24.0
    decode_ready = 31.2 + kv_ms + 0.2   # ~31.567ms
    decode_available = decode_ready + 10.0
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=decode_available,
    )
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=24.0,
        handoff_overhead_ms=0.2,
        queue_state=qs,
    )
    assert result.decision_comparison.pd_split.queue_wait_decode_ms == pytest.approx(10.0)
    assert result.decode.queue_wait_ms == pytest.approx(10.0)


def test_colocated_queue_wait_derived_from_worker_availability():
    # colocated reuses prefill_worker_available_at_ms
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=7.0,
        decode_worker_available_at_ms=0.0,
    )
    result = PDSplitPlanner.plan(_make_plans(), queue_state=qs)
    assert result.decision_comparison.colocated.queue_wait_ms == pytest.approx(7.0)


def test_queue_wait_zero_when_worker_available_before_arrival():
    # worker ready before the request arrives → no queue wait
    qs = QueueState(
        arrival_time_ms=10.0,
        prefill_worker_available_at_ms=5.0,   # available before arrival
        decode_worker_available_at_ms=0.0,
    )
    result = PDSplitPlanner.plan(_make_plans(), queue_state=qs)
    assert result.decision_comparison.pd_split.queue_wait_prefill_ms == pytest.approx(0.0)
    assert result.decision_comparison.colocated.queue_wait_ms == pytest.approx(0.0)


def test_pd_split_queue_waits_independent():
    # prefill queue wait = 2ms; decode worker ready before decode_ready → decode queue wait = 0
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=2.0,
        decode_worker_available_at_ms=0.0,   # available immediately, decode_ready > 0
    )
    result = PDSplitPlanner.plan(_make_plans(), queue_state=qs)
    pd = result.decision_comparison.pd_split
    assert pd.queue_wait_prefill_ms == pytest.approx(2.0)
    assert pd.queue_wait_decode_ms == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Colocated cost breakdown
# ---------------------------------------------------------------------------

def test_ttft_colocated_is_queue_plus_prefill_service():
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=5.0,
        decode_worker_available_at_ms=0.0,
    )
    result = PDSplitPlanner.plan(_make_plans(), queue_state=qs)
    assert result.decision_comparison.colocated.ttft_ms == pytest.approx(31.2 + 5.0)


def test_colocated_tpot_is_decode_service_ms():
    # tpot for colocated = decode_service_ms only; no interference field
    result = PDSplitPlanner.plan(_make_plans())
    assert result.decision_comparison.colocated.tpot_ms == pytest.approx(8.5)


def test_colocated_cost_with_decode_service_adjustment():
    # decode_service_adjustment replaces the old interference_penalty concept
    stm = ServiceTimeModel(
        source="compiler_estimate",
        decode_service_adjustment_ms=5.0,
    )
    result = PDSplitPlanner.plan(_make_plans(), service_time_model=stm)
    col = result.decision_comparison.colocated
    assert col.decode_service_ms == pytest.approx(8.5 + 5.0)
    assert col.tpot_ms == pytest.approx(8.5 + 5.0)
    assert col.total_ms == pytest.approx(31.2 + 8.5 + 5.0)


def test_no_interference_penalty_field_exists():
    result = PDSplitPlanner.plan(_make_plans())
    col = result.decision_comparison.colocated
    assert not hasattr(col, "interference_penalty_ms")


# ---------------------------------------------------------------------------
# PD-split cost breakdown
# ---------------------------------------------------------------------------

def test_pd_split_cost_includes_handoff_and_transfer():
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=24.0,
        handoff_overhead_ms=0.2,
    )
    pd = result.decision_comparison.pd_split
    expected_transfer = 4.0 / 24.0
    assert pd.handoff_overhead_ms == pytest.approx(0.2)
    assert pd.kv_transfer_ms == pytest.approx(expected_transfer)
    assert pd.total_ms == pytest.approx(31.2 + expected_transfer + 0.2 + 8.5)


def test_ttft_pd_split_includes_queue_transfer_and_handoff():
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=2.0,
        decode_worker_available_at_ms=0.0,   # no extra decode queue wait
    )
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=24.0,
        handoff_overhead_ms=0.2,
        queue_state=qs,
    )
    pd = result.decision_comparison.pd_split
    expected_transfer = 4.0 / 24.0
    assert pd.ttft_ms == pytest.approx(31.2 + 2.0 + expected_transfer + 0.2)


def test_tpot_pd_split_is_decode_service_ms_only():
    # tpot = decode_service_ms; decode queue wait appears in ttft, not tpot
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=1000.0,  # large decode queue wait
    )
    result = PDSplitPlanner.plan(_make_plans(), queue_state=qs)
    # tpot must not include the decode queue wait
    assert result.decision_comparison.pd_split.tpot_ms == pytest.approx(8.5)


def test_pd_split_decode_queue_wait_in_ttft_not_tpot():
    kv_ms = 4.0 / 24.0
    decode_ready = 31.2 + kv_ms + 0.2
    decode_available = decode_ready + 3.0
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,
        decode_worker_available_at_ms=decode_available,
    )
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=24.0,
        handoff_overhead_ms=0.2,
        queue_state=qs,
    )
    pd = result.decision_comparison.pd_split
    assert pd.queue_wait_decode_ms == pytest.approx(3.0)
    assert pd.ttft_ms == pytest.approx(31.2 + kv_ms + 0.2 + 3.0)
    assert pd.tpot_ms == pytest.approx(8.5)   # decode queue wait NOT in tpot


def test_pd_split_total_formula_uses_queue_wait_and_service_time():
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=2.0,
        decode_worker_available_at_ms=0.0,
    )
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=24.0,
        handoff_overhead_ms=0.2,
        queue_state=qs,
    )
    pd = result.decision_comparison.pd_split
    kv_ms = 4.0 / 24.0
    expected_total = 2.0 + 31.2 + kv_ms + 0.2 + 0.0 + 8.5
    assert pd.total_ms == pytest.approx(expected_total)


# ---------------------------------------------------------------------------
# Service time model
# ---------------------------------------------------------------------------

def test_service_time_model_applied_to_stage_service_ms():
    stm = ServiceTimeModel(
        source="compiler_estimate",
        prefill_service_adjustment_ms=2.0,
        decode_service_adjustment_ms=1.0,
    )
    result = PDSplitPlanner.plan(_make_plans(), service_time_model=stm)
    assert result.prefill.service_ms == pytest.approx(31.2 + 2.0)
    assert result.decode.service_ms == pytest.approx(8.5 + 1.0)


def test_compiler_estimate_allows_service_adjustment():
    stm = ServiceTimeModel(
        source="compiler_estimate",
        prefill_service_adjustment_ms=3.0,
        decode_service_adjustment_ms=2.0,
    )
    # Must not raise
    result = PDSplitPlanner.plan(_make_plans(), service_time_model=stm)
    assert result.prefill.service_ms == pytest.approx(31.2 + 3.0)


def test_isolated_measurement_allows_service_adjustment():
    stm = ServiceTimeModel(
        source="isolated_single_request_measurement",
        decode_service_adjustment_ms=1.5,
    )
    result = PDSplitPlanner.plan(_make_plans(), service_time_model=stm)
    assert result.decode.service_ms == pytest.approx(8.5 + 1.5)


def test_measured_continuous_batching_rejects_adjustment_to_prevent_double_counting():
    with pytest.raises(ValueError, match="double-count"):
        ServiceTimeModel(
            source="measured_continuous_batching_curve",
            prefill_service_adjustment_ms=5.0,
        )


def test_production_trace_rejects_adjustment_to_prevent_double_counting():
    with pytest.raises(ValueError, match="double-count"):
        ServiceTimeModel(
            source="production_trace_service_time",
            decode_service_adjustment_ms=2.0,
        )


def test_measured_continuous_batching_accepts_zero_adjustments():
    stm = ServiceTimeModel(
        source="measured_continuous_batching_curve",
        prefill_service_adjustment_ms=0.0,
        decode_service_adjustment_ms=0.0,
    )
    result = PDSplitPlanner.plan(_make_plans(), service_time_model=stm)
    assert result.prefill.service_ms == pytest.approx(31.2)


def test_service_time_model_source_recorded_in_breakdowns():
    stm = ServiceTimeModel(source="compiler_estimate")
    result = PDSplitPlanner.plan(_make_plans(), service_time_model=stm)
    assert result.decision_comparison.colocated.service_time_model_source == "compiler_estimate"
    assert result.decision_comparison.pd_split.service_time_model_source == "compiler_estimate"


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------

def test_selected_policy_pd_split_when_lower_total_and_slo_ok():
    # Colocated worker is busy with a prior prefill+decode cycle (~50ms) so the
    # next request queues for 50ms. The pd_split prefill worker finishes its
    # prefill-only cycle sooner and is immediately available. Near-zero KV
    # transfer and zero handoff mean pd_split total < colocated total.
    fast_decode = {**_DECODE_DICT, "cost_summary": {
        **_DECODE_DICT["cost_summary"],
        "pd_split_total_ms": 1.0,
        "colocated_total_ms": 1.0,
    }}
    qs = QueueState(
        arrival_time_ms=0.0,
        prefill_worker_available_at_ms=0.0,   # pd_split prefill worker is free
        decode_worker_available_at_ms=0.0,
        colocated_worker_available_at_ms=50.0,  # colocated worker busy with prior request
    )
    result = PDSplitPlanner.plan(
        _make_plans(decode_dict=fast_decode),
        kv_bandwidth_mb_per_ms=10000.0,   # near-zero KV transfer cost
        handoff_overhead_ms=0.0,
        queue_state=qs,
        slo_ttft_ms=500.0,
        slo_tpot_ms=500.0,
    )
    # col_total = 50 + 31.2 + 1.0 = 82.2ms; pd_total = 0 + 31.2 + ~0 + 0 + 0 + 1.0 = 32.2ms
    assert result.decision_comparison.selected_policy == "pd_split"


def test_selected_policy_colocated_when_transfer_too_expensive():
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=0.001,   # extremely slow → huge transfer cost
        handoff_overhead_ms=100.0,
        slo_ttft_ms=200.0,
        slo_tpot_ms=20.0,
    )
    assert result.decision_comparison.selected_policy == "colocated"


def test_selected_policy_colocated_when_ttft_slo_violated():
    # Default pd_split TTFT ≈ 31.2 + transfer + 0.2 >> 1.0 ms
    result = PDSplitPlanner.plan(_make_plans(), slo_ttft_ms=1.0, slo_tpot_ms=500.0)
    assert result.decision_comparison.selected_policy == "colocated"


def test_selected_policy_colocated_when_tpot_slo_violated():
    # pd_tpot = decode_service_ms = 8.5 > slo_tpot = 5.0 → SLO violated → colocated
    result = PDSplitPlanner.plan(
        _make_plans(),
        slo_ttft_ms=500.0,
        slo_tpot_ms=5.0,
    )
    assert result.decision_comparison.selected_policy == "colocated"


def test_selected_policy_colocated_when_equal_cost():
    # Equal total costs: pd_split is NOT strictly lower → colocated wins.
    zero_transfer = {**_PREFILL_DICT, "kv_plan": {
        "layout": "paged",
        "kv_byte_estimate_mb": 0.0,
        "layout_reason": "",
        "truth_boundary": "",
    }}
    result = PDSplitPlanner.plan(
        _make_plans(prefill_dict=zero_transfer),
        kv_bandwidth_mb_per_ms=24.0,
        handoff_overhead_ms=0.0,
        slo_ttft_ms=500.0,
        slo_tpot_ms=500.0,
    )
    # colocated total == pd_split total; pd_total_lower is False → colocated
    assert result.decision_comparison.selected_policy == "colocated"


# ---------------------------------------------------------------------------
# Goodput proxy
# ---------------------------------------------------------------------------

def test_goodput_proxy_is_inverse_of_selected_total():
    result = PDSplitPlanner.plan(_make_plans())
    cmp = result.decision_comparison
    if cmp.selected_policy == "colocated":
        expected = 1.0 / max(cmp.colocated.total_ms, 1e-6)
    else:
        expected = 1.0 / max(cmp.pd_split.total_ms, 1e-6)
    assert cmp.goodput_proxy == pytest.approx(expected)


def test_goodput_proxy_positive():
    result = PDSplitPlanner.plan(_make_plans())
    assert result.decision_comparison.goodput_proxy > 0.0


# ---------------------------------------------------------------------------
# total_compiler_cost_ms
# ---------------------------------------------------------------------------

def test_total_compiler_cost_ms_is_sum_of_stages():
    result = PDSplitPlanner.plan(
        _make_plans(),
        kv_bandwidth_mb_per_ms=24.0,
        handoff_overhead_ms=0.2,
    )
    expected = 31.2 + (4.0 / 24.0) + 0.2 + 8.5
    assert result.total_compiler_cost_ms == pytest.approx(expected)


def test_total_compiler_cost_excludes_queue_waits():
    # Queue waits must not affect total_compiler_cost_ms.
    base = PDSplitPlanner.plan(_make_plans())
    kv_ms = 4.0 / 24.0
    decode_ready = 31.2 + kv_ms + 0.2
    with_queues = PDSplitPlanner.plan(
        _make_plans(),
        queue_state=QueueState(
            arrival_time_ms=0.0,
            prefill_worker_available_at_ms=10.0,
            decode_worker_available_at_ms=decode_ready + 10.0,
        ),
    )
    assert base.total_compiler_cost_ms == pytest.approx(with_queues.total_compiler_cost_ms)


# ---------------------------------------------------------------------------
# Rejection / validation
# ---------------------------------------------------------------------------

def test_rejects_missing_prefill():
    decode_only = [_dict_to_spec(_DECODE_DICT)]
    with pytest.raises(ValueError, match="prefill"):
        PDSplitPlanner.plan(decode_only)


def test_rejects_missing_decode():
    prefill_only = [_dict_to_spec(_PREFILL_DICT)]
    with pytest.raises(ValueError, match="decode"):
        PDSplitPlanner.plan(prefill_only)


def test_rejects_duplicate_prefill():
    p1 = _dict_to_spec(_PREFILL_DICT)
    p2 = _dict_to_spec(_PREFILL_DICT)
    d = _dict_to_spec(_DECODE_DICT)
    with pytest.raises(ValueError, match="prefill"):
        PDSplitPlanner.plan([p1, p2, d])


def test_rejects_wrong_function_names():
    wrong = {**_PREFILL_DICT, "function_name": "encode"}
    p = _dict_to_spec(wrong)
    d = _dict_to_spec(_DECODE_DICT)
    with pytest.raises(ValueError, match="prefill"):
        PDSplitPlanner.plan([p, d])


def test_rejects_empty_plan_list():
    with pytest.raises(ValueError, match="prefill"):
        PDSplitPlanner.plan([])


# ---------------------------------------------------------------------------
# Truth boundaries
# ---------------------------------------------------------------------------

def test_truth_boundaries_present():
    result = PDSplitPlanner.plan(_make_plans())
    assert result.truth_boundary == "pd_split_schedule_static_plan_not_live_cluster_execution"
    assert result.kv_transfer.truth_boundary == "kv_transfer_cost_model_not_measured_network"
    assert result.decision_comparison.truth_boundary == "goodput_proxy_not_cluster_throughput"
    assert result.decision_comparison.colocated.truth_boundary == (
        "prefill_decode_cost_model_based_on_compiler_estimates"
    )
    assert result.decision_comparison.pd_split.truth_boundary == (
        "kv_transfer_cost_model_not_measured_network"
    )
    assert result.prefill.truth_boundary == "pd_split_schedule_static_plan_not_live_cluster_execution"
    assert result.decode.truth_boundary == "pd_split_schedule_static_plan_not_live_cluster_execution"


# ---------------------------------------------------------------------------
# Immutability / no mutation
# ---------------------------------------------------------------------------

def test_does_not_mutate_runtime_execution_plans():
    plans = _make_plans()
    pre_prefill_cost = plans[0].service_ms
    pre_decode_cost = plans[1].service_ms
    pre_prefill_name = plans[0].function_name
    pre_decode_name = plans[1].function_name
    pre_prefill_kv = plans[0].kv_byte_estimate_mb

    PDSplitPlanner.plan(plans)

    assert plans[0].service_ms == pre_prefill_cost
    assert plans[1].service_ms == pre_decode_cost
    assert plans[0].function_name == pre_prefill_name
    assert plans[1].function_name == pre_decode_name
    assert plans[0].kv_byte_estimate_mb == pre_prefill_kv


def test_result_is_immutable():
    result = PDSplitPlanner.plan(_make_plans())
    with pytest.raises((AttributeError, TypeError)):
        result.total_compiler_cost_ms = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Order independence
# ---------------------------------------------------------------------------

def test_order_independent_prefill_decode():
    # Plans provided decode-first — planner must find them by name.
    p = _dict_to_spec(_PREFILL_DICT)
    d = _dict_to_spec(_DECODE_DICT)
    result = PDSplitPlanner.plan([d, p])
    assert result.prefill.compiler_cost_ms == pytest.approx(31.2)
    assert result.decode.compiler_cost_ms == pytest.approx(8.5)
