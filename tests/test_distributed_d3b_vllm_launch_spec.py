"""D3B: vLLM Distributed Launch-Spec Materialization and Fail-Closed Validation
-- focused tests.

Covers the positive materialization chain for the real D2/D3A TP1 and TP2
compiler plans, plus every Part N fail-closed negative test. No test in
this file ever starts a real vLLM server or a real subprocess for a
rejected launch spec.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest

from deployment.execution_plan.loader import ExecutionPlanError, load_execution_plan
from deployment.vllm_adapter.distributed_argument_registry import (
    build_mock_registry_without,
    check_argument,
)
from deployment.vllm_adapter.distributed_capability_inventory import (
    discover_argument_registry,
    discover_environment,
)
from deployment.vllm_adapter.distributed_cli import build_cli
from deployment.vllm_adapter.distributed_launch_spec import (
    D3B_REACHABLE_STATES,
    ExecutionReadinessState,
    WholeModelTPEvidenceStatus,
)
from deployment.vllm_adapter.distributed_materializer import (
    UnknownDistributedStrategyError,
    materialize_launch_spec,
)
from deployment.vllm_adapter.distributed_preflight import (
    PreflightInputs,
    check_model_locally_resolvable,
    check_port_available,
    run_preflight,
)
from deployment.vllm_adapter.distributed_provenance import compute_provenance_counters
from deployment.vllm_adapter.distributed_rank_placement import build_rank_placement

REPO_ROOT = Path(__file__).resolve().parents[1]
D2_RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
TP1_PLAN_PATH = D2_RESULTS_DIR / "real_qwen_tp1_execution_plan.json"
TP2_PLAN_PATH = D2_RESULTS_DIR / "real_qwen_tp2_execution_plan.json"

pytestmark = pytest.mark.skipif(
    not TP2_PLAN_PATH.exists() or not TP1_PLAN_PATH.exists(),
    reason="requires the D2 compiler-exported real-Qwen TP1/TP2 plan artifacts",
)


def _base_preflight_inputs(**overrides) -> PreflightInputs:
    base = dict(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        model_locally_resolvable=True,
        vllm_installed=True,
        all_cli_arguments_supported=True,
        unsupported_cli_arguments=(),
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        world_size=2,
        visible_gpu_count=1,
        cuda_available=True,
        per_rank_gpu_memory_mb=(3700.0,),
        estimated_model_footprint_mb=1400.0,
        gpu_memory_utilization=0.9,
        dtype="float16",
        supported_dtypes=("auto", "bfloat16", "float", "float16", "float32", "half"),
        bf16_hardware_supported=True,
        max_model_len=2048,
        model_max_position_embeddings=32768,
        port=8000,
        port_available=True,
        master_address="127.0.0.1",
        rank_placement_valid=True,
        rank_placement_errors=(),
        rank_ids_contiguous=True,
        no_duplicate_physical_device=True,
        placement_count_equals_world_size=True,
        environ_conflicts=(),
        distributed_executor_backend="mp",
        supported_executor_backends=("external_launcher", "mp", "ray", "uni"),
        whole_model_tp_evidence_established=False,
    )
    base.update(overrides)
    return PreflightInputs(**base)


# ---------------------------------------------------------------------------
# Positive path: real D2/D3A TP1 and TP2 plans on this real single-GPU host.
# ---------------------------------------------------------------------------


def test_tp2_materializes_and_preflight_rejects_on_one_gpu_host():
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.spec.tensor_parallel_size == 2
    assert bundle.spec.pipeline_parallel_size == 1
    assert bundle.spec.world_size == 2
    assert bundle.preflight.passed is False
    assert bundle.preflight.primary_reason == "insufficient_visible_gpu_count"
    assert bundle.spec.execution_readiness_state == ExecutionReadinessState.PREFLIGHT_REJECTED.value
    assert bundle.spec.whole_model_tp_evidence_status == (
        WholeModelTPEvidenceStatus.NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY.value
    )
    assert bundle.spec.d3b_mode == "planning_only"
    # Never downgraded despite hardware insufficiency.
    assert bundle.spec.tensor_parallel_size == 2


def test_tp1_preflight_succeeds_and_reaches_dry_run_validated():
    bundle = materialize_launch_spec(TP1_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.spec.tensor_parallel_size == 1
    assert bundle.spec.world_size == 1
    assert bundle.preflight.passed is True
    assert bundle.preflight.rejection_reasons == ()
    assert bundle.spec.execution_readiness_state == ExecutionReadinessState.DRY_RUN_VALIDATED.value
    assert bundle.dry_run.passed is True


def test_every_materialized_field_has_a_recorded_provenance_source():
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    for name, entry in bundle.spec.field_provenance.items():
        assert entry.source is not None
        assert entry.reason.strip(), f"field {name} has an empty provenance reason"


def test_spec_is_json_serializable():
    import json

    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    payload = bundle.spec.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["schema_version"]
    assert payload["source_execution_plan_id"] == bundle.plan.plan_id


def test_d3b_never_reaches_execution_ready_or_started():
    for path in (TP1_PLAN_PATH, TP2_PLAN_PATH):
        bundle = materialize_launch_spec(path, repo_root=REPO_ROOT)
        assert bundle.spec.execution_readiness_state in {s.value for s in D3B_REACHABLE_STATES}
        assert bundle.spec.execution_readiness_state != ExecutionReadinessState.EXECUTION_READY.value
        assert bundle.spec.execution_readiness_state != ExecutionReadinessState.EXECUTION_STARTED.value


def test_no_subprocess_launched_for_rejected_tp2_spec(monkeypatch):
    # Capability discovery legitimately shells out to `nvidia-smi -L` (a
    # read-only probe); what must never happen is the *materialized vLLM
    # server command itself* being launched for a rejected spec.
    real_popen = subprocess.Popen

    def _guarded_popen(args, *a, **kwargs):
        argv = args if isinstance(args, (list, tuple)) else [args]
        if any("vllm.entrypoints.openai.api_server" in str(a) for a in argv):
            raise AssertionError("the materialized vLLM server command must never be launched for a rejected spec")
        return real_popen(args, *a, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _guarded_popen)
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.preflight.passed is False  # sanity: this is indeed the rejected case


# ---------------------------------------------------------------------------
# Negative tests (Part N).
# ---------------------------------------------------------------------------


def test_negative_tp2_with_one_visible_gpu():
    inputs = _base_preflight_inputs(visible_gpu_count=1, tensor_parallel_size=2, world_size=2)
    result = run_preflight(inputs)
    assert result.passed is False
    assert "insufficient_visible_gpu_count" in result.rejection_reasons
    assert result.primary_reason == "insufficient_visible_gpu_count"


def test_negative_world_size_mismatch():
    inputs = _base_preflight_inputs(world_size=3, tensor_parallel_size=2, pipeline_parallel_size=1)
    result = run_preflight(inputs)
    assert "world_size_tp_pp_mismatch" in result.rejection_reasons


def test_negative_tp_pp_mismatch_malformed_plan_rejected_by_loader():
    malformed = {
        "schema": "execution_plan", "schema_version": "2.0.0", "plan_id": "bad",
        "provenance": {"compiler_tool": "t", "model_spec_ref": "", "capability_bundle": {}},
        "model_identity": {}, "global_decisions": {}, "function_plans": [],
        "distributed": {
            "strategy": "tensor_parallel", "world_size": 2, "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1, "ranks": [{"rank_id": 0, "logical_device": "d0"}],
            "tensor_shards": [], "collectives": [],
        },
    }
    with pytest.raises(ExecutionPlanError):
        from deployment.execution_plan.loader import validate_execution_plan

        validate_execution_plan(malformed)


def test_negative_missing_rank_placement():
    result = build_rank_placement(compiler_rank_ids=(0,), world_size=2, visible_gpu_count=1)
    assert result.valid is False
    assert result.placement_count_equals_world_size is False


def test_negative_duplicate_rank_placement():
    result = build_rank_placement(compiler_rank_ids=(0, 0, 1), world_size=2, visible_gpu_count=2)
    # duplicate ranks collapse the contiguity check
    assert result.rank_ids_contiguous is False


def test_negative_two_tp_ranks_never_mapped_to_one_gpu():
    result = build_rank_placement(compiler_rank_ids=(0, 1), world_size=2, visible_gpu_count=1)
    physical = [p.physical_device_index for p in result.placements if p.physical_device_index is not None]
    assert len(physical) == len(set(physical))  # never duplicated
    assert result.no_duplicate_physical_device is True
    # rank 1 correctly left unresolved rather than fabricated onto GPU 0
    rank1 = next(p for p in result.placements if p.rank_id == 1)
    assert rank1.physical_device_index is None


def test_negative_unsupported_cli_flag_is_never_silently_emitted():
    real_registry = discover_argument_registry()
    mocked = build_mock_registry_without(["tensor_parallel_size"], base_registry=real_registry)
    record = check_argument("tensor_parallel_size", registry=mocked, value_source="compiler_plan")
    assert record.installed_version_support_status == "unsupported"

    fields = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
        "trust_remote_code": False, "dtype": "float16", "seed": 1234, "revision": None,
        "served_model_name": "x", "host": "127.0.0.1", "port": 8000,
        "master_address": "127.0.0.1", "master_port": 29501, "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1, "data_parallel_size": 1, "distributed_executor_backend": "mp",
        "max_model_len": 2048, "max_num_seqs": 4, "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.9, "enable_prefix_caching": True, "enable_chunked_prefill": True,
        "world_size": 2,
        "_sources": {k: "compiler_plan" for k in [
            "model", "tokenizer", "trust_remote_code", "dtype", "seed", "revision",
            "served_model_name", "host", "port", "master_address", "master_port",
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
            "distributed_executor_backend", "max_model_len", "max_num_seqs",
            "max_num_batched_tokens", "gpu_memory_utilization", "enable_prefix_caching",
            "enable_chunked_prefill",
        ]},
    }
    cli = build_cli(
        fields, registry=mocked, environment={"X": "1"}, working_directory=".", rank_placements=[]
    )
    assert cli.all_arguments_supported is False
    assert "tensor_parallel_size" in cli.unsupported_arguments
    assert "--tensor-parallel-size" not in cli.argv  # never silently included


def test_negative_unsupported_dtype():
    inputs = _base_preflight_inputs(dtype="int4_made_up", supported_dtypes=("auto", "float16"))
    result = run_preflight(inputs)
    assert "unsupported_dtype" in result.rejection_reasons


def test_negative_invalid_model_identifier():
    assert check_model_locally_resolvable("this/does-not-exist-anywhere-12345") is False


def test_negative_invalid_port():
    inputs = _base_preflight_inputs(port=99999)
    result = run_preflight(inputs)
    assert "invalid_port" in result.rejection_reasons


def test_negative_port_already_occupied():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    occupied_port = sock.getsockname()[1]
    try:
        assert check_port_available("127.0.0.1", occupied_port) is False
        inputs = _base_preflight_inputs(port=occupied_port, port_available=False)
        result = run_preflight(inputs)
        assert "port_already_occupied" in result.rejection_reasons
    finally:
        sock.close()


def test_negative_missing_vllm_installation():
    inputs = _base_preflight_inputs(vllm_installed=False)
    result = run_preflight(inputs)
    assert "vllm_not_installed" in result.rejection_reasons


def test_negative_malformed_distributed_plan_gap_in_shards():
    malformed = {
        "schema": "execution_plan", "schema_version": "2.0.0", "plan_id": "bad2",
        "provenance": {"compiler_tool": "t", "model_spec_ref": "", "capability_bundle": {}},
        "model_identity": {}, "global_decisions": {}, "function_plans": [],
        "distributed": {
            "strategy": "tensor_parallel", "world_size": 2, "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "ranks": [{"rank_id": 0, "logical_device": "d0"}, {"rank_id": 1, "logical_device": "d1"}],
            "tensor_shards": [
                {"tensor_id": "t", "partition_axis": 0, "partition_count": 2, "shard_index": 0,
                 "range_start": 0, "range_end": 400},
                {"tensor_id": "t", "partition_axis": 0, "partition_count": 2, "shard_index": 1,
                 "range_start": 450, "range_end": 896},  # gap between 400 and 450
            ],
            "collectives": [{"collective_id": "c0", "sequence_id": 0, "kind": "all_reduce",
                              "participants": [0, 1], "tensor_id": "t", "reduction": "sum"}],
        },
    }
    with pytest.raises(ExecutionPlanError, match="gap"):
        from deployment.execution_plan.loader import validate_execution_plan

        validate_execution_plan(malformed)


def test_negative_unknown_distributed_strategy(tmp_path):
    import json

    plan_dict = json.loads(TP2_PLAN_PATH.read_text())
    plan_dict["distributed"]["strategy"] = "pipeline_parallel_v9_made_up"
    bad_plan_path = tmp_path / "bad_strategy_plan.json"
    bad_plan_path.write_text(json.dumps(plan_dict))
    with pytest.raises(UnknownDistributedStrategyError):
        materialize_launch_spec(bad_plan_path, repo_root=REPO_ROOT)


def test_negative_operator_level_evidence_never_marked_whole_model_ready():
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.spec.whole_model_tp_evidence_status == (
        WholeModelTPEvidenceStatus.NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY.value
    )
    assert bundle.spec.execution_readiness_state != ExecutionReadinessState.EXECUTION_READY.value


def test_negative_tp2_never_silently_downgraded_to_tp1():
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.plan.distributed.tensor_parallel_size == 2
    assert bundle.spec.tensor_parallel_size == 2  # not silently downgraded to 1 despite 1-GPU host

    counters = compute_provenance_counters(
        plan_id=bundle.plan.plan_id,
        spec_source_execution_plan_id=bundle.spec.source_execution_plan_id,
        selected_candidate_id=bundle.selected_candidate_id,
        spec_source_candidate_id=bundle.spec.source_candidate_id,
        expected_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        spec_model=bundle.spec.model,
        plan_tensor_parallel_size=bundle.plan.distributed.tensor_parallel_size,
        spec_tensor_parallel_size=bundle.spec.tensor_parallel_size,
        plan_pipeline_parallel_size=bundle.plan.distributed.pipeline_parallel_size,
        spec_pipeline_parallel_size=bundle.spec.pipeline_parallel_size,
        plan_world_size=bundle.plan.distributed.world_size,
        spec_world_size=bundle.spec.world_size,
        plan_rank_ids=tuple(r.rank_id for r in bundle.plan.distributed.ranks),
        spec_rank_ids=tuple(p.rank_id for p in bundle.spec.rank_placements),
        unsupported_arguments=bundle.cli.unsupported_arguments,
        field_provenance=bundle.spec.field_provenance,
        execution_readiness_state=bundle.spec.execution_readiness_state,
        preflight_passed=bundle.preflight.passed,
        subprocess_launch_attempts_for_rejected_specs=0,
        tracked_pids_still_alive=(),
    )
    assert counters.silent_downgrade_count == 0
    assert counters.all_zero()


def test_negative_unsupported_executor_backend():
    inputs = _base_preflight_inputs(
        distributed_executor_backend="totally_bogus_backend",
        supported_executor_backends=("external_launcher", "mp", "ray", "uni"),
    )
    result = run_preflight(inputs)
    assert "unsupported_distributed_executor_backend" in result.rejection_reasons


def test_negative_incompatible_installed_vllm_version_via_mock_registry():
    real_registry = discover_argument_registry()
    incompatible = build_mock_registry_without(
        ["tensor_parallel_size", "pipeline_parallel_size", "distributed_executor_backend"],
        base_registry=real_registry,
    )
    assert incompatible["vllm_version"] != real_registry["vllm_version"]
    for dest in ("tensor_parallel_size", "pipeline_parallel_size", "distributed_executor_backend"):
        record = check_argument(dest, registry=incompatible, value_source="compiler_plan")
        assert record.installed_version_support_status == "unsupported"


def test_negative_attempted_launch_while_preflight_rejected_raises_provenance_bypass(monkeypatch):
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.preflight.passed is False

    counters = compute_provenance_counters(
        plan_id=bundle.plan.plan_id,
        spec_source_execution_plan_id=bundle.spec.source_execution_plan_id,
        selected_candidate_id=bundle.selected_candidate_id,
        spec_source_candidate_id=bundle.spec.source_candidate_id,
        expected_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        spec_model=bundle.spec.model,
        plan_tensor_parallel_size=bundle.plan.distributed.tensor_parallel_size,
        spec_tensor_parallel_size=bundle.spec.tensor_parallel_size,
        plan_pipeline_parallel_size=bundle.plan.distributed.pipeline_parallel_size,
        spec_pipeline_parallel_size=bundle.spec.pipeline_parallel_size,
        plan_world_size=bundle.plan.distributed.world_size,
        spec_world_size=bundle.spec.world_size,
        plan_rank_ids=tuple(r.rank_id for r in bundle.plan.distributed.ranks),
        spec_rank_ids=tuple(p.rank_id for p in bundle.spec.rank_placements),
        unsupported_arguments=bundle.cli.unsupported_arguments,
        field_provenance=bundle.spec.field_provenance,
        # Simulate a hypothetical bypass bug for the counter's own detection logic only.
        execution_readiness_state="EXECUTION_READY",
        preflight_passed=bundle.preflight.passed,
        subprocess_launch_attempts_for_rejected_specs=0,
        tracked_pids_still_alive=(),
    )
    assert counters.preflight_bypass_count > 0  # the counter correctly flags this as a bypass

    # And the real pipeline never actually produces that state:
    assert bundle.spec.execution_readiness_state == ExecutionReadinessState.PREFLIGHT_REJECTED.value


def test_no_force_or_bypass_parameter_exists_anywhere_on_the_adapter():
    import inspect

    from deployment.vllm_adapter.backend_adapter import VLLMDistributedAdapter
    from deployment.vllm_adapter import distributed_materializer, distributed_preflight

    for obj in (VLLMDistributedAdapter.materialize_from_execution_plan, materialize_launch_spec):
        sig = inspect.signature(obj)
        for name in sig.parameters:
            assert "force" not in name.lower()
            assert "ignore_preflight" not in name.lower()
            assert "bypass" not in name.lower()
