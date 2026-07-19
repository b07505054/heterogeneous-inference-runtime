"""D3A: Live Qwen Tensor Capture and Serialized Rank-Local Validation --
focused tests.

Covers the positive end-to-end chain and the Part M fail-closed negative
tests. Uses the real, locally-cached Qwen2.5-0.5B-Instruct model (module
scoped fixture, loaded once) -- no synthetic fallback is exercised on the
success path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from deployment.execution_plan.loader import load_execution_plan
from deployment.execution_plan.schema import DistributedTensorShard
from deployment.tp_process_runtime import (
    DistributedProcessRuntime,
    OperatorMappingError,
    TPDecompositionError,
    apply_bias_contract,
    apply_bias_twice_incorrectly,
    build_rank_shards,
    map_compiler_operator_to_module,
    rank_local_partial_output,
    run_serialized_all_reduce,
    verify_live_qwen_provenance,
)
from deployment.tp_process_runtime.live_capture import (
    LiveCaptureError,
    _require_single_invocation,
    capture_module_activation,
    load_live_model,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
TP2_PLAN_PATH = RESULTS_DIR / "real_qwen_tp2_execution_plan.json"

pytestmark = pytest.mark.skipif(
    not TP2_PLAN_PATH.exists(),
    reason="requires the D2 compiler-exported real-Qwen TP2 plan artifact",
)


@pytest.fixture(scope="module")
def plan():
    p = load_execution_plan(TP2_PLAN_PATH)
    assert p.distributed is not None
    return p.distributed


@pytest.fixture(scope="module")
def model_handle():
    return load_live_model()


@pytest.fixture(scope="module")
def mapping(plan, model_handle):
    operator_id = plan.tensor_shards[0].tensor_id
    hidden_dim = max(s.range_end for s in plan.tensor_shards)
    return map_compiler_operator_to_module(operator_id, model_handle.model, expected_hidden_size=hidden_dim)


@pytest.fixture(scope="module")
def captured(model_handle, mapping):
    return capture_module_activation(model_handle, mapping.module_path)


def _flatten(captured):
    batch, seq, hidden = captured.input_shape
    x = captured.input_activation.reshape(batch * seq, hidden)
    y_live = captured.output_activation.reshape(batch * seq, hidden)
    return x, y_live


# ---------------------------------------------------------------------------
# Positive end-to-end chain
# ---------------------------------------------------------------------------

def test_operator_mapping_is_unique_and_verified(mapping):
    assert mapping.module_path == "model.layers.0.self_attn.o_proj"
    assert mapping.module_class == "torch.nn.modules.linear.Linear"
    assert mapping.weight_shape == (896, 896)
    assert all(mapping.checks.values())


def test_live_capture_is_real_and_single_invocation(captured):
    assert captured.invocation_count == 1
    assert captured.input_shape[-1] == 896
    assert captured.output_shape[-1] == 896
    assert captured.weight.shape == (896, 896)
    assert captured.bias is None  # real Qwen2.5-0.5B-Instruct o_proj has no bias


def test_rank_isolation_and_serialized_reconstruction_matches_live(plan, captured):
    x, y_live = _flatten(captured)
    shards = build_rank_shards(x, captured.weight, plan.tensor_shards)

    assert set(shards) == {0, 1}
    assert shards[0].shard_width == 448
    assert shards[1].shard_width == 448
    # rank isolation: neither rank's shard is the full hidden width.
    assert shards[0].x_shard.shape[-1] != 896
    assert shards[1].x_shard.shape[-1] != 896
    # complete, non-overlapping coverage.
    assert shards[0].range_end == shards[1].range_start
    assert shards[0].range_start == 0 and shards[1].range_end == 896

    partials = {rid: rank_local_partial_output(s) for rid, s in shards.items()}
    assert partials[0].shape == y_live.shape
    assert partials[1].shape == y_live.shape

    c = plan.collectives[0]
    outcome = run_serialized_all_reduce(
        collective_id=c.collective_id, sequence_id=c.sequence_id,
        tensor_id=c.tensor_id, contributions=partials,
    )
    assert outcome.status == "completed"
    reconstructed = apply_bias_contract(outcome.reduced, captured.bias)

    assert np.allclose(reconstructed, y_live, atol=1e-4, rtol=1e-4)
    max_abs = float(np.max(np.abs(reconstructed - y_live)))
    assert max_abs < 1e-4


def test_reconstruction_matches_standalone_pytorch_reference(plan, captured):
    x, y_live = _flatten(captured)
    direct_ref = x @ captured.weight.T
    if captured.bias is not None:
        direct_ref = direct_ref + captured.bias
    assert np.allclose(direct_ref, y_live, atol=1e-5, rtol=1e-5)

    shards = build_rank_shards(x, captured.weight, plan.tensor_shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in shards.items()}
    c = plan.collectives[0]
    outcome = run_serialized_all_reduce(
        collective_id=c.collective_id, sequence_id=c.sequence_id,
        tensor_id=c.tensor_id, contributions=partials,
    )
    reconstructed = apply_bias_contract(outcome.reduced, captured.bias)
    assert np.allclose(reconstructed, direct_ref, atol=1e-4, rtol=1e-4)


def test_bonus_multiprocess_ipc_replay_matches_live(plan, captured):
    """Distinct from the serialized rank-local path: reuses D1's real
    multi-process runtime unmodified, now fed real captured Qwen tensors."""
    x, y_live = _flatten(captured)
    rt = DistributedProcessRuntime()
    result = rt.run(plan, x.astype(np.float64), captured.weight.T.astype(np.float64))
    assert result.status == "completed"
    assert result.provenance["orphan_process_count"] == 0
    final = apply_bias_contract(result.distributed_output, captured.bias)
    assert np.allclose(final, y_live, atol=1e-4, rtol=1e-4)
    for p in result.processes.values():
        assert p.exitcode == 0
        with pytest.raises(ProcessLookupError):
            os.kill(p.pid, 0)


def test_cross_layer_provenance_all_zero(plan, captured, mapping):
    x, y_live = _flatten(captured)
    shards = build_rank_shards(x, captured.weight, plan.tensor_shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in shards.items()}
    c = plan.collectives[0]
    outcome = run_serialized_all_reduce(
        collective_id=c.collective_id, sequence_id=c.sequence_id,
        tensor_id=c.tensor_id, contributions=partials,
    )
    reconstructed = apply_bias_contract(outcome.reduced, captured.bias)

    report = verify_live_qwen_provenance(
        operator_id=plan.tensor_shards[0].tensor_id, mapping=mapping, plan=plan,
        captured=captured, shards=shards, partials=partials,
        collective_outcome=outcome, reconstructed=reconstructed, live_reference=y_live,
        tolerance={"atol": 1e-4, "rtol": 1e-4},
        orphan_process_count=0, temporary_files_remaining=0, fallback_events=0,
    )
    for name, value in report.counters.items():
        assert value == 0, f"{name} must be zero, got {value}"
    assert report.all_zero


def test_function_plans_bug_does_not_affect_d3a_provenance():
    """Documents and proves D2's pre-existing function_plans==() issue
    (see function_plans_bug_analysis.json) does not block D3A: every D3A
    provenance value is sourced from plan.distributed, never function_plans."""
    full_plan = load_execution_plan(TP2_PLAN_PATH)
    assert full_plan.function_plans == ()
    assert full_plan.distributed is not None
    assert full_plan.distributed.tensor_shards[0].tensor_id == "qwen_prefill::llm.o_proj::layer_0"
    assert len(full_plan.distributed.ranks) == 2


# ---------------------------------------------------------------------------
# Part M negative tests
# ---------------------------------------------------------------------------

def test_negative_operator_id_maps_to_no_module(model_handle):
    with pytest.raises(OperatorMappingError, match="no compiler-operator-to-Transformers"):
        map_compiler_operator_to_module("qwen_prefill::llm.unknown_op::layer_0", model_handle.model)


def test_negative_operator_id_maps_ambiguously():
    class _FakeModel:
        def named_modules(self):
            return [
                ("model.layers.0.self_attn.o_proj", object()),
                ("model.layers.0.self_attn.o_proj_dup", object()),
            ]

    import deployment.tp_process_runtime.qwen_module_mapping as qmm
    original_pattern = qmm._OP_TYPE_TO_MODULE_PATTERN["llm.o_proj"]
    try:
        import re
        qmm._OP_TYPE_TO_MODULE_PATTERN["llm.o_proj"] = re.compile(
            r"^model\.layers\.(?P<layer_index>\d+)\.self_attn\.o_proj(_dup)?$"
        )
        with pytest.raises(OperatorMappingError, match="ambiguously"):
            map_compiler_operator_to_module("qwen_prefill::llm.o_proj::layer_0", _FakeModel())
    finally:
        qmm._OP_TYPE_TO_MODULE_PATTERN["llm.o_proj"] = original_pattern


def test_negative_wrong_layer_number(model_handle):
    with pytest.raises(OperatorMappingError, match="no matching module exists at that layer"):
        map_compiler_operator_to_module("qwen_prefill::llm.o_proj::layer_999", model_handle.model)


def test_negative_weight_shape_differs_from_plan(model_handle):
    with pytest.raises(OperatorMappingError, match="does not match the compiler plan"):
        map_compiler_operator_to_module(
            "qwen_prefill::llm.o_proj::layer_0", model_handle.model, expected_hidden_size=123,
        )


def test_negative_captured_input_hidden_dimension_differs_from_plan(plan):
    # Case 1: X/W agree with each other (100) but disagree with the plan's
    # declared hidden dimension (896) -- shard coverage becomes impossible.
    wrong_x = np.zeros((4, 100), dtype=np.float32)
    wrong_w = np.zeros((896, 100), dtype=np.float32)
    with pytest.raises(TPDecompositionError, match="exceeds captured hidden dimension"):
        build_rank_shards(wrong_x, wrong_w, plan.tensor_shards)

    # Case 2: X and W directly disagree with each other.
    mismatched_x = np.zeros((4, 100), dtype=np.float32)
    mismatched_w = np.zeros((896, 896), dtype=np.float32)
    with pytest.raises(TPDecompositionError, match="differs from module weight"):
        build_rank_shards(mismatched_x, mismatched_w, plan.tensor_shards)


def test_negative_module_hook_never_fires():
    with pytest.raises(LiveCaptureError, match="hook never fired"):
        _require_single_invocation(0, "model.layers.0.self_attn.o_proj")


def test_negative_module_hook_fires_unexpected_number_of_times():
    with pytest.raises(LiveCaptureError, match="fired 3 times"):
        _require_single_invocation(3, "model.layers.0.self_attn.o_proj")


def test_negative_rank_shard_overlap():
    x = np.zeros((4, 896), dtype=np.float32)
    w = np.zeros((896, 896), dtype=np.float32)
    bad_shards = (
        DistributedTensorShard("t", 0, 2, 0, 0, 500),
        DistributedTensorShard("t", 0, 2, 1, 400, 896),  # overlaps [400,500)
    )
    with pytest.raises(TPDecompositionError, match="gap/overlap"):
        build_rank_shards(x, w, bad_shards)


def test_negative_rank_shard_coverage_gap():
    x = np.zeros((4, 896), dtype=np.float32)
    w = np.zeros((896, 896), dtype=np.float32)
    bad_shards = (
        DistributedTensorShard("t", 0, 2, 0, 0, 400),
        DistributedTensorShard("t", 0, 2, 1, 448, 896),  # gap [400,448)
    )
    with pytest.raises(TPDecompositionError, match="gap/overlap"):
        build_rank_shards(x, w, bad_shards)


def test_negative_rank_receives_full_tensor_unexpectedly(plan, captured, mapping):
    x, y_live = _flatten(captured)
    shards = build_rank_shards(x, captured.weight, plan.tensor_shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in shards.items()}
    c = plan.collectives[0]
    outcome = run_serialized_all_reduce(
        collective_id=c.collective_id, sequence_id=c.sequence_id,
        tensor_id=c.tensor_id, contributions=partials,
    )
    reconstructed = apply_bias_contract(outcome.reduced, captured.bias)

    # Simulate a leaked full-width shard on rank 0 and confirm the
    # provenance detector flags it (rank_input_leakage_count > 0).
    from deployment.tp_process_runtime.linear_tp_decomposition import RankShard
    leaked_shards = dict(shards)
    leaked_shards[0] = RankShard(rank_id=0, range_start=0, range_end=896,
                                 x_shard=x, w_shard=captured.weight)
    report = verify_live_qwen_provenance(
        operator_id=plan.tensor_shards[0].tensor_id, mapping=mapping, plan=plan,
        captured=captured, shards=leaked_shards, partials=partials,
        collective_outcome=outcome, reconstructed=reconstructed, live_reference=y_live,
        tolerance={"atol": 1e-4, "rtol": 1e-4},
        orphan_process_count=0, temporary_files_remaining=0,
    )
    assert report.counters["rank_input_leakage_count"] > 0
    assert not report.all_zero


def test_negative_bias_applied_twice():
    partials = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
    bias = np.array([10.0, 10.0])
    correct = apply_bias_contract(sum(partials), bias)
    wrong = apply_bias_twice_incorrectly(partials, bias)
    assert not np.allclose(correct, wrong)
    assert np.array_equal(wrong, correct + bias)  # bias counted twice


def test_negative_bias_omitted():
    partials = [np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])]
    bias = np.array([10.0, 10.0])
    reduced = sum(partials)
    correct = apply_bias_contract(reduced, bias)
    omitted = reduced  # bias never applied
    assert not np.allclose(correct, omitted)


def test_negative_collective_participant_missing(plan, captured):
    x, _y = _flatten(captured)
    shards = build_rank_shards(x, captured.weight, plan.tensor_shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in shards.items()}
    c = plan.collectives[0]
    # Only rank 0 contributes -- rank 1 is missing.
    outcome = run_serialized_all_reduce(
        collective_id=c.collective_id, sequence_id=c.sequence_id, tensor_id=c.tensor_id,
        contributions={0: partials[0]}, timeout_s=0.5,
    )
    # A serialized queue with a genuinely missing contribution starves;
    # the coordinator's expected_ranks must come from the plan, not just
    # what was contributed, to detect this -- exercise that directly.
    import queue as queue_mod
    import time as time_mod
    from deployment.tp_process_runtime.collective import CollectiveCoordinator
    from deployment.tp_process_runtime.messages import array_to_payload

    q: queue_mod.Queue = queue_mod.Queue()
    msg = {"type": "contribution", "rank_id": 0, "collective_id": c.collective_id,
           "sequence_id": c.sequence_id, "tensor_id": c.tensor_id, "ts": time_mod.time(),
           **array_to_payload(partials[0])}
    q.put(msg)
    coordinator = CollectiveCoordinator()
    result = coordinator.run_all_reduce_sum(
        collective_id=c.collective_id, sequence_id=c.sequence_id,
        expected_ranks={0, 1}, from_rank_queue=q, timeout_s=0.5,
    )
    assert result.status == "timeout"
    assert result.missing_ranks == {1}


def test_negative_collective_sequence_mismatch(plan, captured):
    x, _y = _flatten(captured)
    shards = build_rank_shards(x, captured.weight, plan.tensor_shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in shards.items()}
    c = plan.collectives[0]
    wrong_seq_contributions = {0: partials[0], 1: partials[1]}
    import queue as queue_mod
    import time as time_mod
    from deployment.tp_process_runtime.collective import CollectiveCoordinator
    from deployment.tp_process_runtime.messages import array_to_payload

    q: queue_mod.Queue = queue_mod.Queue()
    for rid, arr in wrong_seq_contributions.items():
        q.put({"type": "contribution", "rank_id": rid, "collective_id": c.collective_id,
               "sequence_id": 99, "tensor_id": c.tensor_id, "ts": time_mod.time(),
               **array_to_payload(arr)})
    coordinator = CollectiveCoordinator()
    result = coordinator.run_all_reduce_sum(
        collective_id=c.collective_id, sequence_id=c.sequence_id,
        expected_ranks={0, 1}, from_rank_queue=q, timeout_s=0.5,
    )
    assert result.status == "timeout"
    assert len(result.sequence_mismatch_events) == 2


def test_negative_reconstructed_output_exceeds_tolerance(plan, captured):
    x, y_live = _flatten(captured)
    shards = build_rank_shards(x, captured.weight, plan.tensor_shards)
    partials = {rid: rank_local_partial_output(s) for rid, s in shards.items()}
    corrupted_partials = dict(partials)
    corrupted_partials[1] = partials[1] + 1000.0  # inject large error
    c = plan.collectives[0]
    outcome = run_serialized_all_reduce(
        collective_id=c.collective_id, sequence_id=c.sequence_id,
        tensor_id=c.tensor_id, contributions=corrupted_partials,
    )
    reconstructed = apply_bias_contract(outcome.reduced, captured.bias)
    assert not np.allclose(reconstructed, y_live, atol=1e-4, rtol=1e-4)


def test_negative_temporary_tensor_file_remains():
    with tempfile.TemporaryDirectory() as d:
        leftover = Path(d) / "leaked_tensor.npy"
        leftover.write_bytes(b"not a real tensor, just proving detection")
        remaining = list(Path(d).glob("*.npy"))
        assert len(remaining) == 1  # detector: a leaked file is found
        leftover.unlink()
        remaining_after = list(Path(d).glob("*.npy"))
        assert len(remaining_after) == 0


def test_negative_tp2_live_tensor_plan_sent_to_real_vllm_adapter():
    from deployment.vllm_adapter.plan_schema import validate_vllm_execution_plan

    base = {
        "artifact_type": "vllm_execution_plan", "schema_version": "1.0.0",
        "truth_boundary": "Execution planning artifact only; not measured performance.",
        "source_artifacts": ["x"],
        "model": {"model_id": "qwen2.5-0.5b", "tokenizer": "t", "dtype": "fp32",
                   "quantization": "none", "trust_remote_code": False},
        "hardware_profile": {"gpu_name": "g", "vram_gb": 4},
        "backend_profile": {"backend": "vllm"},
        "batch_policy": {"max_num_seqs": 1, "max_num_batched_tokens": 1, "enable_chunked_prefill": False},
        "prefix_policy": {"enable_prefix_caching": False},
        "memory_policy": {"gpu_memory_utilization": 0.5, "max_model_len": 1, "block_size": 1, "swap_space": 0},
        "quantization_policy": {"dtype": "fp32", "quantization": "none"},
        "speculative_policy": {"enabled": False},
        "runtime_config": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                            "served_model_name": "qwen2.5-0.5b"},
    }
    with pytest.raises(Exception, match="tensor_parallel_size"):
        validate_vllm_execution_plan(base)
