"""D1: Compiler-Planned TP=2 Multi-Process Simulation -- focused tests.

Covers Part L requirements: runtime schema (load/reject/no-downgrade/legacy/
real-vLLM-adapter-reject), process runtime (distinct OS processes, clean
shutdown, exception propagation, no orphans, shard isolation), collective
(all_reduce correctness, missing-participant timeout, duplicate/wrong-
sequence/wrong-shape rejection), and the full end-to-end TP2 slice.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deployment.execution_plan.loader import ExecutionPlanError, load_execution_plan, parse_execution_plan
from deployment.execution_plan.schema import (
    DistributedCollectiveStep,
    DistributedPlan,
    DistributedRankPlacement,
    DistributedTensorShard,
)
from deployment.tp_process_runtime import (
    CollectiveCoordinator,
    DistributedProcessRuntime,
    DistributedRuntimeError,
    serial_matmul_reference,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "runtime_paths" / "distributed_d1_tp2_multiprocess"
TP1_PLAN_PATH = RESULTS_DIR / "compiler_tp1_plan.json"
TP2_PLAN_PATH = RESULTS_DIR / "compiler_tp2_plan.json"

pytestmark = pytest.mark.skipif(
    not TP2_PLAN_PATH.exists(),
    reason="requires the compiler-exported D1 TP2 plan artifact; run "
           "ml-graph-compiler-runtime's emit-distributed-execution-plan first",
)


def _make_problem(seed: int = 0, m: int = 4, k: int = 16, n: int = 4):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2, 2, size=(m, k))
    b = rng.uniform(-2, 2, size=(k, n))
    return a, b


def _tp2_plan() -> DistributedPlan:
    plan = load_execution_plan(TP2_PLAN_PATH)
    assert plan.distributed is not None
    return plan.distributed


# ---------------------------------------------------------------------------
# Runtime schema / loader
# ---------------------------------------------------------------------------

def test_load_valid_tp2_plan():
    plan = load_execution_plan(TP2_PLAN_PATH)
    assert plan.distributed is not None
    assert plan.distributed.world_size == 2
    assert plan.distributed.tensor_parallel_size == 2
    assert len(plan.distributed.ranks) == 2
    assert len(plan.distributed.collectives) == 1


def test_reject_malformed_tp2_plan_unknown_collective_kind():
    import json
    payload = json.loads(TP2_PLAN_PATH.read_text())
    payload["distributed"]["collectives"][0]["kind"] = "reduce_scatter"
    with pytest.raises(ExecutionPlanError, match="unknown or unsupported kind"):
        parse_execution_plan(payload)


def test_reject_silent_downgrade_missing_rank():
    """A TP2-declared plan (world_size=2) with only one rank present must be
    rejected outright -- the loader must never quietly execute it as TP1."""
    import json
    payload = json.loads(TP2_PLAN_PATH.read_text())
    payload["distributed"]["ranks"].pop()
    with pytest.raises(ExecutionPlanError, match="contiguous set"):
        parse_execution_plan(payload)


def test_preserve_legacy_tp1_plan_behavior():
    plan = load_execution_plan(TP1_PLAN_PATH)
    assert plan.distributed is None
    # A pre-D1 plan with no "distributed" key at all must parse identically.
    import json
    payload = json.loads(TP1_PLAN_PATH.read_text())
    assert "distributed" not in payload
    parsed = parse_execution_plan(payload)
    assert parsed.distributed is None


def test_reject_tp2_on_real_vllm_adapter_path():
    """The pre-existing real-vLLM adapter path must keep rejecting
    tensor_parallel_size != 1; D1 does not lift this rejection."""
    from deployment.vllm_adapter.plan_schema import validate_vllm_execution_plan

    base = {
        "artifact_type": "vllm_execution_plan",
        "schema_version": "1.0.0",
        "truth_boundary": "Execution planning artifact only; not measured performance.",
        "source_artifacts": ["x"],
        "model": {"model_id": "m", "tokenizer": "t", "dtype": "fp16", "quantization": "none",
                   "trust_remote_code": False},
        "hardware_profile": {"gpu_name": "g", "vram_gb": 4},
        "backend_profile": {"backend": "vllm"},
        "batch_policy": {"max_num_seqs": 1, "max_num_batched_tokens": 1, "enable_chunked_prefill": False},
        "prefix_policy": {"enable_prefix_caching": False},
        "memory_policy": {"gpu_memory_utilization": 0.5, "max_model_len": 1, "block_size": 1, "swap_space": 0},
        "quantization_policy": {"dtype": "fp16", "quantization": "none"},
        "speculative_policy": {"enabled": False},
        "runtime_config": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                            "served_model_name": "m"},
    }
    with pytest.raises(Exception, match="tensor_parallel_size"):
        validate_vllm_execution_plan(base)


# ---------------------------------------------------------------------------
# Process runtime
# ---------------------------------------------------------------------------

def test_two_distinct_os_processes_start_and_shutdown_cleanly():
    a, b = _make_problem(seed=1)
    rt = DistributedProcessRuntime()
    result = rt.run(_tp2_plan(), a, b)
    pids = [p.pid for p in result.processes.values()]
    assert len(pids) == 2
    assert len(set(pids)) == 2, "expected two distinct OS process IDs"
    assert result.status == "completed"
    for p in result.processes.values():
        assert p.exitcode == 0
        assert p.alive_after_teardown is False
    assert result.provenance["orphan_process_count"] == 0


def test_no_orphan_processes_after_run():
    a, b = _make_problem(seed=2)
    rt = DistributedProcessRuntime()
    result = rt.run(_tp2_plan(), a, b)
    for p in result.processes.values():
        assert p.pid is not None
        with pytest.raises(ProcessLookupError):
            os.kill(p.pid, 0)


def test_rank_local_shard_isolation():
    """Each rank must only ever report a K-slice shape, never the full K."""
    a, b = _make_problem(seed=3, k=16)
    rt = DistributedProcessRuntime()
    result = rt.run(_tp2_plan(), a, b)
    shard_events = [e for e in result.trace.events if e.get("event") == "shard_received"]
    assert len(shard_events) == 2
    for ev in shard_events:
        assert tuple(ev["a_shape"])[-1] == 8  # half of K=16, never the full 16
        assert tuple(ev["b_shape"])[0] == 8


def test_child_exception_propagates_to_parent():
    """A plan whose declared shard coverage doesn't match the real problem
    size must fail closed in the parent before any child ever misbehaves."""
    plan = _tp2_plan()
    a, b = _make_problem(seed=4, k=10)  # 10 is not what the TP2 plan's shards declare (16)
    rt = DistributedProcessRuntime()
    with pytest.raises(DistributedRuntimeError, match="shard coverage"):
        rt.run(plan, a, b)


# ---------------------------------------------------------------------------
# Collective
# ---------------------------------------------------------------------------

def test_all_reduce_correctness_against_serial_reference():
    a, b = _make_problem(seed=5)
    rt = DistributedProcessRuntime()
    result = rt.run(_tp2_plan(), a, b)
    ref = serial_matmul_reference(a, b)
    assert np.allclose(result.distributed_output, ref, rtol=1e-9, atol=1e-9)
    assert result.distributed_output.shape == ref.shape


def test_all_reduce_correctness_nontrivial_seeded_case():
    a, b = _make_problem(seed=12345, m=6, k=32, n=5)
    plan = DistributedPlan(
        strategy="tensor_parallel", world_size=2, tensor_parallel_size=2,
        pipeline_parallel_size=1,
        ranks=(DistributedRankPlacement(0, "simulated_cpu_process_0"),
               DistributedRankPlacement(1, "simulated_cpu_process_1")),
        tensor_shards=(
            DistributedTensorShard("partial_output", 0, 2, 0, 0, 16),
            DistributedTensorShard("partial_output", 0, 2, 1, 16, 32),
        ),
        collectives=(DistributedCollectiveStep("all_reduce_0", 0, "all_reduce", (0, 1),
                                                "partial_output", "sum"),),
        truth_boundary="test",
    )
    rt = DistributedProcessRuntime()
    result = rt.run(plan, a, b)
    ref = serial_matmul_reference(a, b)
    assert np.allclose(result.distributed_output, ref, rtol=1e-9, atol=1e-9)
    assert result.provenance == {
        "rank_mismatch_count": 0, "missing_rank_count": 0, "unexpected_rank_count": 0,
        "shard_mismatch_count": 0, "collective_sequence_mismatch_count": 0,
        "missing_collective_participant_count": 0, "unexpected_collective_participant_count": 0,
        "fallback_count": 0, "silent_downgrade_count": 0, "orphan_process_count": 0,
    }


def test_missing_participant_timeout_is_real_deadlock_detection():
    import time
    a, b = _make_problem(seed=6)
    rt = DistributedProcessRuntime()
    t0 = time.time()
    result = rt.run(_tp2_plan(), a, b, collective_timeout_s=1.5, force_skip_collective_rank=1)
    elapsed = time.time() - t0
    assert result.status == "timeout"
    assert elapsed >= 1.4, "timeout must be a real wall-clock wait, not instantaneous"
    assert result.deadlock is not None
    assert result.deadlock["missing_ranks"] == [1]
    assert result.provenance["missing_collective_participant_count"] == 1
    for p in result.processes.values():
        assert p.alive_after_teardown is False
        with pytest.raises(ProcessLookupError):
            os.kill(p.pid, 0)


def test_duplicate_participant_rejection():
    coordinator = CollectiveCoordinator()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from deployment.tp_process_runtime.messages import array_to_payload
    for rank_id in (0, 0):  # rank 0 contributes twice
        q.put({"type": "contribution", "rank_id": rank_id, "collective_id": "c0",
               "sequence_id": 0, "tensor_id": "t", "ts": 0.0,
               **array_to_payload(np.ones((2, 2)))})
    q.put({"type": "contribution", "rank_id": 1, "collective_id": "c0", "sequence_id": 0,
           "tensor_id": "t", "ts": 0.0, **array_to_payload(np.ones((2, 2)))})
    outcome = coordinator.run_all_reduce_sum(
        collective_id="c0", sequence_id=0, expected_ranks={0, 1},
        from_rank_queue=q, timeout_s=2.0,
    )
    assert outcome.status == "completed"
    assert len(outcome.duplicate_events) == 1


def test_wrong_sequence_rejection():
    coordinator = CollectiveCoordinator()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from deployment.tp_process_runtime.messages import array_to_payload
    q.put({"type": "contribution", "rank_id": 0, "collective_id": "c0", "sequence_id": 99,
           "tensor_id": "t", "ts": 0.0, **array_to_payload(np.ones((2, 2)))})
    outcome = coordinator.run_all_reduce_sum(
        collective_id="c0", sequence_id=0, expected_ranks={0}, from_rank_queue=q, timeout_s=1.0,
    )
    assert outcome.status == "timeout"
    assert len(outcome.sequence_mismatch_events) == 1


def test_wrong_tensor_shape_rejection():
    coordinator = CollectiveCoordinator()
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    from deployment.tp_process_runtime.messages import array_to_payload
    q.put({"type": "contribution", "rank_id": 0, "collective_id": "c0", "sequence_id": 0,
           "tensor_id": "t", "ts": 0.0, **array_to_payload(np.ones((2, 2)))})
    q.put({"type": "contribution", "rank_id": 1, "collective_id": "c0", "sequence_id": 0,
           "tensor_id": "t", "ts": 0.0, **array_to_payload(np.ones((3, 3)))})
    outcome = coordinator.run_all_reduce_sum(
        collective_id="c0", sequence_id=0, expected_ranks={0, 1}, from_rank_queue=q, timeout_s=2.0,
    )
    assert outcome.status == "shape_mismatch"


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_compiler_plan_to_verified_distributed_output():
    plan = load_execution_plan(TP2_PLAN_PATH)
    a, b = _make_problem(seed=999, m=4, k=16, n=4)
    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, a, b)

    assert result.status == "completed"
    assert result.all_ranks_completed
    assert result.all_collectives_completed

    ref = serial_matmul_reference(a, b)
    assert np.allclose(result.distributed_output, ref, rtol=1e-9, atol=1e-9)

    for key, value in result.provenance.items():
        assert value == 0, f"provenance counter {key} must be zero, got {value}"
