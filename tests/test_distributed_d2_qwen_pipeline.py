"""D2: Real Qwen Pipeline Distributed Strategy Planning -- focused tests.

Covers Part L runtime tests and Part J negative tests that are specific to
D2 (real-Qwen-plan consumption, cross-layer provenance, operator/shape
mismatch rejection). Compiler-side D2 negative tests (non-divisible Qwen
dimension, unsupported operator, missing shape metadata, no distributed
capability) live in ml-graph-compiler-runtime's
DistributedStrategyPlanningTest.cpp and
RunDistributedStrategyPlanningPipelineTest.cmake.
"""

from __future__ import annotations

import os
import copy
from pathlib import Path

import numpy as np
import pytest

from deployment.execution_plan.loader import ExecutionPlanError, load_execution_plan, parse_execution_plan
from deployment.tp_process_runtime import (
    DistributedProcessRuntime,
    DistributedRuntimeError,
    build_qwen_derived_workload,
    serial_matmul_reference,
    verify_cross_layer_provenance,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
TP1_PLAN_PATH = RESULTS_DIR / "real_qwen_tp1_execution_plan.json"
TP2_PLAN_PATH = RESULTS_DIR / "real_qwen_tp2_execution_plan.json"

pytestmark = pytest.mark.skipif(
    not TP2_PLAN_PATH.exists(),
    reason="requires the compiler-exported D2 real-Qwen TP2 plan artifact; run "
           "ml-graph-compiler-runtime's compile-for-target against the real "
           "per-layer Qwen ONNX graph with the D2 opt-in profile first",
)


def _tp2_plan():
    plan = load_execution_plan(TP2_PLAN_PATH)
    assert plan.distributed is not None
    return plan


def test_load_real_qwen_tp2_plan():
    plan = _tp2_plan()
    assert plan.model_identity["model_id"] == "qwen2.5-0.5b"
    assert plan.model_identity["hidden_size"] == 896
    assert plan.distributed.world_size == 2
    assert plan.distributed.tensor_parallel_size == 2
    operator_id = plan.distributed.tensor_shards[0].tensor_id
    assert "llm.o_proj" in operator_id
    assert "qwen_prefill" in operator_id


def test_real_qwen_tp1_plan_has_no_distributed_capability_disabled():
    """The non-opt-in profile's exported plan must carry no distributed
    section at all -- proves the pass is a no-op (not merely 'selected TP1
    silently') when distributed capability was never declared."""
    plan = load_execution_plan(TP1_PLAN_PATH)
    assert plan.distributed is None


def test_materialize_exact_planned_ranks():
    plan = _tp2_plan()
    workload = build_qwen_derived_workload(plan.distributed)
    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, workload.a, workload.b)
    assert result.status == "completed"
    pids = [p.pid for p in result.processes.values()]
    assert len(pids) == plan.distributed.world_size == 2
    assert len(set(pids)) == 2
    assert set(result.processes.keys()) == {r.rank_id for r in plan.distributed.ranks}
    for p in result.processes.values():
        assert p.exitcode == 0
        assert p.alive_after_teardown is False


def test_execute_qwen_derived_operator_dimensions():
    plan = _tp2_plan()
    workload = build_qwen_derived_workload(plan.distributed)
    assert workload.hidden_dim == 896
    shard_widths = sorted(s.range_end - s.range_start for s in plan.distributed.tensor_shards)
    assert shard_widths == [448, 448]

    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, workload.a, workload.b)
    shard_events = [e for e in result.trace.events if e.get("event") == "shard_received"]
    assert len(shard_events) == 2
    for ev in shard_events:
        assert tuple(ev["a_shape"])[-1] == 448  # half of real hidden_size=896, never the full 896


def test_collective_and_serial_reference_correctness():
    plan = _tp2_plan()
    workload = build_qwen_derived_workload(plan.distributed, seed=4242)
    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, workload.a, workload.b)
    assert result.status == "completed"
    assert result.all_collectives_completed

    ref = serial_matmul_reference(workload.a, workload.b)
    assert np.allclose(result.distributed_output, ref, rtol=1e-9, atol=1e-9)
    assert result.distributed_output.shape == (workload.sequence_length, workload.hidden_dim)

    for key, value in result.provenance.items():
        assert value == 0, f"provenance counter {key} must be zero, got {value}"


def test_cross_layer_provenance_complete_and_matches():
    plan = _tp2_plan()
    workload = build_qwen_derived_workload(plan.distributed, seed=99)
    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, workload.a, workload.b)

    report = verify_cross_layer_provenance(plan.distributed, result, workload.hidden_dim)
    assert report.operator_id_match
    assert report.world_size_match
    assert report.rank_ids_match
    assert report.shard_ranges_match
    assert report.collective_id_match
    assert report.sequence_id_match
    assert report.participant_set_match
    assert report.no_silent_downgrade
    assert report.no_synthetic_fallback_dimensions
    assert report.all_match
    assert report.mismatch_count == 0


def test_zero_orphan_processes_after_real_qwen_run():
    plan = _tp2_plan()
    workload = build_qwen_derived_workload(plan.distributed)
    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, workload.a, workload.b)
    assert result.provenance["orphan_process_count"] == 0
    for p in result.processes.values():
        assert p.pid is not None
        with pytest.raises(ProcessLookupError):
            os.kill(p.pid, 0)


# ---------------------------------------------------------------------------
# Part J negative tests (D2-specific)
# ---------------------------------------------------------------------------

def test_reject_collective_referencing_unknown_operator_tensor():
    plan = _tp2_plan()
    workload = build_qwen_derived_workload(plan.distributed)
    tampered = copy.deepcopy(plan.distributed)
    bad_collective = tampered.collectives[0]
    object.__setattr__(bad_collective, "tensor_id", "unknown::operator::not_in_plan")
    tampered = tampered.__class__(**{**tampered.__dict__, "collectives": (bad_collective,)})

    rt = DistributedProcessRuntime()
    with pytest.raises(DistributedRuntimeError, match="unknown tensor_id"):
        rt.run(tampered, workload.a, workload.b)


def test_runtime_rejects_dimensions_differing_from_compiler_plan():
    """The real Qwen plan declares hidden_size=896 (448/448 shards); handing
    the runtime a workload with a different K must fail closed, not silently
    reshape or truncate."""
    plan = _tp2_plan()
    rng = np.random.default_rng(7)
    wrong_a = rng.uniform(-1, 1, size=(8, 100))   # 100 != 896
    wrong_b = rng.uniform(-1, 1, size=(100, 100))
    rt = DistributedProcessRuntime()
    with pytest.raises(DistributedRuntimeError, match="shard coverage"):
        rt.run(plan.distributed, wrong_a, wrong_b)


def test_runtime_mismatched_operator_id_detected_by_cross_layer_check():
    """Simulates the runtime being handed a workload whose declared hidden
    dimension does not match what the compiler plan says -- cross-layer
    provenance must flag it rather than silently accept it."""
    plan = _tp2_plan()
    workload = build_qwen_derived_workload(plan.distributed)
    rt = DistributedProcessRuntime()
    result = rt.run(plan.distributed, workload.a, workload.b)

    # Deliberately assert against a wrong hidden_dim, simulating a runtime
    # bookkeeping bug that used a synthetic default instead of the plan's
    # real 896.
    report = verify_cross_layer_provenance(plan.distributed, result, workload_hidden_dim=16)
    assert not report.no_synthetic_fallback_dimensions
    assert not report.all_match
    assert report.mismatch_count >= 1


def test_duplicate_rank_id_rejected():
    plan = _tp2_plan()
    tampered = copy.deepcopy(plan.distributed)
    ranks = list(tampered.ranks)
    ranks[1] = ranks[0]  # duplicate rank_id 0 twice, rank_id 1 missing
    tampered = tampered.__class__(**{**tampered.__dict__, "ranks": tuple(ranks)})
    workload = build_qwen_derived_workload(plan.distributed)
    rt = DistributedProcessRuntime()
    with pytest.raises(DistributedRuntimeError, match="contiguous"):
        rt.run(tampered, workload.a, workload.b)


def test_reject_tp2_real_qwen_plan_on_real_vllm_adapter_path():
    """Even a real-Qwen TP2 plan must still be rejected by the untouched
    real-vLLM adapter path in D2."""
    from deployment.vllm_adapter.plan_schema import validate_vllm_execution_plan

    base = {
        "artifact_type": "vllm_execution_plan",
        "schema_version": "1.0.0",
        "truth_boundary": "Execution planning artifact only; not measured performance.",
        "source_artifacts": ["x"],
        "model": {"model_id": "qwen2.5-0.5b", "tokenizer": "t", "dtype": "fp16",
                   "quantization": "none", "trust_remote_code": False},
        "hardware_profile": {"gpu_name": "g", "vram_gb": 4},
        "backend_profile": {"backend": "vllm"},
        "batch_policy": {"max_num_seqs": 1, "max_num_batched_tokens": 1, "enable_chunked_prefill": False},
        "prefix_policy": {"enable_prefix_caching": False},
        "memory_policy": {"gpu_memory_utilization": 0.5, "max_model_len": 1, "block_size": 1, "swap_space": 0},
        "quantization_policy": {"dtype": "fp16", "quantization": "none"},
        "speculative_policy": {"enabled": False},
        "runtime_config": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                            "served_model_name": "qwen2.5-0.5b"},
    }
    with pytest.raises(Exception, match="tensor_parallel_size"):
        validate_vllm_execution_plan(base)
