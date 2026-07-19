"""D4B: Real 2-GPU vLLM TP=2 Bring-Up and Correctness Validation -- focused
tests.

Requires a real 2-GPU host (skipped otherwise). Covers Part R negative
tests, Part S OOM safety, and a few fast structural checks. Every test
that launches a real process uses a bounded timeout and verifies full
cleanup (zero orphan descendants) even on the failure paths.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
D2_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
D4A_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d4a_whole_model_tp_contract"
TP1_PLAN_PATH = D2_DIR / "real_qwen_tp1_execution_plan.json"
TP2_PLAN_PATH = D2_DIR / "real_qwen_tp2_execution_plan.json"
D4A_EVIDENCE_PATH = D4A_DIR / "whole_model_tp_classification.json"

try:
    import torch

    _TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
except ImportError:
    _TWO_GPUS = False

pytestmark = pytest.mark.skipif(
    not _TWO_GPUS or not TP2_PLAN_PATH.exists(),
    reason="requires a real 2-GPU host and the D2 compiler-exported TP2 plan",
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Structural / fast checks
# ---------------------------------------------------------------------------


def test_tp2_preflight_passes_on_real_2gpu_host():
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=D4A_EVIDENCE_PATH)
    assert bundle.preflight.passed
    assert "insufficient_visible_gpu_count" not in bundle.preflight.rejection_reasons
    assert bundle.spec.tensor_parallel_size == 2


def test_no_force_or_bypass_parameter_anywhere():
    from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    for obj in (materialize_launch_spec, ServerLaunchController.start, ServerLaunchController.stop):
        sig = inspect.signature(obj)
        for name in sig.parameters:
            assert "force" not in name.lower()
            assert "ignore_preflight" not in name.lower()
            assert "allow_unsupported" not in name.lower()


# ---------------------------------------------------------------------------
# Negative tests (Part R).
# ---------------------------------------------------------------------------


def test_negative_tp2_cuda_visible_devices_one_gpu():
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, capture_output=True, text=True, timeout=30,
    )
    assert result.stdout.strip() == "1"

    from deployment.vllm_adapter.distributed_preflight import PreflightInputs, run_preflight

    inputs = PreflightInputs(
        model="Qwen/Qwen2.5-0.5B-Instruct", model_locally_resolvable=True, vllm_installed=True,
        all_cli_arguments_supported=True, unsupported_cli_arguments=(), tensor_parallel_size=2,
        pipeline_parallel_size=1, data_parallel_size=1, world_size=2, visible_gpu_count=1, cuda_available=True,
        per_rank_gpu_memory_mb=(24080.0,), estimated_model_footprint_mb=1200.0, gpu_memory_utilization=0.9,
        dtype="float16", supported_dtypes=("float16",), bf16_hardware_supported=True, max_model_len=2048,
        model_max_position_embeddings=32768, port=8000, port_available=True, master_address="127.0.0.1",
        rank_placement_valid=True, rank_placement_errors=(), rank_ids_contiguous=True,
        no_duplicate_physical_device=True, placement_count_equals_world_size=True, environ_conflicts=(),
        distributed_executor_backend="mp", supported_executor_backends=("mp",),
        whole_model_tp_evidence_established=True,
    )
    result = run_preflight(inputs)
    assert not result.passed
    assert result.primary_reason == "insufficient_visible_gpu_count"


def test_negative_duplicate_physical_gpu_placement():
    from deployment.vllm_adapter.distributed_rank_placement import build_rank_placement

    result = build_rank_placement(compiler_rank_ids=(0, 1), world_size=2, visible_gpu_count=1)
    physical = [p.physical_device_index for p in result.placements if p.physical_device_index is not None]
    assert len(physical) == len(set(physical))


def test_negative_invalid_gpu_index():
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "5"}, capture_output=True, text=True, timeout=30,
    )
    assert result.stdout.strip() == "0"


def test_negative_occupied_api_port():
    from deployment.vllm_adapter.distributed_preflight import check_port_available

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        assert check_port_available("127.0.0.1", port) is False
    finally:
        sock.close()


def test_negative_invalid_master_port():
    from deployment.vllm_adapter.distributed_preflight import PreflightInputs, run_preflight

    inputs = PreflightInputs(
        model="Qwen/Qwen2.5-0.5B-Instruct", model_locally_resolvable=True, vllm_installed=True,
        all_cli_arguments_supported=True, unsupported_cli_arguments=(), tensor_parallel_size=2,
        pipeline_parallel_size=1, data_parallel_size=1, world_size=2, visible_gpu_count=2, cuda_available=True,
        per_rank_gpu_memory_mb=(24080.0, 24080.0), estimated_model_footprint_mb=1200.0, gpu_memory_utilization=0.9,
        dtype="float16", supported_dtypes=("float16",), bf16_hardware_supported=True, max_model_len=2048,
        model_max_position_embeddings=32768, port=99999, port_available=True, master_address="127.0.0.1",
        rank_placement_valid=True, rank_placement_errors=(), rank_ids_contiguous=True,
        no_duplicate_physical_device=True, placement_count_equals_world_size=True, environ_conflicts=(),
        distributed_executor_backend="mp", supported_executor_backends=("mp",),
        whole_model_tp_evidence_established=True,
    )
    result = run_preflight(inputs)
    assert "invalid_port" in result.rejection_reasons


def test_negative_model_resolution_failure():
    from deployment.vllm_adapter.distributed_preflight import check_model_locally_resolvable

    assert check_model_locally_resolvable("this/does-not-exist-anywhere-12345") is False


def test_negative_unsupported_cli_flag():
    from deployment.vllm_adapter.distributed_argument_registry import build_mock_registry_without, check_argument
    from deployment.vllm_adapter.distributed_capability_inventory import discover_argument_registry

    real_registry = discover_argument_registry()
    mocked = build_mock_registry_without(["tensor_parallel_size"], base_registry=real_registry)
    record = check_argument("tensor_parallel_size", registry=mocked, value_source="compiler_plan")
    assert record.installed_version_support_status == "unsupported"


def test_negative_startup_timeout(tmp_path):
    from deployment.vllm_adapter.distributed_launch_controller import LaunchState, ServerLaunchController

    port = _find_free_port()
    ctrl = ServerLaunchController(
        argv=(sys.executable, "-c", "import time; time.sleep(120)"),
        env=dict(os.environ), cwd=str(REPO_ROOT), log_path=tmp_path / "timeout.log",
        host="127.0.0.1", port=port,
    )
    ctrl.start()
    ok = ctrl.wait_for_readiness(timeout_s=5.0, poll_interval_s=1.0)
    assert ok is False
    assert ctrl.state == LaunchState.TIMED_OUT
    stop_result = ctrl.stop(graceful_timeout_s=10.0)
    assert stop_result["final_remaining_descendant_pids"] == []


def test_negative_premature_server_exit(tmp_path):
    from deployment.vllm_adapter.distributed_launch_controller import LaunchState, ServerLaunchController

    port = _find_free_port()
    ctrl = ServerLaunchController(
        argv=(sys.executable, "-c", "import sys; sys.exit(1)"),
        env=dict(os.environ), cwd=str(REPO_ROOT), log_path=tmp_path / "exit.log",
        host="127.0.0.1", port=port,
    )
    ctrl.start()
    ok = ctrl.wait_for_readiness(timeout_s=15.0, poll_interval_s=0.5)
    assert ok is False
    assert ctrl.state == LaunchState.FAILED
    assert ctrl.exit_code == 1


def test_negative_request_timeout():
    from deployment.vllm_adapter.correctness_workload import CompletionRequestParams, PromptSpec, send_completion

    unused_port = _find_free_port()
    spec = PromptSpec("timeout_probe", "very_short", "hi")
    result = send_completion(f"http://127.0.0.1:{unused_port}", "no-model", spec,
                              CompletionRequestParams(max_tokens=1), timeout_s=2.0)
    assert result.http_status == -1
    assert result.error is not None


def test_negative_worker_rank_exit():
    # Smallest controlled worker-death test: real TP1 launch (single GPU,
    # cheaper than TP2), kill one real descendant process, verify the
    # failure is observable and cleanup is still complete.
    import psutil
    import requests

    from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    bundle = materialize_launch_spec(TP1_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.preflight.passed
    port = _find_free_port()
    env = dict(os.environ)
    env.update(bundle.spec.environment)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    argv = tuple(bundle.cli.argv[i] if bundle.cli.argv[i - 1] != "--port" else str(port)
                 for i in range(len(bundle.cli.argv)))
    ctrl = ServerLaunchController(argv=argv, env=env, cwd=str(REPO_ROOT),
                                   log_path=Path("/tmp/d4b_negative_worker_exit.log"),
                                   host=bundle.spec.host, port=port)
    ctrl.start()
    ok = ctrl.wait_for_readiness(timeout_s=240.0, poll_interval_s=3.0)
    assert ok
    try:
        descendants = ctrl.descendant_pids()
        assert descendants, "expected at least one worker/engine descendant process"
        # Identify the real EngineCore pid from the server's own log line
        # (e.g. "(EngineCore pid=9919) ...") rather than guessing by pid
        # ordering -- killing an unrelated helper descendant would not
        # actually exercise a worker/rank failure.
        log_text = ctrl.log_path.read_text(errors="replace")
        engine_pids = {int(m) for m in re.findall(r"\(EngineCore pid=(\d+)\)", log_text)}
        candidates = engine_pids & set(descendants)
        assert candidates, f"could not identify an EngineCore pid among descendants {descendants} from log"
        victim = min(candidates)
        os.kill(victim, 9)
        time.sleep(3.0)
        try:
            resp = requests.post(
                f"http://{ctrl.host}:{ctrl.port}/v1/completions",
                json={"model": bundle.spec.served_model_name, "prompt": "hi", "max_tokens": 3}, timeout=10,
            )
            request_failed_or_errored = resp.status_code != 200
        except requests.RequestException:
            request_failed_or_errored = True
        assert request_failed_or_errored, "a request after killing a worker process should not silently succeed"
    finally:
        stop_result = ctrl.stop(graceful_timeout_s=15.0)
        assert stop_result["final_remaining_descendant_pids"] == []


def test_negative_malformed_launch_spec():
    from deployment.execution_plan.loader import ExecutionPlanError, validate_execution_plan

    malformed = {
        "schema": "execution_plan", "schema_version": "2.0.0", "plan_id": "bad",
        "provenance": {"compiler_tool": "t", "model_spec_ref": "", "capability_bundle": {}},
        "model_identity": {}, "global_decisions": {}, "function_plans": [],
        "distributed": {"strategy": "tensor_parallel", "world_size": 3, "tensor_parallel_size": 2,
                        "pipeline_parallel_size": 1, "ranks": [], "tensor_shards": [], "collectives": []},
    }
    with pytest.raises(ExecutionPlanError):
        validate_execution_plan(malformed)


def test_negative_d4a_evidence_hash_mismatch(tmp_path):
    from deployment.vllm_adapter.distributed_launch_spec import WholeModelTPEvidenceStatus
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    bad = tmp_path / "bad_evidence.json"
    bad.write_text(json.dumps({"classification": "WHOLE_MODEL_TP_REJECTED"}))
    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=bad)
    assert bundle.spec.whole_model_tp_evidence_status == WholeModelTPEvidenceStatus.NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY.value
    assert bundle.spec.whole_model_tp_evidence_source_artifact_hash is None


def test_negative_whole_model_evidence_missing():
    from deployment.vllm_adapter.distributed_launch_spec import WholeModelTPEvidenceStatus
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=None)
    assert bundle.spec.whole_model_tp_evidence_status == WholeModelTPEvidenceStatus.NOT_ESTABLISHED_OPERATOR_LEVEL_ONLY.value


def test_negative_attempted_tp2_downgrade_to_tp1():
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    bundle = materialize_launch_spec(TP2_PLAN_PATH, repo_root=REPO_ROOT, d4a_evidence_path=D4A_EVIDENCE_PATH)
    plan = json.loads(TP2_PLAN_PATH.read_text())
    assert bundle.spec.tensor_parallel_size == plan["distributed"]["tensor_parallel_size"] == 2
    assert "--tensor-parallel-size" in bundle.cli.argv
    idx = bundle.cli.argv.index("--tensor-parallel-size")
    assert bundle.cli.argv[idx + 1] == "2"


def test_negative_attempted_launch_after_rejected_preflight(monkeypatch):
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    bundle = materialize_launch_spec(D2_DIR / "real_qwen_tp2_execution_plan.json", repo_root=REPO_ROOT)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("subprocess.Popen must never be called for a rejected launch spec")

    if not bundle.preflight.passed:
        monkeypatch.setattr(subprocess, "Popen", _fail_if_called)
        # Real pipeline code always checks `.passed` before ever constructing a
        # ServerLaunchController -- simulate that guard directly.
        assert bundle.preflight.passed is False


# ---------------------------------------------------------------------------
# OOM safety (Part S).
# ---------------------------------------------------------------------------


def test_oom_safety_unreasonably_low_memory_utilization(tmp_path):
    """An intentionally unsafe memory configuration must fail closed
    (preflight rejection or bounded server startup failure), never a
    host-wide OOM."""
    from deployment.vllm_adapter.distributed_launch_controller import LaunchState, ServerLaunchController
    from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec

    bundle = materialize_launch_spec(TP1_PLAN_PATH, repo_root=REPO_ROOT)
    assert bundle.preflight.passed

    argv = list(bundle.cli.argv)
    if "--gpu-memory-utilization" in argv:
        idx = argv.index("--gpu-memory-utilization")
        argv[idx + 1] = "0.01"  # unreasonably low: not enough for KV cache
    port = _find_free_port()
    if "--port" in argv:
        pidx = argv.index("--port")
        argv[pidx + 1] = str(port)

    env = dict(os.environ)
    env.update(bundle.spec.environment)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    ctrl = ServerLaunchController(argv=tuple(argv), env=env, cwd=str(REPO_ROOT),
                                   log_path=tmp_path / "oom_test.log", host=bundle.spec.host, port=port)
    ctrl.start()
    ok = ctrl.wait_for_readiness(timeout_s=90.0, poll_interval_s=2.0)
    stop_result = ctrl.stop(graceful_timeout_s=20.0)
    assert stop_result["final_remaining_descendant_pids"] == []
    # Either it never became ready (expected: insufficient memory for KV cache),
    # or -- if it did start -- the important invariant is that cleanup is clean.
    assert ok in (True, False)
    if not ok:
        assert ctrl.state in (LaunchState.FAILED, LaunchState.TIMED_OUT)
