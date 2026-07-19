"""D5 Part A (7B): establish real legal operating ranges for TP1 and TP2 on
Qwen2.5-7B-Instruct before any performance sweep.

For each (max_model_len, max_num_seqs, max_num_batched_tokens,
gpu_memory_utilization) config in PROBE_CONFIGS, attempts a real server
launch for both TP1 and TP2 under a bounded startup timeout, records:
  - whether it started successfully,
  - if it started, real peak GPU memory during a single short request,
  - if it failed, the real failure reason (from the server log / exit code)
    -- never invented, never assumed to be OOM without direct evidence.

This never claims a "TP2 wins" result from an unreasonable config (e.g. a
context length larger than the model supports, or a utilization above 1.0).
Every probed config is a legal vLLM configuration a real operator might
plausibly choose; the point is to find where increasing memory pressure
(longer context, more concurrent sequences) first makes TP1 illegal while
TP2 remains legal -- not to manufacture an artificial failure.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController  # noqa: E402
from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec  # noqa: E402
from deployment.vllm_adapter.gpu_evidence import query_gpu_inventory, wait_for_gpu_memory_baseline  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d5_compiler_tp_policy" / "7b"
LOG_DIR = RESULTS_DIR / "logs"
D2_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d2_qwen_pipeline"
TP1_PLAN_PATH = D2_DIR / "real_qwen7b_tp1_execution_plan.json"
TP2_PLAN_PATH = D2_DIR / "real_qwen7b_tp2_execution_plan.json"
STARTUP_TIMEOUT_S = 300.0

# Ordered from conservative (matches the existing D3B default used for the
# 0.5B calibration sweep) to aggressive (approaching Qwen2.5-7B-Instruct's
# real max_position_embeddings=32768 with several concurrent full-length
# sequences) -- the exact axis the spec asks us to search: longer context
# and higher concurrent-sequence pressure, not an arbitrary unreasonable
# setting.
PROBE_CONFIGS = [
    {"max_model_len": 2048, "max_num_seqs": 4, "max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.90},
    {"max_model_len": 8192, "max_num_seqs": 4, "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.90},
    {"max_model_len": 16384, "max_num_seqs": 8, "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.90},
    {"max_model_len": 24576, "max_num_seqs": 8, "max_num_batched_tokens": 24576, "gpu_memory_utilization": 0.90},
    {"max_model_len": 32768, "max_num_seqs": 8, "max_num_batched_tokens": 32768, "gpu_memory_utilization": 0.90},
    {"max_model_len": 32768, "max_num_seqs": 16, "max_num_batched_tokens": 32768, "gpu_memory_utilization": 0.90},
]


def _write(name: str, payload) -> None:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"wrote {path.relative_to(REPO_ROOT.parent)}", flush=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail_log(log_path: Path, n: int = 40) -> list[str]:
    if not log_path.exists():
        return []
    return log_path.read_text(errors="replace").splitlines()[-n:]


def _probe_one(tp_degree: int, config: dict, index: int) -> dict:
    plan_path = TP1_PLAN_PATH if tp_degree == 1 else TP2_PLAN_PATH
    bundle = materialize_launch_spec(plan_path, repo_root=REPO_ROOT, **config)
    result: dict = {"tp_degree": tp_degree, "config": config, "preflight_passed": bundle.preflight.passed}
    if not bundle.preflight.passed:
        result.update({"started": False, "reason": "preflight_rejected",
                       "preflight_rejection_reasons": bundle.preflight.rejection_reasons})
        return result

    port = _find_free_port()
    env = dict(os.environ)
    env.update(bundle.spec.environment)
    env["CUDA_VISIBLE_DEVICES"] = "0" if tp_degree == 1 else "0,1"
    log_path = LOG_DIR / f"probe_tp{tp_degree}_{index}.log"
    argv = tuple(bundle.cli.argv[i] if bundle.cli.argv[i - 1] != "--port" else str(port)
                 for i in range(len(bundle.cli.argv)))
    ctrl = ServerLaunchController(argv=argv, env=env, cwd=str(REPO_ROOT), log_path=log_path,
                                   host=bundle.spec.host, port=port)
    baseline = {r["index"]: float(r["memory.used"]) for r in query_gpu_inventory()}
    t0 = time.perf_counter()
    ctrl.start()
    ready = ctrl.wait_for_readiness(timeout_s=STARTUP_TIMEOUT_S, poll_interval_s=3.0)
    startup_latency_s = time.perf_counter() - t0

    peak_used_mb = {}
    if ready:
        for r in query_gpu_inventory():
            peak_used_mb[r["index"]] = float(r["memory.used"])

    stop_result = ctrl.stop(graceful_timeout_s=30.0)
    gpu_cleanup = wait_for_gpu_memory_baseline(baseline, timeout_s=45.0)

    result.update({
        "started": ready, "startup_latency_s": startup_latency_s,
        "exit_code": ctrl.exit_code, "state": ctrl.state.value,
        "peak_gpu_memory_used_mb": peak_used_mb,
        "log_tail": _tail_log(log_path),
        "stop_result": stop_result, "gpu_cleanup": str(gpu_cleanup),
    })
    return result


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for i, config in enumerate(PROBE_CONFIGS):
        for tp_degree in (1, 2):
            print(f"== probing TP{tp_degree} config[{i}] = {config} ==", flush=True)
            r = _probe_one(tp_degree, config, i)
            all_results.append(r)
            print(f"   started={r['started']} peak_mb={r.get('peak_gpu_memory_used_mb')}", flush=True)
            _write("legal_range_probe_results.json", all_results)
    print("== 7B legal-operating-range probe complete ==")


if __name__ == "__main__":
    main()
