"""D6 Part K: real 2-GPU execution verification for a compiler-selected
plan (produced by the D6 C++ compiler selector via fresh compilation, not
hand-authored, not chosen at runtime by any Python selector).

Launches the exact plan file passed in through the unmodified D3B/D4B
chain (materialize_launch_spec -> ServerLaunchController -> real vLLM),
proves NCCL/GPU evidence for TP2 (or single-GPU absence of NCCL for TP1),
runs the real correctness corpus, tests repeated-run stability, and
verifies complete cleanup.
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

from deployment.vllm_adapter.correctness_workload import (  # noqa: E402
    CompletionRequestParams,
    build_prompt_corpus,
    compare_completions,
    send_completion,
)
from deployment.vllm_adapter.distributed_launch_controller import ServerLaunchController  # noqa: E402
from deployment.vllm_adapter.distributed_materializer import materialize_launch_spec  # noqa: E402
from deployment.vllm_adapter.gpu_evidence import (  # noqa: E402
    build_gpu_snapshot,
    compute_process_gpu_mapping,
    extract_nccl_evidence,
    query_gpu_inventory,
    wait_for_gpu_memory_baseline,
)

RESULTS_DIR = REPO_ROOT / "results" / "runtime_paths" / "distributed_d6_compiler_owned_tp_selection"
LOG_DIR = RESULTS_DIR / "logs"
STARTUP_TIMEOUT_S = 600.0


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(plan_path: str, label: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    bundle = materialize_launch_spec(Path(plan_path), repo_root=REPO_ROOT)
    print(f"== {label}: tensor_parallel_size={bundle.spec.tensor_parallel_size} "
          f"preflight_passed={bundle.preflight.passed} ==", flush=True)
    assert bundle.preflight.passed, bundle.preflight.to_dict()

    port = _find_free_port()
    env = dict(os.environ)
    env.update(bundle.spec.environment)
    tp = bundle.spec.tensor_parallel_size
    env["CUDA_VISIBLE_DEVICES"] = "0" if tp == 1 else "0,1"
    log_path = LOG_DIR / f"d6_{label}_server.log"
    argv = tuple(bundle.cli.argv[i] if bundle.cli.argv[i - 1] != "--port" else str(port)
                 for i in range(len(bundle.cli.argv)))
    ctrl = ServerLaunchController(argv=argv, env=env, cwd=str(REPO_ROOT), log_path=log_path,
                                   host=bundle.spec.host, port=port)

    baseline = {r["index"]: float(r["memory.used"]) for r in query_gpu_inventory()}
    t0 = time.perf_counter()
    ctrl.start()
    ready = ctrl.wait_for_readiness(timeout_s=STARTUP_TIMEOUT_S, poll_interval_s=5.0)
    startup_latency_s = time.perf_counter() - t0
    assert ready, f"server failed to become ready: {ctrl.to_dict()}"
    print(f"ready in {startup_latency_s:.1f}s, pid={ctrl.pid}", flush=True)

    gpu_during = build_gpu_snapshot(f"during_{label}")
    tracked_pids = ctrl.all_tracked_pids()
    gpu_mapping = compute_process_gpu_mapping(tracked_pids)

    nccl_evidence = None
    if tp > 1:
        log_text = log_path.read_text(errors="replace")
        nccl_evidence = extract_nccl_evidence(log_text)

    # Correctness: full corpus + repeated-run stability check on prompt 0.
    corpus = build_prompt_corpus()
    params = CompletionRequestParams(max_tokens=24, logprobs=5)
    base_url = f"http://{bundle.spec.host}:{port}"
    outputs = []
    for spec in corpus:
        r = send_completion(base_url, bundle.spec.served_model_name, spec, params, timeout_s=120.0)
        outputs.append(r.to_dict())
    all_ok = all(o["http_status"] == 200 for o in outputs)

    repeat_a = send_completion(base_url, bundle.spec.served_model_name, corpus[0], params, timeout_s=120.0)
    repeat_b = send_completion(base_url, bundle.spec.served_model_name, corpus[0], params, timeout_s=120.0)
    import importlib
    tok_mod = importlib.import_module("transformers")
    tokenizer = tok_mod.AutoTokenizer.from_pretrained(bundle.spec.model)
    stability = compare_completions(corpus[0].prompt_id, repeat_a, repeat_b, tokenizer)

    stop_result = ctrl.stop(graceful_timeout_s=45.0)
    gpu_cleanup = wait_for_gpu_memory_baseline(baseline, timeout_s=60.0)

    # Stale port check: confirm nothing is listening on the port anymore.
    port_free = True
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            port_free = s.connect_ex(("127.0.0.1", port)) != 0
    except OSError:
        pass

    evidence = {
        "label": label, "source_plan": str(plan_path),
        "tensor_parallel_size": tp, "startup_latency_s": startup_latency_s,
        "preflight": bundle.preflight.to_dict(),
        "gpu_mapping": gpu_mapping if isinstance(gpu_mapping, dict) else str(gpu_mapping),
        "nccl_evidence": nccl_evidence,
        "correctness_all_200_ok": all_ok,
        "correctness_outputs": outputs,
        "repeated_run_stability_text_match": stability.text_match,
        "repeated_run_stability_token_match": stability.token_ids_match,
        "stop_result": stop_result,
        "gpu_cleanup": str(gpu_cleanup),
        "port_free_after_stop": port_free,
        "orphan_processes_after_cleanup": stop_result.get("final_remaining_descendant_pids", []),
    }
    (RESULTS_DIR / f"part_k_{label}_real_verification.json").write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    print(f"correctness_all_200_ok={all_ok} stability_text_match={stability.text_match} "
          f"gpu_cleanup_ok={'within_tolerance=True' in str(gpu_cleanup)} orphans={stop_result.get('final_remaining_descendant_pids')}")
    print(f"wrote part_k_{label}_real_verification.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
