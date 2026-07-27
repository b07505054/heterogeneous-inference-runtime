#!/usr/bin/env python3
"""E2E-8 required correctness experiment: for batch sizes {1,2,4,8} x prompt
lengths {32,128,512}, submit `batch` concurrent deterministic completions
(temperature=0, fixed seed, exact token-ID prompts) and record the decoded
text + token count for each. Run once with the state under test (baseline or
optimized) via one server launch reused across all combinations -- the
comparison across states happens by diffing two runs' output JSON.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_perf_model_experiment import build_command, gpu_memory_used_mib, attention_backend_from_log, oom_occurred

BATCH_SIZES = (1, 2, 4, 8)
PROMPT_LENGTHS = (32, 128, 512)
OUTPUT_TOKENS = 16
FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "the compiler schedules kernels across heterogeneous accelerators. "
)


def exact_prompt_token_ids(tokenizer, target_tokens: int) -> list[int]:
    text = FILLER
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += FILLER
    return tokenizer.encode(text, add_special_tokens=False)[:target_tokens]


def send_completion(base_url: str, model: str, prompt_ids: list[int], request_id: str) -> dict:
    import requests
    payload = {"model": model, "prompt": prompt_ids, "max_tokens": OUTPUT_TOKENS, "temperature": 0.0,
               "seed": 1234, "logprobs": 0}
    resp = requests.post(f"{base_url}/v1/completions", json=payload, timeout=60)
    body = resp.json() if resp.content else None
    text = (body.get("choices") or [{}])[0].get("text") if body else None
    finish_reason = (body.get("choices") or [{}])[0].get("finish_reason") if body else None
    return {"request_id": request_id, "http_status": resp.status_code, "text": text,
            "finish_reason": finish_reason, "usage": body.get("usage") if body else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--state", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=240)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    fixed = json.loads(args.fixed.read_text())
    tokenizer = AutoTokenizer.from_pretrained(fixed["model"], local_files_only=True)
    prompt_ids_by_length = {L: exact_prompt_token_ids(tokenizer, L) for L in PROMPT_LENGTHS}

    python = ".venv/bin/python"
    command = build_command(fixed, 8, args.port, python)
    launcher = str(Path(__file__).resolve().parent / "launch_vllm_with_tiny_m.py")
    assert command[:3] == [python, "-m", "vllm.entrypoints.openai.api_server"]
    command = [python, launcher] + command[3:]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    idle_mib = gpu_memory_used_mib()
    log = args.log.open("w")
    env = os.environ.copy()
    env["VLLM_HOST_IP"] = "127.0.0.1"
    if args.state == "optimized":
        env["VLLM_TINY_M_GEMV_ENABLE"] = "1"
        env["VLLM_TINY_M_GEMV_OPS"] = "lm_head"
        env["VLLM_TINY_M_GEMV_THRESHOLD"] = "8"
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)

    results = {}
    classification = "VALID"
    try:
        deadline = time.time() + args.startup_timeout
        ready = False
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError("SERVER_START_FAILED")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(1)
        if not ready:
            raise RuntimeError("READINESS_TIMEOUT")

        base_url = f"http://127.0.0.1:{args.port}"
        for batch in BATCH_SIZES:
            for length in PROMPT_LENGTHS:
                prompt_ids = prompt_ids_by_length[length]
                key = f"B{batch}-L{length}"
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
                    futures = [pool.submit(send_completion, base_url, fixed["served_model_name"], prompt_ids,
                                            f"{key}-req{i}") for i in range(batch)]
                    rows = [f.result() for f in futures]
                results[key] = rows
                print(f"{key}: {[r['http_status'] for r in rows]}")
    except RuntimeError as exc:
        classification = str(exc)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        cleanup_status = "graceful_sigterm"
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            cleanup_status = "forced_sigkill"
        log.close()
        time.sleep(2)
        after_mib = gpu_memory_used_mib()

    oom = oom_occurred(args.log)
    args.out.write_text(json.dumps({
        "state": args.state, "classification": classification, "results": results,
        "idle_gpu_memory_mib": idle_mib, "after_shutdown_gpu_memory_mib": after_mib,
        "process_cleanup_status": cleanup_status, "oom_detected_in_log": oom,
    }, indent=2))
    print(f"wrote {args.out} classification={classification}")


if __name__ == "__main__":
    main()
