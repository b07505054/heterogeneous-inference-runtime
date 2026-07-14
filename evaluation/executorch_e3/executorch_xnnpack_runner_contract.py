#!/usr/bin/env python3
"""E3 contract adapter for the common ExecuTorch/XNNPACK executor_runner.

This adapter does not implement MatMul/Bias/ReLU and does not choose a candidate.
It validates artifact hashes, forwards to the single ExecuTorch runner binary, and
emits a provenance/timing self-report consumed by the E3 harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

EXPECTED_EXECUTORCH_TAG = "v1.3.1"
EXPECTED_EXECUTORCH_COMMIT = "e2f18eb23c45bd22ca332b0b8b49a81de304b472"
EXPECTED_XNNPACK_COMMIT = "1adaa7c709d4839d29e1f219cb962b01c9e6a905"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_samples(log: str) -> tuple[list[float], float | None, list[str]]:
    samples = [float(x) for x in re.findall(r"Iteration \d+ of \d+: ([0-9.]+) ms", log)]
    m = re.search(r"Model loaded in ([0-9.]+) ms", log)
    thread_log = re.findall(r"Resetting threadpool[^\n]*", log)
    return samples, (float(m.group(1)) if m else None), thread_log


def resolve_runner(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).resolve()
    env = os.environ.get("EXECUTORCH_RUNNER_BIN")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().with_name("executor_runner")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor_runner")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_a", required=True)
    ap.add_argument("--input_b", required=True)
    ap.add_argument("--input_bias", required=True)
    ap.add_argument("--requested_threads", required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--repeats", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--result_json", required=True)
    ap.add_argument("--affinity", default="0-3")
    args = ap.parse_args()

    adapter = Path(__file__).resolve()
    runner = resolve_runner(args.executor_runner)
    model = Path(args.model_path).resolve()
    inputs = [Path(args.input_a).resolve(), Path(args.input_b).resolve(), Path(args.input_bias).resolve()]
    output = Path(args.output).resolve()
    result_json = Path(args.result_json).resolve()
    for path in [runner, model, *inputs]:
        if not path.exists():
            raise SystemExit(f"missing required artifact: {path}")

    total = args.warmups + args.repeats
    if total <= 0 or args.warmups < 0 or args.repeats <= 0:
        raise SystemExit("invalid warmup/repeat counts")
    outbase = output.with_suffix(output.suffix + ".executor_runner")
    produced = Path(str(outbase) + "-0.bin")
    if produced.exists():
        produced.unlink()
    if output.exists():
        output.unlink()

    cmd = [
        "taskset", "-c", args.affinity,
        str(runner),
        "--model_path", str(model),
        "--inputs", ",".join(str(p) for p in inputs),
        "--num_executions", str(total),
        "--print_output", "none",
        "--output_file", str(outbase),
    ]
    requested: int | str
    if args.requested_threads == "default":
        requested = "default"
    else:
        requested = int(args.requested_threads)
        if requested not in (1, 4):
            raise SystemExit("requested_threads must be 1, 4, or default")
        cmd += ["--cpu_threads", str(requested)]

    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    log = proc.stdout + proc.stderr
    samples, load_ms, thread_log = parse_samples(log)
    if produced.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, output)

    report = {
        "schema": "e3_executorch_xnnpack_runner_report",
        "runner_contract": "executorch_xnnpack_runner_contract",
        "backend": "xnnpack",
        "adapter_path": str(adapter),
        "runner_path": str(runner),
        "runner_sha256": sha_file(adapter),
        "runner_binary_sha256": sha_file(runner),
        "pte_path": str(model),
        "pte_sha256": sha_file(model),
        "input_hashes": {
            "a": sha_file(inputs[0]),
            "b": sha_file(inputs[1]),
            "bias": sha_file(inputs[2]),
        },
        "executorch_tag": EXPECTED_EXECUTORCH_TAG,
        "executorch_commit": EXPECTED_EXECUTORCH_COMMIT,
        "xnnpack_commit": EXPECTED_XNNPACK_COMMIT,
        "requested_threads": requested,
        "observed_thread_log": thread_log,
        "observed_thread_classification": "executor_runner_threadpool_log" if thread_log else "not_reported_by_executor_runner",
        "process_lifetime_class": "single_process_single_model_load",
        "timing_boundary": "executor_runner_iteration_ms" if len(samples) == total else "executor_runner_iteration_ms_unreported",
        "timing_samples_complete": len(samples) == total,
        "warmup_count": args.warmups,
        "timed_repeat_count": args.repeats,
        "raw_samples_ms": samples,
        "warm_samples_ms": samples[args.warmups:],
        "load_time_ms": load_ms,
        "end_to_end_invocation_ms": elapsed_ms,
        "exit_status": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "output_path": str(output),
        "output_sha256": sha_file(output) if output.exists() else None,
    }
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    if not output.exists():
        raise SystemExit("executor_runner did not produce output")


if __name__ == "__main__":
    main()
