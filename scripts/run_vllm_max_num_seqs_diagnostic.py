#!/usr/bin/env python3
"""Diagnostic-only real-vLLM runner with request, server-metric, and GPU timelines."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.backends.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from benchmark.metrics import latency_summary_ms


def server_command(fixed, value, port):
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", fixed["model"], "--tokenizer", fixed["tokenizer"], "--dtype", fixed["dtype"], "--max-model-len", str(fixed["max_model_len"]), "--gpu-memory-utilization", str(fixed["gpu_memory_utilization"]), "--block-size", str(fixed["block_size"]), "--max-num-batched-tokens", str(fixed["max_num_batched_tokens"]), "--tensor-parallel-size", str(fixed["tensor_parallel_size"]), "--pipeline-parallel-size", str(fixed["pipeline_parallel_size"]), "--served-model-name", fixed["served_model_name"], "--host", "127.0.0.1", "--port", str(port), "--no-enable-chunked-prefill", "--no-enable-prefix-caching", "--max-num-seqs", str(value)]
    return command


def gpu_sample():
    query = "utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu"
    try:
        text = subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL).strip()
        values = [x.strip() for x in text.split(",")]
        return dict(zip(("gpu_utilization_percent", "memory_used_mib", "power_watts", "sm_clock_mhz", "temperature_c"), (float(x) for x in values)))
    except Exception as exc:
        return {"error": str(exc)}


def prometheus_sample(port):
    wanted = ("num_requests_running", "num_requests_waiting", "gpu_cache_usage_perc", "kv_cache_usage_perc", "generation_tokens_total", "prompt_tokens_total", "iteration_tokens_total")
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1) as response:
            return _parse_prometheus(response.read().decode(), wanted)
    except Exception as exc:
        return {"error": str(exc)}


def _parse_prometheus(text, wanted):
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split()[0].rsplit(":", 1)[-1]
        for name in wanted:
            if metric_name == name:
                match = re.search(r"(?:^|\})\s+([-+0-9.eE]+)$", line)
                if match:
                    result.setdefault(name, []).append(float(match.group(1)))
    return {name: sum(values) for name, values in result.items()}


def concurrency(intervals):
    events = []
    for start, end in intervals:
        events.extend(((start, 1), (end, -1)))
    events.sort(key=lambda item: (item[0], item[1]))
    running = maximum = 0
    area = 0.0
    previous = events[0][0]
    for timestamp, delta in events:
        area += running * (timestamp - previous)
        running += delta
        maximum = max(maximum, running)
        previous = timestamp
    span = events[-1][0] - events[0][0]
    return maximum, area / span if span else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--workload", choices=("S2", "S3"), required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--client-concurrency", type=int)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--mode", choices=("runtime_equivalent", "direct_control"), default="runtime_equivalent")
    args = parser.parse_args()
    fixed = json.loads(args.fixed.read_text())
    manifest = json.loads(args.workload_manifest.read_text())
    workload = next(x for x in manifest["workloads"] if x["workload_id"] == args.workload)
    client_concurrency = args.client_concurrency or workload["concurrency"]
    command = server_command(fixed, args.max_num_seqs, args.port)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log = args.log.open("w")
    env = os.environ.copy()
    env["VLLM_HOST_IP"] = "127.0.0.1"
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    startup_start = time.perf_counter()
    samples = []
    stop = False

    def sample():
        while not stop:
            samples.append({"time": time.perf_counter(), "gpu": gpu_sample(), "vllm": prometheus_sample(args.port)})
            time.sleep(0.25)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    rows = []
    classification = "VALID"
    startup_seconds = None
    trace_start = trace_end = None
    try:
        deadline = time.time() + args.startup_timeout
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError("SERVER_START_FAILED")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2) as response:
                    if response.status == 200:
                        startup_seconds = time.perf_counter() - startup_start
                        break
            except Exception:
                time.sleep(1)
        if startup_seconds is None:
            raise RuntimeError("READINESS_TIMEOUT")
        backend = OpenAICompatibleBackend(OpenAICompatibleConfig(base_url=f"http://127.0.0.1:{args.port}", model=fixed["served_model_name"], concurrency=client_concurrency, timeout_s=120, record_timeline=True))
        for request in workload["requests"][:workload["warmup_requests"]]:
            backend.execute(request, True)
        measured = workload["requests"][workload["warmup_requests"]:]
        trace_start = time.perf_counter()

        def execute(request, submitted):
            result = backend.execute(request)
            result["timeline"]["submit_time"] = submitted
            result["request_id"] = request["metadata"]["request_id"]
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=client_concurrency) as executor:
            futures = []
            for request in measured:
                submitted = time.perf_counter()
                futures.append(executor.submit(execute, request, submitted))
            rows = [future.result() for future in futures]
        trace_end = time.perf_counter()
    except RuntimeError as exc:
        classification = str(exc)
    finally:
        stop = True
        sampler.join(3)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.close()
    good = [row for row in rows if row.get("ok")]
    intervals = [(row["timeline"]["connection_start_time"], row["timeline"]["completion_time"]) for row in good]
    maximum, average = concurrency(intervals) if intervals else (0, 0)
    total_tokens = sum(row.get("output_tokens", 0) for row in good)
    wall = trace_end - trace_start if trace_start is not None and trace_end is not None else None
    result = {
        "mode": args.mode,
        "workload_id": args.workload,
        "workload_sha256": workload["sha256"],
        "max_num_seqs": args.max_num_seqs,
        "client_concurrency": client_concurrency,
        "session": args.session,
        "classification": classification if len(good) == len(rows) else "REQUEST_FAILURE",
        "request_count": len(rows),
        "success_count": len(good),
        "failure_count": len(rows) - len(good),
        "ttft_ms": latency_summary_ms(row["ttft_ms"] for row in good),
        "tpot_ms": latency_summary_ms(row["tpot_ms"] for row in good),
        "e2e_ms": latency_summary_ms(row["e2e_latency_ms"] for row in good),
        "output_token_throughput": total_tokens / wall if wall else None,
        "request_throughput": len(good) / wall if wall else None,
        "trace_wall_seconds": wall,
        "actual_maximum_in_flight": maximum,
        "actual_average_in_flight": average,
        "startup_seconds": startup_seconds,
        "server_pid": process.pid,
        "command": command,
        "request_timelines": [{"request_id": row["request_id"], **row["timeline"], "ttft_ms": row["ttft_ms"], "tpot_ms": row["tpot_ms"], "output_tokens": row["output_tokens"]} for row in good],
        "samples": samples,
        "raw_log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
