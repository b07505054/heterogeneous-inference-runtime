#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
OUT_DIR = ROOT / "results/qwen_no_quant"
TRIAL_DIR = OUT_DIR / "repeatability_trials"
RAW_PATH = OUT_DIR / "repeatability_raw.json"
SUMMARY_PATH = OUT_DIR / "repeatability_summary.md"

BASE_URL = "http://127.0.0.1:8000"
WARMUP = 4
CONCURRENCY = 1
TIMEOUT_S = 180
WORKLOADS = ["short", "shared_prefix", "no_shared_prefix"]
TRACE = {
    "short": "results/qwen_no_quant/traces/short.jsonl",
    "shared_prefix": "results/qwen_no_quant/traces/shared_prefix.jsonl",
    "no_shared_prefix": "results/qwen_no_quant/traces/no_shared_prefix.jsonl",
}
ORDER = {
    1: ["baseline", "compiler"],
    2: ["compiler", "baseline"],
    3: ["baseline", "compiler"],
}
PATHS = {
    "baseline": {
        "artifact_label": "baseline-conservative-fixed",
        "claimed_server": "vllm_no_quant_baseline_conservative_fixed_repeatability",
        "command_file": OUT_DIR / "baseline_conservative_fixed_command.txt",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "compiler": {
        "artifact_label": "compiler-guided-fixed",
        "claimed_server": "vllm_no_quant_compiler_guided_fixed_repeatability",
        "command_file": OUT_DIR / "compiler_guided_fixed_command.txt",
        "model": "qwen2.5-0.5b",
    },
}


def run(cmd, *, check=False, capture=True, shell=False, env=None):
    return subprocess.run(
        cmd,
        shell=shell,
        text=True,
        capture_output=capture,
        check=check,
        env=env,
    )


def kill_vllm():
    for pattern in [
        "vllm.entrypoints.openai.api_server",
        "vllm serve",
        "vllm/v1/engine",
    ]:
        run(["pkill", "-f", pattern])
    time.sleep(3)


def gpu_snapshot():
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
    )
    line = proc.stdout.strip().splitlines()[0]
    name, used, total = [part.strip() for part in line.split(",")]
    return {
        "name": name,
        "memory_used_mib": int(used),
        "memory_total_mib": int(total),
        "raw": line,
    }


def wait_gpu_idle(max_used_mib=50, timeout_s=90):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        last = gpu_snapshot()
        if last["memory_used_mib"] <= max_used_mib:
            return last
        time.sleep(5)
    return last or gpu_snapshot()


def health_ready(timeout_s=180):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return True, None
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(3)
    return False, last_error


def read_command(path: Path) -> str:
    text = path.read_text().strip()
    if not text:
        raise RuntimeError(f"empty command file: {path}")
    return text


def load_result(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def completed_unit(label: str, trial: int) -> bool:
    return all(
        (TRIAL_DIR / f"trial{trial}_{label}_{workload}.json").exists()
        and (TRIAL_DIR / f"trial{trial}_{label}_{workload}.json").stat().st_size > 0
        for workload in WORKLOADS
    )


def existing_run_record(label: str, trial: int, server_commands: dict[str, str]) -> dict:
    run_record = {
        "trial": trial,
        "path": label,
        "artifact_label": PATHS[label]["artifact_label"],
        "server_command": server_commands[label],
        "server_log": str(TRIAL_DIR / f"trial{trial}_{label}_server.log"),
        "gpu_memory_ready": None,
        "server_ready": True,
        "server_ready_error": None,
        "workloads": {},
        "status": "ok",
        "resumed_from_existing_files": True,
        "gpu_memory_after": None,
        "server_returncode": None,
    }
    for workload in WORKLOADS:
        out = TRIAL_DIR / f"trial{trial}_{label}_{workload}.json"
        result = load_result(out)
        run_record["workloads"][workload] = {
            "benchmark_command": result.get("command"),
            "output": str(out),
            "returncode": 0,
            "stdout": str(out),
            "stderr": "",
            "elapsed_s": None,
            "result": result,
            "metrics": result.get("metrics", {}),
            "server_metadata": result.get("server_metadata", {}),
            "resumed_from_existing_file": True,
        }
    return run_record


def metric_value(result: dict, key: str, stat: str):
    return result.get("metrics", {}).get(key, {}).get(stat)


def tokens_per_second(result: dict):
    return result.get("metrics", {}).get("tokens_per_second")


def stdev(values):
    vals = [float(value) for value in values if value is not None]
    if len(vals) < 2:
        return None
    return statistics.stdev(vals)


def mean(values):
    vals = [float(value) for value in values if value is not None]
    return statistics.mean(vals) if vals else None


def pct_delta(compiler, baseline, lower_is_better=True):
    if compiler is None or baseline in (None, 0):
        return None
    if lower_is_better:
        return (compiler - baseline) / baseline * 100.0
    return (baseline - compiler) / baseline * 100.0


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def gpu_pair(run_record: dict) -> str:
    ready = run_record.get("gpu_memory_ready") or {}
    after = run_record.get("gpu_memory_after") or {}
    ready_used = ready.get("memory_used_mib")
    after_used = after.get("memory_used_mib")
    if ready_used is None or after_used is None:
        return "existing/existing"
    return f"{ready_used}/{after_used}"


def benchmark_cmd(label, workload, output):
    cfg = PATHS[label]
    return [
        ".venv/bin/python",
        "scripts/benchmark_openai_compatible_server.py",
        "--base-url",
        BASE_URL,
        "--model",
        cfg["model"],
        "--trace",
        TRACE[workload],
        "--concurrency",
        str(CONCURRENCY),
        "--warmup",
        str(WARMUP),
        "--timeout-s",
        str(TIMEOUT_S),
        "--claimed-server",
        cfg["claimed_server"],
        "--output",
        str(output),
    ]


def validate_existing_artifacts():
    missing = []
    for cfg in PATHS.values():
        if not cfg["command_file"].exists():
            missing.append(str(cfg["command_file"]))
    for trace in TRACE.values():
        if not (ROOT / trace).exists():
            missing.append(trace)
    if missing:
        raise SystemExit("missing required existing artifacts: " + ", ".join(missing))


def aggregate(raw):
    summary = {}
    for workload in WORKLOADS:
        summary[workload] = {}
        for label in ["baseline", "compiler"]:
            rows = []
            for run_record in raw["runs"]:
                if run_record["path"] != label or run_record.get("status") != "ok":
                    continue
                workload_record = run_record["workloads"].get(workload, {})
                result = workload_record.get("result")
                if not result:
                    continue
                metrics = result.get("metrics", {})
                rows.append(
                    {
                        "trial": run_record["trial"],
                        "ttft_mean_ms": metric_value(result, "ttft_ms", "mean"),
                        "ttft_p50_ms": metric_value(result, "ttft_ms", "p50"),
                        "ttft_p95_ms": metric_value(result, "ttft_ms", "p95"),
                        "tpot_mean_ms": metric_value(result, "tpot_ms", "mean"),
                        "tpot_p50_ms": metric_value(result, "tpot_ms", "p50"),
                        "tpot_p95_ms": metric_value(result, "tpot_ms", "p95"),
                        "e2e_mean_ms": metric_value(result, "e2e_latency_ms", "mean"),
                        "e2e_p50_ms": metric_value(result, "e2e_latency_ms", "p50"),
                        "e2e_p95_ms": metric_value(result, "e2e_latency_ms", "p95"),
                        "tokens_per_second": tokens_per_second(result),
                        "success_count": metrics.get("success_count"),
                        "error_count": metrics.get("error_count"),
                    }
                )
            aggregate_metrics = {}
            for key in ["ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms", "tokens_per_second"]:
                vals = [row[key] for row in rows]
                aggregate_metrics[key] = {"mean": mean(vals), "stdev": stdev(vals)}
            summary[workload][label] = {"trials": rows, "aggregate": aggregate_metrics}
        deltas = {}
        for key, lower in [
            ("ttft_mean_ms", True),
            ("tpot_mean_ms", True),
            ("e2e_mean_ms", True),
            ("tokens_per_second", False),
        ]:
            baseline = summary[workload]["baseline"]["aggregate"][key]["mean"]
            compiler = summary[workload]["compiler"]["aggregate"][key]["mean"]
            deltas[key] = pct_delta(compiler, baseline, lower_is_better=lower)
        summary[workload]["delta_percent"] = deltas
    return summary


def write_summary(raw, summary, server_commands):
    lines = [
        "# Qwen No-Quant Repeatability Study",
        "",
        "Compiler-guided no-quant Qwen uses original Qwen weights. Differences come from execution/runtime policy, not model weight optimization.",
        "",
        f"- Created UTC: `{raw['created_utc']}`",
        f"- Host: `{raw['host']}`",
        f"- Runtime: `{raw['cwd']}`",
        f"- Virtual env: `{raw['environment']['virtual_env']}`",
        f"- Python: `{raw['environment']['python_version']}` at `{raw['environment']['python']}`",
        f"- Warmup: `{WARMUP}`",
        f"- Concurrency: `{CONCURRENCY}`",
        f"- Port/base URL: `{BASE_URL}`",
        "",
        "## Commands",
        "",
    ]
    for label in ["baseline", "compiler"]:
        lines.extend(
            [
                f"### {PATHS[label]['artifact_label']}",
                "```bash",
                server_commands[label],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "Remaining command/config difference: compiler-guided uses `--served-model-name qwen2.5-0.5b`; baseline does not. Other low-memory runtime flags are equivalent, with minor ordering differences only.",
            "",
            "Served-model-name affects OpenAI model routing by changing the model id accepted by `/v1/chat/completions` and returned by `/v1/models`. In these runs, baseline benchmark requests used `Qwen/Qwen2.5-0.5B-Instruct`; compiler-guided requests used `qwen2.5-0.5b`. The served model root remained `Qwen/Qwen2.5-0.5B-Instruct`, so routing points at the same original weights.",
            "",
            "## Per-Trial Results",
            "",
            "| Workload | Path | Trial | TTFT mean/p50/p95 ms | TPOT mean/p50/p95 ms | E2E mean/p50/p95 ms | tok/s | success/error | GPU ready/after MiB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for workload in WORKLOADS:
        for label in ["baseline", "compiler"]:
            for row in summary[workload][label]["trials"]:
                run_record = next(
                    record
                    for record in raw["runs"]
                    if record["trial"] == row["trial"] and record["path"] == label
                )
                lines.append(
                    f"| `{workload}` | `{PATHS[label]['artifact_label']}` | {row['trial']} | "
                    f"{fmt(row['ttft_mean_ms'])}/{fmt(row['ttft_p50_ms'])}/{fmt(row['ttft_p95_ms'])} | "
                    f"{fmt(row['tpot_mean_ms'])}/{fmt(row['tpot_p50_ms'])}/{fmt(row['tpot_p95_ms'])} | "
                    f"{fmt(row['e2e_mean_ms'])}/{fmt(row['e2e_p50_ms'])}/{fmt(row['e2e_p95_ms'])} | "
                    f"{fmt(row['tokens_per_second'])} | {row['success_count']}/{row['error_count']} | "
                    f"{gpu_pair(run_record)} |"
                )
    lines.extend(
        [
            "",
            "## Aggregate Delta",
            "",
            "Positive latency delta means compiler-guided is slower. Positive throughput delta means compiler-guided has lower throughput.",
            "",
            "| Workload | Metric | Baseline mean | Baseline stdev | Compiler mean | Compiler stdev | Delta % |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for workload in WORKLOADS:
        for key, display in [
            ("ttft_mean_ms", "TTFT mean ms"),
            ("tpot_mean_ms", "TPOT mean ms"),
            ("e2e_mean_ms", "E2E mean ms"),
            ("tokens_per_second", "tok/s"),
        ]:
            baseline = summary[workload]["baseline"]["aggregate"][key]
            compiler = summary[workload]["compiler"]["aggregate"][key]
            delta = summary[workload]["delta_percent"][key]
            lines.append(
                f"| `{workload}` | {display} | {fmt(baseline['mean'])} | {fmt(baseline['stdev'])} | "
                f"{fmt(compiler['mean'])} | {fmt(compiler['stdev'])} | {fmt(delta)}% |"
            )
    lines.extend(["", "## Conclusion", ""])
    for workload in WORKLOADS:
        delta = summary[workload]["delta_percent"]["e2e_mean_ms"]
        baseline = summary[workload]["baseline"]["aggregate"]["e2e_mean_ms"]
        compiler = summary[workload]["compiler"]["aggregate"]["e2e_mean_ms"]
        pooled_sd = math.sqrt(((baseline["stdev"] or 0) ** 2 + (compiler["stdev"] or 0) ** 2) / 2.0)
        pooled_mean = mean([baseline["mean"], compiler["mean"]])
        pooled_cv = (pooled_sd / pooled_mean * 100.0) if pooled_mean else None
        if delta is None:
            judgment = "insufficient data"
        elif pooled_cv is not None and abs(delta) <= max(2.0 * pooled_cv, 1.0):
            judgment = "likely within benchmark noise"
        else:
            judgment = "larger than observed trial-to-trial noise in this 3-trial sample"
        lines.append(
            f"- `{workload}`: E2E mean delta `{fmt(delta)}%`; pooled E2E CV `{fmt(pooled_cv)}%`; judgment: {judgment}."
        )
    lines.extend(
        [
            "",
            "Do not claim speedup from these results. This repeatability pass is only validating whether the prior 2-3% difference is stable or benchmark noise.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def main():
    TRIAL_DIR.mkdir(parents=True, exist_ok=True)
    validate_existing_artifacts()
    server_commands = {label: read_command(cfg["command_file"]) for label, cfg in PATHS.items()}
    raw = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": run(["hostname"], check=True).stdout.strip(),
        "cwd": str(ROOT),
        "environment": {
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "offline_env": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            },
        },
        "order": ORDER,
        "workloads": WORKLOADS,
        "warmup": WARMUP,
        "concurrency": CONCURRENCY,
        "base_url": BASE_URL,
        "server_commands": server_commands,
        "runs": [],
    }
    for trial in [1, 2, 3]:
        for label in ORDER[trial]:
            if completed_unit(label, trial):
                print(
                    f"[repeatability] trial={trial} path={PATHS[label]['artifact_label']}: preserving existing complete files",
                    flush=True,
                )
                raw["runs"].append(existing_run_record(label, trial, server_commands))
                RAW_PATH.write_text(json.dumps(raw, indent=2))
                continue
            cfg = PATHS[label]
            artifact_label = cfg["artifact_label"]
            print(f"[repeatability] trial={trial} path={artifact_label}: killing vLLM", flush=True)
            kill_vllm()
            gpu_ready = wait_gpu_idle()
            if gpu_ready["memory_used_mib"] > 50:
                raise SystemExit(f"GPU not idle before trial {trial} {label}: {gpu_ready}")
            server_cmd = server_commands[label]
            env = os.environ.copy()
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            print(
                f"[repeatability] trial={trial} path={artifact_label}: GPU ready {gpu_ready['memory_used_mib']} MiB; starting server",
                flush=True,
            )
            log_path = TRIAL_DIR / f"trial{trial}_{label}_server.log"
            log_f = log_path.open("w")
            proc = subprocess.Popen(
                server_cmd,
                shell=True,
                cwd=str(ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            ready, ready_error = health_ready()
            run_record = {
                "trial": trial,
                "path": label,
                "artifact_label": artifact_label,
                "server_command": server_cmd,
                "server_log": str(log_path),
                "gpu_memory_ready": gpu_ready,
                "server_ready": ready,
                "server_ready_error": ready_error,
                "workloads": {},
            }
            try:
                if not ready:
                    run_record["status"] = "server_not_ready"
                    raw["runs"].append(run_record)
                    raise SystemExit(
                        f"server not ready for trial {trial} {label}: {ready_error}; see {log_path}"
                    )
                for workload in WORKLOADS:
                    out = TRIAL_DIR / f"trial{trial}_{label}_{workload}.json"
                    cmd = benchmark_cmd(label, workload, out)
                    print(
                        f"[repeatability] trial={trial} path={artifact_label} workload={workload}: benchmark",
                        flush=True,
                    )
                    started = time.time()
                    bench = run(cmd, env=env)
                    ended = time.time()
                    workload_record = {
                        "benchmark_command": cmd,
                        "output": str(out),
                        "returncode": bench.returncode,
                        "stdout": bench.stdout.strip(),
                        "stderr": bench.stderr.strip(),
                        "elapsed_s": round(ended - started, 6),
                    }
                    if out.exists():
                        result = load_result(out)
                        workload_record["result"] = result
                        workload_record["metrics"] = result.get("metrics", {})
                        workload_record["server_metadata"] = result.get("server_metadata", {})
                    run_record["workloads"][workload] = workload_record
                    if bench.returncode != 0:
                        run_record["status"] = "benchmark_failed"
                        raw["runs"].append(run_record)
                        raise SystemExit(
                            f"benchmark failed trial {trial} {label} {workload}: {bench.stderr}"
                        )
                run_record["status"] = "ok"
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=10)
                log_f.close()
                time.sleep(5)
                kill_vllm()
                gpu_after = wait_gpu_idle()
                run_record["gpu_memory_after"] = gpu_after
                run_record["server_returncode"] = proc.returncode
                raw["runs"].append(run_record)
                RAW_PATH.write_text(json.dumps(raw, indent=2))
                print(
                    f"[repeatability] trial={trial} path={artifact_label}: done; GPU after {gpu_after['memory_used_mib']} MiB",
                    flush=True,
                )
    summary = aggregate(raw)
    raw["summary"] = summary
    RAW_PATH.write_text(json.dumps(raw, indent=2))
    write_summary(raw, summary, server_commands)
    print(f"[repeatability] wrote {RAW_PATH}")
    print(f"[repeatability] wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
