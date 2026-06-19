import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(args):
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None
    return (completed.stdout or completed.stderr or "").strip() or None


def git_commit_hash():
    return run_command(["git", "rev-parse", "--short", "HEAD"])


def git_dirty():
    return bool(run_command(["git", "status", "--short"]))


def parse_ncu_csv(text):
    rows = []
    csv_lines = [
        line
        for line in text.splitlines()
        if line.startswith('"') or line.startswith("ID,") or line.startswith("Process ID,")
    ]
    if not csv_lines:
        return rows

    reader = csv.DictReader(csv_lines)
    for row in reader:
        metric_name = row.get("Metric Name") or row.get("Metric Name ")
        metric_value = row.get("Metric Value") or row.get("Metric Value ")
        if metric_name:
            rows.append({
                "kernel_name": row.get("Kernel Name") or row.get("Kernel Name "),
                "metric_name": metric_name,
                "metric_unit": row.get("Metric Unit") or row.get("Metric Unit "),
                "metric_value": metric_value,
            })
    return rows


def tail(text, max_lines=80):
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def classify_status(returncode, log_text, metric_rows):
    if metric_rows:
        return "captured"
    if "ERR_NVGPUCTRPERM" in log_text:
        return "permission_blocked"
    if returncode == 0:
        return "captured_without_parsed_metrics"
    return "failed"


def write_markdown_report(path, payload):
    capture = payload["nsight_compute_capture"]
    benchmark = payload.get("benchmark_summary") or {}
    lines = [
        "# Nsight Compute RMSNorm Capture",
        "",
        f"Status: `{capture['status']}`",
        "",
        "## Command",
        "",
        "```bash",
        " ".join(capture.get("command") or []),
        "```",
        "",
        "## Environment",
        "",
        f"- ncu: `{capture.get('ncu_path')}`",
        f"- ncu version: `{capture.get('ncu_version')}`",
        f"- Commit: `{payload['environment'].get('commit_hash')}`",
        f"- Git dirty: `{payload['environment'].get('git_dirty')}`",
        f"- Return code: `{capture.get('returncode')}`",
        "",
        "## Benchmark Summary",
        "",
        f"- Shape: `{benchmark.get('representative_shape')}`",
        f"- Custom latency ms: `{benchmark.get('custom_latency_ms')}`",
        f"- PyTorch latency ms: `{benchmark.get('fallback_latency_ms')}`",
        f"- Speedup: `{benchmark.get('speedup')}`",
        f"- Correct: `{benchmark.get('correct')}`",
        "",
        "## Nsight Metrics",
        "",
    ]

    metric_rows = capture.get("metric_rows", [])
    if metric_rows:
        lines.extend([
            "| Kernel | Metric | Value | Unit |",
            "|---|---|---:|---|",
        ])
        for row in metric_rows[:40]:
            lines.append(
                f"| `{row.get('kernel_name')}` | `{row.get('metric_name')}` | "
                f"{row.get('metric_value')} | `{row.get('metric_unit')}` |"
            )
    elif capture["status"] == "permission_blocked":
        lines.extend([
            "Nsight Compute launched, but NVIDIA performance counters are locked on this machine.",
            "",
            "Enable access with the NVIDIA ERR_NVGPUCTRPERM instructions, then rerun this script.",
        ])
    else:
        lines.append("No metrics were parsed from the Nsight Compute output.")

    lines.extend([
        "",
        "## Profiler Output Tail",
        "",
        "```text",
        capture.get("log_tail") or "",
        "```",
    ])
    write_text(path, "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cuda_transformer/rmsnorm_nsight_compute_capture.json")
    parser.add_argument("--report-output", default="results/cuda_transformer/rmsnorm_nsight_compute_capture.md")
    parser.add_argument("--raw-output", default="results/cuda_transformer/rmsnorm_nsight_compute_raw.csv")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    args = parser.parse_args()

    output = ROOT / args.output
    report_output = ROOT / args.report_output
    raw_output = ROOT / args.raw_output
    benchmark_output = output.with_name(output.stem + "_benchmark.json")
    benchmark_report = output.with_name(output.stem + "_benchmark.md")

    ncu_path = shutil.which("ncu")
    payload = {
        "artifact_type": "nsight_compute_kernel_capture",
        "source": "scripts/capture_rmsnorm_nsight_compute.py",
        "environment": {
            "commit_hash": git_commit_hash(),
            "git_dirty": git_dirty(),
            "python": sys.executable,
        },
        "benchmark_summary": None,
        "nsight_compute_capture": {
            "requested": True,
            "available": ncu_path is not None,
            "ncu_path": ncu_path,
            "ncu_version": run_command([ncu_path, "--version"]) if ncu_path else None,
            "status": "unavailable" if not ncu_path else "not_run",
            "metrics": [item for item in args.metrics.split(",") if item],
            "command": None,
            "returncode": None,
            "duration_seconds": None,
            "metric_rows": [],
            "raw_output": display_path(raw_output),
            "log_tail": None,
        },
    }

    if not ncu_path:
        write_json(output, payload)
        write_markdown_report(report_output, payload)
        print(json.dumps(payload, indent=2))
        return 0

    raw_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ncu_path,
        "--target-processes",
        "all",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        args.metrics,
        "--log-file",
        str(raw_output),
        "--force-overwrite",
        sys.executable,
        str(ROOT / "scripts" / "benchmark_rmsnorm_cuda.py"),
        "--tokens",
        str(args.tokens),
        "--hidden",
        str(args.hidden),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
        "--output",
        str(benchmark_output),
        "--report-output",
        str(benchmark_report),
    ]
    payload["nsight_compute_capture"]["command"] = command

    start = time.time()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    duration = time.time() - start

    log_text = ""
    if raw_output.exists():
        log_text = raw_output.read_text(encoding="utf-8", errors="replace")
    combined_output = "\n".join(part for part in [completed.stdout, completed.stderr, log_text] if part)
    metric_rows = parse_ncu_csv(log_text)

    benchmark_summary = None
    if benchmark_output.exists():
        benchmark_payload = json.loads(benchmark_output.read_text(encoding="utf-8"))
        kernel_benchmarks = benchmark_payload.get("kernel_benchmarks") or []
        if kernel_benchmarks:
            benchmark_summary = kernel_benchmarks[0]

    capture = payload["nsight_compute_capture"]
    capture.update({
        "status": classify_status(completed.returncode, combined_output, metric_rows),
        "returncode": completed.returncode,
        "duration_seconds": round(duration, 3),
        "metric_rows": metric_rows,
        "stdout_tail": tail(completed.stdout or ""),
        "stderr_tail": tail(completed.stderr or ""),
        "log_tail": tail(log_text),
    })
    payload["benchmark_summary"] = benchmark_summary

    write_json(output, payload)
    write_markdown_report(report_output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
