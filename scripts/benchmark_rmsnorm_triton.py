import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_rmsnorm_cuda import (  # noqa: E402
    bandwidth_gbps,
    cuda_version_from_nvcc,
    driver_version,
    git_commit_hash,
    git_dirty,
    rmsnorm_perf_model,
    summarize_times,
    torch_rmsnorm,
    write_json,
    write_text,
)


DEFAULT_OUTPUT = ROOT / "results" / "cuda_transformer" / "rmsnorm_triton_benchmark.json"
DEFAULT_REPORT = ROOT / "results" / "cuda_transformer" / "rmsnorm_triton_benchmark_report.md"


def unavailable_report(reason, output, report_output):
    payload = {
        "artifact_type": "runtime_kernel_profile",
        "source": "scripts/benchmark_rmsnorm_triton.py",
        "profile_status": "unavailable",
        "reason": reason,
        "kernel_benchmarks": [
            {
                "fusion_candidate": "rmsnorm",
                "custom_kernel": "fused_rmsnorm_triton",
                "fallback_kernel": "torch_rmsnorm",
                "custom_latency_ms": None,
                "fallback_latency_ms": None,
                "speedup": None,
                "correct": None,
                "selection_ready": False,
            }
        ],
    }
    write_json(output, payload)
    write_markdown_report(report_output, payload)
    return payload


def percentile(values, p):
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p / 100))
    return ordered[idx]


def write_markdown_report(path, payload):
    lines = [
        "# Triton RMSNorm Benchmark Report",
        "",
        f"Status: `{payload.get('profile_status')}`",
    ]
    if payload.get("reason"):
        lines.extend(["", f"Reason: `{payload['reason']}`"])
    if payload.get("profile_status") == "measured":
        env = payload["environment"]
        lines.extend([
            "",
            "## Environment",
            "",
            f"- GPU: `{env.get('gpu_name')}`",
            f"- CUDA: `{env.get('cuda_version')}`",
            f"- PyTorch: `{env.get('pytorch_version')}`",
            f"- Triton: `{env.get('triton_version')}`",
            "",
            "## Shape Sweep",
            "",
            "| Tokens | Hidden | Triton p50 ms | PyTorch p50 ms | Speedup | Correct |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for row in payload["sweep"]:
            shape = row["shape"]
            lines.append(
                f"| {shape['tokens']} | {shape['hidden']} | {row['custom_p50_ms']} | "
                f"{row['fallback_p50_ms']} | {row['speedup']} | {row['correct']} |"
            )
    write_text(path, "\n".join(lines) + "\n")


def measure(torch, fn, warmup, runs):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def representative_row(rows):
    selectable = [row for row in rows if row["selection_ready"]]
    return max(selectable, key=lambda row: row["speedup"], default=None) or min(
        rows,
        key=lambda row: row["custom_latency_ms"],
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton RMSNorm against PyTorch.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tokens", default="1,16,128")
    parser.add_argument("--hidden", default="768,1024,4096,8192")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    try:
        import torch
        import triton
        import triton.language as tl
    except ImportError as exc:
        payload = unavailable_report(f"import failed: {exc}", args.output, args.report_output)
        print(json.dumps(payload, indent=2))
        return 0

    if not torch.cuda.is_available():
        payload = unavailable_report("CUDA is not available on this machine", args.output, args.report_output)
        print(json.dumps(payload, indent=2))
        return 0

    @triton.jit
    def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, hidden: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden
        x = tl.load(x_ptr + row * hidden + offsets, mask=mask, other=0.0)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        ss = tl.sum(x * x, axis=0) / hidden
        inv = tl.rsqrt(ss + eps)
        y = x * inv * w
        tl.store(y_ptr + row * hidden + offsets, y, mask=mask)

    def triton_rmsnorm(torch, x, weight, eps):
        y = torch.empty_like(x)
        hidden = x.shape[-1]
        block = triton.next_power_of_2(hidden)
        rmsnorm_kernel[(x.shape[0],)](x, weight, y, hidden, eps, block, num_warps=8)
        return y

    rows = []
    token_sizes = [int(item) for item in args.tokens.split(",") if item]
    hidden_sizes = [int(item) for item in args.hidden.split(",") if item]
    for tokens in token_sizes:
        for hidden in hidden_sizes:
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
            weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
            expected = torch_rmsnorm(torch, x, weight, args.eps)
            actual = triton_rmsnorm(torch, x, weight, args.eps)
            torch.cuda.synchronize()
            correct = bool(torch.allclose(expected, actual, rtol=1e-4, atol=1e-4))
            fallback_times = measure(torch, lambda: torch_rmsnorm(torch, x, weight, args.eps), args.warmup, args.runs)
            triton_times = measure(torch, lambda: triton_rmsnorm(torch, x, weight, args.eps), args.warmup, args.runs)
            fallback_stats = summarize_times(fallback_times)
            triton_stats = summarize_times(triton_times)
            custom_mean = statistics.mean(triton_times)
            fallback_mean = statistics.mean(fallback_times)
            perf = rmsnorm_perf_model(tokens, hidden)
            rows.append({
                "fusion_candidate": "rmsnorm",
                "shape": {"tokens": tokens, "hidden": hidden, "dtype": "float32"},
                "custom_kernel": "fused_rmsnorm_triton",
                "fallback_kernel": "torch_rmsnorm",
                "custom_latency_ms": round(custom_mean, 6),
                "fallback_latency_ms": round(fallback_mean, 6),
                "custom_p50_ms": triton_stats["p50_ms"],
                "fallback_p50_ms": fallback_stats["p50_ms"],
                "custom_p95_ms": triton_stats["p95_ms"],
                "fallback_p95_ms": fallback_stats["p95_ms"],
                "speedup": round(fallback_mean / custom_mean, 4) if custom_mean else None,
                "custom_effective_bandwidth_gbps": round(bandwidth_gbps(perf["bytes_total"], custom_mean), 3),
                "fallback_effective_bandwidth_gbps": round(bandwidth_gbps(perf["bytes_total"], fallback_mean), 3),
                "correct": correct,
                "selection_ready": correct and custom_mean < fallback_mean,
            })

    summary = representative_row(rows)
    payload = {
        "artifact_type": "runtime_kernel_profile",
        "source": "scripts/benchmark_rmsnorm_triton.py",
        "profile_status": "measured",
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "nvcc_version": cuda_version_from_nvcc(),
            "pytorch_version": torch.__version__,
            "triton_version": triton.__version__,
            "driver_version": driver_version(),
            "commit_hash": git_commit_hash(),
            "git_dirty": git_dirty(),
        },
        "benchmark_config": {
            "tokens": token_sizes,
            "hidden": hidden_sizes,
            "warmup": args.warmup,
            "runs": args.runs,
            "eps": args.eps,
            "dtype": "float32",
        },
        "kernel_benchmarks": [
            {
                "fusion_candidate": "rmsnorm",
                "custom_kernel": "fused_rmsnorm_triton",
                "fallback_kernel": "torch_rmsnorm",
                "custom_latency_ms": summary["custom_latency_ms"],
                "fallback_latency_ms": summary["fallback_latency_ms"],
                "speedup": summary["speedup"],
                "correct": summary["correct"],
                "selection_ready": summary["selection_ready"],
                "representative_shape": summary["shape"],
            }
        ],
        "sweep": rows,
    }
    write_json(args.output, payload)
    write_markdown_report(args.report_output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
