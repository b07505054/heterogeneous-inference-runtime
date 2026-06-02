import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "cuda_transformer_kernels"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    idx = int(round((len(values) - 1) * p / 100))
    return values[idx]


def rmsnorm_perf_model(tokens, hidden, dtype_bytes=4):
    elements = tokens * hidden
    # This kernel reads input once for the reduction, reads input again for the
    # final write, reads weight once, and writes output once.
    bytes_per_element = dtype_bytes * 4
    bytes_total = elements * bytes_per_element
    flops_per_token = hidden * 4
    flops_total = tokens * flops_per_token
    return {
        "bytes_per_token": hidden * bytes_per_element,
        "bytes_total": bytes_total,
        "flops_per_token": flops_per_token,
        "flops_total": flops_total,
        "arithmetic_intensity_flops_per_byte": flops_total / max(bytes_total, 1),
        "memory_bound": True,
        "model_notes": [
            "RMSNorm is dominated by input/weight reads and output writes.",
            "The custom kernel reads input twice: once for sum-of-squares reduction and once for output.",
            "Low arithmetic intensity makes effective bandwidth and launch overhead key latency drivers.",
        ],
    }


def bandwidth_gbps(bytes_total, latency_ms):
    if not latency_ms:
        return None
    return bytes_total / (latency_ms * 1_000_000)


def summarize_times(times):
    return {
        "mean_ms": round(statistics.mean(times), 6),
        "p50_ms": round(percentile(times, 50), 6),
        "p95_ms": round(percentile(times, 95), 6),
        "min_ms": round(min(times), 6),
        "max_ms": round(max(times), 6),
    }


def write_markdown_report(path, payload):
    lines = [
        "# CUDA RMSNorm Benchmark Report",
        "",
        f"Status: `{payload.get('profile_status')}`",
        f"Device: `{payload.get('device', 'n/a')}`",
        "",
        "| Tokens | Hidden | Custom ms | PyTorch ms | Speedup | Custom GB/s | Correct |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("sweep", []):
        shape = row["shape"]
        lines.append(
            "| {tokens} | {hidden} | {custom} | {fallback} | {speedup} | {bw} | {correct} |".format(
                tokens=shape["tokens"],
                hidden=shape["hidden"],
                custom=row["custom_latency_ms"],
                fallback=row["fallback_latency_ms"],
                speedup=row["speedup"],
                bw=row["custom_effective_bandwidth_gbps"],
                correct=row["correct"],
            )
        )

    lines.extend([
        "",
        "## Roofline Notes",
        "",
        "- RMSNorm is memory-bound for this kernel model.",
        "- The kernel reads input for reduction, reads input again for output, reads weight, and writes output.",
        "- Low arithmetic intensity means bandwidth, reduction efficiency, and launch/framework overhead dominate latency.",
        "- PyTorch baseline includes framework dispatch overhead and a less shape-specialized RMSNorm expression.",
        "",
        "## Optional Nsight Compute",
        "",
        "Run Nsight Compute on the benchmark command when available and attach metrics such as achieved occupancy, DRAM throughput, and SM efficiency.",
    ])
    write_text(path, "\n".join(lines) + "\n")


def unavailable_report(reason, output):
    payload = {
        "artifact_type": "runtime_kernel_profile",
        "source": "scripts/benchmark_rmsnorm_cuda.py",
        "profile_status": "unavailable",
        "reason": reason,
        "kernel_benchmarks": [
            {
                "fusion_candidate": "rmsnorm",
                "custom_kernel": "fused_rmsnorm_cuda",
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
    return payload


def torch_rmsnorm(torch, x, weight, eps):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cuda_transformer/rmsnorm_benchmark.json")
    parser.add_argument("--report-output", default="results/cuda_transformer/rmsnorm_benchmark_report.md")
    parser.add_argument("--tokens", default="1,16,128")
    parser.add_argument("--hidden", default="768,1024,4096,8192")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    output = ROOT / args.output

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        payload = unavailable_report(f"PyTorch import failed: {exc}", output)
        print(json.dumps(payload, indent=2))
        return 0

    if not torch.cuda.is_available():
        payload = unavailable_report("CUDA is not available on this machine", output)
        print(json.dumps(payload, indent=2))
        return 0

    extension = load(
        name="fused_rmsnorm_cuda_ext",
        sources=[
            str(KERNEL_DIR / "rmsnorm_extension.cpp"),
            str(KERNEL_DIR / "rmsnorm_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

    rows = []
    token_sizes = [int(item) for item in args.tokens.split(",") if item]
    hidden_sizes = [int(item) for item in args.hidden.split(",") if item]

    for tokens in token_sizes:
        for hidden in hidden_sizes:
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
            weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

            torch_out = torch_rmsnorm(torch, x, weight, args.eps)
            custom_out = extension.fused_rmsnorm_forward(x, weight, args.eps)
            torch.cuda.synchronize()
            correct = bool(torch.allclose(torch_out, custom_out, rtol=1e-4, atol=1e-4))

            fallback_times = measure(
                torch,
                lambda: torch_rmsnorm(torch, x, weight, args.eps),
                args.warmup,
                args.runs,
            )
            custom_times = measure(
                torch,
                lambda: extension.fused_rmsnorm_forward(x, weight, args.eps),
                args.warmup,
                args.runs,
            )

            fallback_mean = statistics.mean(fallback_times)
            custom_mean = statistics.mean(custom_times)
            perf = rmsnorm_perf_model(tokens, hidden)
            custom_stats = summarize_times(custom_times)
            fallback_stats = summarize_times(fallback_times)
            rows.append({
                "fusion_candidate": "rmsnorm",
                "shape": {
                    "tokens": tokens,
                    "hidden": hidden,
                    "dtype": "float32",
                },
                "custom_kernel": "fused_rmsnorm_cuda",
                "fallback_kernel": "torch_rmsnorm",
                "custom_latency_ms": round(custom_mean, 6),
                "fallback_latency_ms": round(fallback_mean, 6),
                "custom_p50_ms": custom_stats["p50_ms"],
                "custom_p95_ms": custom_stats["p95_ms"],
                "custom_min_ms": custom_stats["min_ms"],
                "custom_max_ms": custom_stats["max_ms"],
                "fallback_p50_ms": fallback_stats["p50_ms"],
                "fallback_p95_ms": fallback_stats["p95_ms"],
                "fallback_min_ms": fallback_stats["min_ms"],
                "fallback_max_ms": fallback_stats["max_ms"],
                "speedup": round(fallback_mean / custom_mean, 4) if custom_mean else None,
                "bytes_per_token": perf["bytes_per_token"],
                "bytes_total": perf["bytes_total"],
                "flops_per_token": perf["flops_per_token"],
                "flops_total": perf["flops_total"],
                "arithmetic_intensity_flops_per_byte": round(perf["arithmetic_intensity_flops_per_byte"], 6),
                "custom_effective_bandwidth_gbps": round(bandwidth_gbps(perf["bytes_total"], custom_mean), 3),
                "fallback_effective_bandwidth_gbps": round(bandwidth_gbps(perf["bytes_total"], fallback_mean), 3),
                "roofline_classification": "memory_bound",
                "nsight_compute": {
                    "available": False,
                    "achieved_occupancy": None,
                    "dram_throughput_gbps": None,
                    "sm_efficiency": None,
                    "note": "Optional: collect with ncu on the same benchmark command.",
                },
                "correct": correct,
                "selection_ready": correct and custom_mean < fallback_mean,
            })

    selectable = [row for row in rows if row["selection_ready"]]
    best = max(selectable, key=lambda row: row["speedup"], default=None)
    summary_row = best or min(rows, key=lambda row: row["custom_latency_ms"])

    payload = {
        "artifact_type": "runtime_kernel_profile",
        "source": "scripts/benchmark_rmsnorm_cuda.py",
        "profile_status": "measured",
        "device": torch.cuda.get_device_name(0),
        "benchmark_config": {
            "tokens": token_sizes,
            "hidden": hidden_sizes,
            "warmup": args.warmup,
            "runs": args.runs,
            "eps": args.eps,
        },
        "roofline_model": rmsnorm_perf_model(
            summary_row["shape"]["tokens"],
            summary_row["shape"]["hidden"],
        ),
        "kernel_benchmarks": [
            {
                "fusion_candidate": "rmsnorm",
                "custom_kernel": "fused_rmsnorm_cuda",
                "fallback_kernel": "torch_rmsnorm",
                "custom_latency_ms": summary_row["custom_latency_ms"],
                "fallback_latency_ms": summary_row["fallback_latency_ms"],
                "speedup": summary_row["speedup"],
                "correct": summary_row["correct"],
                "selection_ready": summary_row["selection_ready"],
                "representative_shape": summary_row["shape"],
                "custom_effective_bandwidth_gbps": summary_row["custom_effective_bandwidth_gbps"],
                "fallback_effective_bandwidth_gbps": summary_row["fallback_effective_bandwidth_gbps"],
                "bytes_per_token": summary_row["bytes_per_token"],
                "flops_per_token": summary_row["flops_per_token"],
                "arithmetic_intensity_flops_per_byte": summary_row["arithmetic_intensity_flops_per_byte"],
            }
        ],
        "sweep": rows,
    }
    write_json(output, payload)
    write_markdown_report(ROOT / args.report_output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
