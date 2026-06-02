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


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    idx = int(round((len(values) - 1) * p / 100))
    return values[idx]


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
    parser.add_argument("--tokens", default="1,16,128,512")
    parser.add_argument("--hidden", default="768,1024,2048,4096")
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
                "custom_p95_ms": round(percentile(custom_times, 95), 6),
                "fallback_p95_ms": round(percentile(fallback_times, 95), 6),
                "speedup": round(fallback_mean / custom_mean, 4) if custom_mean else None,
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
            }
        ],
        "sweep": rows,
    }
    write_json(output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
