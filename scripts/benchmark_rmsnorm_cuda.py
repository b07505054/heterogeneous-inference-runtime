import argparse
import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "cuda_transformer_kernels"

# Block-size policy lab, Phase 1: the launch block size is the ONLY
# controlled variable. 256 is the original fixed launch configuration and
# remains the default policy.
DEFAULT_BLOCK_SIZE = 256

CSV_FIELDS = [
    "tokens",
    "hidden",
    "block_size",
    "dtype",
    "custom_median_ms",
    "custom_p50_ms",
    "custom_p95_ms",
    "fallback_median_ms",
    "fallback_p50_ms",
    "fallback_p95_ms",
    "speedup",
    "correct",
]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            shape = row["shape"]
            writer.writerow({
                "tokens": shape["tokens"],
                "hidden": shape["hidden"],
                "block_size": row["block_size"],
                "dtype": shape["dtype"],
                "custom_median_ms": row["custom_p50_ms"],
                "custom_p50_ms": row["custom_p50_ms"],
                "custom_p95_ms": row["custom_p95_ms"],
                "fallback_median_ms": row["fallback_p50_ms"],
                "fallback_p50_ms": row["fallback_p50_ms"],
                "fallback_p95_ms": row["fallback_p95_ms"],
                "speedup": row["speedup"],
                "correct": row["correct"],
            })


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def git_commit_hash():
    return run_command(["git", "rev-parse", "--short", "HEAD"])


def git_dirty():
    output = run_command(["git", "status", "--short"])
    return bool(output)


def driver_version():
    output = run_command([
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ])
    if not output:
        return None
    return output.splitlines()[0].strip()


def gpu_uuid():
    output = run_command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"])
    return output.splitlines()[0].strip() if output else None


def cuda_version_from_nvcc():
    output = run_command(["nvcc", "--version"])
    if not output:
        return None
    for line in output.splitlines():
        if "release" in line:
            return line.strip()
    return output.splitlines()[-1].strip() if output.splitlines() else None


def nsight_compute_metadata(with_nsight):
    ncu_path = shutil.which("ncu")
    if not with_nsight:
        return {
            "requested": False,
            "available": ncu_path is not None,
            "path": ncu_path,
            "reason": "not requested",
        }
    if not ncu_path:
        return {
            "requested": True,
            "available": False,
            "path": None,
            "reason": "ncu not found",
        }
    return {
        "requested": True,
        "available": True,
        "path": ncu_path,
        "version": run_command([ncu_path, "--version"]),
        "note": "ncu is available; collect detailed occupancy/DRAM metrics with the benchmark command under Nsight Compute.",
    }


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


def representative_row(rows):
    # The legacy summary stays comparable with pre-sweep artifacts: only the
    # default block size (the original fixed launch configuration) is
    # considered representative.
    default_rows = [row for row in rows if row["block_size"] == DEFAULT_BLOCK_SIZE]
    candidates = default_rows or rows
    for row in candidates:
        shape = row["shape"]
        if shape["tokens"] == 16 and shape["hidden"] == 4096 and shape["dtype"] == "float32":
            return row
    selectable = [row for row in candidates if row["selection_ready"]]
    return max(selectable, key=lambda row: row["speedup"], default=None) or min(
        candidates,
        key=lambda row: row["custom_latency_ms"],
    )


def write_markdown_report(path, payload):
    env = payload.get("environment", {})
    config = payload.get("benchmark_config", {})
    nsight = payload.get("nsight_compute", {})
    lines = [
        "# CUDA RMSNorm Benchmark Report",
        "",
        f"Status: `{payload.get('profile_status')}`",
        "",
        "## Environment",
        "",
        f"- GPU: `{env.get('gpu_name')}`",
        f"- CUDA version: `{env.get('cuda_version')}`",
        f"- NVCC version: `{env.get('nvcc_version')}`",
        f"- PyTorch version: `{env.get('pytorch_version')}`",
        f"- Driver version: `{env.get('driver_version')}`",
        f"- Commit: `{env.get('commit_hash')}`",
        f"- Git dirty: `{env.get('git_dirty')}`",
        f"- Warmup runs: `{config.get('warmup')}`",
        f"- Timed runs: `{config.get('runs')}`",
        f"- Dtype: `{config.get('dtype')}`",
        "",
        "## Shape x Block-Size Sweep",
        "",
        "| Tokens | Hidden | Block | Custom ms | PyTorch ms | Speedup | Custom GB/s | Correct |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("sweep", []):
        shape = row["shape"]
        lines.append(
            "| {tokens} | {hidden} | {block} | {custom} | {fallback} | {speedup} | {bw} | {correct} |".format(
                tokens=shape["tokens"],
                hidden=shape["hidden"],
                block=row["block_size"],
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
        f"- Requested: `{nsight.get('requested')}`",
        f"- Available: `{nsight.get('available')}`",
        f"- Reason: `{nsight.get('reason')}`",
        f"- Path: `{nsight.get('path')}`",
        "",
        "When available, run Nsight Compute on the same benchmark command and attach metrics such as achieved occupancy, DRAM throughput, and SM efficiency.",
    ])
    write_text(path, "\n".join(lines) + "\n")


def unavailable_report(reason, output, report_output=None, nsight=None):
    payload = {
        "artifact_type": "runtime_kernel_profile",
        "source": "scripts/benchmark_rmsnorm_cuda.py",
        "profile_status": "unavailable",
        "reason": reason,
        "environment": {
            "commit_hash": git_commit_hash(),
            "git_dirty": git_dirty(),
        },
        "nsight_compute": nsight or nsight_compute_metadata(False),
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
    if report_output:
        write_markdown_report(report_output, payload)
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
    parser.add_argument("--csv-output", default="results/cuda_transformer/rmsnorm_block_size_sweep.csv")
    parser.add_argument("--tokens", default="1,16,128")
    parser.add_argument("--hidden", default="768,1024,4096,8192")
    parser.add_argument("--block-sizes", default="64,128,256,512")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--with-nsight", action="store_true")
    parser.add_argument("--selected-candidate-id")
    parser.add_argument("--proof-output")
    args = parser.parse_args()

    output = ROOT / args.output
    report_output = ROOT / args.report_output
    csv_output = ROOT / args.csv_output
    nsight = nsight_compute_metadata(args.with_nsight)

    try:
        import torch
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        payload = unavailable_report(f"PyTorch import failed: {exc}", output, report_output, nsight)
        print(json.dumps(payload, indent=2))
        return 0

    if not torch.cuda.is_available():
        payload = unavailable_report("CUDA is not available on this machine", output, report_output, nsight)
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
    source_hash = hashlib.sha256(
        (KERNEL_DIR / "rmsnorm_kernel.cu").read_bytes()
        + (KERNEL_DIR / "rmsnorm_extension.cpp").read_bytes()
    ).hexdigest()

    rows = []
    token_sizes = [int(item) for item in args.tokens.split(",") if item]
    hidden_sizes = [int(item) for item in args.hidden.split(",") if item]
    block_sizes = [int(item) for item in args.block_sizes.split(",") if item]
    unsupported = sorted(set(block_sizes) - {64, 128, 256, 512})
    if unsupported:
        raise SystemExit(f"unsupported CUDA RMSNorm block sizes: {unsupported}")

    for tokens in token_sizes:
        for hidden in hidden_sizes:
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float32)
            weight = torch.randn(hidden, device="cuda", dtype=torch.float32)

            torch_out = torch_rmsnorm(torch, x, weight, args.eps)
            torch.cuda.synchronize()

            # The PyTorch baseline is block-size independent: measure once
            # per shape and reuse across block-size rows.
            fallback_times = measure(
                torch,
                lambda: torch_rmsnorm(torch, x, weight, args.eps),
                args.warmup,
                args.runs,
            )
            fallback_mean = statistics.mean(fallback_times)
            fallback_stats = summarize_times(fallback_times)
            perf = rmsnorm_perf_model(tokens, hidden)

            for block_size in block_sizes:
                custom_out = extension.fused_rmsnorm_forward(
                    x, weight, args.eps, block_size
                )
                torch.cuda.synchronize()
                correct = bool(
                    torch.allclose(torch_out, custom_out, rtol=1e-4, atol=1e-4)
                )
                abs_error = (torch_out - custom_out).abs()
                rel_error = abs_error / torch_out.abs().clamp_min(1e-12)

                custom_times = measure(
                    torch,
                    lambda: extension.fused_rmsnorm_forward(
                        x, weight, args.eps, block_size
                    ),
                    args.warmup,
                    args.runs,
                )

                custom_mean = statistics.mean(custom_times)
                custom_stats = summarize_times(custom_times)
                rows.append({
                    "candidate_id": f"cuda_rmsnorm_fp32_bs{block_size}_v1",
                    "operator": "rmsnorm",
                    "semantics": "weighted_rmsnorm",
                    "backend": "cuda",
                    "kernel_family": "custom_cuda_rmsnorm",
                    "kernel_entry_point": "fused_rmsnorm_forward",
                    "dtype": "fp32",
                    "tokens": tokens,
                    "hidden": hidden,
                    "epsilon": args.eps,
                    "source_hash": source_hash,
                    "fusion_candidate": "rmsnorm",
                    "shape": {
                        "tokens": tokens,
                        "hidden": hidden,
                        "dtype": "float32",
                    },
                    "block_size": block_size,
                    "num_warps": None,
                    "num_stages": None,
                    "custom_kernel": "fused_rmsnorm_cuda",
                    "fallback_kernel": "torch_rmsnorm",
                    "custom_latency_ms": round(custom_mean, 6),
                    "fallback_latency_ms": round(fallback_mean, 6),
                    "custom_median_ms": custom_stats["p50_ms"],
                    "custom_p50_ms": custom_stats["p50_ms"],
                    "custom_p95_ms": custom_stats["p95_ms"],
                    "custom_min_ms": custom_stats["min_ms"],
                    "custom_max_ms": custom_stats["max_ms"],
                    "fallback_median_ms": fallback_stats["p50_ms"],
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
                    "nsight_compute": nsight,
                    "correct": correct,
                    "max_absolute_error": float(abs_error.max().item()),
                    "max_relative_error": float(rel_error.max().item()),
                    "failure_reason": None,
                    "selection_ready": correct,
                })

    summary_row = representative_row(rows)

    payload = {
        "artifact_type": "runtime_kernel_profile",
        "source": "scripts/benchmark_rmsnorm_cuda.py",
        "profile_status": "measured",
        "device": torch.cuda.get_device_name(0),
        "environment": {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "nvcc_version": cuda_version_from_nvcc(),
            "pytorch_version": torch.__version__,
            "driver_version": driver_version(),
            "commit_hash": git_commit_hash(),
            "git_dirty": git_dirty(),
            "compute_capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
            "gpu_uuid": gpu_uuid(),
        },
        "benchmark_config": {
            "tokens": token_sizes,
            "hidden": hidden_sizes,
            "block_sizes": block_sizes,
            "default_block_size": DEFAULT_BLOCK_SIZE,
            "controlled_variable": "block_size_only",
            "warmup": args.warmup,
            "runs": args.runs,
            "eps": args.eps,
            "dtype": "float32",
            "rtol": 1e-4,
            "atol": 1e-4,
        },
        "nsight_compute": nsight,
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
                "representative_block_size": summary_row["block_size"],
                "custom_effective_bandwidth_gbps": summary_row["custom_effective_bandwidth_gbps"],
                "fallback_effective_bandwidth_gbps": summary_row["fallback_effective_bandwidth_gbps"],
                "bytes_per_token": summary_row["bytes_per_token"],
                "flops_per_token": summary_row["flops_per_token"],
                "arithmetic_intensity_flops_per_byte": summary_row["arithmetic_intensity_flops_per_byte"],
            }
        ],
        "sweep": rows,
        "exact_candidates": rows,
    }
    write_json(output, payload)
    write_csv(csv_output, rows)
    write_markdown_report(report_output, payload)
    if args.selected_candidate_id:
        matching = [row for row in rows if row["candidate_id"] == args.selected_candidate_id]
        if len(matching) != 1 or len(rows) != 1:
            raise SystemExit("selected candidate must identify the single executed shape/configuration")
        proof = {
            "selected_candidate_id": args.selected_candidate_id,
            "executed_candidate_id": matching[0]["candidate_id"],
            "selected_block_size": matching[0]["block_size"],
            "executed_block_size": matching[0]["block_size"],
            "redecision_count": 0,
            "correct": matching[0]["correct"],
        }
        if args.proof_output:
            write_json(Path(args.proof_output), proof)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
