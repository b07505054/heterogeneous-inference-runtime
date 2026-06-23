"""Measured GPU batch-scaling benchmark for a synthetic decode block.

This script measures real CUDA latency for a small synthetic transformer
decode block (attention + MLP) across batch sizes and context lengths. It is
NOT a full pretrained model (not Qwen), NOT vLLM/TensorRT-LLM, and NOT a
production serving benchmark. It exists to produce real GPU latency samples
that can later calibrate the runtime's CostModel; it does not itself change
any scheduler or cost-model behavior.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_TYPE = "gpu_decode_batch_scaling_benchmark"
TRUTH_BOUNDARY_NOTE = (
    "Decode/prefill latency numbers are real measurements on the reported GPU "
    "for a synthetic transformer decode block. The workload is synthetic "
    "(not a pretrained model such as Qwen), and this is not a production "
    "LLM-serving benchmark, not vLLM, and not TensorRT-LLM. This artifact "
    "does not currently feed the runtime scheduler or CostModel."
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def timestamp_utc():
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    idx = int(round((len(values) - 1) * p / 100))
    return values[idx]


def summarize_ms(times_ms):
    return {
        "mean_ms": round(statistics.mean(times_ms), 6),
        "p50_ms": round(percentile(times_ms, 50), 6),
        "p95_ms": round(percentile(times_ms, 95), 6),
        "min_ms": round(min(times_ms), 6),
        "max_ms": round(max(times_ms), 6),
    }


def provenance():
    return {
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "timestamp_utc": timestamp_utc(),
    }


def unavailable_payload(reason, config):
    return {
        "artifact_type": ARTIFACT_TYPE,
        "source": "scripts/benchmark_gpu_decode_batch_scaling.py",
        "status": "unavailable",
        "truth_boundary": "not_applicable",
        "reason": reason,
        "hardware": None,
        "software": {
            "python_version": platform.python_version(),
        },
        "config": config,
        "results": {"decode": [], "prefill": []},
        "derived": {"decode": [], "prefill": []},
        "provenance": provenance(),
        "note": (
            "This artifact was generated without CUDA/torch available. It must "
            "not be treated as a measured GPU result and should not be "
            "committed in place of a real GPU run."
        ),
    }


class SyntheticDecodeBlock:
    """A minimal attention+MLP block sized like one transformer layer's decode step."""

    def __init__(self, torch, hidden_size=4096, n_heads=32, mlp_ratio=4, dtype=None, device="cuda"):
        self.torch = torch
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.dtype = dtype
        self.device = device
        nn = torch.nn
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False).to(device=device, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)
        self.mlp_in = nn.Linear(hidden_size, mlp_ratio * hidden_size, bias=False).to(device=device, dtype=dtype)
        self.mlp_out = nn.Linear(mlp_ratio * hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype)

    def _attention(self, x, kv_cache):
        torch = self.torch
        batch, seq, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq, self.hidden_size)
        return self.out_proj(attn_out)

    def forward(self, x, kv_cache=None):
        torch = self.torch
        x = x + self._attention(x, kv_cache)
        mlp_hidden = torch.nn.functional.gelu(self.mlp_in(x))
        x = x + self.mlp_out(mlp_hidden)
        return x

    def make_kv_cache(self, batch_size, context_tokens):
        torch = self.torch
        shape = (batch_size, self.n_heads, context_tokens, self.head_dim)
        k = torch.randn(shape, device=self.device, dtype=self.dtype)
        v = torch.randn(shape, device=self.device, dtype=self.dtype)
        return k, v


def measure_ms(torch, fn, warmup, runs):
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


def memory_stats(torch):
    return {
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def derived_metrics(rows):
    if not rows:
        return []
    by_context = {}
    for row in rows:
        by_context.setdefault(row["context_tokens"], []).append(row)

    derived = []
    for context_tokens, group in by_context.items():
        group = sorted(group, key=lambda r: r["batch_size"])
        baseline = next((r for r in group if r["batch_size"] == 1), group[0])
        baseline_latency = baseline["latency_ms"]["mean_ms"]
        baseline_tps = baseline["tokens_per_second"]
        for row in group:
            latency_growth = (
                row["latency_ms"]["mean_ms"] / baseline_latency if baseline_latency else None
            )
            throughput_gain = (
                row["tokens_per_second"] / baseline_tps if baseline_tps else None
            )
            note = None
            if throughput_gain is not None and latency_growth is not None and row["batch_size"] > 1:
                if throughput_gain < row["batch_size"] * 0.7:
                    note = "diminishing_return"
            derived.append(
                {
                    "batch_size": row["batch_size"],
                    "context_tokens": context_tokens,
                    "throughput_gain_vs_batch1": round(throughput_gain, 4) if throughput_gain is not None else None,
                    "latency_growth_vs_batch1": round(latency_growth, 4) if latency_growth is not None else None,
                    "diminishing_return": note,
                }
            )
    return derived


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def dtype_from_name(torch, name):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/llm_runtime_artifacts/gpu_decode_batch_scaling_gtx1650maxq.json",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--context-tokens", default="128,512,1024,2048")
    parser.add_argument("--prefill-tokens", default="128,512,1024,2048,4096")
    parser.add_argument("--include-prefill", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--n-heads", type=int, default=32)
    args = parser.parse_args()

    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    batch_sizes = parse_int_list(args.batch_sizes)
    context_tokens_list = parse_int_list(args.context_tokens)
    prefill_tokens_list = parse_int_list(args.prefill_tokens) if args.include_prefill else []

    config = {
        "batch_sizes": batch_sizes,
        "context_tokens": context_tokens_list,
        "prefill_tokens": prefill_tokens_list,
        "include_prefill": args.include_prefill,
        "dtype": args.dtype,
        "hidden_size": args.hidden_size,
        "n_heads": args.n_heads,
        "warmup": args.warmup,
        "runs": args.runs,
    }

    try:
        import torch
    except ImportError as exc:
        payload = unavailable_payload(f"PyTorch import failed: {exc}", config)
        write_json(output, payload)
        print(json.dumps(payload, indent=2))
        return 0

    if not torch.cuda.is_available():
        payload = unavailable_payload("CUDA is not available on this machine", config)
        write_json(output, payload)
        print(json.dumps(payload, indent=2))
        return 0

    dtype = dtype_from_name(torch, args.dtype)
    device = "cuda"
    block = SyntheticDecodeBlock(
        torch,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        dtype=dtype,
        device=device,
    )

    decode_rows = []
    prefill_rows = []

    for context_tokens in context_tokens_list:
        for batch_size in batch_sizes:
            torch.cuda.reset_peak_memory_stats()
            kv_cache = block.make_kv_cache(batch_size, context_tokens)
            decode_input = torch.randn(batch_size, 1, args.hidden_size, device=device, dtype=dtype)

            def decode_step():
                with torch.no_grad():
                    block.forward(decode_input, kv_cache=kv_cache)

            times_ms = measure_ms(torch, decode_step, args.warmup, args.runs)
            mem = memory_stats(torch)
            latency_stats = summarize_ms(times_ms)
            tokens_per_second = (
                batch_size * 1000.0 / latency_stats["mean_ms"] if latency_stats["mean_ms"] else None
            )
            decode_rows.append(
                {
                    "batch_size": batch_size,
                    "context_tokens": context_tokens,
                    "latency_ms": latency_stats,
                    "tokens_per_second": round(tokens_per_second, 3) if tokens_per_second else None,
                    "memory": mem,
                }
            )

    if args.include_prefill:
        for prefill_tokens in prefill_tokens_list:
            for batch_size in batch_sizes:
                torch.cuda.reset_peak_memory_stats()
                prefill_input = torch.randn(
                    batch_size, prefill_tokens, args.hidden_size, device=device, dtype=dtype
                )

                def prefill_step():
                    with torch.no_grad():
                        block.forward(prefill_input, kv_cache=None)

                times_ms = measure_ms(torch, prefill_step, args.warmup, args.runs)
                mem = memory_stats(torch)
                latency_stats = summarize_ms(times_ms)
                tokens_per_second = (
                    batch_size * prefill_tokens * 1000.0 / latency_stats["mean_ms"]
                    if latency_stats["mean_ms"]
                    else None
                )
                prefill_rows.append(
                    {
                        "batch_size": batch_size,
                        "prefill_tokens": prefill_tokens,
                        "latency_ms": latency_stats,
                        "tokens_per_second": round(tokens_per_second, 3) if tokens_per_second else None,
                        "memory": mem,
                    }
                )

    capability = torch.cuda.get_device_capability(0)

    payload = {
        "artifact_type": ARTIFACT_TYPE,
        "source": "scripts/benchmark_gpu_decode_batch_scaling.py",
        "status": "measured",
        "truth_boundary": "measured",
        "description": TRUTH_BOUNDARY_NOTE,
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version_torch": torch.version.cuda,
            "cuda_home": os.environ.get("CUDA_HOME"),
            "device_capability": f"{capability[0]}.{capability[1]}",
        },
        "software": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "dtype": args.dtype,
        },
        "config": config,
        "results": {
            "decode": decode_rows,
            "prefill": prefill_rows,
        },
        "derived": {
            "decode": derived_metrics(decode_rows),
            "prefill": derived_metrics(
                [
                    {
                        "batch_size": row["batch_size"],
                        "context_tokens": row["prefill_tokens"],
                        "latency_ms": row["latency_ms"],
                        "tokens_per_second": row["tokens_per_second"],
                    }
                    for row in prefill_rows
                ]
            )
            if prefill_rows
            else [],
        },
        "provenance": provenance(),
    }

    write_json(output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
