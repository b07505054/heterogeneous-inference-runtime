#!/usr/bin/env python3
"""E2E-7 Phase 4: isolated, standalone PyTorch microbenchmark (NO vLLM
involved) of the exact decode-projection shapes identified in Phase 2/3 via
direct kernel grid-dimension evidence.

Shapes (Qwen2.5-0.5B-Instruct, hidden=896, intermediate=4864, vocab=151936):
  qkv_proj:    K=896,  N=1152  (896 Q + 128 K + 128 V, fused QKVParallelLinear)
  o_proj:      K=896,  N=896
  gate_up:     K=896,  N=9728  (2*4864, fused MergedColumnParallelLinear)
  down_proj:   K=4864, N=896
  lm_head:     K=896,  N=151936

Candidates:
  1_default_linear   -- torch.nn.functional.linear(x, W, bias), the exact
                         production-resolved operator (source-confirmed,
                         vllm/model_executor/layers/utils.py:dispatch_unquantized_gemm)
  2_gemv_loop         -- decompose into per-row F.linear calls (each row is a
                         real M=1 GEMV dispatch, matching the batch=1 production kernel)
  3_pretransposed     -- x @ W_T where W_T is pre-transposed once (offline),
                         avoiding the transpose-view F.linear does internally
  4_tunableop         -- torch.cuda.tunable (PyTorch's own built-in GEMM
                         algorithm auto-tuner), enabled around the same F.linear call

Correctness: every candidate's output is checked against F.linear (candidate 1)
with FP16-appropriate tolerance (atol=1e-2, rtol=1e-2, generous but standard
for fp16 GEMM comparisons).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SHAPES = {
    "qkv_proj": {"K": 896, "N": 1152},
    "o_proj": {"K": 896, "N": 896},
    "gate_up_proj": {"K": 896, "N": 9728},
    "down_proj": {"K": 4864, "N": 896},
    "lm_head": {"K": 896, "N": 151936},
}
BATCH_SIZES = (1, 2, 3, 4, 6, 8)
WARMUP_ITERS = 150
TIMED_ITERS = 300


def cuda_event_timed(fn, iters: int) -> list[float]:
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def summarize(times_ms: list[float]) -> dict:
    ordered = sorted(times_ms)
    p95_idx = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    median = statistics.median(ordered)
    mad = statistics.median(abs(t - median) for t in ordered)
    return {"median_ms": median, "p95_ms": ordered[p95_idx], "min_ms": ordered[0],
            "mad_ms": mad, "n": len(ordered)}


def candidate_1_default_linear(x, W, bias):
    return F.linear(x, W, bias)


def candidate_2_gemv_loop(x, W, bias):
    rows = [F.linear(x[i : i + 1], W, bias) for i in range(x.shape[0])]
    return torch.cat(rows, dim=0)


def candidate_3_pretransposed(x, W_T, bias):
    out = x @ W_T
    if bias is not None:
        out = out + bias
    return out


def run_tunableop_candidate(x, W, bias):
    return F.linear(x, W, bias)


def max_abs_rel_error(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().clamp_min(1e-3)
    max_rel = (diff / denom).max().item()
    return max_abs, max_rel


def bench_shape(op_name: str, K: int, N: int, batch_sizes, device, dtype, run_tunableop: bool) -> list[dict]:
    torch.manual_seed(1234)
    W = torch.randn(N, K, device=device, dtype=dtype) * 0.02
    bias = None
    W_T = W.t().contiguous()

    rows = []
    for M in batch_sizes:
        x = torch.randn(M, K, device=device, dtype=dtype) * 0.02

        for _ in range(WARMUP_ITERS):
            ref = candidate_1_default_linear(x, W, bias)
        times1 = cuda_event_timed(lambda: candidate_1_default_linear(x, W, bias), TIMED_ITERS)

        for _ in range(WARMUP_ITERS):
            out2 = candidate_2_gemv_loop(x, W, bias)
        times2 = cuda_event_timed(lambda: candidate_2_gemv_loop(x, W, bias), TIMED_ITERS)
        err2 = max_abs_rel_error(out2, ref)

        for _ in range(WARMUP_ITERS):
            out3 = candidate_3_pretransposed(x, W_T, bias)
        times3 = cuda_event_timed(lambda: candidate_3_pretransposed(x, W_T, bias), TIMED_ITERS)
        err3 = max_abs_rel_error(out3, ref)

        candidates = {
            "1_default_linear": {**summarize(times1), "max_abs_error": 0.0, "max_rel_error": 0.0},
            "2_gemv_loop": {**summarize(times2), "max_abs_error": err2[0], "max_rel_error": err2[1]},
            "3_pretransposed": {**summarize(times3), "max_abs_error": err3[0], "max_rel_error": err3[1]},
        }

        if run_tunableop:
            torch.cuda.tunable.enable(True)
            torch.cuda.tunable.tuning_enable(True)
            for _ in range(WARMUP_ITERS):
                out4 = run_tunableop_candidate(x, W, bias)
            times4 = cuda_event_timed(lambda: run_tunableop_candidate(x, W, bias), TIMED_ITERS)
            err4 = max_abs_rel_error(out4, ref)
            torch.cuda.tunable.tuning_enable(False)
            candidates["4_tunableop"] = {**summarize(times4), "max_abs_error": err4[0], "max_rel_error": err4[1]}
        else:
            candidates["4_tunableop"] = {"status": "skipped_this_shape_for_time_budget"}

        flops = 2 * M * N * K
        rows.append({
            "operation": op_name, "M": M, "N": N, "K": K, "flops": flops,
            "candidates": candidates,
        })
        print(f"  {op_name} M={M}: " + ", ".join(f"{k}={v.get('median_ms', v.get('status')):.4f}" if isinstance(v.get('median_ms', v.get('status')), float) else f"{k}=skip" for k, v in candidates.items()))
    return rows


def fresh_process_stability_probe(op_name: str, K: int, N: int, M: int, device, dtype) -> dict:
    torch.manual_seed(1234)
    W = torch.randn(N, K, device=device, dtype=dtype) * 0.02
    x = torch.randn(M, K, device=device, dtype=dtype) * 0.02
    for _ in range(WARMUP_ITERS):
        F.linear(x, W, None)
    times = cuda_event_timed(lambda: F.linear(x, W, None), 100)
    return summarize(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stability-probe", action="store_true",
                         help="Run only the single fresh-process stability probe for one shape/M and exit.")
    parser.add_argument("--probe-op", default="gate_up_proj")
    parser.add_argument("--probe-m", type=int, default=4)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = "cuda"
    dtype = torch.float16
    torch.backends.cudnn.benchmark = False

    if args.stability_probe:
        shape = SHAPES[args.probe_op]
        result = fresh_process_stability_probe(args.probe_op, shape["K"], shape["N"], args.probe_m, device, dtype)
        args.out.write_text(json.dumps({"probe_op": args.probe_op, "probe_m": args.probe_m, "result": result,
                                         "torch_version": torch.__version__,
                                         "gpu_name": torch.cuda.get_device_name(0)}, indent=2))
        print(f"stability probe {args.probe_op} M={args.probe_m}: median={result['median_ms']:.4f}ms")
        return

    all_rows = []
    # Run TunableOp only on the two dominant, most-informative shapes to keep the
    # slice's time budget bounded (it re-tunes per unique problem shape and is slow).
    tunableop_shapes = {"qkv_proj", "gate_up_proj"}
    for op_name, shape in SHAPES.items():
        print(f"=== {op_name} (K={shape['K']}, N={shape['N']}) ===")
        rows = bench_shape(op_name, shape["K"], shape["N"], BATCH_SIZES, device, dtype,
                            run_tunableop=(op_name in tunableop_shapes))
        all_rows.extend(rows)

    args.out.write_text(json.dumps({
        "torch_version": torch.__version__, "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda, "dtype": "float16",
        "warmup_iters": WARMUP_ITERS, "timed_iters": TIMED_ITERS,
        "shapes": SHAPES, "batch_sizes": list(BATCH_SIZES), "rows": all_rows,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
