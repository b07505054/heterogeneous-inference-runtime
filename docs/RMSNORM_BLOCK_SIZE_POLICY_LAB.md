# RMSNorm CUDA Block-Size Policy Lab — Phase 1

## Truth boundary

Custom CUDA RMSNorm results are **kernel-level correctness and
microbenchmark evidence only** — not an end-to-end Qwen or vLLM execution
path. No winning block size may be claimed unless supported by measured
results from an actual CUDA-host run of the sweep.

## Experiment design

**Controlled variable: launch block size only** — 64, 128, 256, or 512
threads per block, as compile-time template specializations with an
explicit dispatcher (`cuda_transformer_kernels/rmsnorm_kernel.cu`).
Everything else is deliberately fixed at the Phase-1 baseline:

- algorithm: one CUDA block per token row, warp shuffle reduction +
  shared-memory block reduction, two-pass input access (read for
  sum-of-squares, read again for the normalized write);
- dtype: FP32 input, weight, accumulation, and output;
- memory access: scalar loads/stores (no float2/float4 vectorization);
- no register caching of the input, no shared-memory reduction
  replacement, no split-row decomposition, no fused residual, no
  FP16/BF16, no auto-tuning (all explicitly out of Phase-1 scope).

The default block size is **256** — identical to the original fixed
`kThreadsPerBlock = 256` launch configuration — so existing callers,
execution plans, and artifacts are unaffected unless a block size is
explicitly requested.

## Hypotheses (to be tested, not assumed)

- **Small hidden (768/1024):** large blocks waste threads — with
  hidden = 768 and block 512, ~40% of lanes in the grid-stride loop do at
  most one element, while the block-level reduction still synchronizes all
  16 warps. Smaller blocks (64/128) may reduce reduction/synchronization
  overhead and win.
- **Large hidden (4096/8192):** each thread covers more elements; larger
  blocks expose more memory-level parallelism per row and may achieve
  higher effective bandwidth for the two row passes.
- **tokens = 1:** exactly one block is resident — the GPU is inherently
  underutilized regardless of block size (one SM active out of many), so
  latency is dominated by launch overhead and single-SM bandwidth. Block
  size effects may be visible but total device utilization stays low;
  results at tokens = 1 say little about batched behavior.

## Why higher occupancy does not automatically imply lower latency

Occupancy measures how many warps *can* be resident, not how fast the work
finishes. This kernel is memory-bound (arithmetic intensity « 1 FLOP/byte):
once enough warps are in flight to saturate DRAM bandwidth for the resident
rows, additional occupancy adds no throughput — while larger blocks make
each row's `__syncthreads()`-separated reduction wider and its tail effects
worse. A 512-thread block on hidden = 768 has high theoretical occupancy
per SM and mostly idle lanes. Latency follows from achieved bandwidth,
reduction critical path, and launch overhead — not from the occupancy
number.

## Nsight Compute counters to inspect (when profiling on a CUDA host)

- `sm__warps_active.avg.pct_of_peak_sustained_active` — achieved occupancy.
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` and
  `dram__bytes.sum` — whether the two-pass row traffic saturates DRAM.
- `l1tex__t_bytes.sum` / `lts__t_bytes.sum` — cache behavior of the second
  input pass (the reload may hit in L2 for small rows).
- `smsp__cycles_active.avg` vs `gpu__time_duration.sum` — reduction/sync
  stalls versus total kernel time.
- `launch__registers_per_thread`, `launch__occupancy_limit_*` — what bounds
  residency at each block size.

Capture via `scripts/capture_rmsnorm_nsight_compute.py` or by running the
benchmark command under `ncu`.

## How to run

```bash
# Full matrix: tokens {1,16,128} x hidden {768,1024,4096,8192} x block {64,128,256,512}
scripts/run_rmsnorm_block_size_sweep.sh

# Individually:
.venv/bin/python scripts/test_rmsnorm_cuda_correctness.py \
  --tokens 1,16,128 --hidden 768,1024,4096,8192 --block-sizes 64,128,256,512
.venv/bin/python scripts/benchmark_rmsnorm_cuda.py \
  --block-sizes 64,128,256,512 \
  --csv-output results/cuda_transformer/rmsnorm_block_size_sweep.csv
```

Correctness reports max absolute error, max relative error, and pass/fail
per (tokens, hidden, block_size); it also verifies that unsupported block
sizes (e.g. 96) and non-contiguous inputs are rejected. The benchmark warms
up before measuring, times with CUDA events (extension compilation,
allocation, and host–device transfers excluded from kernel timing),
measures the PyTorch RMSNorm baseline once per shape, and emits JSON, CSV
(median/p50/p95, speedup, correctness per row), and a markdown report.
Both scripts skip cleanly (correctness: `SKIP`, exit 0; benchmark:
`profile_status: "unavailable"` artifact) when CUDA is absent.

## Execution-plan integration (optional, default-off)

`CustomCudaBackendAdapter` honors an optional
`benchmark_config["rmsnorm_block_size"]` on the
`CUSTOM_CUDA_MICROBENCHMARK` path: when present, the materialized benchmark
command gains `--block-sizes N` and the config records
`kernel_policy: {block_size: N}`. When absent — the case for every existing
plan — commands and config are byte-identical to the previous behavior.
