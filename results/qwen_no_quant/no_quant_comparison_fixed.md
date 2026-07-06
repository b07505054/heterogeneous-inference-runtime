# Qwen No-Quant Fixed Fair Benchmark

Compiler-guided no-quant Qwen uses the original Qwen weights. Differences come from execution/runtime policy, not model weight optimization.

## Fixed Compiler-Guided Command

```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.75 --block-size 16 --max-num-seqs 4 --max-num-batched-tokens 2048 --tensor-parallel-size 1 --pipeline-parallel-size 1 --served-model-name qwen2.5-0.5b
```

## Results

| Workload | Path | TTFT p50 / p95 / mean ms | TPOT p50 / p95 / mean ms | E2E p50 / p95 / mean ms | Throughput tok/s | Success | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| `short` | `baseline_conservative_fixed` | 145.642 / 147.306 / 145.322 | 12.003 / 12.107 / 12.006 | 901.790 / 910.018 / 901.704 | 70.977 | 28 | 0 |
| `short` | `compiler_guided_fixed` | 152.421 / 154.667 / 152.642 | 12.328 / 12.423 / 12.319 | 928.659 / 936.930 / 928.733 | 68.911 | 28 | 0 |
| `shared_prefix` | `baseline_conservative_fixed` | 150.238 / 151.865 / 150.394 | 12.230 / 12.315 / 12.234 | 921.076 / 927.606 / 921.129 | 69.480 | 28 | 0 |
| `shared_prefix` | `compiler_guided_fixed` | 156.376 / 156.823 / 156.148 | 12.530 / 12.594 / 12.530 | 945.644 / 949.969 / 945.567 | 67.684 | 28 | 0 |
| `no_shared_prefix` | `baseline_conservative_fixed` | 153.754 / 155.256 / 153.865 | 12.393 / 12.459 / 12.390 | 934.864 / 939.751 / 934.456 | 68.489 | 28 | 0 |
| `no_shared_prefix` | `compiler_guided_fixed` | 158.134 / 159.893 / 158.222 | 12.646 / 12.714 / 12.649 | 954.632 / 960.626 / 955.082 | 67.010 | 28 | 0 |

## GPU Memory

- Idle before/after fixed rerun: 3 MiB / 4096 MiB.
- Baseline-conservative-fixed ready/after: 2897 / 2899 MiB.
- Compiler-guided-fixed ready/after: 2897 / 2899 MiB.
