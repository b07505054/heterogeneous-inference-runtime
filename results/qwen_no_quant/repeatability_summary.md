# Qwen No-Quant Repeatability Study

Compiler-guided no-quant Qwen uses original Qwen weights. Differences come from execution/runtime policy, not model weight optimization.

- Created UTC: `2026-07-06T21:26:27.155812+00:00`
- Host: `allen-ZenBook-UX534FTC-UX534FT`
- Runtime: `/home/allen/Desktop/Project/heterogeneous-inference-runtime`
- Virtual env: `/home/allen/Desktop/Project/heterogeneous-inference-runtime/.venv`
- Python: `3.12.13` at `/home/allen/Desktop/Project/heterogeneous-inference-runtime/.venv/bin/python`
- Warmup: `4`
- Concurrency: `1`
- Port/base URL: `http://127.0.0.1:8000`

## Commands

### baseline-conservative-fixed
```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct --gpu-memory-utilization 0.75 --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 16 --dtype float16 --tensor-parallel-size 1 --pipeline-parallel-size 1
```

### compiler-guided-fixed
```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.75 --block-size 16 --max-num-seqs 4 --max-num-batched-tokens 2048 --tensor-parallel-size 1 --pipeline-parallel-size 1 --served-model-name qwen2.5-0.5b
```

Remaining command/config difference: compiler-guided uses `--served-model-name qwen2.5-0.5b`; baseline does not. Other low-memory runtime flags are equivalent, with minor ordering differences only.

Served-model-name affects OpenAI model routing by changing the model id accepted by `/v1/chat/completions` and returned by `/v1/models`. In these runs, baseline benchmark requests used `Qwen/Qwen2.5-0.5B-Instruct`; compiler-guided requests used `qwen2.5-0.5b`. The served model root remained `Qwen/Qwen2.5-0.5B-Instruct`, so routing points at the same original weights.

## Per-Trial Results

| Workload | Path | Trial | TTFT mean/p50/p95 ms | TPOT mean/p50/p95 ms | E2E mean/p50/p95 ms | tok/s | success/error | GPU ready/after MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `short` | `baseline-conservative-fixed` | 1 | 152.760/152.527/154.587 | 12.761/12.763/12.889 | 956.693/956.342/965.380 | 66.897 | 28/0 | existing/existing |
| `short` | `baseline-conservative-fixed` | 2 | 146.389/146.475/147.877 | 12.036/12.045/12.085 | 904.678/905.557/908.956 | 70.743 | 28/0 | 3/3 |
| `short` | `baseline-conservative-fixed` | 3 | 146.958/147.180/148.035 | 12.074/12.087/12.113 | 907.590/908.549/911.180 | 70.516 | 28/0 | 3/3 |
| `short` | `compiler-guided-fixed` | 1 | 156.855/156.881/158.794 | 13.042/13.023/13.213 | 978.474/976.850/990.243 | 65.408 | 28/0 | existing/existing |
| `short` | `compiler-guided-fixed` | 2 | 142.975/143.100/144.132 | 11.897/11.899/11.944 | 892.498/892.800/896.363 | 71.709 | 28/0 | 3/3 |
| `short` | `compiler-guided-fixed` | 3 | 147.903/147.933/148.996 | 12.115/12.121/12.154 | 911.168/911.569/914.318 | 70.240 | 28/0 | 3/3 |
| `shared_prefix` | `baseline-conservative-fixed` | 1 | 155.563/155.691/156.287 | 12.973/12.987/13.024 | 972.887/974.103/975.658 | 65.784 | 28/0 | existing/existing |
| `shared_prefix` | `baseline-conservative-fixed` | 2 | 147.738/147.666/148.411 | 12.125/12.127/12.150 | 911.600/911.885/913.870 | 70.206 | 28/0 | 3/3 |
| `shared_prefix` | `baseline-conservative-fixed` | 3 | 149.245/149.501/150.378 | 12.185/12.193/12.217 | 916.905/917.239/919.668 | 69.800 | 28/0 | 3/3 |
| `shared_prefix` | `compiler-guided-fixed` | 1 | 160.336/160.074/161.789 | 13.392/13.383/13.539 | 1004.024/1004.372/1012.789 | 63.743 | 28/0 | existing/existing |
| `shared_prefix` | `compiler-guided-fixed` | 2 | 145.230/145.290/146.516 | 12.004/12.002/12.037 | 901.459/901.642/904.621 | 70.996 | 28/0 | 3/3 |
| `shared_prefix` | `compiler-guided-fixed` | 3 | 149.617/149.714/150.221 | 12.191/12.191/12.217 | 917.652/917.729/919.658 | 69.743 | 28/0 | 3/3 |
| `no_shared_prefix` | `baseline-conservative-fixed` | 1 | 158.100/158.013/159.303 | 13.139/13.162/13.215 | 985.854/987.433/990.673 | 64.918 | 28/0 | existing/existing |
| `no_shared_prefix` | `baseline-conservative-fixed` | 2 | 149.145/149.240/149.836 | 12.158/12.161/12.173 | 915.119/915.128/916.681 | 69.936 | 28/0 | 3/3 |
| `no_shared_prefix` | `baseline-conservative-fixed` | 3 | 150.554/150.509/151.227 | 12.223/12.223/12.244 | 920.598/920.759/922.145 | 69.520 | 28/0 | 3/3 |
| `no_shared_prefix` | `compiler-guided-fixed` | 1 | 162.310/162.235/163.499 | 13.689/13.667/13.864 | 1024.723/1022.877/1036.907 | 62.456 | 28/0 | existing/existing |
| `no_shared_prefix` | `compiler-guided-fixed` | 2 | 146.685/146.675/147.296 | 12.054/12.055/12.075 | 906.088/906.351/907.493 | 70.633 | 28/0 | 3/3 |
| `no_shared_prefix` | `compiler-guided-fixed` | 3 | 150.721/150.846/151.442 | 12.234/12.238/12.264 | 921.487/921.696/923.650 | 69.453 | 28/0 | 3/3 |

## Aggregate Delta

Positive latency delta means compiler-guided is slower. Positive throughput delta means compiler-guided has lower throughput.

| Workload | Metric | Baseline mean | Baseline stdev | Compiler mean | Compiler stdev | Delta % |
|---|---|---:|---:|---:|---:|---:|
| `short` | TTFT mean ms | 148.702 | 3.525 | 149.245 | 7.037 | 0.365% |
| `short` | TPOT mean ms | 12.290 | 0.408 | 12.351 | 0.608 | 0.497% |
| `short` | E2E mean ms | 922.987 | 29.227 | 927.380 | 45.222 | 0.476% |
| `short` | tok/s | 69.386 | 2.158 | 69.119 | 3.297 | 0.385% |
| `shared_prefix` | TTFT mean ms | 150.849 | 4.152 | 151.728 | 7.771 | 0.583% |
| `shared_prefix` | TPOT mean ms | 12.428 | 0.473 | 12.529 | 0.753 | 0.813% |
| `shared_prefix` | E2E mean ms | 933.797 | 33.956 | 941.045 | 55.139 | 0.776% |
| `shared_prefix` | tok/s | 68.597 | 2.445 | 68.161 | 3.877 | 0.635% |
| `no_shared_prefix` | TTFT mean ms | 152.600 | 4.815 | 153.238 | 8.111 | 0.418% |
| `no_shared_prefix` | TPOT mean ms | 12.507 | 0.548 | 12.659 | 0.896 | 1.219% |
| `no_shared_prefix` | E2E mean ms | 940.524 | 39.353 | 950.766 | 64.510 | 1.089% |
| `no_shared_prefix` | tok/s | 68.125 | 2.785 | 67.514 | 4.420 | 0.897% |

## Conclusion

- `short`: E2E mean delta `0.476%`; pooled E2E CV `4.115%`; judgment: likely within benchmark noise.
- `shared_prefix`: E2E mean delta `0.776%`; pooled E2E CV `4.885%`; judgment: likely within benchmark noise.
- `no_shared_prefix`: E2E mean delta `1.089%`; pooled E2E CV `5.650%`; judgment: likely within benchmark noise.

Do not claim speedup from these results. This repeatability pass is only validating whether the prior 2-3% difference is stable or benchmark noise.
