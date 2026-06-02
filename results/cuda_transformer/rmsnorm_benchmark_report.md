# CUDA RMSNorm Benchmark Report

Status: `measured`
Device: `NVIDIA GeForce GTX 1650 with Max-Q Design`

| Tokens | Hidden | Custom ms | PyTorch ms | Speedup | Custom GB/s | Correct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 768 | 0.032968 | 0.093578 | 2.8385 | 0.373 | True |
| 1 | 1024 | 0.032771 | 0.087741 | 2.6774 | 0.5 | True |
| 1 | 4096 | 0.03156 | 0.089118 | 2.8238 | 2.077 | True |
| 1 | 8192 | 0.033632 | 0.08897 | 2.6454 | 3.897 | True |
| 16 | 768 | 0.032454 | 0.089096 | 2.7453 | 6.058 | True |
| 16 | 1024 | 0.03268 | 0.089155 | 2.7281 | 8.021 | True |
| 16 | 4096 | 0.031597 | 0.090415 | 2.8615 | 33.186 | True |
| 16 | 8192 | 0.03889 | 0.089868 | 2.3108 | 53.925 | True |
| 128 | 768 | 0.031841 | 0.090669 | 2.8476 | 49.398 | True |
| 128 | 1024 | 0.031981 | 0.089589 | 2.8013 | 65.575 | True |
| 128 | 4096 | 0.084528 | 0.181715 | 2.1498 | 99.24 | True |
| 128 | 8192 | 0.15287 | 0.337329 | 2.2066 | 109.748 | True |

## Roofline Notes

- RMSNorm is memory-bound for this kernel model.
- The kernel reads input for reduction, reads input again for output, reads weight, and writes output.
- Low arithmetic intensity means bandwidth, reduction efficiency, and launch/framework overhead dominate latency.
- PyTorch baseline includes framework dispatch overhead and a less shape-specialized RMSNorm expression.

## Optional Nsight Compute

Run Nsight Compute on the benchmark command when available and attach metrics such as achieved occupancy, DRAM throughput, and SM efficiency.
