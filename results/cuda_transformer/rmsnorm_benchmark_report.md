# CUDA RMSNorm Benchmark Report

Status: `measured`

## Environment

- GPU: `NVIDIA GeForce GTX 1650 with Max-Q Design`
- CUDA version: `12.8`
- NVCC version: `Cuda compilation tools, release 13.1, V13.1.115`
- PyTorch version: `2.11.0+cu128`
- Driver version: `595.71.05`
- Commit: `4350e1b`
- Git dirty: `True`
- Warmup runs: `20`
- Timed runs: `100`
- Dtype: `float32`

## Shape Sweep

| Tokens | Hidden | Custom ms | PyTorch ms | Speedup | Custom GB/s | Correct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 768 | 0.032685 | 0.092775 | 2.8384 | 0.376 | True |
| 1 | 1024 | 0.031624 | 0.08749 | 2.7666 | 0.518 | True |
| 1 | 4096 | 0.029779 | 0.087292 | 2.9313 | 2.201 | True |
| 1 | 8192 | 0.032861 | 0.086717 | 2.6389 | 3.989 | True |
| 16 | 768 | 0.031847 | 0.087757 | 2.7556 | 6.173 | True |
| 16 | 1024 | 0.031602 | 0.087226 | 2.7601 | 8.295 | True |
| 16 | 4096 | 0.030196 | 0.088261 | 2.923 | 34.726 | True |
| 16 | 8192 | 0.038131 | 0.087862 | 2.3042 | 54.998 | True |
| 128 | 768 | 0.03029 | 0.088295 | 2.915 | 51.926 | True |
| 128 | 1024 | 0.030821 | 0.087491 | 2.8387 | 68.043 | True |
| 128 | 4096 | 0.083712 | 0.181661 | 2.1701 | 100.208 | True |
| 128 | 8192 | 0.151385 | 0.336926 | 2.2256 | 110.825 | True |

## Roofline Notes

- RMSNorm is memory-bound for this kernel model.
- The kernel reads input for reduction, reads input again for output, reads weight, and writes output.
- Low arithmetic intensity means bandwidth, reduction efficiency, and launch/framework overhead dominate latency.
- PyTorch baseline includes framework dispatch overhead and a less shape-specialized RMSNorm expression.

## Optional Nsight Compute

- Requested: `True`
- Available: `True`
- Reason: `None`
- Path: `/usr/local/cuda-13.1/bin/ncu`

When available, run Nsight Compute on the same benchmark command and attach metrics such as achieved occupancy, DRAM throughput, and SM efficiency.
