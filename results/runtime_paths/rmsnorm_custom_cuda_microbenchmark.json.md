# CUDA RMSNorm Benchmark Report

Status: `measured`

## Environment

- GPU: `NVIDIA GeForce GTX 1650 with Max-Q Design`
- CUDA version: `12.8`
- NVCC version: `Cuda compilation tools, release 13.1, V13.1.115`
- PyTorch version: `2.11.0+cu128`
- Driver version: `595.71.05`
- Commit: `95cad66`
- Git dirty: `True`
- Warmup runs: `20`
- Timed runs: `100`
- Dtype: `float32`

## Shape x Block-Size Sweep

| Tokens | Hidden | Block | Custom ms | PyTorch ms | Speedup | Custom GB/s | Correct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4096 | 64 | 0.035121 | 0.098232 | 2.797 | 29.856 | True |

## Roofline Notes

- RMSNorm is memory-bound for this kernel model.
- The kernel reads input for reduction, reads input again for output, reads weight, and writes output.
- Low arithmetic intensity means bandwidth, reduction efficiency, and launch/framework overhead dominate latency.
- PyTorch baseline includes framework dispatch overhead and a less shape-specialized RMSNorm expression.

## Optional Nsight Compute

- Requested: `False`
- Available: `True`
- Reason: `not requested`
- Path: `/usr/local/cuda-13.1/bin/ncu`

When available, run Nsight Compute on the same benchmark command and attach metrics such as achieved occupancy, DRAM throughput, and SM efficiency.
