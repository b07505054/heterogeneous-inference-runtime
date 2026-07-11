# CUDA RMSNorm Benchmark Report

Status: `unavailable`

## Environment

- GPU: `None`
- CUDA version: `None`
- NVCC version: `None`
- PyTorch version: `None`
- Driver version: `None`
- Commit: `77de916`
- Git dirty: `True`
- Warmup runs: `None`
- Timed runs: `None`
- Dtype: `None`

## Shape x Block-Size Sweep

| Tokens | Hidden | Block | Custom ms | PyTorch ms | Speedup | Custom GB/s | Correct |
|---:|---:|---:|---:|---:|---:|---:|---:|

## Roofline Notes

- RMSNorm is memory-bound for this kernel model.
- The kernel reads input for reduction, reads input again for output, reads weight, and writes output.
- Low arithmetic intensity means bandwidth, reduction efficiency, and launch/framework overhead dominate latency.
- PyTorch baseline includes framework dispatch overhead and a less shape-specialized RMSNorm expression.

## Optional Nsight Compute

- Requested: `False`
- Available: `False`
- Reason: `not requested`
- Path: `None`

When available, run Nsight Compute on the same benchmark command and attach metrics such as achieved occupancy, DRAM throughput, and SM efficiency.
