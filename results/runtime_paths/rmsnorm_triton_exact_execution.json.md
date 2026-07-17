# Triton RMSNorm Benchmark Report

Status: `measured`

## Environment

- GPU: `NVIDIA GeForce GTX 1650 with Max-Q Design`
- CUDA: `13.0`
- PyTorch: `2.11.0+cu130`
- Triton: `3.6.0`

## Shape Sweep

| Tokens | Hidden | Triton p50 ms | PyTorch p50 ms | Speedup | Correct |
|---:|---:|---:|---:|---:|---:|
| 128 | 8192 | 0.120288 | 0.34 | 2.8237 | True |
