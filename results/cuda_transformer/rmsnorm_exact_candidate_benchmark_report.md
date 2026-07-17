# Exact RMSNorm GPU Candidate Benchmark

Measured operator-level weighted FP32 RMSNorm; not full-model integration.

| Tokens | Hidden | Winner | Backend | p50 ms | p95 ms |
|---:|---:|---|---|---:|---:|
| 1 | 768 | `cuda_rmsnorm_fp32_bs512_v1` | cuda | 0.036032 | 0.042368 |
| 1 | 1024 | `cuda_rmsnorm_fp32_bs256_v1` | cuda | 0.035328 | 0.053792 |
| 1 | 4096 | `cuda_rmsnorm_fp32_bs64_v1` | cuda | 0.034816 | 0.040416 |
| 1 | 8192 | `cuda_rmsnorm_fp32_bs128_v1` | cuda | 0.034816 | 0.054624 |
| 16 | 768 | `cuda_rmsnorm_fp32_bs128_v1` | cuda | 0.035616 | 0.04128 |
| 16 | 1024 | `cuda_rmsnorm_fp32_bs128_v1` | cuda | 0.035232 | 0.04304 |
| 16 | 4096 | `cuda_rmsnorm_fp32_bs64_v1` | cuda | 0.034816 | 0.03904 |
| 16 | 8192 | `cuda_rmsnorm_fp32_bs256_v1` | cuda | 0.034752 | 0.039712 |
| 128 | 768 | `cuda_rmsnorm_fp32_bs512_v1` | cuda | 0.034784 | 0.043072 |
| 128 | 1024 | `cuda_rmsnorm_fp32_bs256_v1` | cuda | 0.034016 | 0.03584 |
| 128 | 4096 | `cuda_rmsnorm_fp32_bs512_v1` | cuda | 0.069632 | 0.078048 |
| 128 | 8192 | `triton_rmsnorm_fp32_block8192_warps4_stages_default_v1` | triton | 0.116768 | 0.122592 |
