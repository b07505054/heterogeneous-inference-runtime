# GPU PGO-like RMSNorm Feedback Report

Status: `passed`

## Technology Gate

- Input: `compiler-emitted HIR RMSNorm op plus runtime shape/workload distribution`
- Decision: `profile-guided kernel selection among CUDA/Triton/PyTorch candidates by shape bucket`
- Metric: `kernel p95 latency, effective bandwidth, TPOT projection, throughput projection`

## Candidate Selection

| Shape | Selected kernel | Backend | Selected p95 ms | Baseline p95 ms | Delta ms | Reason |
|---|---|---|---:|---:|---:|---|
| 128x1024:fp32 | fused_rmsnorm_cuda | CUDA | 0.034144 | 0.093632 | 0.059488 | gpu_pgo_like_lowest_p95_latency |
| 128x4096:fp32 | fused_rmsnorm_cuda | CUDA | 0.085248 | 0.18256 | 0.097312 | gpu_pgo_like_lowest_p95_latency |
| 128x768:fp32 | fused_rmsnorm_cuda | CUDA | 0.031296 | 0.093792 | 0.062496 | gpu_pgo_like_lowest_p95_latency |
| 128x8192:fp32 | fused_rmsnorm_cuda | CUDA | 0.152896 | 0.339712 | 0.186816 | gpu_pgo_like_lowest_p95_latency |
| 16x1024:fp32 | fused_rmsnorm_cuda | CUDA | 0.03312 | 0.09104 | 0.05792 | gpu_pgo_like_lowest_p95_latency |
| 16x4096:fp32 | fused_rmsnorm_cuda | CUDA | 0.031424 | 0.09328 | 0.061856 | gpu_pgo_like_lowest_p95_latency |
| 16x768:fp32 | fused_rmsnorm_cuda | CUDA | 0.033216 | 0.093408 | 0.060192 | gpu_pgo_like_lowest_p95_latency |
| 16x8192:fp32 | fused_rmsnorm_cuda | CUDA | 0.040896 | 0.092448 | 0.051552 | gpu_pgo_like_lowest_p95_latency |
| 1x1024:fp32 | fused_rmsnorm_cuda | CUDA | 0.0328 | 0.09216 | 0.05936 | gpu_pgo_like_lowest_p95_latency |
| 1x4096:fp32 | fused_rmsnorm_cuda | CUDA | 0.031008 | 0.09216 | 0.061152 | gpu_pgo_like_lowest_p95_latency |
| 1x768:fp32 | fused_rmsnorm_cuda | CUDA | 0.036128 | 0.098336 | 0.062208 | gpu_pgo_like_lowest_p95_latency |
| 1x8192:fp32 | fused_rmsnorm_cuda | CUDA | 0.0352 | 0.091488 | 0.056288 | gpu_pgo_like_lowest_p95_latency |

## Serving Impact Projection

- Baseline TPOT p95: `3.244` ms/token
- Projected TPOT p95: `3.182144` ms/token
- TPOT delta: `0.061856` ms/token
- Baseline tokens/sec: `1236.142`
- Projected tokens/sec gain: `24.029`

## Remaining Work

- Replay selected RMSNorm kernel inside a full decode loop instead of projecting from per-kernel p95 latency.
- Add fp16/bf16 candidate rows and vectorized CUDA candidate when kernels are implemented.
- Attach real Nsight Compute occupancy/DRAM/stall metrics when ncu capture is available.
