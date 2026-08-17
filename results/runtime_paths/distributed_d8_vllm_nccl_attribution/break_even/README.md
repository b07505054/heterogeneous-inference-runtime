# Phase 4D TP1/TP2 Break-Even Boundary

This run reuses Phase 4B/4C attribution: measured request IDs, decode-step NVTX ranges, cross-rank NCCL wall unions, wall-clock overlap, and exposed NCCL wall time. Summed per-rank NCCL time is diagnostic only.

## Boundary

Status: `boundary_found`
Success criterion met: `True`
TP1-favorable cells: `3`
TP2-favorable cells: `3`

First TP2-favorable cell:

```json
{
  "hf_model_id": "Qwen/Qwen2.5-7B-Instruct",
  "measured_winner": "tp2",
  "model_key": "qwen2.5-7b",
  "tp_compute_savings_us": 6821.515478691379,
  "workload_id": "in32_out32_c1"
}
```

## End-To-End TPOT

| model | workload | TP1 mean us | TP1 p50 us | TP1 p95 us | TP1 CV | TP2 mean us | TP2 p50 us | TP2 p95 us | TP2 CV | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen2.5-0.5b | in32_out32_c1 | 1772.941 | 1798.472 | 1804.197 | 0.0222 | 1812.237 | 1787.274 | 1872.941 | 0.0244 | tp1 |
| qwen2.5-0.5b | in32_out32_c4 | 2043.633 | 2027.871 | 2264.376 | 0.0775 | 2261.057 | 2119.349 | 2567.297 | 0.0970 | tp1 |
| qwen2.5-0.5b | in32_out32_c8 | 2134.889 | 2159.177 | 2447.754 | 0.0859 | 2298.465 | 2444.724 | 2483.148 | 0.0779 | tp1 |
| qwen2.5-7b | in32_out32_c1 | 15610.407 | 15610.782 | 15613.270 | 0.0002 | 8934.618 | 8933.180 | 8944.297 | 0.0008 | tp2 |
| qwen2.5-7b | in32_out32_c4 | 16275.038 | 16034.398 | 18626.814 | 0.0495 | 10110.278 | 9939.332 | 10403.615 | 0.0647 | tp2 |
| qwen2.5-7b | in32_out32_c8 | 16031.162 | 16043.318 | 16063.908 | 0.0019 | 10011.934 | 10002.425 | 10168.733 | 0.0109 | tp2 |

## Cost Signals

| model | workload | compute saving us | predicted raw NCCL us/step | exposed NCCL us/step | calls/step | bytes/call | residual us | winner |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| qwen2.5-0.5b | in32_out32_c1 | 909.870 | 493.874 | 949.166 | 94.581 | 7168.0 | 0.000 | tp1 |
| qwen2.5-0.5b | in32_out32_c4 | 1406.804 | 683.148 | 1624.228 | 123.500 | 8501.1 | 0.000 | tp1 |
| qwen2.5-0.5b | in32_out32_c8 | 1499.490 | 685.443 | 1663.066 | 123.766 | 8538.1 | 0.000 | tp1 |
| qwen2.5-7b | in32_out32_c1 | 6821.515 | 303.733 | 145.726 | 29.161 | 28672.0 | 0.000 | tp2 |
| qwen2.5-7b | in32_out32_c4 | 6856.623 | 511.342 | 691.863 | 34.303 | 49606.1 | 0.000 | tp2 |
| qwen2.5-7b | in32_out32_c8 | 6876.810 | 591.179 | 857.582 | 34.333 | 60045.4 | 0.000 | tp2 |

## Cause Analysis

- The boundary is primarily caused by model compute increase. The 7B model gains about 6.8 ms/token of estimated TP compute savings, while exposed NCCL is below 0.9 ms/step in the tested cells.
- The 0.5B model remains TP1-favorable because compute savings do not exceed exposed communication penalty.
- Collective count is lower for 7B than 0.5B in this trace, while bytes/call are larger. Bytes/call alone does not explain the decision boundary.
- Concurrency increases exposed NCCL for both models, but it does not change the winner within either model family here.
- Runtime residual is reported as zero in this decomposition because TP2 compute is estimated as unprofiled TP2 TPOT minus measured exposed NCCL; no selector changes are made from this.

## Files

- `workload_matrix.json`
- `end_to_end_results.json`
- `per_cell_cost_breakdown.json`
- `decision_boundary.json`
- `communication_scaling.json`
- `raw/qwen2.5-0.5b/`
- `raw/qwen2.5-7b/`

Selector logic and kernel-selection logic were not modified.
