# Nsight Compute RMSNorm Capture

Status: `permission_blocked`

## Command

```bash
/usr/local/cuda-13.1/bin/ncu --target-processes all --csv --page raw --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct --log-file /tmp/hir-latest-ncu/results/cuda_transformer/rmsnorm_nsight_compute_raw.csv --force-overwrite /home/allen/Desktop/Project/heterogeneous-inference-runtime/.venv-rmsnorm/bin/python /tmp/hir-latest-ncu/scripts/benchmark_rmsnorm_cuda.py --tokens 16 --hidden 4096 --warmup 5 --runs 10 --output /tmp/hir-latest-ncu/results/cuda_transformer/rmsnorm_nsight_compute_capture_benchmark.json --report-output /tmp/hir-latest-ncu/results/cuda_transformer/rmsnorm_nsight_compute_capture_benchmark.md
```

## Environment

- ncu: `/usr/local/cuda-13.1/bin/ncu`
- ncu version: `NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2025 NVIDIA Corporation
Version 2025.4.1.0 (build 37053803) (public-release)`
- Commit: `5bc0d99`
- Git dirty: `False`
- Return code: `1`

## Benchmark Summary

- Shape: `{'tokens': 16, 'hidden': 4096, 'dtype': 'float32'}`
- Custom latency ms: `0.195117`
- PyTorch latency ms: `0.984902`
- Speedup: `5.0478`
- Correct: `True`

## Nsight Metrics

Nsight Compute launched, but NVIDIA performance counters are locked on this machine.

Enable access with the NVIDIA ERR_NVGPUCTRPERM instructions, then rerun this script.

## Profiler Output Tail

```text
==PROF== Connected to process 11156 (/usr/bin/python3.14)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
==PROF== Disconnected from process 11156
```
