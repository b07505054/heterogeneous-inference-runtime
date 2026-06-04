# TVM TensorIR MatMul-Bias-ReLU Benchmark

Status: `ok`

## Environment

- TVM version: `0.24.0`
- LLVM enabled: `True`
- Target: `llvm`
- Platform: `macOS-26.5-arm64-arm-64bit`
- Machine: `arm64`
- Commit: `86b79dc`
- Git dirty: `True`
- Warmup runs: `3`
- Timed repeats: `5`
- Number per repeat: `10`

## Schedule

- Tile M/N/K: `16 / 16 / 8`
- Vectorized N lanes: `16`
- Transformations: `split`, `reorder`, `parallel`, `vectorize`, `reverse_compute_at`

## Shape Sweep

| Shape | Correct | Max abs diff | Unscheduled p50 ms | Scheduled p50 ms | Speedup | Scheduled TensorIR |
|---|---:|---:|---:|---:|---:|---|
| 64x64x64:float32 | True | 0.0 | 0.008465 | 0.001058 | 8.0009 | `results/tvm_tensorir/tir/64_64_64_float32_scheduled.py` |
| 128x128x128:float32 | True | 0.0 | 0.101534 | 0.001596 | 63.6178 | `results/tvm_tensorir/tir/128_128_128_float32_scheduled.py` |
| 256x256x256:float32 | True | 0.0 | 1.021621 | 0.010412 | 98.1196 | `results/tvm_tensorir/tir/256_256_256_float32_scheduled.py` |

## Interview Notes

- This is a real executable TensorIR path, not a static report.
- The unscheduled and scheduled versions share the same fused MatMul-Bias-ReLU semantics.
- The scheduled path exposes hardware-aware loop decisions that can be compared against MLIR/HIR lowering.
- The result is useful as a TVM reference point, while the main compiler story remains MLIR/HIR.
