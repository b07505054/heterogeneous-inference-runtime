# Edge AI Inference Optimization & Profiling Pipeline

## Overview

This project implements an end-to-end edge AI inference optimization and runtime benchmarking pipeline across heterogeneous AI inference backends.

The system evaluates real-world deployment trade-offs across:

- PyTorch eager inference
- ONNX Runtime CPU / CUDA execution providers
- ExecuTorch XNNPACK runtime
- TensorRT FP16 / FP32 inference engines
- Native C++ ONNX Runtime inference
- Quantized INT8 inference
- Multi-threaded CPU execution scaling
- TensorRT dynamic batch scaling

The project simulates a production-style AI deployment workflow similar to modern edge AI and inference infrastructure systems.

The pipeline evaluates:

- Average latency
- p95 latency
- p99 latency
- Throughput (QPS)
- Runtime backend efficiency
- Thread scaling behavior
- Batch scaling behavior
- Quantization trade-offs
- Runtime optimization effectiveness
- GPU inference acceleration

---

## System Pipeline

PyTorch Model
   ↓
ONNX Export
   ↓
ONNX Graph Optimization
   ↓
Quantization (INT8 Dynamic)
   ↓
Backend Conversion / Compilation
   ├── ONNX Runtime
   ├── ExecuTorch XNNPACK
   ├── TensorRT FP16 Engine
   ├── TensorRT FP32 Engine
   └── Native C++ Runtime
   ↓
Benchmarking + Profiling + Validation
   ↓
Nsight Systems GPU Profiling

This simulates a modern AI software deployment toolchain used in production inference systems.

---

## Runtime Backends Evaluated

| Backend | Precision | Device |
|---|---|---|
| PyTorch | FP32 | CPU |
| ONNX Runtime | FP32 | CUDA |
| ONNX Runtime | Optimized FP32 | CUDA |
| ONNX Runtime | INT8 | CPU |
| ExecuTorch | FP32 | XNNPACK |
| TensorRT | FP16 | CUDA |
| TensorRT | FP32 | CUDA |
| Native C++ ONNX Runtime | FP32 | CPU |

---

## Benchmark Setup

- Model: MobileNetV2
- Input Shape: (1, 3, 224, 224)
- Iterations: 100
- Warmup Iterations: 10
- CPU Thread Sweep: 1 / 2 / 4 / 8
- TensorRT Batch Sweep: 1 / 2 / 4 / 8 / 16
- GPU Backend: CUDA
- TensorRT Precision: FP16 / FP32
- ExecuTorch Delegate: XNNPACK

---

## Backend Validation Summary

| Backend | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (QPS) |
|---|---|---|---|---|
| TensorRT FP16 | 1.64 | 1.73 | 1.74 | 608.8 |
| TensorRT FP32 | 1.68 | 1.72 | 1.73 | 596.7 |
| ONNX Runtime Optimized FP32 CUDA | 2.78 | 2.83 | 2.85 | 360.3 |
| ONNX Runtime FP32 CUDA | 2.98 | 3.41 | 3.42 | 335.0 |
| ExecuTorch XNNPACK | 3.81 | 5.16 | 7.40 | 262.5 |
| Thread Scaling 8T | 4.46 | N/A | N/A | 224.3 |
| Thread Scaling 4T | 4.52 | N/A | N/A | 221.4 |
| Thread Scaling 2T | 5.53 | N/A | N/A | 181.0 |
| Native C++ Runtime | 9.25 | N/A | N/A | 108.1 |
| PyTorch FP32 CPU | 16.35 | 18.33 | 19.33 | 61.2 |
| ONNX Runtime INT8 CPU | 73.88 | 100.74 | 124.46 | 13.5 |

---

## TensorRT Batch Scaling Analysis

| Batch Size | Avg Latency (ms) | Throughput (QPS) |
|---|---|---|
| 1 | 1.71 | 585.4 |
| 2 | 1.46 | 1367.1 |
| 4 | 1.80 | 2220.8 |
| 8 | 2.66 | 3010.4 |
| 16 | 4.18 | 3823.7 |

### Insight

TensorRT batch scaling experiments demonstrated:

- Sublinear latency growth as batch size increased
- Significant throughput scaling under larger inference batches
- Improved GPU utilization and kernel amortization
- Reduced relative runtime overhead at larger batch sizes

Throughput improved by over 6.5× from batch-1 to batch-16 inference execution.

This demonstrates typical GPU inference serving behavior where larger batches better utilize available GPU compute resources.

---

## Key Findings

- TensorRT FP16 achieved the best inference performance across all evaluated runtimes
- TensorRT reduced latency by ~1.8× compared to ONNX Runtime CUDA FP32
- TensorRT achieved over 608 QPS under batch-1 inference
- TensorRT batch scaling achieved over 3823 QPS at batch-16 inference
- TensorRT FP16 and FP32 showed similar latency behavior under MobileNetV2 batch-1 inference
- ExecuTorch XNNPACK significantly outperformed PyTorch eager CPU inference
- ONNX Runtime optimized CUDA execution improved throughput compared to baseline CUDA execution
- CPU thread scaling saturated beyond 4 threads
- INT8 dynamic quantization caused severe latency regression on MobileNetV2 under ONNX Runtime CPU

---

## Thread Scaling Analysis

| Threads | Avg Latency (ms) | Throughput (QPS) |
|---|---|---|
| 1 | 9.61 | 104.1 |
| 2 | 5.53 | 181.0 |
| 4 | 4.52 | 221.4 |
| 8 | 4.46 | 224.3 |

### Insight

Performance improvement saturated beyond 4 threads due to:

- CPU scheduling overhead
- Memory bandwidth limitations
- Operator-level parallelism limits
- Runtime synchronization overhead

This demonstrates that increasing thread count does not guarantee proportional inference acceleration.

---

## Quantization Bottleneck Analysis

INT8 dynamic quantization resulted in:

- Significant latency regression
- Severe throughput degradation
- Reduced inference efficiency

### Root Cause

MobileNetV2 relies heavily on convolution operators, which are not efficiently accelerated under:

- ONNX Runtime CPU dynamic quantization path

Dynamic quantization is generally more effective for transformer-style Linear-heavy architectures than Conv-heavy CNN models.

---

## ExecuTorch Integration

This project includes ExecuTorch runtime integration using the XNNPACK delegate.

### Features

- Exported PyTorch model to ExecuTorch `.pte`
- Integrated ExecuTorch Python runtime
- Benchmarked edge-oriented inference performance
- Evaluated latency and throughput against other runtimes

### Result

ExecuTorch XNNPACK achieved:

- 3.81 ms average latency
- 262.5 QPS throughput

This demonstrates efficient edge-oriented inference execution for mobile and constrained-device environments.

---

## TensorRT Integration

This project includes TensorRT FP16 / FP32 engine compilation and benchmarking.

### Features

- ONNX → TensorRT engine conversion
- FP16 optimization
- FP32 baseline comparison
- Dynamic batch inference benchmarking
- CUDA runtime execution
- GPU latency benchmarking
- Throughput profiling
- TensorRT engine profiling

### Result

TensorRT FP16 achieved:

- 1.64 ms average latency
- 608.8 QPS throughput

TensorRT FP32 achieved:

- 1.68 ms average latency
- 596.7 QPS throughput

TensorRT provided the best overall runtime efficiency in this benchmark suite.

---

## Nsight Systems Profiling

This project includes GPU runtime profiling using NVIDIA Nsight Systems.

### Profiling Scope

- CUDA kernel execution timeline
- GPU runtime scheduling
- CUDA stream execution
- Runtime synchronization behavior
- GPU inference execution analysis

### Observation

Nsight Systems profiling showed that MobileNetV2 batch-1 inference was dominated by runtime overhead and sparse GPU utilization rather than tensor-core saturation, explaining the relatively small performance gap between TensorRT FP16 and FP32 execution.

---

## Native C++ ONNX Runtime Inference

This project includes a native C++ ONNX Runtime inference pipeline using CMake.

### Features

- ONNX model loading
- Warmup + timed inference
- CSV benchmark export
- Latency reporting
- Production-style runtime integration

### Build

```bash
cmake -S cpp_inference -B build_cpp
cmake --build build_cpp --config Release
```

### Run

```bash
./build_cpp/edge_onnx_cpp models/mobilenet_v2_optimized.onnx
```

---

## Profiling & Visualization

The project automatically generates:

- Latency comparison plots
- Throughput comparison plots
- p95 latency plots
- p99 latency plots
- Batch scaling plots
- Backend validation summaries

Generated benchmark figures are stored under:

```text
results/
```

---

## Project Structure

```text
heterogeneous-inference-runtime/
│
├── backends/
├── benchmarks/
├── cpp_inference/
├── models/
├── results/
├── scripts/
│
├── backend_validation_runner.py
├── README.md
└── requirements.txt
```

---

## Reproduce Results

### Export ONNX

```bash
python scripts/export_onnx.py
```

### Optimize ONNX

```bash
python scripts/optimize_onnx.py
```

### Quantize INT8

```bash
python scripts/quantize_int8.py
```

### Export ExecuTorch

```bash
python scripts/export_executorch.py
```

### Build TensorRT Engines

```bash
python scripts/build_tensorrt_engine.py
python scripts/build_tensorrt_engine_fp32.py
```

### Run Backend Validation

```bash
python backend_validation_runner.py
```

### Run TensorRT Batch Scaling

```bash
python benchmarks/benchmark_tensorrt_batch.py
```

### Run Nsight Systems Profiling

```bash
nsys profile -o trt_fp16_profile python benchmarks/benchmark_tensorrt.py
nsys profile -o trt_fp32_profile python benchmarks/benchmark_tensorrt_fp32.py
```

### Generate Plots

```bash
python benchmarks/plot_backend_validation.py
python benchmarks/plot_backend_throughput.py
python benchmarks/plot_backend_p95.py
python benchmarks/plot_backend_p99.py
```

---

## Future Work

- Nsight Compute profiling
- Kernel-level CUDA profiling
- TensorRT INT8 calibration
- Transformer model benchmark
- CUDA stream overlap optimization
- GPU memory transfer analysis
- Operator fusion analysis
- Runtime scheduling analysis
- Dynamic shape inference
- vLLM / TensorRT-LLM integration

---

## Key Takeaway

Efficient AI deployment requires more than model accuracy.

Real-world inference systems require optimization across:

- Runtime infrastructure
- Execution backends
- Quantization strategy
- Thread scheduling
- GPU acceleration
- Memory efficiency
- Compiler/runtime integration
- Hardware-aware execution
- Runtime profiling
- Batch scheduling behavior