# Edge AI Inference Optimization & Profiling Pipeline (ONNX Runtime)
## Overview

This project implements an end-to-end edge AI inference optimization pipeline, evaluating performance trade-offs across PyTorch, ONNX Runtime, and quantized models.

The goal is to simulate a real-world AI deployment workflow and analyze:

Inference latency (avg / p95 / p99)
Throughput (QPS)
Memory footprint
Model size
Accuracy consistency
CPU thread scaling behavior

The pipeline is designed to reflect production-level AI software tooling, similar to those used in edge AI systems.

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
Profiling & Evaluation

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
Model: MobileNetV2
Input: (1, 3, 224, 224)
Backend: ONNX Runtime (CPU)
Threads: 1, 2, 4, 8
Iterations: 100 (with warmup)
📊 Benchmark Summary
Model	Backend	Threads	Avg Latency (ms)	P95 Latency (ms)	Throughput (QPS)	Model Size (MB)
PyTorch FP32	PyTorch	N/A	26.12	32.31	38.29	N/A
ONNX FP32	ONNX Runtime	1	9.99	11.02	100.10	0.27
Optimized ONNX FP32	ONNX Runtime	4	4.52	6.11	221.44	13.33
INT8 ONNX	ONNX Runtime	4	105.48	116.02	9.48	3.45
## Key Findings
-  5.6× speedup from PyTorch → optimized ONNX Runtime
-  Throughput improved from 38.3 → 221.4 QPS
-  Model size reduced by 74% via INT8 quantization
-  INT8 caused 22.6× latency regression
-  INT8 reduced Top-1 consistency from 100% → 3%
-  Accuracy / Consistency Validation
Model	Top-1 Consistency with PyTorch
ONNX FP32	100%
Optimized ONNX FP32	100%
INT8 ONNX	3%
## Thread Scaling Analysis

1 → 2 threads: strong improvement
2 → 4 threads: moderate improvement
4 → 8 threads: minimal gain (saturation)
## Insight

Performance improvement saturated beyond 4 threads due to:

CPU scheduling overhead
Memory bandwidth limits
Limited operator-level parallelism

- More threads ≠ better performance in edge environments

## Bottleneck Analysis

INT8 dynamic quantization resulted in:

Significant latency increase
Severe accuracy degradation
Root Cause

MobileNetV2 relies heavily on convolution operators, which are not efficiently accelerated under:

- ONNX Runtime CPU dynamic quantization path

Unlike transformer models:

Dynamic quantization works well on Linear layers
But poorly on Conv-heavy networks
## C++ ONNX Runtime Inference

This project includes a C++ inference benchmark using ONNX Runtime and CMake.

Features
Loads ONNX model
Runs warmup + timed inference
Reports latency
Outputs CSV results
Build & Run
cmake -S cpp_inference -B build_cpp -DONNXRUNTIME_DIR="path\to\onnxruntime"
cmake --build build_cpp --config Release

.\build_cpp\Release\edge_onnx_cpp.exe .\models\mobilenet_v2_optimized.onnx
## Project Structure
edge-onnx-benchmark/
│
├── backends/
├── benchmarks/
├── cpp_inference/
├── models/
├── results/
├── scripts/
│   ├── benchmark.py
│   ├── export_onnx.py
│   ├── optimize_onnx.py
│   ├── quantize_int8.py
│   ├── evaluate_accuracy.py
│   ├── thread_sweep.py
│   ├── plot_thread_sweep.py
│   └── generate_report.py
│
├── cpp_inference/
│   ├── main.cpp
│   └── CMakeLists.txt
│
└── README.md
## Reproduce Results

### Export ONNX

```bash
python scripts/export_onnx.py

### Optimize
python scripts/optimize_onnx.py

### Quantize
python scripts/quantize_int8.py

### Benchmark
python scripts/benchmark.py --backend onnx --threads 4

### Thread sweep
python scripts/thread_sweep.py

### Plot
python scripts/plot_thread_sweep.py

### Accuracy check
python scripts/evaluate_accuracy.py --model-path models/mobilenet_v2_int8.onnx
## Future Work
Static quantization with calibration data
Transformer model benchmark (DistilBERT)
GPU / NPU backend comparison
Operator-level profiling (kernel-level)
Integration with mobile / embedded platforms
## Key Takeaway
Efficient AI deployment is not just about model accuracy —
it requires system-level optimization across model format, runtime, and hardware constraints.