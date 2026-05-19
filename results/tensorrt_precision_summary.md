# TensorRT Backend Validation

## Overview

This document summarizes TensorRT backend validation experiments for the edge inference backend validation suite.

The goal is to evaluate TensorRT as a GPU inference backend and compare it against:
- ONNX Runtime CPU
- ONNX Runtime CUDA Execution Provider
- PyTorch CUDA
- ExecuTorch
- TensorFlow Lite
- Custom CUDA kernels

---

## Backend Validation Goal

This project treats TensorRT as a backend execution target.

The validation workflow is:

```text
ONNX Model
    ↓
TensorRT Engine Build
    ↓
FP32 / FP16 / INT8 Optimization
    ↓
Runtime Benchmark
    ↓
Latency / Throughput / Enqueue / GPU Compute Analysis
```

---

## TensorRT Engine Build

TensorRT engines were generated from MobileNetV2 ONNX models.

Evaluated precision modes:
- FP32
- FP16
- INT8

This validates:
- ONNX parsing
- TensorRT engine creation
- GPU execution
- precision-aware runtime optimization

---

## Runtime Environment

Device:
- NVIDIA GeForce GTX 1650 with Max-Q Design

Execution stack:
- CUDA Toolkit
- TensorRT 10.11
- Windows GPU execution environment

---

## Precision Benchmark Results

| Precision | Throughput | Mean Latency | Median Latency | GPU Compute |
|---|---:|---:|---:|---:|
| FP32 | ~504 qps | ~2.23 ms | ~1.56 ms | ~1.96 ms |
| FP16 | ~589 qps | ~1.92 ms | ~1.51 ms | ~1.68 ms |
| INT8 | ~758 qps | ~1.50 ms | ~1.07 ms | ~1.24 ms |

---

## Key Finding: Precision Optimization

TensorRT INT8 achieved the best performance.

Compared with FP32:
- Throughput improved by approximately 50%
- Mean latency decreased by approximately 33%
- GPU compute time decreased significantly

This shows that precision-aware backend optimization can improve inference performance when supported by the runtime and hardware.

---

## CUDA Graph Runtime Optimization

TensorRT INT8 was also evaluated with CUDA Graph execution.

| Mode | Enqueue Mean | Mean Latency |
|---|---:|---:|
| INT8 | ~0.86 ms | ~1.50 ms |
| INT8 + CUDA Graph | ~0.06 ms | ~1.67 ms |

CUDA Graph reduced enqueue overhead by approximately:

```text
~93%
```

---

## Key Finding: Enqueue Bottleneck

CUDA Graph significantly reduced CPU-side enqueue overhead.

However, end-to-end latency did not improve in this laptop GPU environment.

This suggests that after enqueue overhead was reduced, remaining bottlenecks shifted toward:
- GPU compute variability
- host-device transfer overhead
- WDDM scheduling jitter
- dynamic GPU clock behavior

---

## Runtime Metrics Analyzed

TensorRT profiling exposed:
- throughput
- mean latency
- median latency
- p95 / p99 latency
- enqueue time
- H2D latency
- GPU compute time
- D2H latency
- runtime stability warnings

This provides deeper runtime visibility than average latency alone.

---

## Backend Comparison Insight

TensorRT provided the fastest observed inference path in this project.

Observed latency ranking:
- TensorRT INT8: ~1.50 ms
- TensorRT FP16: ~1.92 ms
- TensorRT FP32: ~2.23 ms
- ONNX Runtime CUDA EP: ~3.41 ms
- ONNX Runtime CPU EP: ~5 ms
- ExecuTorch C++ XNNPACK: ~5.7 ms
- TensorFlow Lite XNNPACK: ~9.69 ms

---

## Deployment Debugging Notes

TensorRT setup required resolving:
- TensorRT DLL path issues
- plugin loading dependency issues
- CUDA / cuDNN runtime dependency configuration
- engine build vs engine load separation

This reflects realistic backend deployment validation work.

---

## Systems Engineering Takeaways

This TensorRT validation phase demonstrates:
- backend runtime validation
- GPU inference benchmarking
- precision-performance trade-off analysis
- CUDA Graph runtime scheduling analysis
- enqueue overhead profiling
- GPU compute bottleneck analysis
- deployment dependency debugging

---

## Summary

This project validates TensorRT as a GPU backend in an edge inference backend validation suite.

The work shows:
- TensorRT engine build from ONNX
- FP32 / FP16 / INT8 backend benchmarking
- CUDA Graph enqueue optimization
- latency / throughput / GPU compute analysis
- runtime bottleneck interpretation

This extends the project from CPU edge runtime benchmarking into GPU inference backend validation.