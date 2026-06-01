# Multi-Runtime Comparison (Edge AI Inference Systems)

##  Overview

This document compares multiple inference runtimes used in modern edge AI deployment workflows.

The comparison focuses on:
- Runtime architecture
- Deployment workflow
- CPU inference performance
- Runtime overhead
- Parallel execution behavior
- Deployment trade-offs

The goal is to understand how different runtimes behave under edge deployment constraints.

---

# 1. Compared Runtimes

This project evaluates:

| Runtime | Role |
|---|---|
| PyTorch Eager | Development baseline |
| TorchScript | Serialized PyTorch runtime |
| ONNX Runtime | Optimized inference engine |
| ExecuTorch Python | Lightweight runtime integration |
| ExecuTorch C++ | Production-oriented edge runtime |

---

# 2. Runtime Architecture Comparison

## PyTorch Eager

Characteristics:
- Dynamic execution graph
- High flexibility
- Research-friendly workflow

Advantages:
- Easy debugging
- Rapid experimentation
- Full framework capability

Limitations:
- Higher runtime overhead
- Less optimized for deployment
- Larger dependency footprint

---

## TorchScript

Characteristics:
- Serialized PyTorch execution graph
- Intermediate deployment representation

Advantages:
- Lower overhead than eager mode
- Better deployment portability

Limitations:
- Less optimized than specialized runtimes
- Limited deployment ecosystem compared to ONNX

---

## ONNX Runtime

Characteristics:
- Optimized inference runtime
- Backend-oriented execution
- Graph optimization support

Advantages:
- Efficient CPU inference
- Operator fusion
- Broad deployment support

Limitations:
- Runtime dependency size
- Limited low-level mobile specialization

---

## ExecuTorch Python Runtime

Characteristics:
- Lightweight runtime integration
- Python-accessible execution interface

Advantages:
- Simplified runtime experimentation
- Lower integration overhead

Limitations:
- Python runtime overhead
- Not ideal for final embedded deployment

---

## ExecuTorch C++ Runtime

Characteristics:
- Production-style deployment runtime
- Minimal runtime overhead
- Backend delegation support

Advantages:
- Lightweight execution
- Direct backend integration
- Edge-device-oriented design

Limitations:
- More complex build infrastructure
- Backend registration requirements
- Lower debugging convenience

---

# 3. Deployment Pipeline Comparison

## PyTorch Workflow

`text
PyTorch Model
    ↓
Direct Execution
`

---

## TorchScript Workflow

`text
PyTorch Model
    ↓
TorchScript Export
    ↓
TorchScript Runtime
`

---

## ONNX Workflow

`text
PyTorch Model
    ↓
ONNX Export
    ↓
ONNX Runtime
    ↓
CPU Execution Provider
`

---

## ExecuTorch Workflow

`text
PyTorch Model
    ↓
ExecuTorch Export (.pte)
    ↓
Backend Delegation (XNNPACK)
    ↓
ExecuTorch Runtime
    ↓
C++ Execution
`

---

# 4. Runtime Performance Results

## Observed Latency

| Runtime | Approximate Latency |
|---|---|
| ONNX Runtime | ~5 ms |
| ExecuTorch Python | ~6 ms |
| ExecuTorch C++ (1 thread) | ~14 ms |
| ExecuTorch C++ (4 threads) | ~5.7 ms |

---

## Key Observation

ExecuTorch C++ achieves latency comparable to ONNX Runtime when:
- Multi-threading is enabled
- XNNPACK backend is properly registered
- Runtime overhead is minimized

---

# 5. Thread Scaling Behavior

ExecuTorch C++ runtime was evaluated under:
- 1 thread
- 2 threads
- 4 threads
- 8 threads

Observed behavior:
- Near-linear speedup up to 4 threads
- Diminishing returns beyond 4 threads

This indicates:
- Effective parallel execution initially
- System saturation at higher thread counts

---

# 6. Runtime Overhead Analysis

## PyTorch

Higher overhead due to:
- Dynamic graph execution
- Framework abstraction layers
- Development-oriented flexibility

---

## ONNX Runtime

Lower overhead due to:
- Static graph optimization
- Operator fusion
- Optimized execution kernels

---

## ExecuTorch C++

Optimized for edge deployment through:
- Lightweight runtime
- Minimal abstraction layers
- Backend-specific delegation
- Reduced deployment footprint

---

# 7. Backend Delegation Comparison

## ONNX Runtime

Uses:
- CPU Execution Provider

Potential support for:
- CUDA
- TensorRT
- DirectML

---

## ExecuTorch

Uses:
- XNNPACK backend

Potential future delegates:
- CoreML
- Vulkan
- Qualcomm AI Engine
- GPU/NPU accelerators

---

# 8. Runtime Deployment Trade-Offs

| Runtime | Flexibility | Deployment Efficiency | Runtime Complexity |
|---|---|---|---|
| PyTorch | High | Low | Low |
| TorchScript | Medium | Medium | Medium |
| ONNX Runtime | Medium | High | Medium |
| ExecuTorch Python | Medium | Medium | Medium |
| ExecuTorch C++ | Lower | Very High | High |

---

# 9. Edge AI Deployment Perspective

This comparison highlights an important deployment principle:

`text
The best runtime depends on deployment constraints
`

Examples:
- Research workflow → PyTorch
- General optimized inference → ONNX Runtime
- Embedded/mobile deployment → ExecuTorch C++

---

# 10. Roofline-Style Interpretation

Observed runtime behavior suggests:

- Initial execution is compute-bound
- Multi-thread scaling improves utilization
- Performance eventually saturates due to:
  - Memory bandwidth pressure
  - Thread contention
  - Cache limitations

This behavior aligns with roofline-style performance analysis.

---

# 11. Runtime Debugging Observations

Observed deployment/debugging issues:
- Backend registration mismatch
- Quantization regression
- Runtime build issues
- Thread scaling saturation

These issues highlight:
- Runtime deployment complexity
- Importance of profiling and validation
- Need for backend/runtime alignment

---

# 12. Key Engineering Takeaways

This project demonstrates:
- Multi-runtime deployment understanding
- Runtime architecture analysis
- Backend delegation analysis
- Thread scaling evaluation
- Runtime-level performance engineering
- Edge AI deployment trade-off analysis

---

# 13. Future Work

Potential future extensions:
- TFLite benchmarking
- MLX benchmarking on Apple Silicon
- CoreML delegate analysis
- Qualcomm AI Engine deployment
- GPU/NPU execution analysis
- Thermal and battery profiling
- Memory footprint analysis

---

# 14. Summary

This project compares modern edge AI inference runtimes from both:
- Systems engineering perspective
- Deployment optimization perspective

The analysis goes beyond raw latency benchmarking and focuses on:
- Runtime behavior
- Backend execution
- Thread scheduling
- Deployment trade-offs
- Runtime bottlenecks

The result is a production-style runtime evaluation workflow for edge AI systems.
