# Runtime Architecture Analysis (ExecuTorch + ONNX Runtime)

##  Overview

This document analyzes the runtime architecture and execution flow of the edge inference pipeline implemented in this project.

The analysis focuses on:
- Runtime execution behavior
- Backend delegation
- CPU execution path
- Thread scheduling
- Runtime bottlenecks
- Deployment trade-offs

The goal is to understand how inference systems behave beyond simple latency benchmarking.

---

# 1. End-to-End Execution Pipeline

The deployment pipeline follows:

`text
PyTorch Model
    ↓
ONNX Export
    ↓
ExecuTorch Export (.pte)
    ↓
Backend Delegation (XNNPACK)
    ↓
C++ Runtime Execution
    ↓
Thread Scheduling
    ↓
Inference Execution
    ↓
Performance Profiling
`

This reflects a production-style edge AI deployment workflow.

---

# 2. ExecuTorch Runtime Architecture

ExecuTorch separates:

- Model representation
- Runtime execution
- Backend delegation
- Memory planning
- Operator execution

The deployed model is serialized into:

`text
.pte (ExecuTorch flatbuffer format)
`

The runtime loads this artifact and dispatches operators to registered execution backends.

---

# 3. XNNPACK Backend Delegation

This project uses:

`text
ExecuTorch + XNNPACK backend
`

XNNPACK provides optimized CPU kernels for neural network operators.

Delegation flow:

`text
ExecuTorch Runtime
    ↓
XNNPACK Backend
    ↓
Optimized CPU Kernels
    ↓
Threadpool Execution
`

The backend accelerates:
- Convolution
- GEMM
- Elementwise operators
- Activation functions

This enables efficient CPU inference on edge devices.

---

# 4. Backend Registration and Runtime Mismatch

One runtime issue encountered during development:

`text
Backend XnnpackBackend is not registered
`

Root cause:
- Model exported with XNNPACK delegate
- Runtime built without XNNPACK backend registration

Resolution:
- Rebuilt ExecuTorch runtime with:
  -DEXECUTORCH_BUILD_XNNPACK=ON

This demonstrates the importance of alignment between:
- Export-time delegates
- Runtime-time backend availability

This issue is representative of real deployment/debugging workflows in edge AI systems.

---

# 5. CPU Execution Path

Inference execution occurs entirely on CPU.

Execution flow:

`text
Model Load
    ↓
Memory Planning
    ↓
Threadpool Initialization
    ↓
Operator Dispatch
    ↓
Kernel Execution
    ↓
Output Synchronization
`

The runtime uses:
- CPU threadpool scheduling
- Parallel operator execution
- Shared memory buffers

---

# 6. Thread Scheduling Analysis

The project evaluated:
- 1 thread
- 2 threads
- 4 threads
- 8 threads

Observed behavior:

| Threads | Avg Latency |
|---|---|
| 1 | ~14 ms |
| 2 | ~8 ms |
| 4 | ~5.7 ms |
| 8 | ~5.8 ms |

Key observations:
- Near-linear speedup up to 4 threads
- Diminishing returns beyond 4 threads

This suggests:
- Effective parallelization initially
- System saturation at higher thread counts

---

# 7. CPU Utilization Analysis

CPU utilization increased with thread count.

Approximate observed behavior:

| Threads | CPU Usage |
|---|---|
| 1 | ~10–40% |
| 2 | ~40–60% |
| 4 | ~70–90% |
| 8 | Near saturation |

Interpretation:
- Single-thread execution primarily occupies one CPU core
- Additional threads improve parallel utilization
- Beyond 4 threads, additional CPU usage provides limited latency improvement

---

# 8. Throughput and Speedup

Throughput scaling:
- Increased nearly linearly up to 4 threads
- Plateaued beyond 4 threads

Speedup behavior:
- ~2.5–3× speedup achieved
- Limited gains beyond saturation point

This behavior is consistent with:
- Thread contention
- Scheduling overhead
- Memory bandwidth limits

---

# 9. Roofline-Style Interpretation

The observed scaling behavior is consistent with the Roofline model:

- Initial performance scaling suggests compute-bound execution
- Saturation at higher thread counts suggests memory-bound behavior

Interpretation:

`text
Compute-bound region
    ↓
Parallel speedup
    ↓
Memory bandwidth pressure
    ↓
Diminishing returns
`

This project does not implement a full roofline model with FLOPs/byte calculation, but the observed runtime behavior aligns with roofline-style performance analysis.

---

# 10. Runtime Comparison

This project compares:

| Runtime | Characteristics |
|---|---|
| PyTorch | High flexibility, higher runtime overhead |
| ONNX Runtime | Optimized CPU inference baseline |
| ExecuTorch Python | Lightweight runtime integration |
| ExecuTorch C++ | Production-oriented deployment runtime |

Observed result:
- ExecuTorch C++ achieves latency comparable to ONNX Runtime when properly parallelized

---

# 11. Quantization Runtime Observations

INT8 quantized ONNX models showed:
- Increased operator count
- DynamicQuantizeLinear overhead
- ConvInteger execution overhead

Observed result:
- INT8 inference became slower than FP32

This highlights an important deployment insight:

`text
Quantization does not guarantee speedup
`

Performance depends on:
- Kernel implementation quality
- Backend optimization
- Hardware support
- Operator fusion effectiveness

---

# 12. Key Engineering Takeaways

This project demonstrates:

- Runtime-level inference understanding
- Backend delegation analysis
- System-level performance profiling
- Multi-thread scaling evaluation
- Runtime debugging and deployment troubleshooting
- Edge AI inference optimization

The project focuses not only on running inference, but on understanding:
- Why performance changes
- Where bottlenecks emerge
- How deployment trade-offs affect runtime behavior

---

# 13. Future Work

Potential future extensions:
- Android deployment
- Apple Silicon benchmarking
- TFLite comparison
- CoreML delegate analysis
- Hardware accelerator profiling
- Full roofline modeling
- Memory bandwidth measurement
- Thermal and power analysis

---

# 14. Summary

This project bridges machine learning deployment and systems engineering.

It combines:
- Edge AI runtime deployment
- C++ runtime execution
- Performance engineering
- Runtime debugging
- System-level analysis

The result is a production-style exploration of modern edge inference systems.
