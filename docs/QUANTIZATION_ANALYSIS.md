# Quantization Analysis (Edge AI Inference)

##  Overview

This document analyzes quantization behavior and deployment trade-offs observed during edge inference benchmarking.

The study focuses on:
- FP32 inference
- INT8 quantized inference
- Runtime operator behavior
- Quantization overhead
- Deployment implications

The goal is to understand how quantization affects real runtime performance in edge AI systems.

---

# 1. Motivation

Quantization is widely used in edge AI deployment because it can:
- Reduce model size
- Reduce memory bandwidth usage
- Improve inference efficiency
- Lower power consumption

However:

`text
Quantization does not always improve runtime performance
`

This project investigates that behavior experimentally.

---

# 2. Compared Models

The project evaluated:

| Model | Precision |
|---|---|
| MobileNetV2 FP32 | Floating point |
| MobileNetV2 Optimized ONNX | Graph-optimized FP32 |
| MobileNetV2 INT8 | Dynamically quantized |

---

# 3. Observed Runtime Results

## FP32 ONNX Runtime

Observed latency:
- ~5 ms

Characteristics:
- Stable execution
- Lower operator overhead
- Efficient kernel execution

---

## Optimized FP32 ONNX Runtime

Observed latency:
- Slightly faster than baseline FP32

Likely reasons:
- Graph simplification
- Reduced runtime overhead
- Operator fusion

---

## INT8 Quantized ONNX Runtime

Observed latency:
- ~100+ ms

Unexpected result:
- INT8 became significantly slower than FP32

---

# 4. Operator-Level Profiling

Profiling identified major runtime contributors:

| Operator | Observation |
|---|---|
| ConvInteger | Dominant runtime cost |
| DynamicQuantizeLinear | Significant overhead |
| Cast | Frequent execution |
| Mul/Add | Increased graph complexity |

---

## Observed Graph Expansion

FP32 model:
- ~100 nodes

INT8 model:
- ~400+ nodes

INT8 quantization introduced:
- Additional conversion operators
- Dynamic quantization steps
- More fragmented execution graph

---

# 5. Root Cause Analysis

The slowdown was likely caused by:

## 1. Dynamic Quantization Overhead

INT8 model relied heavily on:

`text
DynamicQuantizeLinear
`

This introduces:
- Runtime conversion overhead
- Additional memory operations
- Extra operator execution

---

## 2. ConvInteger Execution Cost

INT8 execution used:

`text
ConvInteger
`

Observed behavior suggests:
- CPU kernels were not sufficiently optimized
- Quantized arithmetic overhead exceeded expected gains

---

## 3. Operator Fragmentation

INT8 graph contained:
- More operators
- More synchronization points
- Additional memory movement

This increased runtime overhead significantly.

---

## 4. Hardware Dependency

Quantization effectiveness depends heavily on:
- Hardware instruction support
- Runtime backend optimization
- Kernel implementation quality

On general CPU execution:
- INT8 acceleration may not outperform optimized FP32

---

# 6. Engineering Insight

This project demonstrates an important deployment principle:

`text
Quantization effectiveness is runtime- and hardware-dependent
`

Quantization should be evaluated using:
- Real profiling
- Operator analysis
- Hardware-aware benchmarking

Rather than assuming:
- Smaller precision automatically means faster inference

---

# 7. Memory and Deployment Trade-Offs

Potential advantages of INT8:
- Reduced model size
- Lower memory footprint
- Reduced bandwidth requirements

Potential disadvantages:
- Increased conversion overhead
- Runtime fragmentation
- Kernel inefficiency
- Additional scheduling overhead

---

# 8. Roofline-Style Interpretation

Observed behavior suggests:

- FP32 execution remained relatively compute-efficient
- INT8 introduced additional memory movement and operator overhead
- Execution shifted toward memory-bound behavior

This aligns with:
- Reduced arithmetic efficiency
- Increased runtime overhead
- Higher operational fragmentation

---

# 9. Runtime Implications for Edge AI

This study highlights an important edge deployment lesson:

`text
Model optimization must be evaluated together with runtime behavior
`

Deployment performance depends on:
- Runtime implementation
- Backend optimization
- Hardware capability
- Graph structure
- Operator fusion quality

---

# 10. Future Extensions

Potential future work:
- Static INT8 quantization
- FP16 benchmarking
- Quantization-aware training
- Per-channel quantization
- Hardware accelerator benchmarking
- Mobile-device quantization profiling

---

# 11. Summary

This project demonstrates runtime-aware quantization analysis for edge AI systems.

Key findings:
- INT8 quantization can regress performance
- Runtime profiling is essential
- Operator-level analysis explains deployment behavior
- Quantization must be evaluated together with hardware and runtime constraints

The work focuses not only on applying quantization, but on understanding:
- Why optimization succeeds
- Why optimization fails
- How runtime architecture affects inference performance
