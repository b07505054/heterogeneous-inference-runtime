# Runtime Comparison Analysis

## Overview

This document analyzes multi-runtime inference performance across ONNX Runtime, ExecuTorch, and TFLite.

The goal is to evaluate edge inference deployment trade-offs across different runtime systems under CPU execution.

---

## Compared Runtimes

| Runtime | Backend | Role |
|---|---|---|
| ONNX Runtime | CPU Execution Provider | Optimized CPU inference baseline |
| ExecuTorch Python | Portable / XNNPACK | Lightweight runtime integration |
| ExecuTorch C++ | XNNPACK, 4 threads | Production-style edge runtime |
| TFLite | XNNPACK CPU delegate | Mobile-oriented runtime baseline |

---

## Latency Results

| Runtime | Backend | Avg Latency |
|---|---|---:|
| ONNX Runtime | CPU EP | ~5.00 ms |
| ExecuTorch Python | Portable / XNNPACK | ~6.00 ms |
| ExecuTorch C++ | XNNPACK, 4 threads | ~5.70 ms |
| TFLite | XNNPACK CPU delegate | ~9.69 ms |

---

## Key Findings

### 1. ONNX Runtime achieved the lowest latency

ONNX Runtime achieved the fastest observed latency under the current Windows CPU setup.

This suggests that ONNX Runtime remains a strong optimized CPU inference baseline for MobileNetV2.

---

### 2. ExecuTorch C++ reached comparable performance

ExecuTorch C++ achieved ~5.7 ms latency with XNNPACK and 4 CPU threads.

This is close to ONNX Runtime and demonstrates that ExecuTorch can be competitive when:

- XNNPACK backend is enabled
- Runtime backend registration is correct
- Thread-level parallelism is configured properly

---

### 3. ExecuTorch Python showed modest overhead

ExecuTorch Python achieved ~6 ms latency.

Compared with ExecuTorch C++, the gap is relatively small, but still suggests Python binding/runtime overhead.

---

### 4. TFLite provides a mobile-oriented baseline

TFLite executed successfully with the XNNPACK CPU delegate.

Observed latency:

- avg: ~9.69 ms
- p95: ~12.40 ms
- p99: ~15.65 ms

Although slower in this setup, TFLite adds an important mobile runtime baseline.

---

## Runtime Trade-Offs

| Runtime | Strength | Trade-Off |
|---|---|---|
| ONNX Runtime | Strong CPU performance | Larger general-purpose runtime |
| ExecuTorch C++ | Lightweight edge deployment | More complex build/backend setup |
| ExecuTorch Python | Easy experimentation | Python overhead |
| TFLite | Mobile ecosystem support | Slower in this Windows CPU setup |

---

## Important Benchmark Caveat

This comparison is intended as a runtime-level baseline under Windows CPU execution.

A fully controlled benchmark would require:

- Same threading configuration across all runtimes
- Same preprocessing path
- Same model graph source
- Same input layout
- Same device-level deployment target
- Repeated trials across clean processes

Therefore, the results should be interpreted as practical runtime comparison signals, not absolute universal runtime rankings.

---

## Engineering Interpretation

The results show that runtime choice significantly affects edge inference behavior.

Key lessons:

- Runtime backend configuration matters
- XNNPACK improves CPU execution paths
- C++ runtime integration can reduce deployment overhead
- Mobile-oriented runtimes require device-level validation
- Runtime comparison must consider both latency and deployment constraints

---

## Summary

This experiment expands the project from single-runtime benchmarking into multi-runtime edge inference evaluation.

It demonstrates:

- ONNX Runtime CPU inference benchmarking
- ExecuTorch Python and C++ runtime execution
- TFLite XNNPACK CPU delegate benchmarking
- Runtime trade-off analysis
- Edge deployment performance interpretation