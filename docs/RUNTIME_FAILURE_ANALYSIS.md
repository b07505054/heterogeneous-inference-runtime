# Runtime Failure Analysis (Edge AI Deployment)

## Overview

This document analyzes runtime failures, deployment issues, and debugging scenarios encountered during development of the edge inference pipeline.

The goal is to understand:
- Why deployment systems fail
- How runtime/backend mismatches occur
- How operator execution affects performance
- How system constraints impact inference behavior

These issues are representative of real production edge AI deployment workflows.

---

# 1. XNNPACK Backend Registration Failure

## Observed Error

`text
Backend XnnpackBackend is not registered
`

---

## Root Cause

The model was exported using:

`text
XNNPACK delegate
`

However, the ExecuTorch C++ runtime was initially built without XNNPACK backend support.

This created a mismatch between:
- Export-time backend delegation
- Runtime-time backend registration

---

## Resolution

Rebuilt ExecuTorch runtime with:

`text
-DEXECUTORCH_BUILD_XNNPACK=ON
`

After rebuilding:
- Runtime successfully registered XNNPACK backend
- Model execution succeeded

---

## Engineering Insight

This demonstrates an important edge deployment principle:

`text
Model export configuration must match runtime backend availability
`

This type of issue commonly occurs in:
- Mobile deployment
- Embedded inference systems
- Delegate-based runtimes

---

# 2. INT8 Quantization Performance Regression

## Observed Behavior

INT8 quantized ONNX model became significantly slower than FP32.

Observed latency:
- FP32: ~5 ms
- INT8: ~100+ ms

---

## Operator-Level Analysis

Profiling revealed:

| Operator | Observation |
|---|---|
| ConvInteger | Dominant runtime cost |
| DynamicQuantizeLinear | High execution overhead |
| Cast | Frequent overhead |
| Mul/Add | Increased operator count |

INT8 model introduced:
- Additional quantization operators
- Dynamic conversion overhead
- More fragmented execution graph

---

## Root Cause

Quantization overhead exceeded arithmetic acceleration gains.

Possible reasons:
- Dynamic quantization overhead
- Inefficient CPU kernel implementation
- Limited operator fusion
- Hardware not optimized for INT8 execution

---

## Engineering Insight

This demonstrates:

`text
Quantization does not automatically improve inference performance
`

Performance depends on:
- Backend optimization
- Kernel quality
- Hardware support
- Graph fusion effectiveness

---

# 3. Runtime Saturation and Diminishing Returns

## Observed Behavior

Thread scaling results:

| Threads | Avg Latency |
|---|---|
| 1 | ~14 ms |
| 2 | ~8 ms |
| 4 | ~5.7 ms |
| 8 | ~5.8 ms |

---

## Observed Issue

Latency improvement plateaued beyond 4 threads.

---

## Potential Causes

Likely system-level bottlenecks:
- Thread contention
- Scheduling overhead
- Memory bandwidth saturation
- Cache pressure

---

## Supporting Evidence

Observed:
- CPU utilization increased with thread count
- Throughput plateaued beyond 4 threads
- Speedup stopped scaling linearly

---

## Engineering Insight

This behavior is consistent with:
- Roofline-style saturation
- Transition from compute-bound to memory-bound execution

---

# 4. Windows Build and Infrastructure Failures

## Observed Issues

Development encountered:
- CMake generator mismatch
- Broken build cache
- Missing submodules
- Windows path-length limitations

---

## Root Causes

Examples:
- Previously generated build cache conflicted with new generator settings
- Long Windows paths exceeded MSBuild limitations
- Missing submodules prevented dependency compilation

---

## Resolution

Actions taken:
- Rebuilt from clean build directory
- Used short-path workspace (C:\et)
- Re-cloned repository with recursive submodules

---

## Engineering Insight

Edge AI deployment involves not only ML models, but also:
- Build systems
- Runtime dependencies
- Compiler infrastructure
- Cross-platform tooling

---

# 5. Delegate and Runtime Compatibility

## Observed Risk

Different runtimes support different:
- Operators
- Delegates
- Kernels
- Memory planners

---

## Examples

Potential deployment risks:
- Unsupported operator
- Delegate fallback
- CPU fallback path
- Kernel mismatch

---

## Engineering Insight

Runtime compatibility analysis is critical for:
- Mobile deployment
- Embedded systems
- Edge inference optimization

---

# 6. Deployment Workflow Lessons

This project highlights several important deployment principles:

## 1. Runtime configuration matters
Export configuration and runtime backend support must align.

---

## 2. Quantization is hardware-dependent
INT8 acceleration effectiveness depends heavily on:
- Hardware kernels
- Runtime implementation
- Operator fusion

---

## 3. Scaling eventually saturates
Adding threads does not guarantee linear performance improvement.

---

## 4. Runtime debugging is critical
Successful deployment requires:
- Profiling
- Operator analysis
- Backend inspection
- Runtime validation

---

# 7. Summary

This project demonstrates production-style runtime debugging and deployment troubleshooting for edge AI systems.

The work includes:
- Backend mismatch debugging
- Quantization failure analysis
- Runtime profiling
- Thread scaling analysis
- Infrastructure/build debugging

The project focuses not only on successful inference, but also on understanding:
- Why deployment fails
- Why optimization sometimes regresses
- How runtime architecture affects performance
