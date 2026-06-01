# Thread Scaling and Roofline-Style Analysis (Edge AI Runtime Systems)

##  Overview

This document analyzes thread scaling behavior and runtime saturation characteristics observed during edge inference benchmarking.

The analysis focuses on:
- Latency scaling
- Throughput scaling
- CPU utilization
- Speedup efficiency
- Runtime saturation
- Roofline-style performance interpretation

The goal is to understand how runtime performance evolves as parallelism increases in edge AI systems.

---

# 1. Motivation

Modern edge AI runtimes rely heavily on:
- Multi-threaded execution
- Parallel operator scheduling
- CPU backend optimization

However:

`text
Increasing thread count does not guarantee linear performance improvement
`

This project investigates:
- When scaling helps
- When scaling saturates
- Why diminishing returns emerge

---

# 2. Experimental Setup

## Model

- MobileNetV2
- ExecuTorch deployment format (.pte)

---

## Runtime

- ExecuTorch C++ runtime
- XNNPACK backend
- CPU-only inference

---

## Thread Configurations

Evaluated:
- 1 thread
- 2 threads
- 4 threads
- 8 threads

---

## Collected Metrics

Measured:
- Average latency
- Throughput (ops/sec)
- CPU utilization
- Speedup
- Tail latency behavior

---

# 3. Latency Scaling

Observed latency:

| Threads | Avg Latency |
|---|---|
| 1 | ~14 ms |
| 2 | ~8 ms |
| 4 | ~5.7 ms |
| 8 | ~5.8 ms |

---

## Key Observation

Latency improves significantly from:
- 1 → 2 threads
- 2 → 4 threads

However:
- Improvement plateaus beyond 4 threads

---

## Interpretation

Initial scaling indicates:
- Effective parallel execution
- Good CPU utilization
- Reduced operator execution time

Plateau behavior suggests:
- Runtime saturation
- Resource contention
- Hardware limits

---

# 4. Throughput Scaling

Observed throughput behavior:
- Near-linear increase up to 4 threads
- Plateau beyond 4 threads

---

## Interpretation

This indicates:
- Additional threads initially improve work completion rate
- Beyond saturation point, extra threads provide diminishing benefit

Likely causes:
- Thread synchronization overhead
- Shared resource contention
- Cache pressure
- Memory bandwidth limitations

---

# 5. CPU Utilization Analysis

Observed CPU utilization:

| Threads | Approximate CPU Usage |
|---|---|
| 1 | ~10–40% |
| 2 | ~40–60% |
| 4 | ~70–90% |
| 8 | Near saturation |

---

## Key Observation

CPU utilization scales with thread count.

Interpretation:
- Runtime successfully parallelizes execution
- Additional threads increase hardware utilization
- System approaches saturation at higher thread counts

---

# 6. Speedup Analysis

Observed speedup behavior:

| Threads | Approximate Speedup |
|---|---|
| 1 | 1× |
| 2 | ~1.7× |
| 4 | ~2.5× |
| 8 | ~2.4× |

---

## Key Observation

Speedup initially increases rapidly but eventually plateaus.

This demonstrates:
- Parallel execution benefits
- Non-linear scaling behavior
- Hardware/resource limitations

---

# 7. Tail Latency Behavior

Observed behavior:
- Most executions remained stable
- Occasional latency spikes appeared

Potential causes:
- OS scheduling
- Threadpool overhead
- Background processes
- Cache misses

---

## Engineering Importance

Tail latency matters because:
- Real-time systems care about worst-case latency
- Stable inference is critical for production deployment
- Edge devices often operate under constrained conditions

---

# 8. Runtime Saturation Analysis

Observed saturation beyond 4 threads suggests:

## Potential Bottlenecks

### 1. Memory Bandwidth Saturation

Additional threads increase:
- Memory access pressure
- Shared bandwidth demand

Eventually:
- Memory subsystem becomes limiting factor

---

### 2. Thread Contention

More threads introduce:
- Scheduling overhead
- Synchronization overhead
- Context switching

---

### 3. Cache Pressure

Additional threads:
- Increase cache competition
- Reduce cache locality
- Increase memory traffic

---

# 9. Roofline-Style Interpretation

The observed behavior aligns with Roofline-style performance analysis.

---

## Compute-Bound Region

Initial scaling:
- Strong latency improvement
- Increased throughput
- Efficient parallel execution

Interpretation:
- System benefits from additional compute resources

---

## Transition Region

At higher thread counts:
- Scaling efficiency decreases
- Throughput growth slows
- CPU utilization continues increasing

Interpretation:
- System approaches hardware limits

---

## Memory-Bound Region

Beyond saturation:
- Latency stops improving
- Additional threads provide minimal benefit

Interpretation:
- Execution becomes memory-bound
- Runtime limited by shared resources rather than raw compute

---

# 10. Operational Intensity Perspective

This project does not implement full FLOPs/byte roofline modeling.

However, observed runtime behavior suggests:
- Convolution-heavy workloads initially utilize compute efficiently
- Increasing parallelism eventually amplifies memory pressure
- Scaling efficiency decreases as memory movement dominates execution cost

---

# 11. Runtime Engineering Insights

This project demonstrates several important edge AI runtime principles:

## 1. Parallelism helps — up to a point
Multi-threading improves performance only until shared resources saturate.

---

## 2. Latency alone is insufficient
Performance analysis should include:
- Throughput
- CPU utilization
- Speedup
- Tail behavior

---

## 3. Runtime behavior reflects hardware constraints
Edge AI systems are strongly affected by:
- Memory hierarchy
- Thread scheduling
- Cache behavior
- Bandwidth limits

---

## 4. Profiling is critical
Understanding bottlenecks requires:
- Runtime measurement
- Operator analysis
- System-level profiling

---

# 12. Future Work

Potential future extensions:
- Full roofline model (FLOPs/byte)
- Hardware performance counters
- NUMA analysis
- Cache miss profiling
- Thermal throttling analysis
- Mobile-device benchmarking
- NPU/GPU execution comparison

---

# 13. Summary

This project performs system-level thread scaling analysis for edge AI inference runtimes.

The work combines:
- Runtime benchmarking
- CPU utilization analysis
- Throughput evaluation
- Speedup analysis
- Roofline-style interpretation

The result is a production-style performance engineering workflow for modern edge AI systems.
