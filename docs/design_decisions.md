# Design Decisions

## Keep Runtime Evidence Modular

The repository uses small modules for each runtime concern instead of one monolithic application. CV benchmarking, async video inference, LLM scheduling simulation, native C++ experiments, CUDA kernels, TVM experiments, and agentic evaluation each have separate entry points.

Tradeoff: this makes it easier to demonstrate multiple systems concepts, but it also means there is no single canonical production runtime path.

Assumption: future maintainers should preserve this modular shape unless the project is intentionally productized around one workflow.

## Normalize Benchmark Results

`BenchmarkResult` gives backend adapters a shared output shape: backend, precision, device, average latency, percentile latencies, throughput, and extra metadata.

Tradeoff: a common result type simplifies comparison, but some adapters execute benchmarks while others read historical artifacts. Consumers must inspect `extra` and adapter behavior before treating results as fresh measurements.

Assumption: artifact-backed adapters are acceptable for handoff demos as long as they are clearly labeled.

## Prefer Local Executable Paths Where Practical

The project includes real ONNX Runtime and PyTorch inference paths, a real ONNX Runtime CV backend, native C++ source, and a real CUDA RMSNorm extension path.

Tradeoff: local execution provides stronger evidence but requires dependency setup across Python, native libraries, CUDA, and optional compiler stacks.

Assumption: developers may not have all optional runtimes installed on one machine.

## Use Simulation for Serving-System Concepts

LLM runtime behavior is modeled with local dataclasses, deterministic policies, cost models, and artifact generation rather than production serving frameworks.

Tradeoff: simulations are easy to run and inspect in CI-like environments, but they cannot prove production throughput, memory behavior, or failure semantics.

Assumption: serving-framework reports should be read as conceptual/runtime-model evidence, not as vLLM, SGLang, Triton Server, or TensorRT-LLM benchmark results.

## Separate Scheduler and Paged-Attention Evidence Modes

The README documents two LLM artifact modes: scheduler-focused and paged-attention. This keeps scheduler wins separate from extra paged-attention read-cost accounting.

Tradeoff: maintaining multiple modes increases artifact complexity, but it avoids mixing policy effects with memory-read modeling effects.

Assumption: mode comparisons should continue to state what each mode isolates.

## Bounded Queue for Video Pipeline Backpressure

The async video pipeline uses a bounded `queue.Queue` and drops frames when the queue is full.

Tradeoff: dropping frames protects latency and memory under load, but it sacrifices complete frame coverage.

Assumption: edge CV workloads often prefer recent frames and bounded latency over processing every frame.

## Provider Fallback for ONNX Runtime

`ONNXRuntimeCVBackend` checks available providers and falls back to the configured fallback provider when the requested provider is missing.

Tradeoff: this improves portability across CPU, CoreML, and CUDA environments, but it can hide performance differences if users do not check active/session providers.

Assumption: metrics and docs should include requested provider, active provider, and actual session providers.

## Optional Dependencies Skip or Fail Explicitly

Tests for CUDA and TVM skip when required dependencies are unavailable. Native C++ builds require explicit paths such as `ONNXRUNTIME_DIR`.

Tradeoff: the core repo remains usable without every runtime installed, but coverage varies by environment.

Assumption: maintainers should document skipped coverage when reporting validation status.

## Dataclasses for Runtime State

The LLM runtime model and metrics use dataclasses heavily for request state, cost models, memory planners, KV pages, routing state, and scheduler outputs.

Tradeoff: dataclasses keep simulation state explicit and serializable, but very large stateful classes can still become hard to navigate.

Assumption: future changes should continue preferring simple typed dataclasses and functions over deep inheritance.

## Agentic Evaluation Is Deterministic

The agentic evaluation uses a deterministic policy and a deterministic judge rather than depending on an external LLM.

Tradeoff: deterministic CI is stable and cheap, but it does not measure the variability of a real model-driven agent.

Assumption: this scaffold is intended to evaluate tool-use structure and artifact discipline, not open-ended model intelligence.

## Metrics Are Evidence, Not Universal Claims

The repository contains many historical result artifacts. Documentation should describe where metrics come from and whether they are measured, artifact-backed, simulated, or estimated.

Tradeoff: preserving artifacts is useful for demos and analysis, but stale metrics can be mistaken for current measurements.

Assumption: no new benchmark numbers should be added to documentation unless regenerated and labeled with environment metadata.
