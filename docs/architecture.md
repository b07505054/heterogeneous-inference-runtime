# Architecture

## Purpose

This repository is an edge inference runtime and benchmarking evidence project. It compares heterogeneous inference paths for MobileNetV2-style computer vision workloads, includes native CPU/CUDA experiments, and models LLM-serving runtime behavior such as KV-cache pressure, page prefetch, continuous batching, worker routing, and failover.

The project is best understood as a collection of executable runtime components plus artifact generators. Some paths run real inference locally. Other paths read existing benchmark artifacts or simulate serving behavior. Those boundaries are explicit below.

## Top-Level Modules

### Benchmark backend abstraction

Implemented in `backends/` and driven by `backend_validation_runner.py`.

- `backends/base.py` defines `BenchmarkResult` and the abstract `Backend` interface.
- `ONNXRuntimeBackend` runs ONNX Runtime inference against a local ONNX model and requested execution provider.
- `PyTorchBackend` runs torchvision MobileNetV2 eager inference.
- `CppInferenceBackend` reads a C++ benchmark CSV artifact.
- `ThreadScalingBackend` reads thread-scaling CSV artifacts.
- `TensorRTBackend` reads a TensorRT benchmark CSV artifact.
- `ExecuTorchBackend` is intended to read an ExecuTorch CSV artifact, but currently has implementation issues noted in `technical_debt.md`.

Implemented behavior: ONNX Runtime and PyTorch benchmarks execute local inference.

Artifact-backed behavior: C++ inference, thread scaling, TensorRT, and ExecuTorch adapters summarize pre-existing CSV artifacts rather than building/running those systems inside the adapter.

### Async video inference pipeline

Implemented in `deployment/async_video_pipeline.py` and related deployment modules.

- `VideoFrameSource` wraps OpenCV `VideoCapture`.
- `AsyncVideoInferencePipeline` runs separate capture and inference threads connected by a bounded queue.
- `RuntimeMetrics` tracks frames seen, processed, dropped, FPS, and average latency.
- `PipelineTracer` records Chrome Trace-compatible stage events.
- `create_monitoring_app` exposes FastAPI health, metrics, backend, model, and Prometheus endpoints.
- `export_metrics` writes final pipeline metrics to JSON.

Implemented behavior: threaded capture, bounded queueing, metrics, trace export, monitoring API, mock inference, and ONNX Runtime CV inference.

Simulated behavior: the default `MockCVBackend` returns empty detections and does not run a model. It exists as a placeholder for pipeline validation.

### CV model runtime

Implemented in `deployment/onnx_cv_backend.py`, `deployment/model_registry.py`, and `configs/model_registry.json`.

- The model registry selects active MobileNetV2 ONNX configurations.
- The ONNX CV backend chooses the requested provider when available, otherwise falls back to the configured fallback provider.
- Frames are resized to 224x224, converted BGR to RGB, normalized to float32, transposed to NCHW, and run through ONNX Runtime.
- Output is currently reduced to `top1` classification index.

Implemented behavior: real ONNX Runtime session creation, preprocessing, provider fallback, and inference.

### LLM runtime simulator and artifact generation

Implemented mainly in `deployment/llm_runtime_decision.py`, `deployment/distributed_serving.py`, `scripts/generate_llm_runtime_artifacts.py`, and `scripts/generate_llm_runtime_mode_comparison.py`.

Responsibilities include:

- Request modeling for prompt/output token workloads.
- KV block allocation and pressure-aware admission.
- Paged KV lifecycle tracking and page-release invariants.
- Page prefetch modeling.
- Prefill/decode scheduling and continuous batching-style policy simulation.
- Paged attention read-cost modeling.
- Worker routing policies, KV prefix cache awareness, worker quarantine, retry, and failover simulation.
- JSON, Markdown, Chrome Trace, and comparison artifact generation under `results/llm_runtime_artifacts/`.

Implemented behavior: deterministic local simulation logic, policy comparisons, artifacts, and pytest coverage for key scheduler invariants.

Simulated behavior: vLLM, SGLang, Triton Server, TensorRT-LLM, distributed serving, worker health, TTFT/TPOT, and KV-cache behavior are modeled locally. The repository does not run those production serving frameworks.

### KV page physical-memory microbenchmark

Implemented in `scripts/benchmark_kv_page_microbenchmark.py`, tested in `tests/test_kv_page_microbenchmark.py`.

This is a separate, measured benchmark and is not part of the `llm_runtime_decision.py` simulation described above:

- `KVPagePool` allocates a real tensor (`torch.empty`) sized by page count, KV heads, and head dim, and tracks free/owned pages with plain Python lists/dicts.
- Checkout/release, tensor materialization, paged gather-to-contiguous, and one-token scatter update are timed with `time.perf_counter()` against a CPU/CUDA/MPS device, with device synchronization before/after each timed call.
- An allocator churn stress pass performs randomized checkout/release cycles and reports two things, separate from the steady-state request-loop numbers: free-list fragmentation (`contiguous_free_run_ratio` — largest contiguous run of free page indices as a fraction of pool size; this is free-list index fragmentation, not GPU/CPU allocator memory fragmentation), and raw page-reuse counters (`unique_pages_seen`, `pages_reused`, `page_reuse_events`).
- Each report carries a `provenance` block (git commit, timestamp, invocation args) in addition to the existing `dependency_status` block.

Implemented behavior: real tensor allocation, real timed tensor operations, real device guarding (skips/reports `"unavailable"` cleanly when a requested device is absent).

Truth boundary: this benchmark measures tensor-backed paged KV storage and gather/copy/scatter overhead on the local device. It is not vLLM PagedAttention, not TensorRT-LLM paged attention, and not wired into live Qwen (or any other model's) attention. It does not invoke or claim to invoke a live vLLM/PagedAttention CUDA kernel. It is intentionally decoupled from the `MemoryPlanner`/`PagedKVLifecycle`/`RuntimeScheduler` logical simulation above — extending one does not change numbers produced by the other.

### Native runtime and CUDA experiments

Implemented under `cpp_inference/`, `cuda_backend/`, `cuda_transformer_kernels/`, and `cuda_backend/kernels/`.

- `cpp_inference/` contains C++ ONNX Runtime inference and optional Google Benchmark targets.
- `cuda_backend/runtime/` defines a small C++ backend interface with CPU and CUDA vector-add implementations.
- `cuda_backend/apps/dispatch_main.cpp` selects CPU or CUDA vector-add backend.
- `cuda_transformer_kernels/` provides a PyTorch extension for an FP32 fused RMSNorm CUDA kernel.
- CUDA matmul/vector-add kernels are included as low-level experiments.

Implemented behavior: native source code exists for these paths. CUDA RMSNorm has correctness tests that compile the extension when CUDA is available.

Environment-dependent behavior: native and CUDA paths require local ONNX Runtime, CMake, CUDA, PyTorch extension tooling, and/or Google Benchmark.

### TVM TensorIR experiment

Implemented in `tvm_experiments/matmul_bias_relu.py` and `scripts/benchmark_tvm_matmul_bias_relu.py`.

Responsibilities:

- Build unscheduled and scheduled MatMul-Bias-ReLU TensorIR modules.
- Compile with TVM LLVM target when TVM is available.
- Compare outputs against NumPy.
- Emit benchmark and TensorIR artifacts.

Implemented behavior: optional TVM experiment module and tests are present.

Environment-dependent behavior: tests skip if TVM or LLVM support is unavailable.

### Agentic evaluation scaffold

Implemented in `agentic_eval/`.

- `ArtifactStore` allowlists benchmark artifacts.
- `BenchmarkAgent` deterministically reads artifacts, extracts metrics, filters by p95 constraint, ranks by throughput, and emits a final answer.
- `trace_judge` scores the tool trace and recommendation.

Implemented behavior: deterministic local agent loop for CI-style evaluation.

Simulated behavior: this is an agentic evaluation scaffold, not a production autonomous agent framework or external LLM integration.

## Assumptions

- Python handoff should target Python 3.11.
- Existing files under `results/` are local/historical evidence unless regenerated.
- Metrics in committed artifacts are not revalidated by reading these docs.
- Optional dependencies such as CUDA, TVM, TensorRT, ExecuTorch, and ONNX Runtime C++ are expected to be installed separately when using those paths.
- Documentation does not claim production readiness for simulated LLM serving framework comparisons.
