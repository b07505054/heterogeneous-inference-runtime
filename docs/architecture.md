# Architecture

## Purpose

This repository is a heterogeneous inference runtime and benchmarking evidence
project. It now has two measured backend lanes, Linux vLLM and Apple CoreML,
plus simulator and policy components that prototype future optimization logic.

The runtime should be read as a neutral-graph execution system:

```text
Model Artifact
  ONNX / .mlpackage / vLLM endpoint / future engine / mock
        |
        v
Adapter Registry
        |
        v
Model Adapter
        |
        v
Neutral Runtime Graph
        |
        v
Scheduler / Memory Manager / KV Cache / Prefix Cache /
Speculative Coordinator / Policy
        |
        v
Backend Dispatcher
        |
        v
Backend-specific Executor
```

It is also a measured-baseline-driven policy system:

```text
Measured Baselines
        |
        v
Capability Layer
        |
        v
Optimization Policy Engine
        |
        v
Deployment Planner
        |
        +-- Edge Deployment (CoreML)
        +-- Server Runtime (vLLM)
        +-- Simulator / Policy Evaluation
```

Measured baseline -> capability layer -> policy -> deployment decision is the
central design principle. CoreML and vLLM are treated as measured backends. The
repository optimizes deployment choices on top of them instead of
reimplementing CoreML kernels, vLLM scheduling internals, or production serving
frameworks.

Some paths run real inference locally. Other paths read existing benchmark
artifacts or simulate serving behavior. Those boundaries are explicit below.

For the current Apple/CoreML compiler-runtime integration, `.mlpackage` is the
first-stage executable handoff. ExecutionPlan remains useful as a future richer
contract, but it is not the CoreML v1 runtime centerpiece. A compiler-produced
CoreML candidate is a `.mlpackage` with adjacent `compiler_metadata.json`.
Runtime components consume that package through a model adapter, convert it to
a neutral graph, benchmark it, and then apply policy over measured evidence.

## Layered Architecture

```text
                     +-----------------------------+
                     |     Measured Baselines      |
                     +-----------------------------+
                     | Linux vLLM                  |
                     | Apple CoreML                |
                     +-------------+---------------+
                                   |
                                   v
                     +-----------------------------+
                     |      Capability Layer       |
                     +-----------------------------+
                     | Hardware Capability         |
                     | Backend Capability          |
                     | Kernel Library Capability   |
                     | Measured Support            |
                     +-------------+---------------+
                                   |
                                   v
                     +-----------------------------+
                     |  Optimization Policy Engine |
                     +-----------------------------+
                     | CoreML Edge Policy          |
                     | Server Runtime Policy       |
                     | Future Quant Policy         |
                     | Future KV Policy            |
                     +-------------+---------------+
                                   |
                                   v
                     +-----------------------------+
                     |     Deployment Planner      |
                     +-----------------------------+
                     | Constraint Solver           |
                     | Objective Engine            |
                     | Recommendation Engine       |
                     +-------------+---------------+
                                   |
                                   v
                     +-----------------------------+
                     |      Deployment Plan        |
                     +-------------+---------------+
                                   |
                                   v
                     +-----------------------------+
                     | Runtime / Simulator Layer   |
                     +-----------------------------+
                     | Prefix Cache Simulator      |
                     | Speculative Simulator       |
                     | PD Simulator                |
                     +-----------------------------+
```

### Measured Baselines

Implemented under `benchmark/` and `scripts/`, with details in
`docs/MEASURED_BASELINES.md`.

- Linux/vLLM lane: `scripts/benchmark_openai_compatible_server.py` benchmarks an
  already running OpenAI-compatible server and records TTFT, TPOT, end-to-end
  latency, tokens/sec, success/error counts, and server/model metadata.
- Apple/CoreML lane: `scripts/export_coreml_mobilenetv2.py` and
  `scripts/benchmark_coreml_cv_baseline.py` measure native CoreML MobileNetV2
  `.mlpackage` variants against PyTorch CPU and optional PyTorch MPS.
- Measured artifacts use `artifact_type: "measured_baseline"` and
  `evidence_type: "measured"`.

Generated model packages and measured JSON artifacts remain local under ignored
output directories unless explicitly exported.

### CoreML Compiler/Runtime Contract

The Apple/CoreML path is centered on package execution:

```text
ONNX / model graph
        |
        v
Compiler static optimization using shared capability profiles
        |
        v
CoreML-compatible rewrite / export directive
        |
        v
.mlpackage candidate + compiler_metadata.json
        |
        v
Runtime CoreMLModelAdapter
        |
        v
Neutral Runtime Graph
        |
        v
CoreML benchmark / CoreMLEdgePolicy / Deployment Planner
```

The compiler owns theoretical/static optimization:

- ONNX or model graph analysis.
- Backend support planning.
- Precision, layout, and compression planning.
- CoreML-compatible rewrite or export direction.
- Materializing or directing materialization of `.mlpackage` output.
- Emitting `compiler_metadata.json` beside the package.

The compiler must not claim measured performance. Compiler metadata is
artifact-level provenance, not measured evidence.

The runtime owns best dynamic execution:

- Consuming `.mlpackage`, endpoint, future engine, or mock artifacts through
  model adapters.
- Converting artifacts into a `NeutralRuntimeGraph`.
- Handling memory/current conditions and runtime availability.
- Validating capabilities and looking up measured evidence.
- Applying `CoreMLEdgePolicy`, server policies, and the Deployment Planner.
- Choosing a deployment decision without depending on compiler IR.

Current CoreML comparison paths:

```text
Path A: Direct CoreML baseline
ONNX / PyTorch
  -> direct coremltools export
  -> .mlpackage
  -> CoreML measured baseline

Path B: Compiler CoreML baseline
ONNX / model graph
  -> compiler-produced or compiler-directed .mlpackage
  -> compiler_metadata.json
  -> CoreML measured baseline

Path C: Runtime optimized CoreML
compiler .mlpackage candidates
  -> runtime policy / planner
  -> selected compute_unit / compression / input-size bucket
  -> measured selected result
```

Path A vs Path B measures whether the compiler-produced package changes
latency, package size, memory, or drift compared with direct CoreML export.
Path B vs Path C measures whether runtime dynamic policy selects a better
deployment configuration than the compiler baseline.

No speedup exists until the relevant paths are benchmarked. ExecutionPlan alone
is not measured evidence.

### Capability Layer

Implemented under `capabilities/` as schema definitions, concrete profile
data, and profile loading/validation.

The capability layer is the decision boundary between raw measurements and
policy selection. Policies should never infer hardware or backend support
directly from benchmark code; they should query capability profiles and
measured support records.

`capabilities/schema.py` defines the structure. `capabilities/profiles/`
contains concrete facts for known hardware, backends, and runtime/kernel
availability. `capabilities/profile_loader.py` validates those JSON profiles
and normalizes them against the existing schema classes.

The capability layer has four independent concepts:

- `HardwareCapability`: physical hardware facts only. Examples include Apple
  M5, Apple GPU, Apple ANE, unified memory, NVIDIA GPU family, CUDA compute
  capability, VRAM, and CPU details. No benchmark results belong here.
- `BackendCapability`: runtime/backend support only. Examples include CoreML,
  Metal, MPS, CUDA, vLLM, ONNX Runtime, and TensorRT. Supported features may be
  declared here, but measured performance must not be encoded here.
- `KernelLibraryCapability`: runtime or kernel implementation availability.
  Examples include MatMul, Conv, Attention, Softmax, RMSNorm, FlashAttention,
  PagedAttention, PrefixCache, and Speculative. Availability is categorized as
  `builtin`, `opaque`, `custom`, or `unsupported`. This is runtime/kernel
  availability, not compiler lowering.
- `MeasuredSupport`: experimentally verified support facts only. Examples
  include FP16 benchmark completed, palettization benchmark completed, input
  size 224 measured, CoreML ComputeUnit ALL measured, vLLM TTFT measured, and
  concurrency benchmark completed. Predictions do not belong here.

The compiler-side project already contains richer target-profile and
backend/kernel capability schemas. This runtime repo should reference or adapt
those schemas through JSON/artifact boundaries rather than copying compiler
C++/MLIR implementations into the runtime.

Compiler and runtime must use the same hardware, backend, and kernel capability
facts. For the first Apple/CoreML stage, profiles are centered on the current
Mac hardware and available CoreML/MPS/Metal backend/kernel support. If those
facts change, both sides must be updated together or read from the same shared
profile source. The compiler and runtime must not drift into separate
capability truths.

Capability schema is structure. Capability profiles are concrete facts.
Measured baselines are evidence. Policies consume both profiles and measured
baseline artifacts. Simulators evaluate ideas. These layers must not be merged.

### Optimization Policy Engine

The optimization policy engine is the next layer above measured evidence. It
selects deployment configurations using measured artifacts plus capability
metadata. It should make decisions such as:

- CoreML edge policy: choose compute unit, input-size bucket, and compression
  variant subject to latency, package-size, RSS, and numerical-drift constraints.
- Server runtime policy: choose concurrency/admission/routing behavior for a
  measured OpenAI-compatible/vLLM server.
- Future capability-driven policies: select quantization or KV-related
  deployment choices only when capabilities and measured evidence support them.

Policy artifacts are evidence-backed selections or candidate rankings. They are
not deployment plans, and they are not new measured baselines unless a benchmark
is actually rerun.

Policies produce evidence-backed candidates and policy selections. They should
not directly produce final deployment artifacts because final deployment
recommendations may need to compare multiple runtimes, enforce cross-runtime
constraints, and choose a user-facing objective.

### Deployment Planner

Implemented under `deployment/planner/`.

The deployment planner is the architecture layer above all optimization
policies. It consumes capability profiles, measured baselines, and policy
candidates, then emits a `deployment_plan` artifact.

The planner has three reusable pieces:

- Constraint solver: filters normalized candidates by latency, package size,
  memory, numerical drift, and throughput constraints without hardcoding CoreML
  or vLLM.
- Objective engine: ranks candidates by latency, throughput, memory, package
  size, or a documented balanced score.
- Recommendation engine: explains why the selected candidate won or why no
  candidate was eligible.

The planner does not optimize models, run benchmarks, modify CoreML, modify
vLLM, implement batching, or implement scheduler internals. It is a deployment
recommendation layer over measured evidence.

### Runtime / Simulator Layer

Simulator components remain valuable, but their role is policy prototyping and
future optimization evaluation. They must not be presented as measured
production evidence.

- Prefix cache simulator: models prefix reuse, hit/miss behavior, and routing
  decisions.
- Speculative simulator: estimates draft/verify policy decisions.
- PD simulator: models prefill/decode disaggregation, KV transfer, and queueing
  trade-offs.

These components can inform policy design, but measured claims must come from
the measured baseline layer.

### Model Adapter And Neutral Runtime Graph Layer

The core runtime should not directly depend on ONNX, CoreML, TensorRT, PyTorch,
vLLM, Qwen, Llama, MobileNet, or compiler IR. Format-specific handling belongs
behind model adapters.

Implementation status: Step 1 adds the neutral runtime graph schema, base model
adapter interface, and a test-only mock adapter. Step 2 adds a neutral
`ModelArtifact` descriptor and adapter registry/factory with only the `"mock"`
adapter registered. Step 3 adds the CoreML package metadata contract for
`.mlpackage` plus adjacent `compiler_metadata.json`; it still does not add
CoreML execution or register a CoreML adapter. It does not add vLLM, ONNX,
TensorRT, PyTorch, or compiler adapters, and it does not change runtime or
benchmark behavior.

The selection boundary is:

```text
ModelArtifact(kind, uri, metadata)
        |
        v
Adapter Registry
        |
        v
ModelAdapter
        |
        v
NeutralRuntimeGraph
```

Adapters translate source artifacts into a neutral graph:

- `CoreMLModelAdapter`: consumes `.mlpackage` plus optional
  `compiler_metadata.json`.
- `VLLMEndpointAdapter`: consumes OpenAI-compatible endpoint configuration.
- `MockModelAdapter`: produces deterministic graphs for tests.
- `ONNXModelAdapter`: future optional adapter for ONNX inspection, not a
  runtime-core dependency.

The CoreML adapter contract is:

```text
.mlpackage directory + compiler_metadata.json
        |
        v
CoreMLModelAdapter later
        |
        v
NeutralRuntimeGraph
```

For now, only the metadata parser and fixture exist. They validate that the
package path and metadata file exist and that metadata can map to neutral graph
fields without requiring ExecutionPlan or compiler IR.

A `NeutralRuntimeGraph` should expose neutral concepts:

- `model_family`
- `stages`
- `tensors`
- `memory_requirements`
- `kv_cache_requirements`
- `backend_target`
- `execution_constraints`

Schedulers, memory managers, KV cache, prefix cache, speculative coordination,
policy, and backend dispatch should consume those neutral concepts. They should
not inspect CoreML package internals, ONNX graph internals, vLLM internals,
compiler pass names, or exact model names.

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

Architecture note: this is a legacy format-specific runtime path. Future
runtime orchestration should move model-format assumptions behind model
adapters and expose only `NeutralRuntimeGraph` to scheduler, memory, policy,
and dispatcher components.

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

The simulated `PagedKVLifecycle` in `deployment/llm_runtime_decision.py` reports a `contiguous_free_run_ratio` field with the same name and definition as this microbenchmark's free-list fragmentation metric, computed from the same shape of data (a free page-index list) but from the simulation's logical page table, not from the measured tensor pool above. It also reports `kv_internal_fragmentation_ratio` — true lifetime unused-token-capacity-within-allocated-pages, accumulated across `allocate_range` calls. Both are "Simulated" per the truth-boundary labels; do not blur them with this microbenchmark's measured numbers even though `contiguous_free_run_ratio` shares a name across both. See `docs/design_decisions.md`'s "KV Cache 'Fragmentation' Must Measure Fragmentation" section.

Truth boundary: this benchmark measures tensor-backed paged KV storage and gather/copy/scatter overhead on the local device. It is not vLLM PagedAttention, not TensorRT-LLM paged attention, and not wired into live Qwen (or any other model's) attention. It does not invoke or claim to invoke a live vLLM/PagedAttention CUDA kernel. It is intentionally decoupled from the `MemoryPlanner`/`PagedKVLifecycle`/`RuntimeScheduler` logical simulation above — extending one does not change numbers produced by the other.

Offline calibration boundary: the scheduler already has a local paged-KV cost model; the `KVPagePool` benchmark is a physical measurement layer for offline/manual calibration, not runtime control. A calibrated workflow should run the benchmark on target hardware, inspect report provenance plus p50/p95 movement costs, manually adjust cost-model constants if justified, regenerate LLM runtime artifacts, and compare TPOT, throughput, OOM/reject, and page-lifecycle gates. The runtime scheduler must not load `kv_page_microbenchmark_report.json` directly as an online control signal.

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
