# Data Flow

## Computer Vision Benchmark Flow

Inputs:

- ONNX model files in `models/`.
- Torchvision MobileNetV2 for PyTorch benchmark paths.
- CLI parameters such as backend, model path, thread count, batch size, iteration count, provider, and profiling flag.
- Optional existing CSV artifacts for artifact-backed benchmark adapters.

Processing:

1. `run_pipeline.py` or benchmark scripts parse CLI arguments.
2. `scripts/benchmark.py` runs either PyTorch eager inference or ONNX Runtime CPU inference.
3. `backend_validation_runner.py` instantiates backend adapters and calls `benchmark()`.
4. Executable adapters produce latency lists and calculate average, p50/p95/p99, throughput, memory, and model-size fields where supported.
5. Artifact-backed adapters parse CSV rows and normalize them into `BenchmarkResult`.

Outputs:

- CSV benchmark files under `results/`.
- `results/backend_validation_summary.json` from `backend_validation_runner.py`.
- ONNX Runtime profile JSON files when profiling is enabled.
- Plot scripts can turn result JSON/CSV files into SVG summaries.

Important structures:

- `BenchmarkResult`: normalized backend, precision, device, latency, throughput, and extra metadata.
- Backend adapter classes in `backends/`.

Metrics:

- Average latency in milliseconds.
- p50/p95/p99 latency when available.
- Throughput in queries per second.
- Memory delta and model size in selected benchmark scripts.
- Provider metadata for ONNX Runtime.

Metrics caveat: committed metric values are historical artifacts unless regenerated. This document describes metric fields, not fresh measurements.

## Async Video Pipeline Flow

Inputs:

- Video file, camera index, or other OpenCV-supported source.
- Model registry config from `configs/model_registry.json`.
- CLI options for backend type, ONNX provider, fallback provider, queue size, max frames, API host/port, metrics output, and trace output.

Processing:

1. `ModelRegistry` loads the active or named model config.
2. `AsyncVideoInferencePipeline` creates a `VideoFrameSource`, backend, bounded queue, metrics collector, and optional tracer.
3. Capture thread reads frames and attempts non-blocking enqueue.
4. Full queues cause frame drops and dropped-frame metrics.
5. Inference thread dequeues frames, measures queue wait, invokes `backend.infer(frame)`, records latency, and updates metrics.
6. Optional FastAPI server exposes live state while the pipeline runs.
7. Optional tracer exports Chrome Trace-compatible events.

Outputs:

- Final metrics JSON from `export_metrics`.
- Optional Chrome Trace JSON.
- Optional FastAPI responses:
  - `/health`
  - `/metrics`
  - `/backend`
  - `/model`
  - `/metrics/prometheus`

Important structures:

- Queue item: `{"frame_id", "frame", "timestamp"}`.
- `RuntimeMetrics`: recent latency deque, frame counters, start time.
- Trace event: frame id, stage, start time, duration, thread id, metadata.
- ONNX backend result: backend name, requested/active providers, session providers, latency, and `top1`.

Implemented versus simulated:

- `ONNXRuntimeCVBackend` performs real ONNX Runtime inference.
- `MockCVBackend` is a simulation placeholder and always returns empty detections.

## LLM Runtime Simulation Flow

Inputs:

- Synthetic `Request` rows containing request id, prompt tokens, output tokens, and arrival time.
- Scheduler policy name.
- Memory planner settings: total blocks, block size, and KV MB per block.
- Cost model settings for prefill, decode, and KV updates.
- Optional mode settings for scheduler-focused or paged-attention accounting.

Processing:

1. Request generation creates deterministic synthetic workloads.
2. `RuntimeScheduler` applies a selected policy.
3. `MemoryPlanner` decides admit, delay, or reject based on available KV blocks and pressure level.
4. Prefill/decode steps are simulated with cost-model latencies.
5. Page prefetch and paged attention read-cost models optionally adjust decode evidence.
6. Paged KV lifecycle records page allocation, access, prefetch, release, and invariants.
7. Summarizers compute latency percentiles, throughput, pressure, rejection/OOM counts, and lifecycle checks.
8. Artifact scripts write JSON, Markdown, and trace files.

Outputs:

- Scheduler traces, serving traces, runtime profiles, reports, and mode comparisons under `results/llm_runtime_artifacts/`.
- Chrome Trace-compatible JSON for timeline viewing.
- Reports comparing local policy models to serving-framework concepts.

Important structures:

- `Request`: request id, prompt tokens, output tokens, arrival time.
- `RuntimeRequestState`: per-request progress through waiting, prefill, decode, finished, or rejected states.
- `MemoryPlanner`: free block pool, allocations, pressure, peak blocks, delayed/rejected counters.
- `KVPage` and `PagedKVLifecycle`: page ownership, token ranges, state, allocation/free accounting.
- `PagePrefetchPlanner`: warmed pages, hits, misses, attempts, skips, and waste.
- `SchedulerResult`: scheduler steps, events, placements, KV requests, latencies, counts, throughput, lifecycle summaries, and paged attention summary.
- `DistributedRequest`, `WorkerState`, and `RouteResult`: distributed routing simulation state.

Metrics:

- TTFT and TPOT latency fields.
- Request latency percentiles.
- Tokens per second.
- Decode batch size and efficiency.
- KV pressure, peak KV MB, allocated/freed pages, page leak count.
- Cache hits/misses and prefetch hit rate.
- Worker retry, quarantine, failover, and cache-hit counters.

Metrics caveat: these metrics are produced by local models and synthetic workloads. They are not measurements from production vLLM, SGLang, Triton Server, or TensorRT-LLM.

## Native/CUDA Flow

Inputs:

- C++ executable arguments such as ONNX model path.
- CUDA vector inputs or PyTorch tensors for RMSNorm extension tests.
- Build-time environment variables such as `ONNXRUNTIME_DIR` and `CUDA_HOME`.

Processing:

- C++ ONNX Runtime creates a session, warms up, times inference, and writes a CSV row.
- Google Benchmark target measures ONNX Runtime C++ inference when built.
- CPU/CUDA vector-add dispatch chooses a backend and computes output vectors.
- RMSNorm extension validates tensor shape, dtype, device, and contiguity before launching a CUDA kernel.

Outputs:

- Native benchmark CSV artifacts.
- Console output from sample apps.
- CUDA RMSNorm benchmark/profile JSON and Markdown artifacts when scripts are run.

Implemented versus environment-dependent:

- Source code and tests are implemented.
- Execution depends on local native dependencies and hardware.

## Agentic Eval Flow

Inputs:

- Allowlisted artifacts:
  - `results/backend_validation_summary.json`
  - `results/report.md`
- A task asking for the best backend under a p95 constraint.

Processing:

1. List allowlisted artifacts.
2. Read the backend summary and report.
3. Extract backend metrics.
4. Filter candidates by p95 latency.
5. Compare candidates by throughput with p95 tie-break.
6. Emit final answer.
7. Judge the trace against deterministic expectations.

Outputs:

- `AgentRun` containing task, answer, tool-call trace, and evaluation result.

Assumption:

- The expected answer in `trace_judge.py` is tied to specific committed artifact values. If artifacts change, judge expectations may need updating.
