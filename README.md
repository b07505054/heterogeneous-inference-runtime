# Edge CV Inference Deployment & Profiling Platform

[![Agentic Eval CI](https://github.com/b07505054/heterogeneous-inference-runtime/actions/workflows/agentic-eval-ci.yml/badge.svg)](https://github.com/b07505054/heterogeneous-inference-runtime/actions/workflows/agentic-eval-ci.yml)

## Overview

This project implements an end-to-end edge AI inference deployment, profiling, and runtime benchmarking platform across heterogeneous AI inference backends.

The system evaluates real-world deployment trade-offs across:

- PyTorch eager inference
- ONNX Runtime CPU / CUDA / CoreML execution providers
- ExecuTorch XNNPACK runtime
- TensorRT FP16 / FP32 inference engines
- Native C++ ONNX Runtime inference
- Quantized INT8 inference
- Multi-threaded CPU execution scaling
- TensorRT dynamic batch scaling
- vLLM/SGLang-style prefill/decode scheduling and KV-cache pressure simulation
- Triton Server-style dynamic batching/backend routing comparison artifacts
- Cold-start/model-initialization analysis for model load, backend init, TensorRT
  engine deserialize, first-token warmup, and steady-state TTFT/TPOT
- Async video inference pipelines
- Backend fallback execution
- Runtime monitoring APIs
- Metrics export infrastructure
- Prometheus-compatible metrics endpoint
- Chrome Trace / Perfetto timeline export
- Pipeline-level tracing
- Google Benchmark C++ microbenchmarking
- AddressSanitizer-enabled native runtime validation
- Model registry & versioning
- LlamaForCausalLM backend profiling for compiler/runtime demo artifacts

The LLM runtime artifact generator also emits a serving-framework comparison
track for interview-facing inference systems work:

```text
results/llm_runtime_artifacts/serving_framework_report.json
results/llm_runtime_artifacts/cold_start_report.json
```

That report maps the local scheduler and memory planner onto vLLM-style
continuous batching, SGLang-style request/decode scheduling, Triton
Server-style dynamic batching and backend instance routing, and TensorRT-style
engine/profile backend selection. It reports TTFT, TPOT, p95 latency,
tokens/sec, KV-cache pressure, and memory/SLO signals from the same workload.
The cold-start report separates model artifact load, backend initialization,
TensorRT engine deserialization/context creation, first-request TTFT penalty,
warm TTFT, steady-state TPOT, and concrete initialization-reduction techniques.

The platform simulates a production-style edge computer vision inference system
similar to modern autonomous robotics and edge AI deployment infrastructure.

---

## Agentic Benchmark Evaluation

This repo includes a lightweight tool-using ML benchmarking agent with
trace-based evaluation under `agentic_eval/`. The agent receives a high-level
task, discovers allowlisted benchmark artifacts, chooses read/parse/filter/
compare tools, and recommends the best MobileNetV2 backend under a p95 latency
constraint.

The deterministic CI policy validates the agent loop and judge without requiring
external LLM APIs, CUDA, TensorRT, or ExecuTorch. The judge scores the tool
trace, wrong-file access, p95 constraint handling, throughput tie-break, evidence
quality, and final recommendation. It is intentionally an agentic evaluation
scaffold rather than a production autonomous agent framework.

Run:

```bash
python -m agentic_eval.run_agentic_eval
pytest agentic_eval/tests
```

---

## CUDA RMSNorm Compiler/Runtime Case Study

This repo also provides the runtime evidence source for the MLIR compiler
runtime project. The CUDA RMSNorm path compares a custom CUDA kernel against a
PyTorch RMSNorm baseline, then exports benchmark evidence consumed by the
compiler-side HIR kernel-selection pipeline.

Artifacts:

```text
results/cuda_transformer/rmsnorm_benchmark.json
results/cuda_transformer/rmsnorm_benchmark_report.md
results/cuda_transformer/gpu_pgo_like_rmsnorm_report.json
results/cuda_transformer/gpu_pgo_like_rmsnorm_report.md
```

Benchmark coverage:

```text
tokens: 1, 16, 128
hidden: 768, 1024, 4096, 8192
dtype: float32
baseline: torch_rmsnorm
custom: fused_rmsnorm_cuda
```

Run on the CUDA Linux machine:

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH=$PWD/.venv-rmsnorm/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

.venv-rmsnorm/bin/python -m pytest tests/test_rmsnorm_cuda_correctness.py

.venv-rmsnorm/bin/python scripts/test_rmsnorm_cuda_correctness.py \
  --tokens 1,16,128 \
  --hidden 768,1024,4096,8192

.venv-rmsnorm/bin/python scripts/benchmark_rmsnorm_cuda.py \
  --output results/cuda_transformer/rmsnorm_benchmark.json \
  --report-output results/cuda_transformer/rmsnorm_benchmark_report.md \
  --tokens 1,16,128 \
  --hidden 768,1024,4096,8192 \
  --warmup 20 \
  --runs 100
```

The correctness test covers the fixed FP32 sweep:

```text
tokens = 1, 16, 128
hidden = 768, 1024, 4096, 8192
rtol = 1e-4
atol = 1e-4
```

It checks `torch.allclose`, rejects NaN/Inf outputs, skips cleanly when CUDA is
unavailable, and asserts that non-contiguous inputs are rejected because the
current CUDA kernel is a contiguous row-major implementation.

The report includes latency, p50/p95, speedup, bytes/token, FLOPs/token,
effective bandwidth, arithmetic intensity, environment metadata, and
memory-bound roofline notes. Environment metadata includes GPU name, CUDA
version, NVCC version, PyTorch version, NVIDIA driver version, warmup/timed run
counts, dtype, and the repo commit hash.

Nsight Compute is optional and never blocks the benchmark path:

```bash
.venv-rmsnorm/bin/python scripts/benchmark_rmsnorm_cuda.py \
  --with-nsight
```

If `ncu` is not available, the JSON report records:

```json
{
  "nsight_compute": {
    "requested": true,
    "available": false,
    "reason": "ncu not found"
  }
}
```

The GPU PGO-like report turns runtime benchmark evidence into a compiler-facing
candidate selection table:

```text
compiler-emitted HIR op + runtime shape/workload distribution
  -> benchmark CUDA/Triton/PyTorch RMSNorm candidates
  -> select lowest-correct p95 latency by shape bucket
  -> project TPOT / throughput impact for serving reports
```

Run it after the CUDA/Triton RMSNorm benchmark artifacts exist:

```bash
python3 scripts/generate_gpu_pgo_like_report.py
```

This is intentionally PGO-like rather than traditional CPU binary PGO: the
profile input is GPU kernel latency/bandwidth evidence and serving workload
shape distribution, and the decision is runtime/compiler kernel selection
instead of function/basic-block layout in a CPU binary.

---

## System Pipeline

```text
Camera / Video Stream
        ↓
Frame Capture Thread
        ↓
Async Frame Queue
        ↓
Inference Worker Thread
        ↓
Backend Runtime
   ├── ONNX Runtime
   ├── ExecuTorch
   ├── TensorRT
   └── Native Runtime
        ↓
Metrics Collector
        ↓
Monitoring API
        ↓
Metrics Export + Profiling
```

This architecture simulates production-style asynchronous edge inference deployment systems.

---

## Runtime Backends Evaluated

| Backend | Precision | Device |
|---|---|---|
| PyTorch | FP32 | CPU |
| ONNX Runtime | FP32 | CUDA |
| ONNX Runtime | Optimized FP32 | CUDA |
| ONNX Runtime | INT8 | CPU |
| ONNX Runtime | FP32 | CoreML |
| ExecuTorch | FP32 | XNNPACK |
| TensorRT | FP16 | CUDA |
| TensorRT | FP32 | CUDA |
| Native C++ ONNX Runtime | FP32 | CPU |

---

## Benchmark Summary

| Backend | Avg Latency (ms) | P95 (ms) | P99 (ms) | Throughput (QPS) |
|---|---|---|---|---|
| TensorRT FP16 | 1.64 | 1.73 | 1.74 | 608.8 |
| TensorRT FP32 | 1.68 | 1.72 | 1.73 | 596.7 |
| ONNX Runtime Optimized FP32 CUDA | 2.78 | 2.83 | 2.85 | 360.3 |
| ONNX Runtime FP32 CUDA | 2.98 | 3.41 | 3.42 | 335.0 |
| ExecuTorch XNNPACK | 1.49 | 1.55 | 1.66 | 669.1 |
| Thread Scaling 8T | 4.46 | N/A | N/A | 224.3 |
| Thread Scaling 4T | 4.52 | N/A | N/A | 221.4 |
| Thread Scaling 2T | 5.53 | N/A | N/A | 181.0 |
| Native C++ Runtime | 9.25 | N/A | N/A | 108.1 |
| PyTorch FP32 CPU | 16.88 | 17.08 | 17.68 | 59.3 |
| ONNX Runtime INT8 CPU | 30.36 | 33.78 | 35.26 | 32.9 |

---

## TensorRT Batch Scaling Analysis

| Batch Size | Avg Latency (ms) | Throughput (QPS) |
|---|---|---|
| 1 | 1.71 | 585.4 |
| 2 | 1.46 | 1367.1 |
| 4 | 1.80 | 2220.8 |
| 8 | 2.66 | 3010.4 |
| 16 | 4.18 | 3823.7 |

### Insight

TensorRT batch scaling experiments demonstrated:

- Sublinear latency growth as batch size increased
- Significant throughput scaling under larger inference batches
- Improved GPU utilization and kernel amortization
- Reduced relative runtime overhead at larger batch sizes

Throughput improved by over 6.5× from batch-1 to batch-16 inference execution.

This demonstrates typical GPU inference serving behavior where larger batches better utilize available GPU compute resources.

---

## Async Video Inference Pipeline

This project includes an asynchronous edge video inference deployment pipeline.

### Features

- Webcam / video-stream inference
- Async frame capture
- Queue-based inference scheduling
- Multi-threaded inference execution
- Runtime metrics tracking
- Dropped-frame monitoring
- Backend abstraction layer
- Backend fallback support
- Monitoring API integration
- Metrics export pipeline
- Runtime model registry integration

### Runtime Architecture

```text
Video Source
    ↓
Capture Thread
    ↓
Frame Queue
    ↓
Inference Thread
    ↓
Backend Runtime
    ↓
Metrics Collector
```

### Example Runtime Metrics

```json
{
  "frames_seen": 120,
  "frames_processed": 120,
  "frames_dropped": 0,
  "fps": 29.655,
  "avg_latency_ms": 3.688
}
```

This demonstrates stable real-time edge inference execution at ~30 FPS with zero dropped frames.

---

## Backend Fallback System

This project includes runtime backend fallback support.

### Example

```text
Requested Provider:
CUDAExecutionProvider

Fallback Provider:
CPUExecutionProvider
```

If a preferred execution backend is unavailable, the system automatically falls back to an available runtime backend without crashing inference execution.

### Example Runtime Status

```json
{
  "requested_provider": "FakeCUDAProvider",
  "active_provider": "CPUExecutionProvider",
  "session_providers": [
    "CPUExecutionProvider"
  ]
}
```

This simulates production-style resilient inference deployment infrastructure.

---

## Model Registry & Versioning

This project includes a deployment-oriented model registry system.

### Features

- Active model selection
- Runtime backend configuration
- Backend-specific provider settings
- Fallback-provider configuration
- Runtime model metadata
- Deployment-oriented model abstraction

### Registry Example

```json
{
  "active_model": "mobilenet_v2_onnx_coreml",

  "models": {
    "mobilenet_v2_onnx_coreml": {
      "backend": "onnx",
      "provider": "CoreMLExecutionProvider",
      "fallback_provider": "CPUExecutionProvider"
    }
  }
}
```

### Example Runtime Model Endpoint

```json
{
  "name": "mobilenet_v2_onnx_coreml",
  "backend": "onnx",
  "provider": "CoreMLExecutionProvider",
  "fallback_provider": "CPUExecutionProvider",
  "precision": "FP32"
}
```

This simulates production-style deployment configuration and runtime model selection systems.

---

## Monitoring API

This project includes a runtime monitoring API using FastAPI.

### Supported Endpoints

| Endpoint | Description |
|---|---|
| `/health` | Runtime health status |
| `/metrics` | Live FPS / latency metrics |
| `/metrics/prometheus` | Prometheus scrape-compatible runtime metrics |
| `/backend` | Active backend runtime status |
| `/model` | Active model registry configuration |

### Example Metrics Endpoint

```json
{
  "frames_seen": 286,
  "frames_processed": 286,
  "frames_dropped": 0,
  "fps": 29.918,
  "avg_latency_ms": 2.787
}
```

This enables real-time runtime observability for edge inference systems.

### Prometheus Metrics Endpoint

The monitoring API also exposes Prometheus text-format metrics:

```bash
curl http://127.0.0.1:8000/metrics/prometheus
```

Example output:

```text
edge_frames_seen_total{backend="MockCVBackend",active_provider="unknown"} 466
edge_frames_processed_total{backend="MockCVBackend",active_provider="unknown"} 466
edge_frames_dropped_total{backend="MockCVBackend",active_provider="unknown"} 0
edge_pipeline_fps{backend="MockCVBackend",active_provider="unknown"} 30.231
edge_avg_latency_ms{backend="MockCVBackend",active_provider="unknown"} 0.01
```

This bridges offline benchmarking with production-style observability.

---

## Metrics Export Infrastructure

The pipeline automatically exports runtime execution metrics to:

```text
results/video_pipeline_metrics.json
```

### Example Export

```json
{
  "metrics": {
    "frames_seen": 120,
    "frames_processed": 120,
    "frames_dropped": 0,
    "fps": 29.655,
    "avg_latency_ms": 3.688
  },

  "backend": {
    "name": "ONNXRuntimeCVBackend",
    "requested_provider": "CPUExecutionProvider",
    "active_provider": "CPUExecutionProvider"
  }
}
```

This enables reproducible runtime analysis and deployment reporting.

---

## ExecuTorch Integration

This project includes ExecuTorch runtime integration using the XNNPACK delegate.

### Features

- Exported PyTorch model to ExecuTorch `.pte`
- Integrated ExecuTorch Python runtime
- Benchmarked edge-oriented inference performance
- Evaluated latency and throughput against other runtimes

### Result

ExecuTorch XNNPACK achieved:

- 1.49 ms average latency
- 669.1 QPS throughput

This demonstrates efficient edge-oriented inference execution for mobile and constrained-device environments.

---

## TensorRT Integration

This project includes TensorRT FP16 / FP32 engine compilation and benchmarking.

### Features

- ONNX → TensorRT engine conversion
- FP16 optimization
- FP32 baseline comparison
- Dynamic batch inference benchmarking
- CUDA runtime execution
- GPU latency benchmarking
- Throughput profiling
- TensorRT engine profiling

### Result

TensorRT FP16 achieved:

- 1.64 ms average latency
- 608.8 QPS throughput

TensorRT FP32 achieved:

- 1.68 ms average latency
- 596.7 QPS throughput

TensorRT provided the best overall GPU runtime efficiency in this benchmark suite.

---

## Nsight Systems Profiling

This project includes GPU runtime profiling using NVIDIA Nsight Systems.

### Profiling Scope

- CUDA kernel execution timeline
- GPU runtime scheduling
- CUDA stream execution
- Runtime synchronization behavior
- GPU inference execution analysis

### Observation

Nsight Systems profiling showed that MobileNetV2 batch-1 inference was dominated by runtime overhead and sparse GPU utilization rather than tensor-core saturation, explaining the relatively small performance gap between TensorRT FP16 and FP32 execution.

---

## Native C++ ONNX Runtime Inference

This project includes a native C++ ONNX Runtime inference pipeline using CMake.

### Features

- ONNX model loading
- Warmup + timed inference
- CSV benchmark export
- Latency reporting
- Production-style runtime integration
- AddressSanitizer build option for native runtime safety checks
- Google Benchmark target for standardized C++ latency measurement

### AddressSanitizer Validation

Prepare the ONNX Runtime C++ package:

```bash
tar -xzf third_party/onnxruntime-osx-arm64-1.26.0.tgz -C third_party
```

```bash
cmake -S cpp_inference -B build/cpp_asan \
  -DONNXRUNTIME_DIR=$PWD/third_party/onnxruntime-osx-arm64-1.26.0 \
  -DENABLE_ASAN=ON \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/cpp_asan

ASAN_OPTIONS=abort_on_error=1 \
./build/cpp_asan/edge_onnx_cpp models/mobilenet_v2_optimized.onnx
```

### Google Benchmark

```bash
cmake -S cpp_inference -B build/cpp_bench \
  -DONNXRUNTIME_DIR=$PWD/third_party/onnxruntime-osx-arm64-1.26.0 \
  -DENABLE_GOOGLE_BENCHMARK=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/cpp_bench

./build/cpp_bench/benchmark_onnx_cpp \
  --benchmark_out=results/google_benchmark_onnx_cpp.json \
  --benchmark_out_format=json
```

Example result:

```text
BM_ONNXRuntimeCpp/0    2.24 ms
```

---

## Profiling & Visualization

The project automatically generates:

- Latency comparison plots
- Throughput comparison plots
- p95 latency plots
- p99 latency plots
- Batch scaling plots
- Runtime metrics exports
- Backend validation summaries
- ONNX Runtime operator profile summaries
- Chrome Trace / Perfetto-compatible runtime timelines
- Pipeline-level capture / queue / inference traces
- LlamaForCausalLM TTFT / TPOT scaling artifacts for MPS, CPU, and CUDA when
  available

Generated benchmark figures are stored under:

```text
results/
```

### ONNX Runtime Operator Profiling

```bash
.venv/bin/python scripts/benchmark.py \
  --backend onnx \
  --model-path models/mobilenet_v2_optimized.onnx \
  --batch-size 1 \
  --threads 4 \
  --iterations 100 \
  --profile

PROFILE=$(ls -t results/onnxruntime_profile*.json | head -1)

.venv/bin/python scripts/analyze_onnx_profile.py \
  --profile "$PROFILE" \
  --output-json results/onnx_operator_profile_summary.json
```

Example operator summary:

```text
FusedConv    total=260.8260 ms count=5400 avg=0.0483 ms
Conv         total=28.4450 ms count=840 avg=0.0339 ms
Gemm         total=8.2530 ms count=120 avg=0.0688 ms
```

### Chrome Trace / Perfetto Export

```bash
PROFILE=$(ls -t results/onnxruntime_profile*.json | head -1)

.venv/bin/python scripts/export_chrome_trace.py \
  --profile "$PROFILE" \
  --output-trace results/chrome_trace_onnx_cpu.json \
  --output-summary results/chrome_trace_onnx_cpu_summary.json
```

Open the trace in:

```text
https://ui.perfetto.dev
```

### Pipeline-Level Trace

```bash
PYTHONPATH=$PWD .venv/bin/python deployment/async_video_pipeline.py \
  --source data/synthetic_input.mp4 \
  --backend mock \
  --max-frames 300 \
  --trace-output results/pipeline_trace_mock.json
```

The exported trace includes:

- `capture`
- `queue_enqueue`
- `queue_wait`
- `inference`
- `metrics_update`

This shows per-frame runtime behavior beyond aggregate latency numbers.

---

## LLM Runtime Artifact Export

This repo is also the runtime producer for the external
`mini-llm-serving-runtime-demo` workbench. The demo does not import the Python
runtime directly; it consumes committed artifact snapshots from:

```text
results/llm_runtime_artifacts/
```

Key outputs:

- `runtime_profile.json`: aggregate serving-path metrics
- `prefill_decode_benchmark.json`: prefill and decode phase timing
- `scheduler_trace.json`: request admission, queue, prefill, and decode events
- `kv_cache_trace.json`: KV-cache allocation, page lifecycle, and block-usage trace
- `plan_benchmark_results.json`: measured plan comparison for Metal, CPU, and
  hybrid candidates
- `scheduler_decision_report.json`: baseline-vs-cost-aware scheduling comparison
  driven by a runtime cost model, KV memory planner, and profiling calibration
- `page_prefetch_report.json`: vLLM-style allocated KV page prefetch policy
  comparison against the no-prefetch scheduler
- `page_prefetch_trace.json`: scheduler and serving events showing page prefetch
  attempts, hits, misses, and pressure skips
- `distributed_serving_report.json`: distributed serving routing comparison
  across round-robin, least-queue, and KV-aware policies
- `fault_tolerance_report.json`: worker timeout, retry, quarantine, and
  failover behavior with latency impact
- `grpc_contract_report.json`: protobuf contract coverage for distributed
  serving control-plane messages
- `llm_runtime_chrome_trace.json`: Perfetto-compatible runtime timeline
- `real_llama_profile.json`: HuggingFace `LlamaForCausalLM` backend profile with
  TTFT, TPOT, batch/sequence scaling, and operator bottleneck breakdown

The current LLM artifact generator now runs an actual scheduling decision loop:

```text
synthetic request stream
  -> FCFS fixed-batch baseline scheduler
  -> collect observed prefill/decode latency samples
  -> calibrate runtime cost model
  -> run cost-aware memory-pressure scheduler
  -> run TensorRT-LLM-aligned local in-flight scheduler candidate
  -> select the better policy from measured throughput, p95 latency, and KV usage
```

The scheduler is implemented in `deployment/llm_runtime_decision.py`. It includes:

- a table-free runtime cost model for prefill, decode, and KV update cost
- a KV block memory planner with allocation, free, peak usage, and pressure tracking
- explicit admission decisions: `admit`, `delay`, or `reject`
- pressure levels: `low`, `medium`, `high`, and `critical`
- a baseline FCFS policy
- a cost-aware memory-pressure policy that forms larger decode batches when KV
  capacity and shape compatibility allow it
- a vLLM-style page prefetch candidate that warms already allocated KV pages
  when memory pressure is below budget
- a local policy named `inflight_paged_kv_continuous_batching` with
  TensorRT-LLM-aligned concepts, not TensorRT-LLM internals
- request lifecycle states: `waiting`, `prefill`, `decode`, `finished`, and
  `rejected`
- event-loop scheduler ticks that choose `prefill_chunk`, `decode_batch`,
  `mixed_step`, `drain_decode`, or `reject_or_delay`
- chunked prefill with `prefill_chunk_tokens = 256`
- page-level KV lifecycle metadata: page id, owner request id, token range,
  resident/prefetched state, and last access step
- projected-pressure batch limiting, so the scheduler can reject a batch
  candidate before admitting it would push KV usage into a higher pressure band
- a profiling feedback step that calibrates the cost model from observed samples

The in-flight policy is deliberately scoped as a local runtime policy
implementation:

```text
Implemented a TensorRT-LLM-aligned local runtime policy with in-flight batching,
paged KV cache orchestration, memory-pressure-aware admission, and TTFT/TPOT
validation.
```

It does not claim to modify TensorRT-LLM internals, implement a TensorRT-LLM
engine, or provide real multi-GPU/multi-node TensorRT-LLM serving. Distributed
fields such as worker id and device id are trace hooks for future integration.

This moves the LLM runtime path from artifact description toward runtime
decision-making: the generated report records which scheduler policy won and why.
For the committed artifact snapshot, the optimized scheduler records
`pressure_limited_candidates`, showing that memory pressure directly changed
batching decisions rather than only being reported after the fact.

The in-flight policy is selected only when its gate passes:

```text
TPOT p95 improves or stays within 3%
throughput improves
OOM/rejection does not regress
TTFT p95 stays within tolerance
lifecycle invariants pass
```

The artifact allows older policies to remain selected when those gates fail.
Even when not selected, the in-flight candidate trace remains embedded in
`scheduler_trace.json` for validation.

The page prefetch path is intentionally labeled as vLLM-style rather than a
real vLLM fork. Its gate is:

```text
vLLM-style request/decode trace + KV block allocations
  -> prefetch next decode KV pages under memory-pressure budget
  -> measure hit rate, wasted prefetch blocks, TPOT, throughput, and OOM rate
```

The distributed serving path is also artifact-backed rather than a production
cluster claim:

```text
vLLM-style request trace + worker/KV residency state
  -> round-robin vs least-queue vs KV-aware routing
  -> measure TTFT, TPOT, throughput, cache hit rate, and queue wait
```

Fault tolerance is exercised by injecting a worker timeout, retrying the
request, quarantining the worker, and recording failover plus latency impact.
The protobuf schema in `protos/distributed_serving.proto` defines the control
plane contract; it is not presented as a production gRPC deployment.

Generate deterministic serving artifacts:

```bash
.venv/bin/python scripts/generate_llm_runtime_artifacts.py \
  --output-dir results/llm_runtime_artifacts
```

Generate real backend Llama profiling artifacts:

```bash
.venv/bin/python scripts/profile_tiny_llama_backends.py \
  --profile-mode hf \
  --hf-model-id hf-internal-testing/tiny-random-LlamaForCausalLM \
  --output-dir results/llm_runtime_artifacts \
  --devices mps,cpu,cuda \
  --batch-sizes 1,2 \
  --sequence-lengths 64,128 \
  --decode-steps 4 \
  --warmup 1 \
  --runs 2
```

`--hf-model-id` accepts either a HuggingFace model id or a local model path.
The committed artifact uses a tiny public Llama model to keep the repo
reproducible, while preserving the same profiling path for larger local
Llama-family models.

### Transformer Kernel Feedback

This repo now also produces runtime kernel profile evidence for
compiler-side kernel selection. The first transformer kernel target is
RMSNorm:

```text
PyTorch RMSNorm fallback
  vs
custom fused_rmsnorm_cuda extension
  -> results/cuda_transformer/rmsnorm_benchmark.json
  -> ml-graph-compiler-runtime kernel-selection metadata
```

Run the benchmark on a CUDA-capable machine:

```bash
python3 scripts/benchmark_rmsnorm_cuda.py \
  --output results/cuda_transformer/rmsnorm_benchmark.json
```

### Optional Triton RMSNorm Candidate

The same RMSNorm compiler candidate can be benchmarked against a Triton kernel:

```bash
python3 scripts/benchmark_rmsnorm_triton.py \
  --output results/cuda_transformer/rmsnorm_triton_benchmark.json \
  --report-output results/cuda_transformer/rmsnorm_triton_benchmark_report.md
```

If Triton or CUDA is unavailable, the script writes an `unavailable` profile
instead of failing CI. On a CUDA machine with Triton installed, it records
correctness, latency, bandwidth, and `selection_ready` metadata for
`fused_rmsnorm_triton` versus `torch_rmsnorm`.

If CUDA or PyTorch is unavailable, the script still emits an explicit
`profile_status: unavailable` artifact. The compiler should then retain the
fallback kernel instead of claiming that a custom CUDA kernel won without
runtime evidence.

---

## Project Structure

```text
heterogeneous-inference-runtime/
│
├── backends/
├── benchmarks/
├── configs/
├── cpp_inference/
├── data/
├── deployment/
├── models/
├── results/
├── scripts/
├── third_party/
│
├── backend_validation_runner.py
├── README.md
└── requirements.txt
```

---

## Reproduce Results

### Run Async Video Pipeline

```bash
python -m deployment.async_video_pipeline \
  --source 0 \
  --backend onnx \
  --model mobilenet_v2_onnx_coreml \
  --enable-api
```

### Query Monitoring API

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/backend
curl http://127.0.0.1:8000/model
curl http://127.0.0.1:8000/metrics
curl http://127.0.0.1:8000/metrics/prometheus
```

### Run Monitoring API Smoke Test

```bash
.venv/bin/python scripts/smoke_test_monitoring_api.py
```

### Run Backend Validation

```bash
python backend_validation_runner.py
```

### Run TensorRT Batch Scaling

```bash
python benchmarks/benchmark_tensorrt_batch.py
```

### Run Nsight Systems Profiling

```bash
nsys profile -o trt_fp16_profile python benchmarks/benchmark_tensorrt.py
nsys profile -o trt_fp32_profile python benchmarks/benchmark_tensorrt_fp32.py
```

---

## Future Work

- Nsight Compute profiling
- TensorRT INT8 calibration
- Dynamic backend scheduling
- Runtime backend auto-selection
- CUDA stream overlap optimization
- Operator fusion analysis
- Edge model hot-swapping
- Multi-camera inference
- Distributed edge inference
- vLLM / TensorRT-LLM integration

---

## Key Takeaway

Efficient edge AI deployment requires more than model accuracy.

Production-grade inference systems require optimization across:

- Runtime infrastructure
- Execution backends
- Asynchronous scheduling
- Backend fallback
- Model versioning
- Runtime monitoring
- Quantization strategy
- Thread scheduling
- GPU acceleration
- Profiling infrastructure
- Batch scheduling behavior
- Hardware-aware execution
