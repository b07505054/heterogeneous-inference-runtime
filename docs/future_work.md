# Future Work

## Roadmap

Phase 1 is complete:

- Linux measured baseline for an external OpenAI-compatible vLLM server.
- Simulator and policy-evaluation components for prefix cache, speculative
  decoding, prefill/decode planning, and paged-KV behavior.
- Runtime execution plan schema, compiler-runtime adapter, backend dispatcher,
  typed runtime decisions, and execution trace recorder.
- vLLM execution-plan validation and materialization helpers under
  `deployment/vllm_adapter/`.
- Runtime profile trace generation from compiler artifacts, including committed
  iPhone A17 Pro and frontend-normalized Qwen trace artifacts.

Phase 2 is in progress:

- Server runtime policy that uses measured vLLM/OpenAI-compatible artifacts to
  guide concurrency, admission, and routing decisions.
- A durable feedback database for measured vLLM runtime metrics. Existing
  measured baselines and trace artifacts are evidence inputs, not a long-lived
  feedback store.
- ~~End-to-end evaluation of compiler-materialized vLLM runtime config against
  a comparable measured baseline.~~ **Done** for the no-quant Qwen 2.5-0.5B /
  GTX 1650 Max-Q case: see `docs/MEASURED_BASELINES.md` ("No-Quant Qwen
  Compiler-Guided vLLM Evidence") and `results/qwen_no_quant/`. Result:
  compiler-guided no-quant matches the manually-tuned conservative baseline
  within ~1% E2E across three repeatability trials — benchmark noise, not a
  speedup. Remaining: extend this comparison to a quantized (AWQ/GPTQ) path
  once one exists (see Phase 3/C below).

Phase 3:

- Extend the completed fused-operator capability-driven quantization policy to
  full models and graph-wide mixed precision.
- Capability-driven deployment policy that joins hardware capability, backend
  capability, kernel-library capability, and measured support.
- Extend the existing Model Adapter and Neutral Runtime Graph architecture so
  runtime orchestration can consume more real source artifacts through neutral
  model-family/stage/tensor/memory/KV/backend-target concepts instead of ONNX,
  TensorRT, PyTorch, vLLM, or model-name internals.

Phase 4:

- Future measured optimizations, added only when the optimization can be
  benchmarked and exported through the measured baseline schema.
- Richer compiler-runtime contracts that keep ExecutionPlan as the main
  vLLM-facing artifact.

## vLLM Execution Planning

ExecutionPlan is now the active compiler-runtime handoff. The remaining work is
to close the loop from planning artifact to measured runtime feedback:

```text
Client Requests
  -> compiler / execution planner
  -> hardware profile + backend profile + workload facts
  -> ExecutionPlan
  -> vLLM runtime config
  -> measured runtime metrics
  -> feedback database
```

Implemented pieces:

- `deployment/execution_plan/schema.py` defines the runtime-facing
  ExecutionPlan v2 contract.
- `deployment/vllm_adapter/plan_schema.py` validates compiler-produced vLLM
  execution plans and rejects embedded measured-performance claims.
- `deployment/vllm_adapter/config_materializer.py` materializes vLLM runtime
  configuration from planning artifacts.
- Runtime profile traces can be generated from compiler artifacts and are
  explicitly labeled as offline simulation artifacts unless a benchmark
  produced measured data.

Required future work:

- Define a durable feedback database schema for measured TTFT, TPOT, E2E
  latency, throughput, success/error counts, and memory pressure.
- Ensure compiler and runtime read the same NVIDIA/CUDA hardware profile and
  vLLM backend capability profile, or update their profile copies together with
  matching profile IDs and digests.
- Benchmark compiler-materialized runtime configs against comparable measured
  vLLM/OpenAI-compatible baselines before claiming speedup.

Do not claim vLLM speedup until a planned runtime config is benchmarked against
a comparable measured vLLM baseline.

### Qwen GTX 1650 Phase C: Quantized (AWQ/GPTQ) — Not Implemented

The no-quant Qwen A/B comparison (above) is complete and measured. A quantized
Phase C is not implemented anywhere in this repo or in
`ml-graph-compiler-runtime`. Minimum remaining work, in order:

1. A real AWQ/GPTQ export step in `ml-graph-compiler-runtime` (or a dedicated
   script here) producing a quantized Qwen weight artifact from the original
   `Qwen/Qwen2.5-0.5B-Instruct` checkpoint. No such tool exists today — the
   only AWQ/GPTQ code present on this machine is inside third-party packages
   (`torchao.prototype.awq`/`gptq`), unused by any project script.
2. A compiler target profile that declares int4/AWQ backend support.
   `nvidia_gtx1650_maxq.json` currently declares `supportedQuantModes:
   ["none"]` (Turing, cc 7.5, no native INT4 tensor cores) and cannot honestly
   produce an AWQ `QuantizationDecision` as-is.
3. Extend `deployment/vllm_adapter/config_materializer.py` to emit
   `--quantization awq|gptq` and point `--model` at the quantized artifact
   path instead of the HF repo id.
4. A repeatability benchmark pass for the quantized path mirroring
   `results/qwen_no_quant/repeatability_summary.md`, with its own measured
   evidence directory (e.g. `results/qwen_quant/`).

## Neutral Runtime Graph

The neutral runtime graph layer exists under `deployment/model_adapter/` with a
test-only `MockModelAdapter`, registry helpers, and pytest coverage. The next
step is to add real optional source adapters without leaking source-format
details into the runtime core.

The graph should describe:

- Model family.
- Stages.
- Tensors.
- Memory requirements.
- KV-cache requirements.
- Backend target.
- Execution constraints.

Adapters should translate source artifacts into this graph:

- `MockModelAdapter`: implemented for deterministic tests.
- `VLLMEndpointAdapter`: future OpenAI-compatible endpoint configuration.
- `ONNXModelAdapter`: future optional adapter.

Runtime core should not know exact model names such as Qwen, Llama, or
MobileNet; source format internals; compiler IR; or backend-specific package
internals.

## Clarify Runtime Modes

- Keep the README matrix and high-level docs current so each path is labeled as
  measured live, artifact-backed, simulated, or optional/hardware-dependent.
- Add this label to `BenchmarkResult.extra` for every backend adapter.
- Rename or document artifact-backed adapters so users do not assume they execute live benchmarks.

## Strengthen Testing

- Add unit tests for metrics, tracing, model registry, Prometheus formatting, and metrics export.
- Add mocked tests for ONNX provider fallback.
- Add deterministic tests for queue drops and stop behavior in the async video pipeline.
- Add CSV schema tests for artifact-backed backend adapters.
- Add smoke tests for artifact generators that validate required JSON keys without asserting fragile benchmark values.

## Improve Backend Adapters

- Fix `ExecuTorchBackend` constructor/state issues.
- Add existence and schema checks for all CSV-backed adapters.
- Include a `source_type` field such as `live`, `artifact`, or `simulated`.
- Consider a shared helper for percentile and throughput calculations.
- Keep backend classes small; prefer helper functions for parsing and validation.

## Make Metrics More Reproducible

- Add environment metadata to every generated benchmark artifact, including Python version, package versions, platform, CPU/GPU info where available, command line, timestamp, and git commit.
- Avoid updating benchmark artifacts without recording the environment.
- Add a script that validates whether committed result artifacts match the current schema.
- Label modeled or estimated metrics as modeled/estimated in generated reports.

## Add Offline KV Cost Calibration Helper

- Add a helper that reads `kv_page_microbenchmark_report.json` and prints suggested scheduler constants for manual review.
- Candidate suggestions may cover `page_read_ms`, `non_contiguous_segment_penalty_ms`, `kv_update_ms_per_block`, and prefetch hit/miss terms.
- The helper must not edit source files, write runtime config, or make `RuntimeScheduler` consume benchmark JSON directly.
- Validate any accepted constants by regenerating LLM runtime artifacts and comparing TPOT, throughput, OOM/reject, and page-lifecycle gates.

## Split Large LLM Runtime Files

Potential module boundaries:

- `requests.py`: request and state dataclasses.
- `memory.py`: `MemoryPlanner`, `KVPage`, and paged lifecycle.
- `prefetch.py`: prefetch planner and summaries.
- `scheduler.py`: runtime scheduler policies.
- `cost_model.py`: prefill/decode/paged-attention cost models.
- `distributed.py`: worker routing and failover simulation.
- `reports.py`: summarization and artifact formatting.

This should be done carefully with tests in place because the current large files encode many implicit policy contracts.

## Productize The Video Pipeline

- Add typed result structures for CV inference output.
- Add batch inference support if the target use case needs throughput over per-frame freshness.
- Add graceful shutdown behavior for API server and worker threads.
- Add optional frame sampling or latest-frame replacement for overloaded pipelines.
- Add structured logging instead of periodic `print` calls.
- Add confidence labels or class mapping for MobileNetV2 outputs.

## Native And CUDA Hardening

- Add CUDA error checking macros to CUDA backend and RMSNorm kernel launch paths.
- Add automated CMake build documentation and CI matrix entries for native paths where feasible.
- Avoid repeated CSV headers in native benchmark output.
- Add a small correctness test for CPU/CUDA vector add.
- Record CUDA driver, runtime, GPU, and compiler metadata in native benchmark artifacts.

## TVM Path Cleanup

- Validate TVM APIs against the supported TVM version.
- Pin or document the TVM version used for the TensorIR experiment.
- Add a minimal smoke script that reports dependency/API availability.
- Keep TVM optional, but make failures actionable when a developer intentionally runs that path.

## Documentation Improvements

- Add a quickstart for each major workflow:
  - CV benchmark.
  - Async video pipeline.
  - LLM artifact generation.
  - Agentic eval.
  - CUDA RMSNorm.
  - Native C++ ONNX Runtime.
- Add a source-of-truth artifact map for `results/`.
- Add a "do not compare directly" note for measured versus simulated metrics.

## Real Serving Integrations

If the project moves beyond simulation:

- Add a real vLLM or SGLang adapter behind a small interface.
- Feed real serving traces into the existing summarizers.
- Keep synthetic simulator tests for policy logic, but label real serving measurements separately.
- Add workload capture/replay support so simulated and real runs can use the same request sequence.

## Assumptions For Next Maintainer

- Python 3.11 should be the default development target.
- Favor simple typed functions and dataclasses.
- Do not add large framework abstractions until multiple real callers need them.
- Preserve the measured/artifact/simulated distinction in code and docs.
- Run tests after non-trivial changes and document skipped optional-runtime coverage.
