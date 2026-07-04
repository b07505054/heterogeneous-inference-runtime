# Future Work

## Roadmap

Phase 1 is complete:

- Linux measured baseline for an external OpenAI-compatible vLLM server.
- Apple CoreML measured baseline for native MobileNetV2 `.mlpackage` variants.
- Simulator and policy-evaluation components for prefix cache, speculative
  decoding, prefill/decode planning, and paged-KV behavior.

Phase 2:

- CoreML edge policy that selects among measured compute-unit, input-size, and
  compression variants.
- Server runtime policy that uses measured vLLM/OpenAI-compatible artifacts to
  guide concurrency, admission, and routing decisions.

Phase 3:

- Capability-driven quantization policy.
- Capability-driven deployment policy that joins hardware capability, backend
  capability, kernel-library capability, and measured support.
- Model Adapter and Neutral Runtime Graph architecture, so runtime orchestration
  consumes neutral model-family/stage/tensor/memory/KV/backend-target concepts
  instead of ONNX, CoreML, TensorRT, PyTorch, vLLM, or model-name internals.

Phase 4:

- Future measured optimizations, added only when the optimization can be
  benchmarked and exported through the measured baseline schema.
- Richer compiler-runtime contracts such as ExecutionPlan-driven execution,
  after the first CoreML package-centered path is measured and stable.

## CoreML Compiler Runtime Integration

The first Apple/CoreML compiler-runtime milestone should use `.mlpackage` as
the executable handoff:

```text
ONNX / model graph
  -> compiler static optimization from shared capability profiles
  -> compiler-produced or compiler-directed .mlpackage
  -> compiler_metadata.json beside the package
  -> runtime CoreMLModelAdapter
  -> NeutralRuntimeGraph
  -> CoreML benchmark and runtime policy
```

ExecutionPlan artifacts remain useful future contracts, but they are not the
first-stage CoreML runtime centerpiece.

Required future work:

- Define the CoreML compiler candidate bundle:
  `model.mlpackage` plus `compiler_metadata.json`.
- Add a runtime `CoreMLModelAdapter` that validates compiler metadata and
  produces a neutral runtime graph without depending on compiler IR.
- Keep direct CoreML baselines, compiler-produced package baselines, and
  runtime-policy-selected package results as separate measured paths.
- Ensure compiler and runtime read the same hardware/backend/kernel capability
  profile source, or update their profile copies together with matching profile
  IDs and digests.

Do not claim compiler CoreML speedup until a compiler-produced `.mlpackage` is
benchmarked against a direct CoreML export. Do not claim runtime policy speedup
until the selected runtime configuration is benchmarked.

## Neutral Runtime Graph

Add a neutral runtime graph layer before implementing more model-specific
runtime paths.

The graph should describe:

- Model family.
- Stages.
- Tensors.
- Memory requirements.
- KV-cache requirements.
- Backend target.
- Execution constraints.

Adapters should translate source artifacts into this graph:

- `CoreMLModelAdapter`: `.mlpackage` plus optional `compiler_metadata.json`.
- `VLLMEndpointAdapter`: OpenAI-compatible endpoint configuration.
- `MockModelAdapter`: deterministic tests.
- `ONNXModelAdapter`: future optional adapter.

Runtime core should not know exact model names such as Qwen, Llama, or
MobileNet; source format internals; compiler IR; or backend-specific package
internals.

## Clarify Runtime Modes

- Add a small matrix in the README that labels each path as measured live, artifact-backed, simulated, or optional/hardware-dependent.
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
