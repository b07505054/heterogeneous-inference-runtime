# Future Work

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
