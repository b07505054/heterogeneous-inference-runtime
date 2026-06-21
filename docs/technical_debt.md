# Technical Debt

## Source Issues And Bugs

- `backends/executorch_backend.py` references `self.backend_name`, but the constructor does not define it. The file also appears to contain an indented stray `import csv` after a `return`, suggesting a copy/paste or formatting error.
- `backend_validation_runner.py` imports `TensorRTBackend` but does not currently include a TensorRT adapter in its backend list. ExecuTorch is also commented out.
- `cpp_inference/main.cpp` appends a CSV header every run, which can create repeated headers in `results/cpp_benchmark.csv`.
- `AsyncVideoInferencePipeline.capture_loop` defines `enqueue_start` inside the `try` block immediately before `put_nowait`; the variable is only needed for tracing and currently works, but the timing boundaries are fragile.
- `export_metrics` creates only the direct parent directory with `mkdir(exist_ok=True)`, not nested parents.
- Some TVM APIs in `tvm_experiments/matmul_bias_relu.py` look suspicious for modern TVM versions, including `tvm.tirx.const`, `tvm.s_tir.Schedule`, and `schedule.get_sblock`. These may be API drift or typos and should be validated in an environment with TVM installed.
- The CUDA backend vector-add implementation does not check CUDA API return codes.
- The RMSNorm CUDA kernel launch does not check `cudaGetLastError()` after launch.

## Missing Or Thin Tests

- There are no focused unit tests for `RuntimeMetrics`, `PipelineTracer`, `ModelRegistry`, `ONNXRuntimeCVBackend.preprocess`, or FastAPI monitoring endpoints.
- The async video pipeline lacks deterministic tests for queue-full drops, stop behavior, trace export, and metrics export.
- Backend adapters that parse CSV artifacts lack tests for missing files, empty CSVs, invalid columns, and schema drift.
- Provider fallback behavior in `ONNXRuntimeCVBackend` is not directly tested with mocked provider lists.
- Native C++ and CUDA vector-add paths do not appear to have automated build/test coverage in the Python test suite.
- Agentic evaluation tests exist, but expected answer values are coupled to committed artifact numbers.
- Optional TVM and CUDA tests skip cleanly when unavailable, which is good for portability but leaves those paths unvalidated in non-CUDA/non-TVM environments.

## Duplicated Logic

- Percentile helpers are implemented in multiple scripts.
- JSON writing, Markdown report writing, and environment metadata collection appear in several artifact scripts with similar patterns.
- Benchmark scripts repeat warmup/timing/result-summary logic across ONNX, PyTorch, TensorRT, TVM, ExecuTorch, and RMSNorm paths.
- Several plotting scripts likely duplicate result loading and label conventions.
- LLM artifact generation contains many report builders in one large script, making reuse and targeted tests harder.

## Unclear Naming And Boundaries

- The repository title and README describe a runtime platform, but many components are benchmarks, simulations, or artifact adapters rather than a single integrated runtime.
- Names such as `TensorRTBackend` and `ExecuTorchBackend` may imply live execution, but those classes summarize artifacts.
- LLM serving reports compare against serving-framework concepts, but the implementation is a local simulator. This distinction must remain explicit.
- `MockCVBackend` is correctly named, but defaulting to it can surprise users who expect real inference.
- `run_pipeline.py` is a benchmark launcher, while `deployment/async_video_pipeline.py` is the actual video pipeline CLI. The names can be confused.

## Metrics Risks

- Historical metrics under `results/` may become stale relative to current code, hardware, drivers, and dependency versions.
- Some artifacts include benchmark values without a guarantee that they were generated on the current checkout.
- The agentic judge expects exact metric values; regenerating artifacts can break tests or require updating expectations.
- Throughput and latency fields from simulations should not be mixed with measured runtime benchmarks without labels.
- Estimated or modeled serving metrics must be labeled as estimated/modeled when referenced.

## Dependency And Environment Risks

- `requirements.txt` does not pin Python version or exact dependency versions.
- Optional dependencies are broad: CUDA, TensorRT, TVM, ExecuTorch, ONNX Runtime C++ libraries, Google Benchmark, and OpenCV/FastAPI/uvicorn are not all captured in `requirements.txt`.
- The README includes environment-specific commands for CUDA and RMSNorm benchmarking that may not apply to all machines.
- The repository includes a third-party ONNX Runtime tarball, which may become outdated or platform-specific.

## Maintainability Risks

- `deployment/llm_runtime_decision.py` is very large and mixes data models, scheduling policy, memory planning, lifecycle tracking, cost modeling, and summarization.
- `scripts/generate_llm_runtime_artifacts.py` is also very large and combines workload generation, policy gates, reports, trace writing, cold-start modeling, framework comparison, and CLI behavior.
- Many scripts are intended as standalone research/demo entry points, so shared contracts are informal.
- Some files lack docstrings or comments for non-obvious simulation constants.
- Result artifacts are numerous, which makes it harder to know which files are source of truth.

## Future Risks

- If source code and committed artifacts diverge, handoff readers may trust stale results.
- If the LLM simulator is presented as production serving evidence, expectations will be misaligned.
- If optional runtime paths are not regularly validated on matching hardware, they may silently break due to dependency/API drift.
- If more backend adapters are added without clear measured/artifact/simulated labeling, comparison reports may become misleading.
