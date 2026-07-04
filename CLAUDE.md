# CLAUDE.md

## Engineering Constitution

This file is the engineering constitution for this repository. Future work in
CoreML, vLLM, quantization, KV-cache policy, speculative decoding, and runtime
policy should follow these rules before implementation details are chosen.

## Project Identity

This repository is not a GPU simulator, not a CoreML runtime implementation,
and not a vLLM fork.

It is a heterogeneous inference runtime that measures multiple backends and
builds optimization policies on top of measured evidence. CoreML and vLLM are
treated as measured backends. The repository focuses on deployment policy
instead of backend implementation.

The intended architecture is:

```text
Model Artifact
        |
        v
Model Adapter
        |
        v
Neutral Runtime Graph
        |
        v
Scheduler / Memory Manager / KV Cache / Prefix Cache / Policy
        |
        v
Backend Dispatcher
        |
        v
Backend-specific Executor
```

The measured-policy architecture remains:

```text
Measured Baselines
        |
        v
Capability Layer
        |
        v
Optimization Policy Engine
        |
        +-- Edge Deployment (CoreML)
        +-- Server Runtime (vLLM)
        +-- Simulator / Policy Evaluation
```

The project combines real local inference paths, measured backend clients,
native/CUDA experiments, optional compiler experiments, artifact-backed
benchmark adapters, and local simulations for LLM-serving runtime behavior.

For the current Apple/CoreML compiler-runtime path, the executable compiler
output is a CoreML `.mlpackage` plus adjacent `compiler_metadata.json`, not an
ExecutionPlan as the primary runtime artifact. ExecutionPlan artifacts remain
useful for richer future compiler-runtime contracts, but CoreML v1 centers on
package execution and measured runtime policy.

## Repository Philosophy

Optimization should always proceed in this order:

```text
Backend capability
        |
        v
Measured baseline
        |
        v
Capability model
        |
        v
Policy
        |
        v
Deployment decision
```

Do not start from backend reimplementation. The preferred development order is:

```text
Measured Baseline
        |
        v
Capability Layer
        |
        v
Policy Engine
        |
        v
Runtime Integration
        |
        v
(Optional) Backend Extension
```

Backend extension is a last resort, not the default path.

The first-class capability layer lives under `capabilities/` and defines four
separate concepts:

- `HardwareCapability`: physical hardware facts only.
- `BackendCapability`: runtime/backend support only.
- `KernelLibraryCapability`: runtime/kernel availability as `builtin`,
  `opaque`, `custom`, or `unsupported`.
- `MeasuredSupport`: experimentally verified support facts only.

Policies must query this layer instead of inferring support directly from
benchmark scripts. Measured baselines are evidence, capabilities describe what
exists, policies choose among capabilities, and simulators evaluate ideas.
Do not merge those layers.

## Truth Boundary

Always distinguish:

- Implemented live execution: ONNX Runtime/PyTorch benchmark paths, ONNX CV backend, async video pipeline mechanics, metrics/tracing/API, CUDA RMSNorm source/tests when CUDA is available.
- Artifact-backed summaries: several backend adapters read existing CSV/JSON files instead of running benchmarks.
- Simulations: LLM scheduler, KV cache, page prefetch, distributed routing/failover, and serving-framework comparison artifacts.

Every optimization must declare which evidence level it belongs to:

| Level | Evidence type | Definition | Examples |
|---|---|---|---|
| Level 1 | Measured | Real benchmark from an executed command on actual hardware or server runtime. | CoreML measured baseline, vLLM/OpenAI-compatible measured baseline, CUDA RMSNorm benchmark. |
| Level 2 | Artifact | JSON reports, policy reports, capability tables, or exported summaries. They may be derived from measurements, declarations, or simulations and must state their source. | Measured baseline comparison report, capability profile, compiler serving plan, policy report. |
| Level 3 | Simulator | Deterministic local model of runtime behavior, cost, queueing, memory pressure, speculative execution, or routing. | Prefix cache simulator, speculative simulator, PD simulator, KV cost model. |
| Level 4 | Future idea | Planned optimization or design direction that has not been implemented or measured. | Roadmap item, design sketch, future capability-driven optimization. |

Measured evidence must never be mixed with simulator evidence. If a policy uses
both, the output must clearly state which fields came from measured baselines
and which fields came from modeled or simulated inputs.

High-value runtime evidence now includes:

- Native CoreML MobileNetV2 measured baselines, including compute-unit,
  compression, input-size, package-size, latency, RSS, and drift comparisons.
- External OpenAI-compatible/vLLM measured baselines, including throughput,
  TTFT, TPOT, concurrency, success/error counts, and server/model metadata.
- CUDA RMSNorm source and correctness tests when CUDA is available.
- CUDA/Triton/PyTorch RMSNorm benchmark artifacts and GPU PGO-like kernel-selection reports.
- Optional Nsight Compute capture artifacts for representative RMSNorm cases.
- Measured KV page microbenchmark artifacts; keep these distinct from simulated KV-cache scheduler behavior.
- LLM runtime artifact export for prefill/decode, scheduler policy, page prefetch, cold start, serving-framework comparison, and distributed serving/failover.
- vLLM-style and SGLang-style trace adapter reports, Triton Server-style dynamic batching artifacts, and TensorRT-style backend/profile selection artifacts.
- Protobuf control-plane schema and distributed serving reports for routing, retry, quarantine, and failover simulations.
- Agentic benchmark evaluation under `agentic_eval/`.

Do not invent benchmark numbers. If a metric is not freshly measured, say whether it is historical, artifact-backed, modeled, simulated, or estimated.

## Capability-First Rule

Before implementing any optimization, answer these questions in order:

1. Can the hardware support it?
2. Can the backend support it?
3. Does an existing kernel or runtime feature already provide it?
4. Can it be measured with the current benchmark framework?
5. Only then decide whether implementation is necessary.

If capability cannot be established, document the limitation as capability or
future-work evidence instead of implementing around an assumption.

## Backend-First Rule

Before implementing any feature, ask:

1. Does CoreML already support it?
2. Does vLLM already support it?

If the answer is yes, do not reimplement it. Instead:

```text
measure
  -> compare
  -> build policy
  -> document evidence
```

The repository should optimize use of existing backends before considering
custom runtime behavior.

## CoreML Lane

The CoreML lane owns:

- Measured baselines.
- Compute-unit comparison.
- Compression comparison.
- Input-size comparison.
- Deployment policy over measured CoreML variants.

The CoreML lane explicitly does not own:

- CoreML runtime implementation.
- CoreML scheduler implementation.
- Custom CoreML kernels.
- Claims about ANE scheduling or actual ANE use unless separately measured or
  reported by reliable runtime tooling.

Treat CoreML as a measured backend. Do not replace CoreML kernels.

Current CoreML compiler/runtime contract:

```text
ONNX / model graph
        |
        v
Compiler static optimization using shared capability profiles
        |
        v
Compiler materializes or directs materialization of .mlpackage
        |
        v
compiler_metadata.json beside the package
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

The compiler owns theoretical and static optimization: graph analysis, backend
support planning, precision/layout/compression planning, CoreML-compatible
rewrite or export direction, and package materialization. The compiler must not
claim measured performance.

The runtime owns best dynamic execution: consuming `.mlpackage` artifacts or
endpoints through model adapters, converting them into neutral runtime graphs,
validating capabilities, observing memory/current conditions, looking up
measured evidence, selecting policies, and producing deployment decisions. The
core runtime must not depend on compiler IR.

The three CoreML comparison paths are:

- Path A: direct CoreML baseline, `ONNX/PyTorch -> direct coremltools export -> .mlpackage -> measured baseline`.
- Path B: compiler CoreML baseline, `ONNX/model graph -> compiler-produced .mlpackage + compiler_metadata.json -> measured baseline`.
- Path C: runtime optimized CoreML, `compiler .mlpackage candidates -> runtime policy/planner -> selected CoreML config -> measured result`.

Do not claim compiler CoreML speedup until Path B is benchmarked against Path A.
Do not claim runtime policy speedup until Path C is benchmarked against Path B.
Do not treat ExecutionPlan alone as measured evidence.

## Neutral Runtime Graph Boundary

Runtime orchestration should consume a neutral graph abstraction rather than
source-format internals. Model adapters translate artifacts into that graph:

- `CoreMLModelAdapter`: `.mlpackage` plus optional `compiler_metadata.json`.
- `VLLMEndpointAdapter`: OpenAI-compatible endpoint configuration.
- `MockModelAdapter`: deterministic tests.
- `ONNXModelAdapter`: future optional adapter, not a runtime-core dependency.

The runtime may know neutral concepts: model family, stages, tensors, memory
requirements, KV-cache requirements, backend target, and execution constraints.
It must not know exact model names such as Qwen, Llama, or MobileNet; ONNX,
CoreML, TensorRT, PyTorch, or vLLM internals; compiler IR; or compiler pass
names.

## Shared Capability Profile Rule

Compiler and runtime must use the same hardware, backend, and kernel capability
facts. For the first Apple/CoreML stage, profiles are centered on the current
Mac hardware and the available CoreML/MPS/Metal backend/kernel support.

If hardware, backend, or kernel facts change, compiler and runtime profiles
must be updated together or loaded from the same shared profile source. Do not
allow the compiler and runtime to drift into separate capability truths.

## Linux vLLM Lane

The Linux server lane owns:

- Throughput.
- TTFT.
- TPOT.
- Concurrency behavior.
- Batching behavior as observed through measured server runs.
- Routing policy.
- Future runtime policy on top of measured vLLM/OpenAI-compatible baselines.

The Linux server lane explicitly does not own:

- vLLM implementation.
- vLLM kernel development.
- Replacing vLLM internals.

Treat vLLM as an external measured backend. The repo's OpenAI-compatible
benchmark client must not install, start, stop, or manage vLLM unless a
separate operational script is explicitly created and documented as such.

## Simulator Lane

The simulator lane exists for:

- Policy prototyping.
- Future optimization evaluation.
- Cost modeling.
- Ablation studies.
- Invariant validation.

Simulator artifacts must never be presented as production measured evidence.
Simulator results can motivate a measured experiment, but the measured claim
must come from a Level 1 benchmark.

## Future Optimization Workflow

Before implementing any optimization, use this decision tree:

```text
Can the backend already do this?
        |
        +-- YES -> Measure it
                  -> Build capability model
                  -> Build deployment policy
                  -> STOP
        |
        +-- NO  -> Can a thin abstraction solve it?
                  |
                  +-- YES -> Implement abstraction
                  |         -> STOP
                  |
                  +-- NO  -> Implement runtime optimization only when justified
```

Runtime optimization requires a written reason that measured baselines,
capability modeling, and policy selection are insufficient.

## Development Defaults

- Use Python 3.11.
- Prefer dataclasses.
- Avoid unnecessary classes.
- Keep functions under 100 lines when practical.
- Use type hints.
- Prefer simple modular design.
- Avoid over-engineering.
- Use composition over inheritance.
- Avoid giant classes.
- Write tests for non-trivial logic.
- Run tests after changes.
- Explain changes after implementation.

## Environment and Dependencies Policy

- This repo uses a project-local `.venv` only. There is no system/global Python fallback for validation.
- Do not install, upgrade, or uninstall packages (pip, brew, npm, system package managers, etc.) automatically.
- `scripts/check.sh` requires `.venv/bin/python` to exist and fails with a clear message (not a silent fallback) if `.venv` is missing.
- If a required dependency is missing inside `.venv`, stop and report exactly what's missing and which requirements file it belongs in (`requirements.txt` for runtime deps, `requirements-dev.txt` for test-only deps such as `pytest`/`numpy`). Do not work around the gap by installing it yourself.
- `pytest`/`numpy` are hard-required by `scripts/check.sh`; `torch` is soft-checked — its absence is reported as a warning, not a failure, since dependent tests already degrade gracefully (report `status: "unavailable"`) when `torch` is missing.
- Always ask before any pip/brew/npm/system install, even for a single test dependency, even if it seems low-risk.
- `bash scripts/check.sh` is the canonical validation command. It never installs anything; it fails fast with a clear message when `.venv` or a required dependency is missing.
- `.venv-rmsnorm` is a separate, CUDA-specific environment documented in `README.md`'s CUDA RMSNorm section. It is out of scope for `scripts/check.sh` and should not be modified as part of general dependency/test policy changes.

## Coding Guidance

- Preserve the measured/artifact/simulated boundary in names, docs, and result metadata.
- Prefer small pure helpers for parsing, summarizing, and validation.
- Keep benchmark scripts reproducible by recording command, environment, dependency versions, platform, and git commit when practical.
- Avoid broad refactors without first adding characterization tests around existing artifact schemas.
- Keep optional runtime dependencies optional; tests should skip clearly when CUDA, TVM, TensorRT, ExecuTorch, or native ONNX Runtime dependencies are unavailable.

## Common Commands

Run the canonical validation command (single source of truth for local runs,
the Claude Code hook, and CI):

```bash
bash scripts/check.sh
```

This is equivalent to CI and never installs dependencies. It requires a local
`.venv` (`.venv/bin/python`) and runs `pytest -vv agentic_eval/tests tests`
with `PYTHONPATH` set to the repo root.
The `.claude/settings.json` `PostToolUse` hook runs this script automatically
after any Edit/Write/MultiEdit on a `.py` file. CI
(`.github/workflows/agentic-eval-ci.yml`) invokes the same script rather than
embedding its own pytest command. CUDA- and TVM-dependent tests
(`tests/test_rmsnorm_cuda_correctness.py`, `tests/test_tvm_matmul_bias_relu.py`)
skip cleanly on machines without those dependencies; skips there are expected,
not failures.

Run core Python tests directly:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -vv agentic_eval/tests tests
```

Run agentic eval:

```bash
python -m agentic_eval.run_agentic_eval
pytest agentic_eval/tests
```

Run CUDA RMSNorm correctness and benchmark paths on a CUDA-capable machine with the separate `.venv-rmsnorm` environment:

```bash
.venv-rmsnorm/bin/python -m pytest tests/test_rmsnorm_cuda_correctness.py
.venv-rmsnorm/bin/python scripts/test_rmsnorm_cuda_correctness.py
.venv-rmsnorm/bin/python scripts/benchmark_rmsnorm_cuda.py \
  --output results/cuda_transformer/rmsnorm_benchmark.json \
  --report-output results/cuda_transformer/rmsnorm_benchmark_report.md
```

Run optional Triton and Nsight Compute RMSNorm evidence:

```bash
python3 scripts/benchmark_rmsnorm_triton.py \
  --output results/cuda_transformer/rmsnorm_triton_benchmark.json \
  --report-output results/cuda_transformer/rmsnorm_triton_benchmark_report.md
python3 scripts/capture_rmsnorm_nsight_compute.py \
  --output results/cuda_transformer/rmsnorm_nsight_compute_capture.json \
  --report-output results/cuda_transformer/rmsnorm_nsight_compute_capture.md \
  --raw-output results/cuda_transformer/rmsnorm_nsight_compute_raw.csv
```

Run measured KV page microbenchmark evidence:

```bash
python3 scripts/benchmark_kv_page_microbenchmark.py
```

Run the basic benchmark launcher:

```bash
python run_pipeline.py --model mobilenet --threads 4 --batch 1
```

Run async video pipeline with mock backend:

```bash
python -m deployment.async_video_pipeline --source data/synthetic_input.mp4 --backend mock
```

Run async video pipeline with ONNX Runtime backend:

```bash
python -m deployment.async_video_pipeline --source data/synthetic_input.mp4 --backend onnx
```

## High-Risk Areas

- `deployment/llm_runtime_decision.py` and `scripts/generate_llm_runtime_artifacts.py` are large and encode many policy assumptions.
- `backends/executorch_backend.py` needs cleanup before relying on it.
- TVM APIs may need validation against the installed TVM version.
- Historical result artifacts can drift from current code and hardware.
- Agentic eval expected values are coupled to committed artifact contents.
- Do not blur measured KV page microbenchmark results with simulated page-prefetch scheduler reports.

## Documentation Map

- `docs/architecture.md`: purpose, modules, responsibilities, implemented versus simulated behavior.
- `docs/data_flow.md`: inputs, processing flow, outputs, metrics, and important structures.
- `docs/design_decisions.md`: major design choices, tradeoffs, and assumptions.
- `docs/technical_debt.md`: weak spots, missing tests, duplicated logic, unclear naming, and risks.
- `docs/future_work.md`: realistic next steps.

## Portfolio-Level Policy

When this repository is maintained inside the `systems-portfolio` wrapper, follow the root `CLAUDE.md` for shared documentation hierarchy, benchmark honesty, and Git authorship rules. Keep this file focused on repository-specific capabilities, truth boundaries, and validation commands.
