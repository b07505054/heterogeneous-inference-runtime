# CLAUDE.md

## Engineering Constitution

This file is the engineering constitution for this repository. Future work in
vLLM execution planning, quantization, KV-cache policy, speculative decoding,
and runtime policy should follow these rules before implementation details are
chosen.

## Project Identity

This repository is not a GPU simulator and not a vLLM fork.

It is a heterogeneous inference runtime that measures multiple backends and
builds execution plans and optimization policies on top of measured evidence.
The current active direction is vLLM execution planning for NVIDIA/CUDA GPUs.
The repository focuses on runtime configuration, feedback, and policy rather
than backend implementation.

The intended architecture is:

```text
Client Requests
        |
        v
Compiler / Execution Planner
        ^
        |
Hardware Profile + Backend Profile + Workload
        |
        v
Execution Plan
  - Request Grouping
  - Batch Policy
  - Prefix Policy
  - Memory Policy
  - Quantization Policy
  - Speculative Policy
  - Runtime Config
        |
        v
vLLM Backend
        |
        v
NVIDIA GPU
        |
        v
Runtime Metrics
        |
        v
Feedback Database
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
        +-- vLLM Runtime Config
        +-- Simulator / Policy Evaluation
        +-- Archived / Deprioritized Backend Lanes
```

The project combines real local inference paths, measured backend clients,
native/CUDA experiments, execution-planner artifacts, artifact-backed benchmark
adapters, and local simulations for LLM-serving runtime behavior.

ExecutionPlan is the main compiler/runtime contract for current work. In this
repo, "compiler" means execution planner: it turns client requests, workload
metadata, hardware profiles, and backend profiles into runtime configuration
for vLLM. It does not mean a model exporter.

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
| Level 1 | Measured | Real benchmark from an executed command on actual hardware or server runtime. | vLLM/OpenAI-compatible measured baseline, CUDA RMSNorm benchmark. |
| Level 2 | Artifact | JSON reports, policy reports, capability tables, or exported summaries. They may be derived from measurements, declarations, or simulations and must state their source. | Measured baseline comparison report, capability profile, compiler serving plan, policy report. |
| Level 3 | Simulator | Deterministic local model of runtime behavior, cost, queueing, memory pressure, speculative execution, or routing. | Prefix cache simulator, speculative simulator, PD simulator, KV cost model. |
| Level 4 | Future idea | Planned optimization or design direction that has not been implemented or measured. | Roadmap item, design sketch, future capability-driven optimization. |

Measured evidence must never be mixed with simulator evidence. If a policy uses
both, the output must clearly state which fields came from measured baselines
and which fields came from modeled or simulated inputs.

High-value runtime evidence now includes:

- External OpenAI-compatible/vLLM measured baselines, including throughput,
  TTFT, TPOT, concurrency, success/error counts, and server/model metadata.
- No-quant Qwen GTX 1650 Max-Q evidence under `results/qwen_no_quant/`,
  including default vLLM OOM logs, conservative low-memory vLLM comparisons,
  and three-trial repeatability artifacts. Compiler-guided no-quant Qwen uses
  original Qwen weights. Differences come from execution/runtime policy, not
  model weight optimization. Do not claim compiler-optimized weights, AWQ, or
  GPTQ from these artifacts.
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

1. Does vLLM already support it?
2. Can vLLM expose the behavior through runtime configuration or request
   scheduling policy?

If the answer is yes, do not reimplement it. Instead:

```text
measure
  -> compare
  -> build policy
  -> document evidence
```

The repository should optimize use of existing backends before considering
custom runtime behavior.

## vLLM Execution Planning Lane

The active lane owns:

- Request grouping.
- Batch policy.
- Prefix policy.
- Memory policy.
- Quantization policy.
- Speculative policy.
- Runtime config selection for vLLM.
- Feedback from measured runtime metrics.

The active lane explicitly does not own:

- vLLM implementation.
- vLLM kernel development.
- Replacing vLLM internals.
- Claims of speedup without measured vLLM benchmark evidence.

Treat vLLM as the first runtime target. ExecutionPlan is the main
compiler/runtime contract. Hardware profiles are GPU/CUDA-oriented, backend
profiles describe vLLM capabilities, and the feedback database stores measured
runtime metrics for future planning decisions.

## Neutral Runtime Graph Boundary

Runtime orchestration should consume a neutral graph abstraction rather than
source-format internals. Model adapters translate artifacts into that graph:

- `MockModelAdapter`: deterministic tests.
- `VLLMEndpointAdapter`: future OpenAI-compatible endpoint configuration.
- `ONNXModelAdapter`: future optional adapter, not a runtime-core dependency.

The runtime may know neutral concepts: model family, stages, tensors, memory
requirements, KV-cache requirements, backend target, and execution constraints.
It must not know exact model names such as Qwen, Llama, or MobileNet; ONNX,
TensorRT, PyTorch, or vLLM internals; or compiler pass names.

## Shared Capability Profile Rule

Compiler and runtime must use the same hardware, backend, and kernel capability
facts. For the active vLLM stage, profiles are centered on NVIDIA/CUDA GPU
hardware and vLLM backend/runtime capability.

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
