# CLAUDE.md

## Project Handoff Notes

This repository is a heterogeneous inference runtime and benchmarking evidence project. It combines real local inference paths, native/CUDA experiments, optional compiler experiments, artifact-backed benchmark adapters, and local simulations for LLM-serving runtime behavior.

Always distinguish:

- Implemented live execution: ONNX Runtime/PyTorch benchmark paths, ONNX CV backend, async video pipeline mechanics, metrics/tracing/API, CUDA RMSNorm source/tests when CUDA is available.
- Artifact-backed summaries: several backend adapters read existing CSV/JSON files instead of running benchmarks.
- Simulations: LLM scheduler, KV cache, page prefetch, distributed routing/failover, and serving-framework comparison artifacts.

High-value runtime evidence now includes:

- CUDA RMSNorm source and correctness tests when CUDA is available.
- CUDA/Triton/PyTorch RMSNorm benchmark artifacts and GPU PGO-like kernel-selection reports.
- Optional Nsight Compute capture artifacts for representative RMSNorm cases.
- Measured KV page microbenchmark artifacts; keep these distinct from simulated KV-cache scheduler behavior.
- LLM runtime artifact export for prefill/decode, scheduler policy, page prefetch, cold start, serving-framework comparison, and distributed serving/failover.
- vLLM-style and SGLang-style trace adapter reports, Triton Server-style dynamic batching artifacts, and TensorRT-style backend/profile selection artifacts.
- Protobuf control-plane schema and distributed serving reports for routing, retry, quarantine, and failover simulations.
- Agentic benchmark evaluation under `agentic_eval/`.

Do not invent benchmark numbers. If a metric is not freshly measured, say whether it is historical, artifact-backed, modeled, simulated, or estimated.

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
