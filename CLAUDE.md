# CLAUDE.md

## Project Handoff Notes

This repository is a heterogeneous inference runtime and benchmarking evidence project. It combines real local inference paths, native/CUDA experiments, optional compiler experiments, artifact-backed benchmark adapters, and local simulations for LLM-serving runtime behavior.

Always distinguish:

- Implemented live execution: ONNX Runtime/PyTorch benchmark paths, ONNX CV backend, async video pipeline mechanics, metrics/tracing/API, CUDA RMSNorm source/tests when CUDA is available.
- Artifact-backed summaries: several backend adapters read existing CSV/JSON files instead of running benchmarks.
- Simulations: LLM scheduler, KV cache, page prefetch, distributed routing/failover, and serving-framework comparison artifacts.

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

This requires a local `.venv` (`.venv/bin/python`) and runs
`pytest -vv agentic_eval/tests tests` with `PYTHONPATH` set to the repo root.
The `.claude/settings.json` `PostToolUse` hook runs this script automatically
after any Edit/Write/MultiEdit on a `.py` file. CI
(`.github/workflows/agentic-eval-ci.yml`) invokes the same script rather than
embedding its own pytest command. CUDA- and TVM-dependent tests
(`tests/test_rmsnorm_cuda_correctness.py`, `tests/test_tvm_matmul_bias_relu.py`)
skip cleanly on machines without those dependencies; skips there are expected,
not failures.

Run agentic eval:

```bash
python -m agentic_eval.run_agentic_eval
pytest agentic_eval/tests
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

## Documentation Map

- `docs/architecture.md`: purpose, modules, responsibilities, implemented versus simulated behavior.
- `docs/data_flow.md`: inputs, processing flow, outputs, metrics, and important structures.
- `docs/design_decisions.md`: major design choices, tradeoffs, and assumptions.
- `docs/technical_debt.md`: weak spots, missing tests, duplicated logic, unclear naming, and risks.
- `docs/future_work.md`: realistic next steps.
