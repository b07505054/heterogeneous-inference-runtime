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

## Documentation Hierarchy

Truth must flow in the following order:

Code
↓
Artifacts
↓
README.md
↓
CLAUDE.md
↓
docs/

Lower levels must never contradict higher levels.

Documentation must describe reality rather than invent behavior.

If uncertainty exists, trust code and generated artifacts.

Never exaggerate capabilities.

Never claim production behavior unless code and artifacts support it.

## README Contract

README.md exists to answer:

1. What is it?
2. Why is it interesting?
3. How do I run it?
4. What results does it produce?

README should emphasize user-facing understanding.

Avoid implementation details unless necessary.

Avoid maintenance instructions.

## CLAUDE.md Contract

CLAUDE.md exists to answer:

1. How do I maintain it?
2. What commands are canonical?
3. Which components are implemented?
4. Which components are simulated?
5. Which validation commands must pass?
6. What files should not be changed casually?

CLAUDE.md is intended for maintainers and future AI agents.

## docs/ Contract

docs/ exists to answer:

1. Why is it designed this way?
2. What tradeoffs were made?
3. What is measured versus modeled?
4. What assumptions exist?
5. What limitations remain?
6. What future work is possible?

docs/ explains architecture and rationale rather than usage.

## Documentation Principles

Code > Artifacts > README > CLAUDE.md > docs/

Never reverse this order.

Never infer unsupported features.

Never create claims unsupported by code or artifacts.

Prefer conservative wording.

Call synthetic benchmarks synthetic.

Call simulated systems simulated.

Distinguish measured behavior from modeled behavior.

## Git Authorship Policy

The user is the sole maintainer and owner of this repository.

AI agents may modify files as requested.

AI agents must not add AI authorship metadata.

Never add:

* Co-Authored-By entries
* Co-authored-by trailers
* Claude authorship metadata
* AI signatures
* Generated-by-AI footers
* any metadata that makes an AI system appear as a repository contributor

Commit policy:

* By default, do not run git commit.
* If the user explicitly asks in the current conversation to commit, an AI agent may run git add and git commit.
* Commits created by an AI agent must use the user's configured git author and committer identity.
* Commit messages must not mention AI authorship unless the user explicitly asks.
* Before committing, show git status and the staged diff summary when practical.

Push policy:

* By default, do not run git push.
* Only run git push if the user explicitly asks in the current conversation.
* Never force-push unless the user explicitly asks for a force push and the reason is explained.

History policy:

* Do not create branches, rewrite history, rebase, reset, or amend commits unless the user explicitly asks in the current conversation.
* Never rewrite public history without explicit user approval.

Ownership rule:

* The user remains the sole author/maintainer for portfolio presentation purposes.
* No AI system should appear as a repository contributor.
