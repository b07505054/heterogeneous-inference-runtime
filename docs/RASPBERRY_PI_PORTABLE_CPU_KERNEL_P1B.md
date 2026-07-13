# Raspberry Pi Portable CPU Kernel (Phase P1B) — Runtime-Side Summary

Full cross-repo report (compiler + runtime, kernel contract, ExecutionPlan
excerpt, Raspberry Pi execution evidence, tests, known limitations):
`../ml-graph-compiler-runtime/DOC/result/P1B_RASPBERRY_PI_ARM_CPU_KERNEL_REPORT.md`
(sibling repo).

## What this repo added

One real, `ExecutionPlan`-driven CPU kernel dispatch path, scoped to exactly
one op/kernel:

- `native/cpu_kernels/portable_fused_matmul_bias_relu.cpp` — a portable,
  scalar (no NEON, no AVX, no threading) fused MatMul+Bias+ReLU kernel,
  reusing the `bm32_bn32_bk32` tile identity validated by
  `ml-graph-compiler-runtime`'s Phase 1 / R1 CPU schedule-discovery work.
  Builds with a single `g++ -O2 -std=c++17` command — no CMake required,
  which is why it was chosen for a Pi that lacks CMake/Clang but has GCC 14.2.
- `deployment/execution_plan/portable_cpu_kernel_adapter.py` — the adapter:
  validates backend/kernel-id/dtype/rank/shape against the compiler's
  `kernel_selection_contract_v1` decision, then dispatches the real compiled
  kernel via `subprocess`. Never falls back to PyTorch/ONNX Runtime/NumPy/mock.
- `deployment/execution_plan/schema.py` / `stage_builder.py` / `path_builder.py`
  — minimally extended (additive only) to parse `kernel_selection.*` and
  route `hir.fused_matmul_bias_relu` ops to this new adapter, mirroring the
  existing RMSNorm per-op dispatch precedent. All existing vLLM/CUDA routing
  is untouched.
- `tests/test_portable_cpu_kernel_adapter.py`,
  `tests/test_p1b_cross_repo_contract.py` — 16 new tests (accept/reject
  paths, correctness vs. a pure-Python reference, cross-repo kernel-ID and
  target-profile-ID agreement against a freshly-generated compiler plan).
- `results/runtime_paths/portable_cpu_fused_matmul_bias_relu_raspberry_pi_evidence.json`
  — real execution evidence from Raspberry Pi 5 hardware (`100.110.37.6`):
  correctness passed against an independent reference, 20 raw latency
  samples, thermal/governor/affinity provenance. **Functional bring-up and
  correctness evidence only — not a performance claim** (single process, no
  affinity control, `ondemand` governor, no isolation).

## Deployment note

The Raspberry Pi target has no CMake or Clang (GCC 14.2.0 only). The kernel
is deployed as its own single `.cpp` file and compiled directly on the Pi
with plain `g++` — no cross-compilation toolchain and no LLVM/MLIR stack
were installed there. The compiler (`ml-graph-compiler-runtime`) runs only on
the Linux GPU host; its generated `ExecutionPlan` JSON is copied to the Pi
alongside three pure-stdlib Python files (`schema.py`, `loader.py`,
`portable_cpu_kernel_adapter.py`, with two import/path lines adjusted for a
flat, dependency-free layout — commented in-file). See the full report for
why this differs from a literal repo checkout.
