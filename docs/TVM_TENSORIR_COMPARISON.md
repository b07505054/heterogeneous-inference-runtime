# TVM TensorIR Comparison

This repo uses TVM as a focused reference path for AI compiler interviews:

```text
MatMul + Bias + ReLU semantics
  -> TVM TE workload
  -> TensorIR PrimFunc
  -> manual TensorIR schedule
  -> LLVM CPU executable
  -> correctness + latency report
```

The goal is not to replace the MLIR/HIR compiler story. TVM is useful here
because it shows the same optimization problem in another industry-standard
AI compiler stack: define tensor semantics, transform loop schedules, lower to
a backend, then validate with measured runtime evidence.

## What This Adds

- A real TVM TensorIR workload for fused `matmul + bias + relu`.
- An unscheduled baseline and a scheduled variant using `split`, `reorder`,
  `parallel`, `vectorize`, and `reverse_compute_at`.
- LLVM CPU execution through TVM, not a static artifact.
- NumPy correctness validation for both unscheduled and scheduled modules.
- Benchmark JSON, Markdown report, and before/after TensorIR dumps.

## Setup

TVM is optional for normal repo usage. To run this case study, build TVM with
LLVM and expose its Python package and runtime libraries:

```bash
export TVM_HOME=/Users/allen/Documents/Codex/project/deps/tvm-v0.24.0
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$TVM_HOME/build-codex/lib
export DYLD_LIBRARY_PATH=$TVM_HOME/build-codex/lib:${DYLD_LIBRARY_PATH:-}
```

For this local run, TVM v0.24.0 was built from source with LLVM enabled.

## Test

```bash
PYTHONPATH=$TVM_HOME/python:$PYTHONPATH \
TVM_LIBRARY_PATH=$TVM_HOME/build-codex/lib \
DYLD_LIBRARY_PATH=$TVM_HOME/build-codex/lib:${DYLD_LIBRARY_PATH:-} \
.venv/bin/python -m pytest tests/test_tvm_matmul_bias_relu.py
```

The tests skip when TVM is not installed.

## Benchmark

```bash
PYTHONPATH=$TVM_HOME/python:$PYTHONPATH \
TVM_LIBRARY_PATH=$TVM_HOME/build-codex/lib \
DYLD_LIBRARY_PATH=$TVM_HOME/build-codex/lib:${DYLD_LIBRARY_PATH:-} \
.venv/bin/python scripts/benchmark_tvm_matmul_bias_relu.py
```

Outputs:

- `results/tvm_tensorir/matmul_bias_relu_benchmark.json`
- `results/tvm_tensorir/matmul_bias_relu_report.md`
- `results/tvm_tensorir/tir/*_unscheduled.py`
- `results/tvm_tensorir/tir/*_scheduled.py`

## Interview Framing

This case study answers a specific question:

> Can you use a production AI compiler stack to turn tensor semantics into
> scheduled executable code and prove the schedule with correctness and
> benchmark evidence?

The MLIR/HIR path remains the primary compiler project. TVM is the comparison
point that shows fluency with TensorIR scheduling and gives a clean bridge into
TVM, MetaSchedule, TensorIR, and backend-lowering discussions.
