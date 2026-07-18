# Single-node CPU sharding vertical slice

Date: 2026-07-17. This is an Architecture C prototype: an ExecutionPlan-driven
shared-memory linear-operator harness beside vLLM. It is not native vLLM tensor,
pipeline, or data parallelism and is not multi-device or multi-node inference.

## Pipeline and architecture

Actual repository pipeline:

```text
simplified LLM/HIR or narrow StableHLO textual subset
-> serving phase/KV/provider/representation/layout analyses
-> kernel/lowering candidate generation
-> ServingStaticCostModel -> PlanSelection
-> boundary materialization -> ExecutionPlan v2
-> runtime path builder
-> vLLM configuration adapter OR native operator adapter
```

This slice adds:

```text
linear subgraph
-> hir.sharding.* generic MLIR attributes
-> local propagation (linear, bias, activation, unambiguous reshape)
-> ExecutionPlan.global_decisions.cpu_sharding
-> persistent pinned shared-memory CPU workers
-> complete output tensor
```

The vLLM boundary consumes request metadata and plan selection metadata only.
Generated tokens do not depend on the sharded result. Architecture A was
rejected because the installed `vllm==0.24.0` is a CUDA wheel whose CPU platform
detector does not activate on this host. Architecture B was rejected because no
safe installed model-operator replacement interface was demonstrated.

## Environment and terminology

- Intel i5-10210U: 4 physical cores, SMT2, 8 logical CPUs, one NUMA node,
  AVX2/FMA. Rank `i` is pinned to logical CPU `i`; this is not eight physical
  cores and not eight accelerators.
- LLVM/Clang 21.1.8. The project MLIR package contains the old `mesh` dialect,
  `MLIRMeshDialect`, `MLIRMeshToMPI`, `MLIRMPIDialect`, `MLIRMPIToLLVM`, and
  passes `--sharding-propagation`, `--mesh-spmdization`,
  `--convert-mesh-to-mpi`.
- The package has no current upstream `shard` dialect, StableHLO libraries, or
  Shardy/SDY libraries/tools. The repository has only a narrow StableHLO textual
  subset importer; StableHLO is not its production backbone.
- Python 3.12.13, PyTorch 2.11.0+cu130; Gloo available, MPI unavailable;
  vLLM 0.24.0 CUDA wheel; Ray absent.

The new representation uses generic project-owned `hir.sharding.*` attributes,
not a fake dialect. The before/after files are
`mlir/cpu_sharding_linear_before.mlir` and
`mlir/cpu_sharding_linear_after.mlir`.

Example after planning:

```mlir
module attributes {
  hir.sharding.mesh = {name = "cpu_mesh", axis = "cpu", size = 8 : i64},
  hir.sharding.rank_mapping = "rank i -> pinned logical CPU i"
} {
  func.func @linear_bias_relu(...) attributes {
    hir.sharding.strategy = "split_m",
    hir.sharding.tensor_dimension = 0 : i64,
    hir.sharding.uneven_policy = "balanced_remainder",
    hir.sharding.provenance = "compiler_inferred",
    hir.sharding.collective = "none_direct_disjoint_row_assembly"
  }
}
```

Row-parallel execution materializes a shared-memory sum of partial tensors.
Column parallel uses direct disjoint-slice assembly. Neither is network
communication. Split-M uses disjoint output-row assembly and needs no
collective.

## vLLM probe

The local Qwen2.5-0.5B checkpoint was probed with:

```text
VLLM_LOGGING_LEVEL=DEBUG timeout 45 .venv/bin/python -m \
  vllm.entrypoints.openai.api_server \
  --model <local-Qwen2.5-0.5B-snapshot> --tokenizer <same> \
  --device cpu --dtype float32 --max-model-len 32 \
  --tensor-parallel-size 2 --distributed-executor-backend mp
```

The probe exited 1 before engine or worker initialization. Logs say no platform
plugin was found, CUDA had no driver, CPU did not activate, and vLLM selected
`UnspecifiedPlatform`; configuration then raised `Failed to infer device type`.
There were zero ranks, no process group, no weight partitioning/replication
observation, and no collective. CLI argument presence therefore does not prove
CPU TP support. The same installed build provides no demonstrated CPU PP, DP,
Ray worker, or external-launcher execution.

## Correctness

- `33 passed` focused sharding, ExecutionPlan compatibility, and existing vLLM
  adapter tests.
- All four strategies (replicated, split-M, row parallel, column parallel)
  match complete NumPy FP32 reference tensors for divisible and non-divisible
  shapes, with `rtol=1e-5`, `atol=2e-5` in focused tests.
- A separate 1,000-call uneven `17x33 @ 33x29` split-M run passed with persistent
  workers and affinity `{0:[0], ..., 7:[7]}`.
- Both MLIR fixtures parse and verify with LLVM 21 `mlir-opt
  --allow-unregistered-dialect`.
- All 27 compiler CTests pass.
- Full runtime suite: 796 passed, 16 skipped, 39 failed. Failures group into
  pre-existing missing cross-repository capability/native artifacts, one
  import-order assertion, and sandbox-denied localhost socket tests; the new
  focused suite passes.

## Scaling evidence

Command:

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_cpu_sharding.py --calls 50 --warmup 10 \
  --output results/runtime_paths/cpu_sharding_scaling.json
```

Median milliseconds (1 / 2 / 4 / 8 workers):

| MxKxN | 1 | 2 | 4 | 8 | winner |
|---|---:|---:|---:|---:|---:|
| 1x256x256 | 0.031 | 0.050 | 0.086 | 0.116 | 1 |
| 32x512x512 | 0.386 | 0.317 | 0.322 | 0.545 | 2 |
| 128x768x768 | 2.113 | 1.392 | 0.999 | 1.601 | 4 |
| 512x768x768 | 6.903 | 3.969 | 2.644 | 3.176 | 4 |
| 11x257x263 | 0.108 | 0.112 | 0.157 | 0.226 | 1 |

The artifact contains median, p95, variance, dispatch, compute, assembly, and
affinity for every row. Four workers provide up to 2.61x median speedup on the
largest prefill-like case. Eight logical workers never win, consistent with
only four physical cores. Split-M is unsuitable for single-token decode here,
sometimes useful for medium/batched work, and useful for large prefill-like
work.

No calibrated selector was added: the small five-shape matrix is evidence, not
enough independent training and held-out data for an honest regret report.

## Limitations

Not demonstrated: native vLLM CPU TP/PP/DP, generated tokens or logits depending
on this operator, full transformer tensor parallelism, production vLLM serving,
multi-GPU, multi-device, multi-node, NCCL, GPU P2P, real network collectives,
MPI execution, native current-upstream Shard integration, or production Shardy.
TTFT, TPOT, request throughput, model-weight sharding overhead, CPU utilization,
context switches, and end-to-end memory were not reported because the installed
vLLM engine could not initialize on CPU; inventing them from the standalone
operator would be misleading.

## Evidence-backed achievement summary

1. Added a validated eight-logical-rank CPU mesh and explicit uneven partition
   semantics without claiming eight physical cores or devices.
2. Propagated split metadata through a linear/bias/activation/reshape subgraph
   with provenance and explicit unsupported-op replication fallback.
3. Implemented persistent affinity-pinned shared-memory linear execution with
   split-M, row-parallel reduction, and column-parallel direct assembly.
4. Preserved legacy ExecutionPlan v2 compatibility while adding validated
   optional sharding intent and passing 1,000 repeated full-tensor comparisons.
5. Measured 1/2/4/8 logical-worker scaling, including a negative decode result
   and a 2.61x large-prefill operator speedup at four workers.
