# Distributed D1: Compiler-Planned TP=2 Multi-Process Simulation

## 1. Executive result

The compiler (`ml-graph-compiler-runtime`) generates explicit TP1/TP2
distributed candidates, applies fail-closed legality filtering, and exports
a TP=2 `DistributedPlan` inside the real MLIR `ExecutionPlan` schema
(`schema_version 2.0.0`). The runtime (`heterogeneous-inference-runtime`)
loads that exact artifact, launches two real local OS processes (distinct
PIDs, `multiprocessing` "spawn" context), gives each process only its own
K-slice of a sharded matmul, moves each rank's real partial-result bytes
through `multiprocessing.Queue` IPC during a coordinator-run
`all_reduce(sum)`, reconstructs the distributed output explicitly from the
IPC-transferred bytes, and verifies it against an independent single-rank
(TP1) serial reference to a max absolute error of `2.22e-15`. A deliberately
missing-participant scenario produces a real ~2-second timeout, correct
missing-rank identification, and zero process orphans. All ten required
provenance counters are computed from the event log and are all zero on the
successful run.

**D1 acceptance criteria: all 18 satisfied** (see §17).

## 2. Repository state

Both repositories were found fully clean and committed at the start of this
work (contrary to the task's assumption of "substantial modified and
untracked S1-S2.12 work" -- recorded honestly rather than assumed):

| Repo | Branch | HEAD (unchanged before/after) |
|---|---|---|
| `ml-graph-compiler-runtime` | `master` | `dbf7329392bd2c70fa6ef25e359b277d171b3082` |
| `heterogeneous-inference-runtime` | `main` | `34aee51fef08dc447a6a52d938b4867d60eeef70` |

No commits were made. Full detail: `repository_state_before.json`,
`repository_state_after.json` in this result directory.

## 3. Previous baseline and repaired contract tests (Part A)

No stale `ExecutionPlan` contract fixtures were found. All schema/loader/
path-builder/vLLM-adapter/cpu-sharding/attention-adapter Python tests pass
cleanly both before and after this change (846 -> 862 passed, the +16 being
the new D1 suite). The cross-repo subprocess contract tests
(`test_p1b/p1c/p1d_*_contract.py`, 31 cases) skip by design -- they require a
locally built `native/cpu_kernels/portable_fused_matmul_bias_relu` binary
(a Raspberry Pi `kernel_selection_contract_v1` dispatch contract, unrelated
to distributed TP) that this checkout has never built; out of scope for D1,
left unbuilt.

Genuine pre-existing failures found (confirmed present with or without the
new D1 test file, i.e. not caused by this work) and left unmodified:

- Compiler repo: `ImplementationCandidateTest`, `LLMFrontendNormalizationTest`,
  `QwenOnnxServingPlanExportTest` -- unrelated to `ExecutionPlan`/distributed.
- Runtime repo: `test_attention_runtime.py`'s selector-v4 profitability
  regression (`legal_candidate_count` 7 vs 13) -- explicitly out of scope
  ("do not continue selector-v4"); `test_deployment_planner.py` (missing
  local data file `capabilities/profiles/backend/coreml.json`);
  `test_model_adapter_registry.py` (test-ordering `sys.modules` assertion);
  9 `test_native_fused_attention.py` subprocess errors.

Full detail: `test_summary.json`.

## 4. Compiler distributed candidate design

New file `mlir_passes/include/serving/DistributedPlanning.h` +
`lib/serving/DistributedPlanning.cpp` (pure C++, no MLIR IR needed, matching
the existing `ImplementationCandidateTest`/`TargetConstraintsTest` style):

```cpp
struct DistributedCandidate {
  std::string candidate_id;              // "tp1" | "tp2"
  int64_t world_size, tensor_parallel_size, pipeline_parallel_size;
};
std::vector<DistributedCandidate> generateDistributedCandidates();
// -> always exactly {tp1: ws=1,tp=1,pp=1}, {tp2: ws=2,tp=2,pp=1}
```

`checkCandidateLegality(candidate, tensor_dim_k)` and
`validateDistributedPlan(plan)` implement the D1 legality rules explicitly
(§5). `buildDistributedPlan(candidate, tensor_dim_k, tensor_id)` builds a
`DistributedPlan` only for an already-legal candidate (`nullopt` otherwise
-- fail closed, no plan is ever emitted for an illegal candidate).

This is a standalone D1 module, not (yet) wired into the 16-pass Qwen
serving pipeline -- see §16 Known limitations.

## 5. Distributed legality rules

Implemented exactly as specified, split between candidate-level checks
(`checkCandidateLegality`, needs a concrete problem size) and plan-structure
checks (`validateDistributedPlan`, needs a built plan):

- `world_size >= 1`
- `tensor_parallel_size >= 1`
- `pipeline_parallel_size == 1` (D1 scope)
- `world_size == tensor_parallel_size` (D1 scope)
- tensor dimension divisible by `tensor_parallel_size`
- rank IDs unique and contiguous `0..world_size-1`
- collective participants ⊆ declared ranks, no duplicates, non-empty
- collective sequence IDs unique and strictly ordered `0..N-1`
- tensor shard coverage complete and non-overlapping per `(tensor_id, partition_axis)`

Every rule has a dedicated negative test in `DistributedPlanningTest.cpp`
(8/8 pass) and, at the runtime layer, in `loader.py`'s
`_validate_distributed_plan` (exercised directly with real mutated JSON
payloads in a smoke check, plus `test_distributed_tp_process_runtime.py`).

## 6. Exported TP=1 and TP=2 plan examples

Generated by the new `emit-distributed-execution-plan` CLI tool
(`mlir_passes/tools/emit-distributed-execution-plan/main.cpp`) against a
synthetic `M=4,K=16,N=4` sharded-matmul problem:

```
mlir_passes/build/emit-distributed-execution-plan \
  --candidate tp2 --tensor-dim-k 16 --output compiler_tp2_plan.json
```

TP1 (`compiler_tp1_plan.json`) has **no** `distributed` key at all --
verified byte-identical-shape to a legacy pre-D1 plan. TP2
(`compiler_tp2_plan.json`) carries:

```json
"distributed": {
  "strategy": "tensor_parallel", "world_size": 2,
  "tensor_parallel_size": 2, "pipeline_parallel_size": 1,
  "ranks": [
    {"rank_id": 0, "logical_device": "simulated_cpu_process_0"},
    {"rank_id": 1, "logical_device": "simulated_cpu_process_1"}
  ],
  "tensor_shards": [
    {"tensor_id": "partial_output", "partition_axis": 0, "partition_count": 2,
     "shard_index": 0, "range_start": 0, "range_end": 8},
    {"tensor_id": "partial_output", "partition_axis": 0, "partition_count": 2,
     "shard_index": 1, "range_start": 8, "range_end": 16}
  ],
  "collectives": [
    {"collective_id": "all_reduce_0", "sequence_id": 0, "kind": "all_reduce",
     "participants": [0, 1], "tensor_id": "partial_output", "reduction": "sum"}
  ],
  "truth_boundary": "d1_simulated_localhost_multiprocess_ipc_not_real_gpu_not_nccl_not_measured_gpu_performance"
}
```

Full files: `compiler_tp1_plan.json`, `compiler_tp2_plan.json`.

## 7. Runtime process architecture

New package `deployment/tp_process_runtime/`:

- `RankProcessSpec` -- per-rank planned metadata (no tensor bytes).
- `RankMailbox` -- one private `to_rank` queue (input IPC endpoint) per
  rank + one shared `from_rank` queue (output IPC endpoint).
- `rank_worker_main` -- the child-process entry point (module-level,
  `spawn`-picklable). Runs in a **real OS process**, not a thread: launched
  via `multiprocessing.get_context("spawn").Process(...)`, so the child
  starts a fresh interpreter and only ever receives data explicitly `put()`
  onto its mailbox -- never through `fork()`-inherited memory.
- `CollectiveCoordinator` -- runs in the parent; owns `all_reduce(sum)`.
- `DistributedProcessRuntime` -- orchestrator: spawn N processes, confirm
  unique PIDs, dispatch shards, run the collective, broadcast the reduced
  result, collect acks, shut down, join, verify no orphans, compute
  provenance.
- `DistributedExecutionResult` / `DistributedExecutionTrace` -- the result
  and the ordered event log (provenance source of truth).

## 8. Rank-local shard semantics

`A[M,K] @ B[K,N]`, `K` partitioned across ranks per the compiler's
`tensor_shards`. Each rank receives only `A[:, k_start:k_end]` and
`B[k_start:k_end, :]` (never the full `A`/`B`) and computes
`C_partial_rank = A_rank @ B_rank`. `all_reduce(sum)` over all ranks'
`C_partial_rank` reconstructs `C = A @ B` exactly. This mirrors the existing
CPU thread-based `row_parallel` strategy in `deployment/cpu_sharding.py`
(same math, same collective semantics), moved to real OS processes with
real IPC in place of in-process `np.sum` over thread-pool futures.

`test_rank_local_shard_isolation` asserts each rank's reported shard shape
is exactly the half-K slice, never the full `K=16`.

## 9. IPC collective implementation

`CollectiveCoordinator.run_all_reduce_sum` is a **central-coordinator**
all_reduce (documented explicitly as not an efficient ring/tree
implementation): every rank sends its full serialized partial-result bytes
(`ndarray.tobytes()` + shape/dtype) to the parent via the shared queue; the
parent deserializes, validates (participant membership, no duplicates,
correct `collective_id`/`sequence_id`, matching shapes), reduces
(`np.sum`), and broadcasts the reduced bytes back to every rank. Real bytes
move both directions through an OS pipe (`multiprocessing.Queue`), never a
precomputed/serial-in-parent shortcut. Duplicate, out-of-order, or
unexpected-rank contributions are recorded and rejected rather than
silently accepted (`test_duplicate_participant_rejection`,
`test_wrong_sequence_rejection`, `test_wrong_tensor_shape_rejection`).

## 10. End-to-end TP=2 execution trace

`compiler_tp2_plan.json` -> `load_execution_plan()` -> `.distributed` ->
`DistributedProcessRuntime.run()`. Observed on one real run
(`tp2_distributed_result.json`, `rank_process_events.jsonl`,
`collective_events.jsonl`):

- 2 distinct PIDs spawned.
- Each rank: `process_started` -> `shard_received` (8-wide K slice) ->
  `local_compute_done` -> `entered_collective` -> contribution sent ->
  `collective_result_received` -> `rank_done` -> `shutting_down` ->
  clean exit (`exitcode 0`).
- `all_reduce_0` (`sequence_id 0`) completed with 2/2 participants, 128
  bytes contributed per rank (4x4 float64 partial matmul result).
- 16 total trace events, all consistent with the planned provenance chain.

## 11. Correctness results

`correctness_summary.json`:

```json
{
  "distributed_result_matches_serial_reference": true,
  "max_abs_error": 2.220446049250313e-15,
  "max_rel_error": 6.595358580777043e-16,
  "shape_match": true, "dtype_match": true,
  "all_ranks_completed": true, "all_collectives_completed": true,
  "tolerance": {"rtol": 1e-9, "atol": 1e-9}
}
```

`test_all_reduce_correctness_nontrivial_seeded_case` additionally exercises
a larger seeded `6x32x5` problem with the same result. An illegal-dimension
case (`K=10` against a plan declaring `K=16` shards) is rejected in the
parent before any process is asked to compute
(`test_child_exception_propagates_to_parent`, raises
`DistributedRuntimeError: shard coverage ...`).

## 12. Deadlock negative test

Scenario: rank 1 intentionally skips collective `sequence_id 0`
(`force_skip_collective_rank=1`), `collective_timeout_s=2.0`.
`deadlock_negative_test.json`:

```json
{
  "wall_clock_elapsed_s": 2.217215061187744,
  "status": "timeout",
  "deadlock_record": {
    "missing_ranks": [1], "elapsed_s": 2.001934051513672,
    "status": "timeout_detected_deadlock"
  },
  "no_orphans_confirmed": true,
  "assertion_timeout_was_real": true
}
```

The wall-clock elapsed time (2.22s) is close to but slightly above the
configured 2.0s timeout -- a real wait, not an instantaneous/hardcoded
result. Both child processes were terminated
(`terminate()` -> `join()` -> `kill()` fallback) and confirmed dead via
`os.kill(pid, 0)` raising `ProcessLookupError`. `deadlock_count` is never a
hardcoded `0`; it is derived from `outcome.missing_ranks`, itself derived
from which ranks' contribution messages never arrived before the deadline.

## 13. Provenance audit

`provenance_summary.json` (successful TP2 run) -- all ten required counters,
computed from `rank_process_events.jsonl` + `collective_events.jsonl`, are
zero:

```json
{
  "rank_mismatch_count": 0, "missing_rank_count": 0, "unexpected_rank_count": 0,
  "shard_mismatch_count": 0, "collective_sequence_mismatch_count": 0,
  "missing_collective_participant_count": 0, "unexpected_collective_participant_count": 0,
  "fallback_count": 0, "silent_downgrade_count": 0, "orphan_process_count": 0
}
```

The deadlock negative test (§12) is a *separate* artifact and intentionally
shows `missing_collective_participant_count: 1` -- it is not mixed into the
all-zero success claim.

## 14. Process/IPC measurements

`ipc_benchmark.json`, 7 repetitions each, `M=4,K=16,N=4`:

| Metric (median / p95, seconds) | world_size=1 | world_size=2 |
|---|---|---|
| process startup | 0.00307 / 0.00385 | 0.00624 / 0.00710 |
| shard dispatch | 0.00102 / 0.00141 | 0.00119 / 0.00126 |
| collective latency | 0.09534 / 0.09822 | 0.09234 / 0.09370 |
| broadcast + ack | 0.00010 / 0.00012 | 0.00014 / 0.00016 |
| process shutdown | 0.00632 / 0.00679 | 0.00754 / 0.00780 |
| end-to-end | 0.30682 / 0.31133 | 0.30927 / 0.31155 |
| bytes contributed / run | 128 | 256 |

IPC overhead vs. TP1 serial reference (in-process `numpy` matmul, median
`2.83e-6s`): `overhead_ratio ≈ 109,000x` for this tiny problem --
`multiprocessing.Queue`/spawn process overhead completely dominates. This is
reported as an **IPC-cost measurement**, explicitly not a speedup or
scaling claim (there is no speedup claim in D1).

**Truth boundary for all measurements**: single-host CPU multi-process
simulation, localhost IPC only, via `multiprocessing.Queue` (spawn
context); not NCCL; not GPU-to-GPU communication; not real vLLM tensor
parallelism; not representative of multi-GPU scaling.

## 15. Test results

- Compiler: `DistributedPlanningTest` 8/8 pass; full ctest suite 35/38 pass
  (3 pre-existing, unrelated failures, confirmed identical before/after).
- Runtime: `test_distributed_tp_process_runtime.py` 16/16 pass; full suite
  862 passed / 52 skipped / 3 failed / 9 errors (identical pre-existing
  failure/error set with or without the new file).
- Full detail: `test_summary.json`.

## 16. Known limitations

- The distributed candidate/legality/export module is standalone C++, not
  yet wired into the 16-pass Qwen serving pipeline (`ServingPipeline.cpp`,
  `ExecutionPlanBuilder`) -- there is no compiler flag today that produces a
  `distributed` block for a real Qwen `ExecutionPlan`. D2 would integrate it.
- `all_reduce` is the only implemented collective; `kind` values other than
  `"all_reduce"` are explicitly rejected, not implemented.
- The collective is a central-coordinator (star) topology, not a real
  ring/tree collective -- documented, not hidden.
- The runtime executes a synthetic sharded matmul, not real Qwen
  attention/MLP tensors -- no vLLM model weights are involved.
- `p1b/p1c/p1d` cross-repo contract tests remain skipped in this checkout
  (missing unrelated native-kernel build); not evidence of D1 breakage.
- Measurements are on one development machine, 7 repetitions -- indicative,
  not a rigorous statistical benchmark.

## 17. Truth boundary

Allowed D1 claim (verbatim, and true): *"The compiler generated a TP=2
distributed execution plan, and the runtime consumed that plan to launch two
real local OS processes, execute rank-local tensor shards, move real data
through local IPC during a simulated collective, reconstruct the distributed
output, and verify equivalence against a single-rank reference."*

Not claimed, and not implemented: real tensor parallel vLLM, real GPU
tensor parallelism, NCCL, multi-GPU execution, distributed KV cache, real
GPU communication, performance scaling representative of GPUs. See
`truth_boundary.json`.

## 18. Exact next dependency (D2)

Wire `DistributedPlanning` into the real 16-pass Qwen serving pipeline (a
new `DistributedStrategyPlanningPass` emitting the same `distributed` block
from real Qwen tensor shapes, gated behind an explicit compiler flag) so a
*real* Qwen `ExecutionPlan` — not only the D1 synthetic-matmul problem — can
carry a TP=2 candidate. That is the dependency standing between D1 (proven
here) and eventually materializing a selected distributed strategy into
real vLLM `--tensor-parallel-size` configuration for D-stage work beyond
D1, which still requires multiple physical GPUs and remains out of scope
until real hardware rental is explicitly authorized.
