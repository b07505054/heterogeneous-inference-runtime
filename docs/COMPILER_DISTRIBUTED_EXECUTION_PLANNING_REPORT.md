# Compiler-Planned Shared-Memory Multi-Worker Attention

## 1. Verdict

**Distributed maturity Level 5: real-model compiler-planned shared-memory
multi-worker execution.**

The compiler/planner now selects the kernel and schedule, emits exact logical
worker assignments, GQA read ranges, output ownership, communication semantics,
and completion synchronization into ExecutionPlan v2. The runtime consumes
those assignments without repartitioning and dispatches exact work items to
persistent logical workers. A profitability-selected four-worker native AVX2
Qwen prefill and a forced four-worker eight-token causal run were verified.

This is single-process shared-memory execution. It is not multi-process,
multi-node, MPI, NCCL, tensor-parallel serving, network communication, or vLLM
distributed execution.

## 2. Repository state

| Repository | Branch | Starting HEAD | Ending HEAD | Start | End |
|---|---|---|---|---|---|
| heterogeneous-inference-runtime | `main` | `34aee51fef08dc447a6a52d938b4867d60eeef70` | unchanged | clean | dirty with this stage |
| ml-graph-compiler-runtime | `master` | `dbf7329392bd2c70fa6ef25e359b277d171b3082` | unchanged | clean | clean |

No reset, clean, stash, rebase, commit, push, deletion, or overwrite of prior
evidence occurred.

## 3. Prior architecture and runtime audit

Before this stage, `attention_planner.py` selected algorithm, implementation,
strategy, and worker count. `CompilerAttentionRuntime.attention` nevertheless
called `uneven_ranges`, sliced tensors, chose assembly, and implicitly treated
`future.result()` as its barrier. The generic `ThreadPoolExecutor` could not
prove a submitted shard ran on the corresponding logical worker.

| Decision | Before | After |
|---|---|---|
| kernel algorithm | compiler | compiler |
| native implementation | compiler | compiler |
| worker count | compiler/runtime mixed | compiler |
| split dimension | compiler/runtime mixed | compiler |
| exact worker ranges | runtime | compiler |
| output ownership | runtime | compiler |
| barrier requirement | implicit runtime | compiler |
| reduction requirement | implicit runtime | compiler |
| dispatch mechanism | runtime | runtime |
| thread wakeup | runtime | runtime |

`PersistentCPUShardRuntime` creates one persistent single-thread executor per
logical worker. Each worker attempts `sched_setaffinity` to its planned logical
CPU, consumes a private queue, propagates exceptions through `Future`, and is
joined during shutdown. `submit_to(worker_id, ...)` verifies the executing
thread's logical rank. Affinity attempts are recorded; these are logical
workers and no claim is made that the OS always preserves physical-core
placement.

## 4. Planning representation

[`distributed_attention_plan.py`](../deployment/distributed_attention_plan.py)
defines `hir.shared_memory_attention_placement.v1`. Each attention decision
contains:

- parallel strategy and split dimension;
- exact worker and work-item counts;
- deterministic logical worker IDs;
- query-head and query-token ranges;
- GQA KV-head read ranges;
- Q/K/V access modes;
- disjoint logical output slices and element extents;
- worker-private scratch ownership;
- shared-memory copy and reduction semantics;
- completion-counter synchronization;
- predicted compute, dispatch, wakeup, barrier, imbalance, cache, bandwidth,
  and underutilization costs;
- fail-closed no-repartition/no-override flags.

Old plans without this contract remain usable only through an explicitly
counted legacy/manual path. Compiler-selected and forced-test ExecutionPlans
without placement are rejected.

## 5. Worker placement and GQA

Balanced remainder partitioning is deterministic. For Qwen Hq=14, Hkv=2,
workers=4:

| Worker | Query heads | KV heads read | Output ownership |
|---:|---:|---:|---|
| 0 | `[0,4)` | `[0,1)` | heads `[0,4)` |
| 1 | `[4,8)` | `[0,2)` | heads `[4,8)` |
| 2 | `[8,11)` | `[1,2)` | heads `[8,11)` |
| 3 | `[11,14)` | `[1,2)` | heads `[11,14)` |

All query heads are covered once, no output regions overlap, no worker is
empty, and imbalance is at most one head. K/V reads may overlap because they
are immutable shared buffers. Output writes are disjoint and native workers
write directly into their assigned output views, so native split-head requires
zero output copies and no numerical reduction.

## 6. Communication and synchronization

The explicit communication plan is:

```text
Q: shard-local shared-buffer read
K/V: overlapping read-only shared-buffer access
Output: disjoint shared-buffer writes
Copies: zero for native direct output
Reduction: none
Synchronization: one completion-counter barrier before o_proj
```

The main thread submits the plan's exact work items, waits for all futures, and
only then returns the completed tensor. `o_proj` therefore cannot observe a
partially completed output. These operations are shared-buffer access,
dispatch, and synchronization—not network communication or collectives.

## 7. ExecutionPlan and plan-only runtime

The compiler-selected placement is serialized inside each phase decision,
loaded through the normal ExecutionPlan loader, and validated again. Runtime
execution iterates `distributed_execution.workers`; it never calls
`uneven_ranges` on the verified path. Structured provenance records plan ID,
invocation, planned/executed worker, planned/executed shard, symbol, timestamps,
status, and timing.

Acceptance counters for all primary runs:

```text
runtime_repartition_count = 0
runtime_worker_count_override = 0
runtime_strategy_override = 0
manual_shard_count = 0
fallback_count = 0
candidate_mismatch_count = 0
```

## 8. Cost model and candidate rejection

The schedule score is:

```text
predicted total =
  critical-path shard compute
  + dispatch
  + worker wakeup
  + completion barrier
  + balanced-shard imbalance
  + shared-cache contention
  + memory-bandwidth contention
  + algorithm-specific memory/tile cost
```

Critical-path compute uses the largest planned shard rather than dividing
serial cost blindly by W. Calibration rejects worker counts whose shards are
too small to amortize dispatch. Other fail-closed checks cover unavailable
workers/ISA/symbols, workers exceeding independent heads, empty/overlapping
shards, unsupported split dimensions/reductions, missing synchronization, and
ABI mismatch.

## 9. Serial versus worker scaling

The benchmark contains 216 rows, with two warmups and seven measured trials
per candidate. PyTorch intra-op threads were fixed to one. Required dense and
native AVX2 serial/split-head 2/4/8 candidates were measured.

Representative native AVX2 medians:

| Workload | W1 ms | W2 ms | W4 ms | W8 ms | Winner |
|---|---:|---:|---:|---:|---:|
| decode K=128 | 0.135 | 0.176 | 0.316 | 0.647 | 1 |
| decode K=1024 | 0.445 | 0.387 | 0.426 | 0.788 | 2 |
| prefill Q=64 | 0.765 | 0.492 | 0.504 | 0.766 | 2 |
| prefill Q=192 | 5.967 | 3.330 | 2.059 | 2.156 | 4 |

At decode K=128, barrier time rose from 0.078 ms at W1 to 0.483 ms at W8,
making eight workers 4.80× slower. At prefill Q=192, four workers achieved
2.90× speedup and 72.4% parallel efficiency; eight workers were slightly
slower because dispatch/barrier and shared-memory contention outweighed the
smaller shards. Scaling is deliberately non-monotonic.

## 10. Calibration, held-out selection, and fixed policies

Calibration uses decode K=16/64/256/1024 and prefill Q=16/64/256. Held-out
lengths are decode K=24/48/96/192/384/768/1536 and prefill
Q=11/24/48/73/96/192.

| Metric | Calibration | Held-out |
|---|---:|---:|
| workloads | 7 | 13 |
| exact complete-candidate agreement | 85.7% | 92.3% |
| mean regret | 0.33% | 0.36% |
| median regret | 0% | 0% |
| p95/max regret | 2.31% | 4.68% |
| fallback | 0% | 0% |

The only held-out miss was Q=192: the selector chose W8 while W4 was 4.68%
faster in that trial.

Held-out mean regret for fixed worker policies:

| Policy | Mean regret |
|---|---:|
| always W1 | 31.17% |
| always W2 | 25.52% |
| always W4 | 79.87% |
| always W8 | 237.30% |
| compiler-selected | 0.36% |

## 11. Timeline evidence

`worker_timeline.json` records operator submission, each worker's start/end,
duration and status, start/finish skew, and barrier release. The critical path
is the latest finishing planned shard. `worker_scaling_analysis.json` records
speedup, efficiency, dispatch fraction, barrier fraction, and imbalance for
every measured worker count.

## 12. Real Qwen model-forward integration

### Normal eight-token prompt

The four-token workload selected dense serial prefill and native AVX2 serial
decode. All 192 invocations remained compiler-placement-driven:

```text
24 prefill + 168 decode
192 worker events
192/192 outputs entered o_proj
zero repartition, override, fallback, mismatch, or manual shards
```

Baseline and compiler tokens were identical:

```text
[576, 5567, 18404, 264, 501, 5486, 311, 279]
```

Maximum logit difference was `2.574920654296875e-5`.

### Profitability-selected parallel real prefill

At prompt length 64 the compiler selected
`cpu_fused_online_native_avx2_fp32_q1_k32_split_head_w4_v1`.
Across 24 layers it produced 96 exact plan-to-worker events, all outputs entered
`o_proj`, the first greedy token matched (`14252`), and maximum logit difference
was `5.125999450683594e-5`. This is model-forward latency, not serving TTFT.

### Forced worker-shard causal proof

A test-only four-worker native AVX2 plan produced 768 exact worker events over
192 attention invocations and preserved all eight baseline tokens. Applying
`+5.0` only to logical worker 2's owned head shard before `o_proj` changed the
sequence to eight copies of token `284`. Attempting to perturb worker 2 in a
two-worker plan is rejected as unowned/out-of-range. The forced run is labeled
`forced_test_override`, not profitability-selected.

## 13. Mixed execution and negative validation

The mixed stress alternated serial dense, parallel dense, serial native fused,
parallel native fused, prefill, decode, and context lengths for 1,000 calls.
Each of four persistent runtimes executed 250 calls. Results:

```text
nonfinite failures = 0
deadlocks/missed completions = 0
maximum absolute error = 4.172325134277344e-7
worker-pool recreation per call = false
all override/repartition/manual counters = 0
```

Focused negative tests reject overlapping/gapped ownership, count mismatch,
invalid worker IDs, unsupported reduction, missing completion barrier, missing
compiler placement, ABI mismatch, and out-of-range perturbation.

## 14. Tests

```text
.venv/bin/python -m pytest -q \
  tests/test_distributed_attention_plan.py tests/test_attention_runtime.py \
  tests/test_cpu_sharding.py tests/test_native_fused_attention.py \
  tests/test_execution_plan_loader.py
# 90 passed

ctest --test-dir build-mlir -R 'AttentionCPU' --output-on-failure
# 2 passed
```

Both `git diff --check` invocations passed.

## 15. Precise maturity and limitations

Achieved:

- Level 1 persistent shared-memory parallel runtime;
- Level 2 compiler serial/worker-count/split-head selection;
- Level 3 exact compiler worker placement with no runtime repartition;
- Level 4 explicit shared-buffer, no-reduction, completion synchronization;
- Level 5 real Qwen compiler-planned multi-worker attention and causal output.

Not achieved or claimed: Level 6, multi-process execution, multi-node
execution, MPI, NCCL, network collectives, cross-process shared memory, native
vLLM CPU backend, vLLM request scheduling, tensor/pipeline parallel serving,
PagedAttention, or production distributed serving. Eight logical CPU workers
are not eight accelerators.

The attention planner is the project compiler/planning layer in the runtime
repository; the C++/MLIR compiler repository was not modified in this stage.
MatMul split-M already exists in the earlier CPU-sharding prototype but was not
expanded here because attention was the narrow deliverable.

## 16. Resume bullets

- Extended compiler-guided Qwen attention into exact shared-memory execution
  planning, serializing deterministic head/GQA/output assignments and
  completion synchronization through ExecutionPlan v2.
- Implemented targeted persistent logical-worker dispatch with exact
  plan-to-worker/shard/symbol provenance; verified 96 profitability-selected
  parallel events and 768 forced causal events with zero runtime repartition.
- Calibrated serial/2/4/8-worker schedules across 216 measurements, achieving
  92.3% held-out candidate agreement and 0.36% mean regret versus 31–237% for
  fixed worker-count policies.

## 17. Recommended next stage

Repeat the crossover matrix under controlled CPU-frequency and affinity
conditions, then move the stable placement schema into the C++ ExecutionPlan
types/exporter. Only after that should a separate multi-process shared-memory
prototype be considered; network or vLLM-distributed claims require an
independent stage.
