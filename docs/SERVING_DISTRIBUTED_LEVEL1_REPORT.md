# Serving Distributed Level S1 Report

## 1. Executive verdict

**Serving Distributed Level S1 verified for a vLLM-inspired serving-level
planner executed on an eight-logical-core functional CPU cluster.** This is a
single-process, strongly state-isolated multi-instance prototype. It is not
vLLM serving, multi-GPU execution, or multi-node inference.

## 2. Repository state

Starting state:

- `heterogeneous-inference-runtime`: `main`,
  `34aee51fef08dc447a6a52d938b4867d60eeef70`, dirty with preserved Operator O5
  work.
- `ml-graph-compiler-runtime`: `master`,
  `dbf7329392bd2c70fa6ef25e359b277d171b3082`, clean.

Ending state is recorded in
`results/runtime_paths/serving_distributed_level1/repository_state.json`.
Neither HEAD changed. No reset, clean, stash, rebase, commit, push, or deletion
was performed.

## 3–6. Classification, architecture, terminology, cluster

The prior result was incomplete only in its architectural label. It remains
Operator O5 and is preserved verbatim in
`DISTRIBUTED_ARCHITECTURE_CLASSIFICATION.md`.

The two-level path is:

```text
Request → ServingDistributedCompiler → ServingExecutionPlan → CPUReplica
        → local model forward → existing operator compiler
        → OperatorExecutionPlan → LogicalWorkers/native attention
```

The detected host is an Intel i5-10210U with 8 logical CPUs, 4 physical cores,
2 threads/core, one socket and one NUMA node. The default configuration is four
replicas with two **logical-core budgets** each. Affinity was not applied, so
these budgets are not claims of physical isolation.

## 7–10. Replica state, prefix cache, modes, routing

Every `CPUReplica` owns a queue, active-request set, availability clock,
statistics, capacity, and `ReplicaPrefixCache`. Model weights are shared
read-only in the real-Qwen proof; no mutable queue or KV/prefix state is
shared.

The cache hashes complete 16-token blocks with SHA-256 over parent lineage and
tokens, then verifies parent and token equality on lookup. Partial final blocks
are not reusable. LRU eviction is replica-local. `metadata_only` and
`functional_tensor` are distinct modes; all policy and Qwen evidence here uses
`metadata_only`, so it proves neither KV tensor storage nor transfer/reuse.

The compiler generates/evaluates `round_robin`, `least_queue`,
`max_prefix_hit`, and `prefix_queue_cost`. Least-queue uses predicted
availability, not raw queue depth. The combined score is queue wait + uncached
CPU prefill + decode + cache pressure + measured routing overhead.

## 11–15. Legality, cost, plan, exact dispatch, provenance

`ServingExecutionPlan` schema v1 validates cluster/profile identity, enabled
replica, request ID, policy, cache mode, token accounting, backend presence,
and finite nonnegative costs. JSON serialization/deserialization is mandatory.
`PlanOnlyServingRuntime` indexes only the deserialized replica ID. It exposes
no normal rerouting or manual placement path and fails on missing/invalid
plans.

Nested records keep namespaces separate:

```text
request_id → serving_plan_id → replica-N
           → operator_plan_id → native candidate → logical_worker_id
```

The five main serving counters—replica override, reroute, missing plan, manual
assignment, and fallback—were all zero.

## 16–18. Traces and calibration

Saved-seed deterministic traces cover shared, unique, hot, and
capacity-pressure prefixes. Each request records arrival, prompt IDs, length,
and expected output count. Policy sweeps use `derived_from_measured_cpu`
functional service curves, not GPU latency. Large sweeps are modeled;
`qwen_multi_replica_integration.json` is measured real CPU execution.

The lifecycle is ARRIVED → QUEUED → RUNNING_PREFILL → RUNNING_DECODE →
FINISHED, with arrival, planning, queue, execution, first-token and completion
timestamps. Continuous batching is not implemented.

## 19–20. Routing benchmark and trade-off

Across 120-request traces, `prefix_queue_cost` produced:

| Trace | p95 latency (ms) | Hit rate | Mean regret |
|---|---:|---:|---:|
| Shared prefix | 16.740 | 96.7% | 0.0% |
| Unique prefix | 66.020 | 0.0% | 0.0% |
| Hot prefix | 17.590 | 95.8% | 0.0% |
| Capacity pressure | 30.985 | 83.3% | 0.0% |

The oracle uses the same CPU-derived model and current replica state, so zero
regret is evidence of implementation consistency, not broad predictive
generalization. On hot prefixes, round-robin mean regret was 2.16% and
least-queue was 0.77%. Prefix-only overloaded one replica and reached 98.05 ms
p95 despite a 98.3% hit rate.

The explicit conflict tests show:

- best prefix + 50 ms queue: prefix-only chose replica 0; combined chose idle
  replica 1;
- 256-token recomputation versus a 1 ms warm-replica queue: least-queue chose
  replica 0; combined chose warm replica 1.

## 21–22. Topologies and held-out evaluation

For the modeled hot-prefix trace:

| Logical topology | p95 (ms) | Output tok/s | Hit rate |
|---|---:|---:|---:|
| 1 × 8 | 18.570 | 16,606 | 99% |
| 2 × 4 | 19.715 | 16,154 | 98% |
| 4 × 2 | 21.595 | 15,353 | 97% |
| 8 × 1 | 25.920 | 13,910 | 94% |

This CPU model favored fewer replicas because extra per-replica concurrency did
not offset cache fragmentation and reduced per-replica core budget. It is not a
physical-core isolation result.

On 292 held-out requests, combined-policy exact agreement was 100% with 0%
mean regret against its same-model oracle. This does not establish accuracy
against production wall-clock arrivals.

## 23. Real-Qwen multi-replica integration

Qwen2.5-0.5B-Instruct ran four measured requests on two plan-selected CPU
replicas. Two later requests shared a 32-token complete-block prefix. Replica
queues/caches were independent; immutable weights were shared.

- 4/4 serving plans survived exact JSON round-trip.
- planned replica equaled executed replica for 4/4 requests.
- 96 real compiler-attention invocations executed.
- 96/96 returned attention tensors entered `o_proj`.
- serving fallback/reroute/override: 0/0/0.
- operator fallback/repartition/candidate mismatch: 0/0/0.
- generated first tokens: `[99479]`, `[198]`, `[198]`, `[99804]`.
- the final shared-prefix request reused 32 metadata tokens.

This is real Hugging Face Qwen model-forward execution, not a vLLM engine.

## 24. Routing causal verification

The selected replica held 128 cached metadata tokens; the legal non-selected
replica held none. The override comparison therefore added 128 uncached prompt
tokens and the associated predicted CPU prefill cost. Correct routing preserves
model semantics, so tokens were not perturbed. This establishes placement →
replica cache locality → recomputation → cost causality in metadata mode.

## 25. Stress and negative tests

The 1,000-event deterministic replay completed 1,000 requests with zero
duplicates, losses, queue mismatches, leaks, deadlocks, stale states,
reroutes, overrides, fallbacks, or manual assignments.

Negative coverage includes missing/disabled replicas, request mismatch,
impossible token accounting, unknown policy, schema mismatch, duplicate
replica IDs, negative capacity, missing backend, missing plan, and duplicate
request IDs. Focused tests also guard replica/operator namespace separation.

## 26–27. Maturity and limitations

- Operator Distributed: **O5**.
- Serving Distributed: **S1**.

Not demonstrated: processes with OS memory isolation, physical CPU affinity,
continuous batching, chunked prefill, tensor KV reuse, KV transfer,
prefill/decode disaggregation, vLLM engine scheduling, PagedAttention,
tensor/pipeline parallelism, GPU, NCCL/NVLink, multi-node execution, or S2–S6.

## 28. Resume bullets

- Separated operator-shard scheduling from serving request placement and
  introduced versioned, plan-only replica dispatch with nested provenance.
- Built isolated CPU replica queues and lineage-aware block-prefix caches, then
  evaluated queue/cache-aware routing on deterministic traces and four logical
  topologies.
- Routed four real Qwen CPU requests through two independent replica states and
  96 compiler-planned attention calls with exact dispatch, zero fallback, and
  96/96 outputs entering `o_proj`.

## 29. Exact next stage

Serving Distributed S2: continuous batching, chunked prefill, and token-budget
scheduling. Prefill/decode disaggregation remains a later S3 stage.
