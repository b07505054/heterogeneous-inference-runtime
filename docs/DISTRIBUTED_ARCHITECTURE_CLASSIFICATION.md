# Distributed Architecture Classification

## Amendment

The previously verified Distributed Level 5 result is **Operator Distributed
Level O5: compiler-planned intra-process shared-memory multi-worker
execution**.

It proves exact compiler-to-logical-worker placement, shared-buffer semantics,
completion synchronization, native execution provenance, numerical
correctness, causal model dependency, and schedule profitability for one
operator invocation. It does not prove serving-level data-parallel replicas,
prefix-aware request routing, prefill/decode disaggregation, tensor
parallelism, pipeline parallelism, NCCL, multi-GPU execution, or multi-node
execution.

The evidence remains unchanged and operator-scoped:

| Evidence | Preserved result |
|---|---:|
| Candidate measurements | 216 |
| Held-out complete-candidate agreement | 92.3% |
| Mean regret | 0.36% |
| Fixed-policy regret, W1/W2/W4/W8 | 31.17% / 25.52% / 79.87% / 237.30% |
| Normal Qwen worker events | 96/96 exact |
| Forced-test worker events | 768 |
| Maximum Qwen logit difference | `5.126e-5` |
| Mixed operator invocations | 1,000 |
| Focused Python / MLIR CTests | 90 / 2 |

These are not serving-level measurements.

## Terminology

| Term | Meaning |
|---|---|
| LogicalWorker | Persistent local thread participating in one operator call |
| OperatorShard | Head, row, query, or output range for one operator |
| OperatorPlacement | Exact shard-to-logical-worker assignment |
| CPUReplica | Independently queued serving entity with private mutable cache state |
| ServingInstance | Generic replica/phase instance abstraction |
| RequestPlacement | Serving-compiler assignment of one request to one instance |

Compatibility fields in prior operator plans are retained. `worker` in the old
`deployment/distributed_serving.py` simulator referred to a replica-like
entity; new S1 code uses `CPUReplica` and never puts replica IDs in the logical
worker namespace.

## Two independent maturity ladders

The current operator result is **O5**: real-model compiler-planned operator
execution. The serving result described in
`SERVING_DISTRIBUTED_LEVEL1_REPORT.md` is **S1**: independent queues and prefix
states, compiler request placement, and exact runtime dispatch.

S2 continuous batching/chunked prefill, S3 P/D disaggregation, S4 unified
versus P/D ratios, S5 CPU-functional TP/PP semantics, and S6 real
multi-GPU/vLLM execution have not been demonstrated.
