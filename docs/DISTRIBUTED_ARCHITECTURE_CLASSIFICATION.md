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

## Second amendment: S6 is now demonstrated (D4B)

D4B (`docs/DISTRIBUTED_D4B_ACHIEVEMENT_REPORT.md`) closes the S6 gap named
above. A compiler-selected TP=2 plan for a real Qwen2.5-0.5B-Instruct model
was materialized through the existing vLLM launch-spec adapter (D3B) and
executed by a real vLLM 0.24.0 server on two physical GPUs, with:

- two distinct physical GPUs in real use, confirmed by direct
  `nvidia-smi --query-compute-apps` process-to-device evidence (not
  inferred from `--tensor-parallel-size 2`)
- real NCCL communicator initialization for `world_size=2` (direct server
  log evidence, not inferred from server readiness alone)
- output verified consistent with a same-host TP=1 reference across
  token IDs, text, finish reasons, and logprobs

**What S6 still does not mean**, precisely: this is not a speedup claim,
not a general multi-node capability, not evidence that vLLM executed the
compiler's whole-model work items (D4A) individually rather than its own
installed TP implementation, and not a claim about any workload other than
the exact deterministic correctness corpus used. S2-S5 remain
undemonstrated exactly as stated above; D4B does not touch continuous
batching, P/D disaggregation, or CPU-functional TP/PP semantics.

The operator-scoped O5 evidence table above is unchanged and still
describes a different, earlier result (one operator invocation, CPU-only,
no GPU). D4B is real whole-model, real GPU, real NCCL execution — a
strictly later and separate milestone on the S-ladder, not a
reinterpretation of O5.

## Third amendment: a measured TP1/TP2 policy optimization (D5)

D4B proved correct execution; it made no performance claim. D5
(`docs/DISTRIBUTED_D5_COMPILER_TP_POLICY_REPORT.md`) closes that gap with
a real, measured result on the same 2× RTX 4090 host: a compiler cost
model, fit only on calibration-split measurements using pre-execution
features (model weight footprint, KV-cache bytes/token, GPU count,
workload shape), matches an offline oracle's TP1/TP2 choice on 100% of 21
held-out cells spanning two real models (Qwen2.5-0.5B-Instruct,
Qwen2.5-7B-Instruct), beating a fixed always-TP1 policy by 7.33% mean
regret and a fixed always-TP2 policy by 5.67%.

**What D5 does not show**: no memory-capacity-forced TP1/TP2 crossover
was found on this hardware (a 6-configuration probe up to the 7B model's
real 32,768-token context ceiling found every configuration legally
startable on both TP degrees — vLLM's paged KV-cache manager adapts
rather than hard-fails); no interpolation between the two measured model
sizes; no claim beyond this exact 2-GPU, PCIe-only, single-node host.
D5 does not touch S2-S5 as defined above, and does not reinterpret D4B's
correctness result — it is a strictly later, separate measurement on top
of the same execution chain.
