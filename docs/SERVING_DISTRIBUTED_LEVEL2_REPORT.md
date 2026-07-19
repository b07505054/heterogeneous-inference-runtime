# Serving Distributed Level S2 Report

## 1. Executive verdict

**Serving Distributed S2 verified** for replica-local continuous batching,
chunked prefill, compiler-generated token-budget schedules, exact plan-only
execution, modeled policy evaluation, and focused real-Qwen execution.

## 2–4. Repository state and preserved architecture

The runtime repository began on `main` at
`34aee51fef08dc447a6a52d938b4867d60eeef70`; the compiler repository began on
`master` at `dbf7329392bd2c70fa6ef25e359b277d171b3082`. Existing dirty Operator
O5 and Serving S1 work was preserved. Ending state is in
`results/runtime_paths/serving_distributed_level2/repository_state.json`.

Operator O5 remains the intra-operator worker/shard layer. S1
`ServingExecutionPlan` remains request-to-replica ownership. S2 adds a separate
`ScheduleStepPlan` and cannot move a request between replicas.

## 5–7. Audit, request state, and replica scheduler state

The audit found that S1 executed a whole-request callback. Older continuous
batching code was a disconnected simulator, and vLLM
`max_num_seqs`/`max_num_batched_tokens`/chunked-prefill fields were
configuration metadata, not exact step-plan execution.

`RequestExecutionState` tracks arrival, ownership, prefix match, prefill and
decode progress, phase, first token, completion, chunks, token times, and
operator provenance. Validation forbids out-of-range progress, decode before
prefill, and premature FINISHED.

Each `ReplicaSchedulerState` independently owns requests, logical token and
sequence budgets, step/version clocks, terminal IDs, and statistics. Logical
token budget, logical-core budget, and operator worker count remain distinct.

## 8–10. Continuous batching, chunks, and budgets

Future arrivals become ready while older requests decode. Each step may contain
both decode items and exact prefill ranges. Finished requests leave without
draining the remaining batch; the 19 focused tests include changing membership
across three consecutive steps.

Prefill items begin exactly at `prefill_completed_tokens`, are positive, and
are bounded by remaining prompt, maximum chunk, token budget, and sequence
budget. Full prefix hits enter DECODE directly. Tests cover one-token chunks,
partial/full hits, uneven final chunks, gaps, overlaps, and overflow.

Metadata-mode chunking proves scheduler progression, legality, accounting, and
model invocation sequencing. It does not prove production PagedAttention KV
block persistence.

## 11–12. ScheduleStepPlan and validation

Schema v1 serializes replica, monotonic step ID, scheduler-state version,
candidate, exact request IDs, phases, token ranges, budget accounting, and
explainable costs. Normal execution is:

```text
compiler object → canonical JSON → normal deserializer → plan-only runtime
```

Validation rejects stale steps, foreign replicas, duplicate/missing/finished
requests, illegal phases/ranges, budget overflow, invalid costs, and schema
mismatch.

## 13–15. Candidates, legality, and cost

Implemented candidates:

- `decode_first`: decodes, then remaining prefill budget;
- `prefill_first`: prefill chunks, then remaining decode budget;
- `chunked_balanced`: configurable decode reservation, then chunks;
- `slo_aware`: deterministic urgency from TTFT and decode-gap targets.

All candidates return a legal exact plan or `NO_READY_REQUESTS`. Stable arrival,
request ID, policy ID ordering resolves ties.

The analytical CPU-derived score includes step compute, TTFT, decode-gap and
starvation penalties, a KV-pressure placeholder, and batching benefit. It is
intentionally simple; poor held-out regret is reported rather than hidden.

## 16–17. Plan-only runtime and provenance

The runtime validates and applies each item without rebuilding, phase changes,
token changes, or unplanned admission. All six mutation counters remained zero.

Provenance is:

```text
request → S1 serving plan → replica → scheduler version/step
→ exact phase/range → model invocation → operator plan
→ native candidate → logical-worker events
```

Scheduler step IDs, model invocation IDs, and worker event identities are
separate namespaces.

## 18–19. Execution modes and traces

`modeled_service_time` drives decode-heavy, prefill-heavy, mixed, long-prefill,
arrival-burst, prefix-reuse, adversarial, topology, and stress sweeps.
`real_qwen` drives cross-layer correctness. Modeled results are not presented
as measured Qwen latency.

The declared functional SLO is TTFT ≤ 50 ms and maximum decode gap ≤ 10 ms.
Goodput counts requests satisfying both per modeled second.

## 20–22. Modeled workload results

Representative p95 results:

| Workload/policy | TTFT ms | ITL ms | Utilization | Steps |
|---|---:|---:|---:|---:|
| Decode-heavy / decode-first | 140.12 | 2.88 | 24.0% | 75 |
| Decode-heavy / balanced | 171.36 | 1.44 | 12.3% | 146 |
| Prefill-heavy / decode-first | 484.60 | 3.77 | 95.1% | 142 |
| Mixed / prefill-first | 241.07 | 4.46 | 74.7% | 82 |
| Mixed / balanced | 248.45 | 4.52 | 85.1% | 72 |

Decode-heavy balanced scheduling halved p95 ITL but increased TTFT and steps.
Mixed prefill-first improved p95 TTFT but used the budget less efficiently.

## 23. Starvation counterexamples

With four long decodes filling `max_num_seqs`, decode-first delayed a
256-token prefill’s first token to 60.34 ms. Prefill-first reduced it to
58.72 ms but increased the maximum active-decode gap from 0.72 to 20.20 ms.

With four full-budget prefill chunks and four active decodes, prefill-first
caused a 22.56 ms maximum initial decode gap. Balanced scheduling reduced it to
2.26 ms, while its p95 TTFT rose from 22.56 to 33.36 ms. Neither fixed policy
is universally best.

## 24–25. Chunk and policy causality

Modeled 64-token whole prefill used one `[0,64)` chunk and eight steps;
8-token chunking used eight contiguous chunks and ten steps. Coverage was
identical with no gap/overlap, while modeled TTFT and decode interleaving
changed.

The real Qwen check compared a 32-token whole prefill with four 8-token chunks.
Both generated `[16731, 16]`; maximum logit differences were `5.8651e-5` and
`1.5259e-5`. Thus chunk size changed invocation structure without changing
greedy semantics.

Fixed-trace policy runs completed the same requests and produced different
plans, TTFT, ITL, utilization, and fairness metrics.

## 26–28. Utilization, topology, and held-out selector quality

On the mixed trace:

| Logical topology | TTFT p95 ms | ITL p95 ms | Budget utilization |
|---|---:|---:|---:|
| 1×8 | 245.39 | 8.54 | 74.7% |
| 2×4 | 247.29 | 4.27 | 78.5% |
| 4×2 | 244.33 | 2.01 | 71.6% |
| 8×1 | 246.65 | 0.62 | 58.6% |

These use logical profiles and modeled service time, not CPU affinity.

Across four held-out shape/arrival families, dominant step-candidate agreement
with the best fixed replay policy was 0%. Mean regret was 41.32%, p95 and
maximum regret 50.53%. The per-step immediate score frequently mixed
`chunked_balanced` and `prefill_first`, while the request-level replay oracle
favored decode-first. This is a measured selector limitation and the clearest
next cost-model issue.

## 29. Real-Qwen S2 integration

Six overlapping requests were round-robin placed across two CPU replicas. One
32-token prompt required four prefill chunks; multiple decode requests
coexisted. Results:

- 18 exact scheduler steps;
- 672 real compiler-attention invocations;
- 672/672 attention outputs entered `o_proj`;
- all six requests finished and produced two greedy tokens;
- serving reroute/override/fallback: zero;
- scheduler rebuild/item/token/phase override/fallback: zero;
- operator fallback/repartition/candidate mismatch: zero.

Generated outputs are preserved in `qwen_s2_integration.json`. The real path
uses per-request contiguous Transformers caches; S1 prefix metadata remains
metadata-only.

## 30–31. Stress and negative validation

The deterministic stress run completed 1,000 requests in 11,010 steps with no
duplicate terminal ID, loss, prefill overlap/gap, early decode, post-finish
schedule, budget overflow, cross-replica execution, deadlock, or runtime
override.

Negative tests cover every required phase/range/budget/ownership/schema error,
including a logical-worker identifier used as a request ID.

## 32. Tests

- Existing Operator O5: 90.
- Existing Serving S1: 14.
- New Serving S2: 19.
- Cross-layer real-Qwen script: 1 successful focused run.
- Total focused Python unit tests: 123.
- MLIR AttentionCPU CTests: 2.

## 33–34. Truth boundary and maturity

Current classification:

```text
Operator Distributed O5
Serving Distributed S1
Serving Distributed S2
```

Not demonstrated: S3 P/D instance separation, KV transfer, S4 P/D topology,
TP, PP, S5 simulation, S6 multi-GPU, NCCL, multi-node execution, vLLM engine
integration, or production PagedAttention.

## 35. Resume bullets

- Added versioned, compiler-generated continuous-batching plans with exact
  request phases and token ranges, enforced by a fail-closed plan-only runtime.
- Implemented contiguous chunked-prefill accounting and four explainable
  token-budget candidates, then quantified TTFT/ITL/starvation trade-offs over
  11,010 deterministic scheduler steps.
- Executed six real Qwen requests through two replica schedulers and 672
  compiler-attention calls with complete request-to-worker provenance and
  672/672 outputs entering `o_proj`.

## 36. Exact next stage

Serving Distributed S3: real CPU prefill/decode instance separation, explicit
KV ownership, a shared-memory KV connector, and handoff synchronization.
