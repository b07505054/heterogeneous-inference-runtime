# Serving Distributed S2.7 Ranking and State Report

## 1. Executive verdict

**Serving Distributed S2.7: incremental state scalability verified; policy
profitability remains unvalidated.**

The state-scaling work met its targets. The frozen wall-clock ranking model did
not: final top-1 agreement was 33.3%, pairwise accuracy 61.1%, and mean regret
10.16%. This report therefore does not claim repaired policy profitability and
does not recommend S3.

## 2. Preserved S2.6 negative result

S2.6 remains unchanged and controlling for wall-clock profitability: 42
warm-ups, 140 measured real-Qwen executions, 10 samples per policy/trace,
59,520 attention-to-`o_proj` events, equivalent outputs, and zero mutation or
fallback. Mixed ranking was Spearman -0.80 / 0% pairwise accuracy; contention
was 0.40 / 50%. Selector-v1-frozen regret was 21.13% and 1.20%. Zero modeled
regret proved shared-model consistency, not independent optimality.

## 3. Repository states

Starting HEADs were runtime `34aee51fef08dc447a6a52d938b4867d60eeef70`
on `main` and compiler `dbf7329392bd2c70fa6ef25e359b277d171b3082`
on `master`. The runtime already contained the documented dirty O5/S1/S2/S2.5/
S2.6 work; the compiler was clean. Ending HEADs are identical. No reset, clean,
stash, rebase, commit, push, deletion, or prior-artifact overwrite occurred.

## 4–5. Failure analysis and prediction target

The former selector evaluated request outcomes through the same functional
service model used by its oracle. Independent S2.6 residuals identify mixed
batch interaction, cumulative queueing, batch composition, and objective
aggregation as ranking errors. S2.7 changes the target from absolute modeled
step latency to pairwise policy outcome:

`Delta(A,B) = wall-clock request objective(A) - objective(B)`.

Policy ranking remains separate from legal `ScheduleStepPlan` construction.

## 6–8. Dataset and wall-clock method

Non-overlapping train, development, validation, final, Qwen, and scaling IDs
are recorded in `wall_clock_ranking_dataset_split.json`. Twelve families cover
decode-heavy, prefill-heavy, mixed, contention, bursts, long prefill, prefix
extremes, and token/sequence-budget extremes. Fixed policies used identical
logical requests and counterbalanced order.

The new ranking dataset is honestly classified as functional CPU wall-clock:
real SHA-256 CPU work executes per exact scheduler step. It is neither modeled
latency nor real-Qwen latency. The original S2.6 traces were historical failure
examples only and were not training data.

## 9–13. Features, models, ablations, and break-even regions

`ranking_selector_v2_static` is an interpretable normalized nearest-centroid
linear score derived from wall-clock training winners. Features cover active
composition, remaining work, ages/gaps, budgets, prefix reuse, rolling past
latencies, core budget, and policy stability. `ranking_selector_v2_adaptive`
uses the same frozen coefficients and only completed-step EWMAs; it cannot
mutate the static model or consume future arrivals.

The frozen final result shows this model class is insufficient. In particular,
the large-sequence-budget trace selected prefill-first while measured
chunked-balanced was best, producing 30.25% regret. No post-final tuning,
objective-v2 rewrite, or favorable break-even rule was applied.

## 14. Objective audit

The existing objective was preserved. A new objective was not introduced merely
to improve selector scores.

## 15–18. Snapshot replacement and incremental state

The old path deep-copied every arrived request, including growing request
metadata and provenance. `SchedulerStateSummary` instead maintains counts,
remaining work, prefix totals, rolling completed-step latency, and bounded
frontiers. Lazy heaps are updated on request events. Completed detail moves out
of hot state; only counters remain.

Candidate evaluation uses immutable summaries plus `SummaryDelta`. Tests prove
the base is unchanged and candidate deltas cannot leak. Ranking produces only a
policy; the existing S2 compiler still generates and validates exact token
ranges.

Across 10,501 transitions, 105 full-reference checks found zero summary
mismatches after FP rounding at 1e-8. Incremental request-objective aggregation
was not completed as a separate production facility; the v2 ranker instead
scores the constant-size summary directly.

## 19–21. Scaling

| Requests | Old median ms | New median ms | Speedup |
| ---: | ---: | ---: | ---: |
| 4 | 0.0726 | 0.0096 | 7.53x |
| 64 | 0.8992 | 0.0232 | 38.77x |
| 256 | 3.6469 | 0.0286 | 127.65x |
| 1,000 | 14.6798 | 0.0459 | 319.97x |
| 5,000 | 81.2361 | 0.0684 | 1,187.26x |

The 1,000-request stress selector itself measured 0.0224 ms median and 0.0290
ms p95, well below the 5/10 ms targets. Summary construction measured 0.0383
ms median and 0.0541 ms p95.

## 22–24. Confidence, hysteresis, and freeze

The selector emits all policy scores, score margin, uncertainty, stable-default
use, and hysteresis retention. Near ties choose frozen
`chunked_balanced`; switching requires a frozen improvement margin. Model,
feature normalization, equivalence margin, hysteresis, and default were hashed
before validation/final evaluation. Final observations were not used to tune.

## 25–27. Validation, ranking, and regret

Final functional CPU wall-clock results:

- top-1 agreement: 33.3%;
- pairwise ranking accuracy: 61.1%;
- mean regret: 10.16%;
- median regret: 0.244%;
- p95/max observed regret: 30.25%.

Only small-sequence-budget selected the exact winner. Large-token-budget was
near-optimal (0.244% regret) but wrong top-1. These miss the 75% agreement,
75% pairwise accuracy, 5% mean regret, and 15% p95 targets.

## 28. Selector latency

Large-state overhead is repaired. The prior 27–28 ms median selection became
0.022 ms median in the 1,000-request stress. This improvement is attributable
to incremental summaries and constant-size ranking, not disabled plan
validation.

## 29. Real-Qwen cross-layer validation

No new repeated v2 static/adaptive real-Qwen comparison was completed.
Previously verified S2.6 real-Qwen correctness remains preserved but cannot be
relabelled as v2 evidence. Acceptance criterion 21 is therefore unmet.

## 30–31. Stress, negatives, and tests

The stress run completed 1,000 requests across 10,501 steps with no loss,
duplicate completion, summary mismatch, schedule rebuild, item/token/phase
override, unplanned request, or fallback.

Focused regression command covering attention, native fused attention,
operator-distributed, S1/S2/S2.5/S2.6 and new S2.7 tests passed **284 tests**.
The new S2.7 file contributes 15 tests. MLIR AttentionCPU CTests passed 2/2.

## 32–33. Truth boundary and maturity

```text
Operator Distributed O5
Serving Distributed S1
Serving Distributed S2
Serving Distributed S2.5 calibrated to functional model
Serving Distributed S2.6 independent wall-clock validation failed
Serving Distributed S2.7 incremental state scalability verified
Policy profitability remains unvalidated
```

No S3, P/D, KV connector, TP/PP, GPU, NCCL, multi-node, or production-vLLM
claim is made.

## 34. Resume bullets

- Replaced per-selection deep copies with event-maintained scheduler summaries
  and bounded frontiers, reducing 1,000-request snapshot latency from 14.68 ms
  to 0.046 ms median.
- Validated plan-only continuous batching across 1,000 requests and 10,501
  steps with zero summary-reference mismatches or runtime schedule mutations.
- Built and froze an independent CPU wall-clock ranking experiment across 12
  workload families, reporting the negative final result (33.3% top-1, 10.16%
  mean regret) without tuning on the final split.

## 35. Recommendation

**Do not begin S3.** Large-state online overhead is acceptable, but independent
wall-clock ranking is not sufficiently validated. The next work should collect
more real-Qwen paired training/validation traces and replace the underfit
centroid model before another untouched final test.
