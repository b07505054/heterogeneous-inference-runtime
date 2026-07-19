# Serving Distributed S2.12 Statistical Closure Report

## 1. Executive verdict

```text
Serving S2.10 remains historical evidence
conservative selector benefit failed broader replication
```

S2.12 completed the frozen, fixed-sample study and found that risk-aware v4
did not outperform the robust `decode_first` baseline on the broader
distribution. The trace-clustered mean difference favored the baseline, its
confidence interval crossed zero, p95 non-inferiority failed, and contention
had a preregistered material regression. Rollout-v2, compiler provenance,
execution correctness, and scalability all remained verified. Do not begin
S3.

## 2–4. Preserved evidence, repository state, and frozen hashes

S2.11 remains unchanged: 0.46% selector mean regret versus 0.69% baseline,
100% tie-aware agreement, but an improvement interval crossing zero and 1.62%
selector p95 versus 1.30% baseline. Its exact rollout and natural native AVX2
split-head/2 evidence are preserved.

Runtime HEAD remained
`34aee51fef08dc447a6a52d938b4867d60eeef70`; compiler HEAD remained
`dbf7329392bd2c70fa6ef25e359b277d171b3082`. No destructive Git operation or
prior-artifact overwrite occurred.

Before measurement, hashes froze selector implementation/configuration,
rollout-v2, three service models, objective, robust baseline, and the trace
registry. No selector-v5, refit, or threshold change was introduced.

## 5. Pre-measurement power analysis

S2.11 estimated a 0.231-percentage-point paired benefit with 0.847-point paired
SD. A normal paired approximation at 95% confidence and 80% power required
about **106 independent traces**. S2.12 preregistered 12 traces, so it met the
execution minimum but was known to be underpowered for such a small effect at
the controlling trace-cluster level.

## 6–9. Preregistration, traces, sample size, and ordering

The full protocol was frozen at SHA-256
`9c18ab09e2bfcb703b47cac4ce43205ff320e1489687e130f0cdce4aeb23f7be`
before measurement. It locked 12 new logical trace hashes, five policy series,
three warm-ups and ten measured repetitions per cell, 780 planned executions,
counterbalanced seed 2120, bootstrap methods, a 0.5-percentage-point p95
tolerance, and no early stopping.

The traces covered decode-heavy, prefill-heavy, mixed, contention, burst,
long-prefill, shared-prefix structure, low-prefix reuse, small/large token
budgets, small sequence budget, and larger active batch. Each family has one
independent trace, not the preferred three; this is disclosed.

All 180 warm-ups and 600 measured executions completed. Each of 60 cells has
exactly ten samples. The frozen order rotated a shuffled base and reversed
after each cycle. Mean objective by position ranged from 3,290.23 to 3,310.04
ms without a monotonic first/last effect.

## 10. Runtime noise

Process CPU time, context switches, page faults, system load, warm-state, and
order position were recorded for every run. Mean within-cell coefficient of
variation was 4.09%; maximum was 10.40%. Between-block variance of block mean
objective was 5,431.68 ms². No valid measurement was removed.

## 11–14. Raw outcomes and paired analyses

The run produced 465,120 attention-to-`o_proj` events, equivalent generated
outputs in every cell, and zero schedule rebuild, item/token/phase override,
unplanned request, or fallback counts.

At the run level (120 paired trace/block observations):

- baseline-minus-selector mean: **-0.286 percentage points**;
- median: -0.016 points;
- bootstrap 95% CI: **[-0.956, +0.320] points**;
- probability of positive mean under the frozen bootstrap: 18.7%.

At the controlling trace level (12 independent aggregates):

- baseline-minus-selector mean: **-0.286 percentage points**;
- median: -0.290 points;
- clustered 95% CI: **[-0.816, +0.230] points**;
- bootstrap probability of positive improvement: 14.3%.

Negative values favor the baseline. Mean-benefit confirmation therefore
failed.

## 15. Tail non-inferiority

| Tail metric | Selector v4 | Baseline |
| --- | ---: | ---: |
| p90 regret | 2.65% | 2.13% |
| p95 regret | 2.65% | 2.13% |
| Maximum | 3.01% | 2.13% |

The p95 difference was +0.522 percentage points, slightly above the frozen
0.5-point tolerance. Its cluster-bootstrap interval was [-0.568, +1.473]
points, so the upper bound also failed non-inferiority.

## 16. Family-level results

Point differences favored v4 for decode-heavy, mixed, arrival-burst,
long-prefill, prefix-heavy, and small-token-budget. They favored the baseline
for prefill-heavy, contention, low-prefix-reuse, large-token-budget,
small-sequence-budget, and larger-active-batch. Contention regressed by 2.10
percentage points, exceeding the frozen 2% material threshold.

One trace per family cannot establish family-specific significance. All
leave-one-trace and leave-one-family intervals remained inconclusive; the
aggregate failure was not reversed by excluding a single trace.

## 17–20. Risk gate, confidence, rollout, and provenance

Across 1,940 selector steps, v4 used the robust default 1,830 times and selected
a candidate 110 times (5.67% coverage). No decision was rejected by the
uncertainty threshold; 1,830 were rejected by the frozen improvement margin.
Ten states had nonzero OOD score. Candidate departures had 51.8% realized win
rate and small positive mean improvement, while default-path outcomes were
negative on average. These diagnostics did not refit the gate.

All confidence observations fell in the 80–100% support bucket, demonstrating
that v4 confidence is not a calibrated win probability. Its observed win rate
was 48.2%. Thresholds remain unchanged.

Rollout-v2 generalized across all 48 new fixed-policy cases:

- first-step agreement: 100%;
- full-sequence agreement: 100%;
- selected-work Jaccard: 1.0;
- step-count error: zero;
- next-state and completion-order agreement: 100%.

The focused compiler regression again produced 24 parallel native AVX2
split-head/2 plans, 48 parallel worker events, and 48/48 `o_proj`
consumptions, with zero fallback, repartition, or mismatch.

## 23–25. Scalability, stress, and tests

The fresh 1,000-request/10,501-step stress completed with zero request loss,
duplicate completion, summary mismatch, rebuild, override, or fallback.

- 1,000-request summary: 0.0337 ms median, 0.0555 ms p95;
- risk-aware selector: 0.0137 ms median, 0.0234 ms p95;
- 5,000-request summary: 0.0293 ms median.

All control-plane targets passed. The final regression passed **315 focused
Python tests** and **2/2 MLIR AttentionCPU CTests**.

## 26–27. Truth boundary and maturity

S2.10 remains accurate historical evidence for its preregistered set. S2.11
showed plausible but inconclusive benefit. S2.12's broader frozen study did
not replicate average profitability or tail non-regression. Execution
correctness, exact rollout, compiler-selected multi-worker provenance, and
scalability remain verified independently of the selector failure.

This stage is not selector-v5, S3, P/D disaggregation, vLLM production,
multi-GPU, NCCL, TP/PP, or multi-node execution.

## 28. Resume bullets

- Completed 600 new measured real-Qwen executions with ten samples in every
  cell; clustered baseline-minus-selector improvement was -0.286 points with
  CI [-0.816, +0.230], so broader profitability failed.
- Tail non-inferiority failed: selector p95 regret was 2.65% versus baseline
  2.13%, and contention crossed the frozen material-regression threshold.
- Preserved 100% rollout-v2 sequence agreement, natural AVX2 split-head/2
  provenance, zero runtime mutation, and 0.0137 ms median selector overhead.

## 29. S3 recommendation

**Do not begin S3.** The controlling prerequisites—positive trace-clustered
mean improvement, p95 non-inferiority, and absence of material family
regression—did not pass. Any future selector repair must use a new version and
new preregistered evidence; S2.12 data must not be used to retroactively tune
v4.
