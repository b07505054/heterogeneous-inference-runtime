# Serving Distributed S2.9 Reconstruction and Validation Report

## 1. Executive verdict

**Serving Distributed S2.9: request-level wall-clock model validated;
profitability selector remains unvalidated.**

Measured scheduler steps reconstruct request outcomes well within the frozen
1 ms tolerance, and execution-shape service models sharply improve independent
step prediction. Static and adaptive selectors still miss profitability
targets. Adaptive selection is rejected. S3 must not begin.

## 2–3. Preserved evidence and repository state

All S2.6–S2.8 positive and negative evidence remains unchanged. S2.8's 520
steps, 48 measured runs, 24 warm-ups, 23,568 `o_proj` events, systematic model
bias, and exploratory selector results are preserved.

Runtime started and ended on `main` at
`34aee51fef08dc447a6a52d938b4867d60eeef70`; compiler started and ended on
`master` at `dbf7329392bd2c70fa6ef25e359b277d171b3082`. No destructive or Git
history-changing operation occurred.

## 4. Timestamp semantics

All reconstructed physical timestamps use `time.perf_counter_ns`. Request
arrival is the physical benchmark origin; admission is the first item callback;
prefill completion is final-prefill commit; token and completion timestamps are
decode-item commits; step end follows scheduler-state commit. Logical scheduler
time is stored separately and never mixed with physical time.

## 5–6. Reconstruction and accuracy

The new schema records each ScheduleItem identity, token range, callback start,
and commit inside its enclosing measured step. Duplicate steps, missing
requests, out-of-step commits, and invalid clock origins fail closed.

Frozen tolerance: 1 ms. Independent results:

| Metric | MAE ms | Median ms | p95 ms | Max ms |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 0.00052 | 0.00047 | 0.00065 | 0.01343 |
| Completion | 0.00548 | 0.00522 | 0.00657 | 0.02066 |
| Max decode gap | 0.00364 | 0.00498 | 0.00674 | 0.01824 |

Reconstruction passes. Request accumulation is therefore validated for these
traces.

## 7–8. Oracle substitution and root cause

Measured reconstruction (Pipeline C) and direct timestamps (Pipeline D)
produced identical fixed-policy orderings on all four traces. The original
model (A) was wrong on three traces. The shape model (B) corrected the
long-prefill ordering substantially but remained wrong on arrival-burst and
prefix-heavy.

This proves accumulation is not the dominant failure. Step-cost fidelity and
policy-sequence ranking remain the controlling errors.

## 9–10. Prefill and mixed residuals

Independent validation:

| Phase | Old MAE ms | v2 MAE ms | Old p95 ms | v2 p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 121.49 | 8.24 | 261.64 | 24.35 |
| Decode | 38.62 | 1.97 | 116.52 | 4.76 |
| Mixed | 92.65 | 6.94 | 258.97 | 15.32 |

The evidence supports missing shape/setup terms rather than token slope alone.
The v2 features include intercept, prefill tokens, decode sequences, scheduled
sequences, model-forward count, maximum query/KV length, phase transitions, and
prefill×decode interaction. No per-op causal coefficient claim is made.

## 11–12. Service-model v2 and policy-sequence cost

Separate versioned least-squares models were fitted for prefill, decode, and
mixed steps using only S2.8 rows. Policy trace cost sums predicted shape-aware
step durations while retaining each policy's actual measured sequence shape for
this validation. A prospective bounded rollout that predicts all future shapes
from online state was not completed.

## 13–15. Dataset, training, and validation

Training uses four S2.8 trace families. Independent evaluation uses new
arrival-burst, long-prefill, prefix-heavy, and budget-pressure request
sequences. There is no trace/request overlap.

However, the coefficient artifact was materialized after final measurements
already existed, even though fitting code reads training rows only and no
post-result tuning occurred. This fails the strict “freeze, then execute final”
protocol. The result is independent holdout validation, not an untouched
post-freeze final test.

## 16. Adaptive failure analysis

Adaptive regret worsened from 3.94% exploratory to 6.78% independent, with 0%
exact-winner agreement. It did not beat static (6.15%). Too few completed
observations, phase mixing, inherited static-rule errors, and policy-induced
sampling bias remain plausible. **Adaptive selector is rejected.**

## 17–19. Static/adaptive selectors and freeze

Historical v3 remains unchanged. No “static_final” or “adaptive_final” selector
is claimed because the required development ablation and pre-run freeze were
not completed. Confidence/default/hysteresis settings remain the exploratory
5%/`chunked_balanced`/8% configuration.

## 20–23. Final methodology and profitability

The new set used four traces, six policies, one warm-up and two measured runs
per policy/trace, counterbalanced. This sample is below the preferred 3/10.

Static independent mean regret: **6.15%**; adaptive: **6.78%**. Both had 0%
exact-winner agreement. Static exceeds the 5% mean-regret target and adaptive is
worse. Tie-aware confidence conclusions are not defensible with two samples.

## 24. Eager-attention boundary

The statistical run uses real eager Qwen. Attention-to-`o_proj` is observed,
but operator-plan and worker-event counts are correctly zero. It validates
request progression and real model execution, not compiler operator plans.

## 25. Compiler-attention provenance

A separate S2.9 compiler-attention run was not performed because changing the
statistical harness would alter its semantics. Earlier Level-5/O5 evidence is
preserved but not relabelled as S2.9. This acceptance item remains incomplete.

## 26. Scalability

Reconstruction is offline and does not enter the selector hot path. Preserved
results remain: summary 0.0459 ms median at 1,000 requests, selector
0.0224/0.0290 ms median/p95, and 5,000-request summary 0.0684 ms median.

## 27–28. Stress, negatives, and tests

The preserved 1,000-request/10,501-step stress remains correct. New tests cover
clock origin, omitted/duplicate steps, commit containment, tolerance, and exact
reconstruction. Focused Python and MLIR totals are reported in the result
artifacts after the final regression run.

## 29–30. Truth boundary and maturity

```text
Operator O5
Serving S1/S2
S2.5 functional calibration
S2.6 wall-clock validation failed
S2.7 incremental scalability verified; profitability unvalidated
S2.8 behavior partially explained
S2.9 request-level wall-clock model validated; profitability unvalidated
```

No S3 or distributed accelerator capability is claimed.

## 31. Resume bullets

- Reconstructed TTFT, completion, and decode gaps from committed real-Qwen
  scheduler items with p95 errors below 0.007 ms against a frozen 1 ms bound.
- Reduced independent prefill/decode/mixed step MAE from 121.49/38.62/92.65 ms
  to 8.24/1.97/6.94 ms using interpretable execution-shape models.
- Rejected an adaptive scheduler honestly after independent regret reached
  6.78%, while preserving sub-0.1 ms incremental scheduling overhead.

## 32. Recommendation

**Do not begin S3.** Next complete a prospective online shape rollout, create
strict pre-run freeze artifacts, collect at least 10 repeated final samples,
and run a separate compiler-attention cross-layer provenance check.
