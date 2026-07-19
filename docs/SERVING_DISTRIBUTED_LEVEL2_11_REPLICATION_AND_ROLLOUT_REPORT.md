# Serving Distributed S2.11 Replication and Rollout Report

## 1. Executive verdict

```text
Serving Distributed S2.11
conservative selector benefit remains plausible
high-confidence replication incomplete
```

The frozen risk-aware selector retained low regret and 100% tie-aware
agreement, but its paired improvement confidence interval crossed zero and its
replication p95 regret exceeded the robust baseline. S2.10 remains valid
historical evidence; S2.11 does not strengthen it into high-confidence
replication. Rollout-v2 fully repaired prospective semantic fidelity, and a
natural compiler-selected multi-worker Qwen path was observed without an
override. Do not begin S3.

## 2–3. Preserved S2.10 evidence and repository states

S2.10 remains unchanged: risk-aware v4 had 100% tie-aware agreement, 0.70%
mean regret, 0.51% median regret, and 1.67% p95/max regret versus the robust
baseline's 1.30% mean and 3.74% p95. Its two-sample limitation, weak rollout,
missing dynamic accuracy, and serial-only focused provenance are preserved.

Runtime HEAD remained
`34aee51fef08dc447a6a52d938b4867d60eeef70`; compiler HEAD remained
`dbf7329392bd2c70fa6ef25e359b277d171b3082`. No reset, clean, stash, commit,
push, deletion, or prior-artifact overwrite occurred.

## 4–6. Preregistration, traces, and repetitions

The protocol was frozen at SHA-256
`92149958a48ce5e6190514294caa29d92cc1ff11442f176c47017aac9447c60e`
before measurement. It locked selector/service hashes, six new trace hashes,
five policies, objective and gate parameters, counterbalanced seed 2110,
three warm-ups, five measured repetitions, paired bootstrap intervals, and
failure rules.

The run completed 90 warm-ups and 150 measured executions across mixed A/B,
contention/decode-heavy, arrival burst, long-prefill, and low-prefix-reuse
families. Five repetitions, rather than the preferred ten, were preregistered
because the full run already required 240 model executions. All 101,640
attention outputs entered `o_proj`; outputs were equivalent within every cell
and all scheduler mutation counters were zero.

## 7–8. Order and runtime noise

The frozen shuffled base was rotated by block and reversed after a complete
rotation. Mean objective by order position ranged from 2,880.93 to 2,899.79
ms, with no monotonic first/last-position trend. Process CPU time, voluntary
and involuntary context switches, page faults, one-minute load, and warm-state
status were recorded per run.

Mean within-cell coefficient of variation was 1.50%; maximum was 2.30%.
Policy differences inside the 2% margin were therefore treated as practical
ties.

## 9–10. Replication profitability and baseline

| Metric | Risk-aware v4 | Robust `decode_first` |
| --- | ---: | ---: |
| Mean regret | 0.46% | 0.69% |
| Median regret | 0.13% | — |
| p90 regret | 0.90% | — |
| p95/max regret | 1.62% | 1.30% |
| Tie-aware agreement | 100% | — |

The selector beat the baseline on four of six trace-level paired differences.
The paired mean difference was -0.23 percentage points, but its preregistered
bootstrap 95% interval was [-0.83, +0.40] percentage points. Because the
interval includes zero and selector p95 was worse, replication is
**inconclusive**, not successful.

## 11–15. Prospective rollout and rollout-v2

The S2.10 rollout feature estimated aggregate remaining steps; it did not
simulate legal ScheduleStepPlans. Rollout-v2 clones only request progress,
then invokes the existing SchedulerCompiler and PlanOnlySchedulerRuntime.
This evidence-backed change aligns arrival visibility, deterministic ordering,
token budgets, prefill/decode transitions, and completion boundaries.

Across 24 fixed-policy trace cases:

- exact first-step agreement: 100%;
- exact full-sequence agreement: 100%;
- selected-work Jaccard similarity: 1.0;
- mean absolute step-count error: 0;
- live-state or candidate-state leakage: zero.

There was no first semantic divergence. Timing variance remains separate.
Rollout-v2 is diagnostic and does not alter frozen selector-v4 decisions or
regret.

## 16–18. Dynamic decisions and confidence

Using the next four scheduler steps and only states shared by at least two
fixed-policy executions produced 81 comparable states and 281 policy pairs.
Dynamic exact-or-tie-aware pairwise accuracy was 72.95%; 58.72% of pairs were
practical ties. This is diagnostic and does not replace trace-level
profitability.

V4 uncertainty is a conservative support/OOD threshold, not a calibrated
probability. Replication data was not used to refit it, so a probability
calibration curve would be misleading.

## 19–20. Natural multi-worker provenance

The legal search covered 312 supported attention shapes; 224 naturally chose
more than one worker. A focused Qwen q=32 prefill naturally selected:

```text
cpu_fused_online_native_avx2_fp32_q1_k32_split_head_w2_v1
```

The run produced 24 parallel prefill operator plans, 24 serial decode plans,
72 total worker events (48 from parallel plans), and 48/48 `o_proj`
consumptions. Parallel heads were assigned [0,7) and [7,14). Median parallel
attention, dispatch, and barrier times were 0.4135, 0.0872, and 0.3207 ms.
Generated tokens were `[16731, 16]`; chunked and whole-prefill outputs matched.
All serving, scheduler, fallback, repartition, and mismatch counters were zero.
No cost, candidate, or worker override was used.

No second existing CPU environment was available without adding a backend.

## 22–24. Scalability, stress, and tests

The fresh 1,000-request/10,501-step run completed all requests with zero loss,
duplicate completion, summary mismatch, rebuild, override, or fallback.

- 1,000-request summary: 0.0334 ms median, 0.0454 ms p95;
- frozen risk-aware selector: 0.0134 ms median, 0.0188 ms p95;
- 5,000-request summary: 0.0290 ms median.

All targets passed. The regression suite passed **310 focused Python tests**
and **2/2 MLIR AttentionCPU CTests**.

## 25–26. Truth boundary and classification

S2.10 profitability remains validated on its preregistered historical set.
S2.11 point estimates support it, but high-confidence replication is
incomplete. Prospective semantic rollout fidelity and natural multi-worker
compiler provenance are now verified. This is still single-node CPU
functional serving—not S3, P/D disaggregation, vLLM production, GPU, NCCL,
TP/PP, or multi-node execution.

## 27. Resume bullets

- Replicated frozen risk-aware v4 over 150 measured real-Qwen runs: 0.46%
  mean regret and 100% tie-aware agreement, but a paired interval crossing
  zero and 1.62% p95 versus the baseline's 1.30% make replication inconclusive.
- Replaced aggregate rollout estimates with exact semantic rollout-v2,
  achieving 100% full-sequence agreement across 24 fixed-policy cases without
  changing the frozen selector.
- Naturally selected native AVX2 split-head/2 for real Qwen q=32 prefill,
  producing 48 parallel worker events and complete compiler-plan-to-`o_proj`
  provenance without overrides.

## 28. S3 recommendation

**Do not begin S3.** Increase replication to at least ten measured samples per
cell and more independent traces until the paired interval establishes
improvement and selector p95 is no worse than the robust baseline. Rollout
fidelity, compiler provenance, and overhead are no longer blockers.
