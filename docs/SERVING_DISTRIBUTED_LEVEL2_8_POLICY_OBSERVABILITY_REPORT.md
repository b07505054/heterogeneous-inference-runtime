# Serving Distributed S2.8 Policy Observability Report

## 1. Executive verdict

**Serving Distributed S2.8: wall-clock policy behavior partially explained;
profitability selector remains unvalidated.**

S2.8 added passive real-Qwen step observability and isolated a substantial
step-latency-model error. Exploratory selector-v3 mean regret was below 5%, but
the required independent training/development/validation/final protocol and
compiler-attention cross-layer final run were not completed. No profitability
or S3 claim is defensible.

## 2. Preserved S2.7 classification

S2.7 remains unchanged: 33.3% top-1, 61.1% pairwise accuracy, 10.16% mean
regret, and 30.25% p95 regret. Incremental summary and large-state overhead
remain verified. No S2.6/S2.7 report or result was overwritten.

## 3. Repository states

Runtime began on `main` at
`34aee51fef08dc447a6a52d938b4867d60eeef70`, with the previously documented
dirty worktree. Compiler began clean on `master` at
`dbf7329392bd2c70fa6ef25e359b277d171b3082`. Ending HEADs are identical. No
reset, clean, stash, rebase, commit, push, deletion, or prior-artifact overwrite
occurred.

## 4–5. Instrumentation and timer accounting

Each real-Qwen step records identity, scheduler state, selected work, actual
Q/KV shape signatures, model-forward count, attention and `o_proj` hooks,
transitions, and eight named timing regions. Timer validation rejects overlap
and excessive unaccounted time.

Across 520 measured steps, maximum unaccounted fraction was
`2.14e-16`; zero steps exceeded the 15% or 0.5 ms tolerance. The near-zero
unaccounted value follows contiguous synchronous CPU timing boundaries; it is
not a claim of per-kernel precision.

## 6–7. Dataset and measured steps

The focused run used four real-Qwen trace families (mixed, contention,
prefill-heavy, decode-heavy), six policies, one warm-up and two measured runs:
24 warm-ups and 48 measured trace/policy executions. It produced 520 measured
steps, 23,568 attention-to-`o_proj` events, equivalent generated outputs, and
zero scheduler-runtime counters.

This is less diverse than the requested twelve-family independent ranking
dataset. It is cross-layer feature discovery, not an untouched final set.

## 8. Existing-model residuals

The calibrated functional model systematically underpredicted real execution:

- prefill mean residual: roughly 98–114 ms; p95 absolute error up to 280 ms;
- mixed mean residual: roughly 48–70 ms; p95 absolute error up to 263 ms;
- decode errors were batch/policy dependent, with p95 up to 354 ms.

Token counts alone therefore do not describe real Qwen step cost.

## 9. Attention versus non-attention cost

Synchronous `self_attn` hooks accumulated 27,208.96 ms from 128,720.98 ms of
measured step time. The attention timer includes `o_proj`; the remainder
includes all non-attention model work plus runtime. No finer per-op attribution
is claimed.

## 10. Policy-induced execution shapes

Policies generated 84–92 steps and 21.4%–44.2% mixed-step frequency. Mean model
forwards ranged from 1.80 to 1.93 per step. Fixed-policy attention invocations
ranged from 3,840 to 3,984. Mean KV length and unused-token budget also varied.
This confirms that policies alter execution shape and forward count, not merely
request-priority timing.

## 11–13. Reconstruction, oracle substitution, and root cause

Pipeline A (modeled latency and modeled accumulation) is the historical failed
path. Pipeline C uses direct Qwen timestamps. Exact Pipeline B reconstruction
was not completed because the initial schema omitted per-request item identities
needed to reconstruct first-token and completion timestamps.

The large systematic step residuals and shape differences strongly implicate
step-latency/shape modeling, but request accumulation cannot yet be independently
cleared. The root cause is therefore **partially isolated**, not proven unique.

## 14–15. Feature ablations and epoch features

The code exposes state, selected composition, Q/KV shapes, forward/attention
counts, transitions, and past-only EWMAs. A non-overlapping feature-ablation
study and structurally shared epoch-sequence predictor were not completed.
Fitting after viewing these cross-layer outcomes would violate the requested
freeze protocol.

## 16–19. Selector v3, confidence, and hysteresis

Separate `ranking_selector_v3_static` and `ranking_selector_v3_adaptive`
implement an interpretable break-even rule set. Adaptive input is limited to
completed past-step EWMAs. Both return policy scores, margin, confidence,
reason, stable-default use, and hysteresis status. Frozen defaults were
`chunked_balanced`, 5% equivalence margin, and 8% hysteresis.

These are exploratory frozen rules, not a wall-clock-trained final model.

## 20–23. Real-Qwen comparison and ranking

Mean real-Qwen objectives identified:

- mixed: SLO-aware best; v3-static regret 9.00%, adaptive 8.64%;
- contention: v3-static best; adaptive regret 0.24%;
- prefill-heavy: decode-first best; v3-static regret 6.05%, adaptive 6.45%;
- decode-heavy: v3-static best; adaptive regret 0.45%.

Across the four traces, mean regret was 3.76% static and 3.94% adaptive.
Static exact-winner rate was 50%; adaptive was 0%. With only two samples and no
independent final split, these do not meet the requested validation standard.

The benchmark used real eager Qwen. Operator-plan counts are correctly recorded
as zero, so this run does not supply compiler-attention worker provenance.
Earlier cross-layer evidence remains preserved but is not relabelled as v3.

## 24. Scalability

The unchanged S2.7 1,000-request results remain 0.0459 ms summary median and
0.0224/0.0290 ms selector median/p95. Constructing, validating, and serializing
an observability record measured 0.0145 ms median and 0.0272 ms p95 over 10,000
samples. The control-plane targets remain satisfied. This is a component
measurement, not a new 1,000-request Qwen stress run.

## 25–26. Stress, negatives, and tests

S2.7's 1,000-request/10,501-step correctness stress remains controlling. Nine
new observability tests cover accounting, future leakage, freeze integrity,
legality, static/adaptive isolation, and hysteresis.

Focused Python regression: **293 passed**. MLIR AttentionCPU: **2/2 passed**.

## 27–28. Truth boundary and maturity

```text
Operator O5
Serving S1/S2
S2.5 functional-model calibration
S2.6 wall-clock validation failed
S2.7 incremental state scalability verified; profitability unvalidated
S2.8 policy behavior partially explained; profitability still unvalidated
```

No S3, P/D, KV transfer, TP/PP, GPU, NCCL, multi-node, or production-vLLM
capability is claimed.

## 29. Resume bullets

- Instrumented 520 real-Qwen scheduler steps with closed timer accounting,
  actual Q/KV shapes, 23,568 attention-to-`o_proj` events, and equivalent
  generated outputs across 48 measured policy/trace runs.
- Identified systematic functional-model underprediction of real prefill and
  mixed steps and quantified policy-induced differences in step count, mixed
  frequency, model forwards, and attention invocations.
- Preserved sub-millisecond incremental scheduling overhead while honestly
  reporting exploratory v3 results: 3.76% mean regret but only 50% exact-winner
  agreement and no independent final split.

## 30. Recommendation

**Do not begin S3.** First add per-request ScheduleItem identities to the
observability dataset, complete measured-step reconstruction, collect
non-overlapping Qwen ranking splits, run feature ablations, freeze a trained v3,
and execute an untouched compiler-attention final comparison.
