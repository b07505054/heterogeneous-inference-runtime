# Serving Distributed S2.5 Calibration Report

## 1–3. Verdict, preserved negative result, repository state

**Serving Distributed S2.5 request-level selector calibrated within the frozen
functional CPU replay model.** Execution correctness and selection
profitability remain separate claims.

The original result is preserved unchanged:

```text
selector v0 dominant-policy agreement: 0%
selector v0 mean request-level regret: 41.32%
```

Therefore the historical statement remains true: *Serving S2 execution
semantics are verified, but the initial scheduling cost model is not profitable
on held-out workload families.*

The runtime repository started at
`34aee51fef08dc447a6a52d938b4867d60eeef70`; the compiler repository started at
`dbf7329392bd2c70fa6ef25e359b277d171b3082`. Ending state is recorded in
`results/runtime_paths/serving_distributed_level2_5/repository_state.json`.

## 4–6. Selector v0 and failure analysis

Selector v0 evaluates one step, selects among four exact `ScheduleStepPlan`s,
and repeats. It was compared against a fixed-policy full-trace oracle. Those
quantities were not commensurate.

| Family | v0 dominant | Oracle | Regret | Primary failure |
|---|---|---|---:|---|
| Mixed | balanced | decode-first | 43.64% | horizon/policy-level mismatch |
| Arrival burst | balanced | decode-first | 45.36% | future arrivals and cumulative delay |
| Prefix reuse | balanced | decode-first | 25.77% | decode-gap/prefix interaction |
| Adversarial | balanced | decode-first | 50.53% | starvation and deferred work |

Root causes are short-horizon bias, omitted-request delay, missing cumulative
age, mismatched units/objectives, and per-step policy switching versus a stable
policy oracle. `ScheduleStepPlan` execution was not a cause.

S2.5 now labels:

- step score: one current exact plan;
- horizon prediction: cloned state after H steps plus terminal cost;
- request objective: completed fixed-policy trace.

## 7–9. Objective, profiles, normalization, and split

`request_objective_v1` normalizes TTFT by 50 ms, decode gap by 10 ms, E2E by
250 ms, and goodput by 50 requests/s. It combines normalized TTFT, decode gap,
E2E, SLO, starvation and fairness penalties minus normalized goodput. Raw
milliseconds, counts, and throughput are never directly summed.

Versioned profiles are balanced-interactive, decode-latency, TTFT, throughput,
and fairness priority. The compiler receives the profile explicitly.

Frozen non-overlapping sets contain four calibration, three development, six
held-out, one stress and one real-Qwen trace identity. IDs and seeds are unique;
parameters were frozen before held-out generation. Held-out variations include
unseen prompt lengths, output lengths, arrival patterns, budgets, sequence
limits, prefix matches, topology profiles, core budgets and SLO scales.

## 10–11. CPU service calibration

Real Qwen2.5-0.5B FP32 measurements covered prefill lengths 4/8/16/32, decode
batch sizes 1/2/4, and three mixed compositions. The mixed curve is fitted
separately; it is not assumed to equal prefill plus decode.

Interpretable nonnegative fits produced:

- mean absolute error: 2.38 ms;
- median absolute error: 1.90 ms;
- p95/maximum absolute error: 5.32 ms.

The coefficients feed `CalibratedQwenCPUServiceModel`, which supplies the
mixed-step compute feature used by request-level policy replay. These focused
points are calibration evidence, not a comprehensive performance model.

## 12–15. Cumulative delay, excluded work, horizon, terminal cost

Every rollout tracks current wait age, time since decode, remaining prefill,
remaining decode and projected terminal time. Scheduled and unscheduled
requests both accrue time because cloned state advances by calibrated step
latency. Evidence records first-step excluded request IDs and their incremental
delay.

Finite horizons 1/2/4/8 are implemented with stable fixed-policy continuation.
The terminal term includes remaining prefill/decode service, projected TTFT
delay, decode-gap delay and completion work. On the saved counterexample,
H=4 without terminal cost selected prefill-first; enabling terminal cost changed
selection to balanced/decode-first-equivalent, preventing deferred work from
appearing free.

## 16–17. Policy plans and epochs

`SchedulerPolicyPlan` is separate from `ScheduleStepPlan`. It records policy,
objective profile, epoch, validity, trigger and selector/cost-model versions.

Compared modes are frozen fixed-trace policy, v0 every-step reselection,
four-step epochs, and event triggers. Supported triggers include arrival, phase
transition, completion, SLO risk and material queue change. Primary v1
evaluation uses one policy epoch per trace; its policy-switch count is zero.

## 18–20. Features, calibration, and ablations

Features are ready/waiting counts, remaining work, oldest age, maximum/mean
decode gap, token/sequence budget, prefix-hit total, KV pressure, core budget
and calibrated mixed latency.

Calibration is interpretable fixed-policy replay plus bounded nonnegative
linear service fits—no learned neural model. Coefficients and normalization are
saved.

Ablations show:

- removing age/deferred-delay behavior recreates frozen v0;
- removing decode-gap features is represented by prefill-first;
- removing remaining-work features is represented by balanced-only;
- removing calibrated mixed latency recreates additive short-horizon behavior;
- topology removal is evaluated separately.

## 21–25. Counterexamples, baselines, oracle, horizons, switching

Saved counterexamples cover immediate-step failure, deferred prefill,
deferred decode, horizon boundary, switching, and topology dependence.

Baselines are always decode-first, prefill-first, balanced, SLO-aware, frozen
v0, deterministic random legal selection, and full fixed-policy replay oracle.
The oracle uses identical traces, topology, legality, objective and calibrated
service model.

Bounded horizons do not reliably match the full oracle: H=1–8 still has ties
and boundary failures. Consequently primary v1 uses calibrated full-trace
request-level prediction for an epoch, not a claim that H=8 alone is solved.
This makes planning more expensive but eliminates v0 oscillation.

## 26–29. Development, held-out, generalization, topology

With frozen parameters:

| Split | Traces | v1 agreement | v1 mean regret | frozen-v0 mean regret |
|---|---:|---:|---:|---:|
| Calibration | 4 | 100% | 0% | 41.21% |
| Development | 3 | 100% | 0% | 42.91% |
| Held-out | 6 | 100% | 0% | 31.48% |
| Unseen generalization | 3 | 100% | 0% | 22.62% |

Median, p95 and maximum v1 held-out regret are all 0%; 100% of traces are
within 5%. This meets the requested modeled targets.

The important limitation is structural: v1 and the replay oracle share the
same calibrated functional model, and v1 evaluates all fixed policies through
completion. Zero modeled regret therefore establishes selector/oracle
consistency, not independent wall-clock optimality. Topology generalization is
functional-model evidence only.

## 30. Real-Qwen latency validation

The real measurement-to-selector chain is:

```text
Qwen CPU points → nonnegative prefill/decode/mixed coefficients
→ CalibratedQwenCPUServiceModel → predicted policy outcome
```

Residuals are reported above and in `service_time_validation.json`.

## 31. Real-Qwen policy comparison

The identical two-replica/six-request trace ran under all four fixed policies,
v0 and v1:

| Mode | Measured model-forward ms | Steps | Attention calls |
|---|---:|---:|---:|
| Decode-first | 5176.2 | 18 | 672 |
| Prefill-first | 5140.0 | 18 | 672 |
| Balanced | 5084.4 | 18 | 672 |
| SLO-aware | 5179.9 | 18 | 672 |
| Selector v0 | 5127.9 | 18 | 672 |
| Selector v1 | 5084.3 | 18 | 672 |

All generated outputs were identical. Every run had 672/672 attention outputs
entering `o_proj`, zero serving/scheduler/operator fallback, zero repartition
and zero mismatch. A single small trace is cross-layer validation, not
statistical proof that v1 is faster.

## 32–33. Stress, negatives, and tests

Selector v1 completed 1,000 requests in 11,010 steps with zero execution
correctness failure and all runtime mutation/fallback counters zero. The H=8
epoch selection evaluated 32 candidate steps across four clones in 282.8 ms
and switched policy zero times. This overhead is too high for per-step use.

Negative tests reject invalid profiles/scales/weights/coefficients, bad
horizons/continuations, split leakage, unavailable candidates and live-state
clone mutation. Frozen v0 remains unchanged.

Final tests:

- Operator O5: 90;
- Serving S1: 14;
- Serving S2 execution: 19;
- Serving S2.5 calibration: 16;
- focused Python total: 139;
- cross-layer Qwen policy runs: 6;
- MLIR AttentionCPU CTests: 2.

## 34–35. Truth boundary and maturity

```text
Operator Distributed O5
Serving Distributed S1
Serving Distributed S2
Serving Distributed S2.5 request-level selector calibrated
```

This classification is limited to frozen functional CPU replay with focused
real-Qwen validation. Real-Qwen statistical profitability remains unproven.
No S3, P/D separation, KV connector/transfer, PagedAttention, vLLM engine,
TP/PP, GPU, NCCL, or multi-node capability is claimed.

## 36. Resume bullets

- Diagnosed a 41.32%-regret step-level LLM scheduler and separated immediate,
  bounded-horizon, and normalized request-level objectives.
- Added calibrated Qwen CPU mixed-step curves, deferred-request penalties,
  terminal remaining-work cost, and versioned policy epochs while preserving
  exact plan-only execution.
- Reached 100% modeled held-out fixed-policy agreement with zero replay regret,
  validated six real-Qwen policy paths with identical outputs, and documented
  the shared-model oracle limitation.

## 37. S3 recommendation

**Do not begin S3 yet.** First validate selector v1 on repeated wall-clock
request traces with confidence intervals and reduce the 282.8 ms epoch-planning
overhead. The remaining limitation is clearly isolated: modeled profitability
meets the target, but independent real-Qwen profitability is not statistically
established.
