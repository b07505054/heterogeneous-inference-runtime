# Serving Distributed S2.10 Policy-Level Validation Report

## 1. Executive verdict

**Serving Distributed S2.10: conservative wall-clock policy selector
validated; exact policy ranking remains imperfect.**

The preregistered risk-aware selector passed tie-aware/regret targets and beat
the frozen robust fixed baseline. The unrestricted pairwise selector did not.
Prospective rollout fidelity remains poor, and the focused compiler-attention
shape selected serial execution, leaving multi-worker event provenance absent.
S3 is not recommended.

## 2–3. Preserved evidence and repository state

All S2.6–S2.9 evidence remains unchanged. S2.9 reconstruction and service-model
metrics, static profitability failure, adaptive rejection, eager boundary, and
freeze limitation are preserved.

Both repository HEADs remained runtime
`34aee51fef08dc447a6a52d938b4867d60eeef70` and compiler
`dbf7329392bd2c70fa6ef25e359b277d171b3082`. No destructive Git or artifact
operation occurred.

## 4. Policy-level error propagation

Per-run artifacts record signed/absolute residual sums, variance, lag-1
autocorrelation, and early/late residual means. Combined with S2.9 substitution,
the evidence shows that small step errors accumulate and that phase/position
correlation matters; aggregate MAE alone is insufficient.

## 5. Residual substitution

Actual sequence plus actual latency matches direct timestamps. Actual sequence
plus the v2 shape model improves latency prediction but still misranks
arrival-burst and prefix-heavy. A prospective predicted-sequence plus empirical
bucket pipeline was not completed. Both latency residual and sequence
prediction therefore remain material.

## 6. Rollout fidelity

The aggregate initial estimate predicted 1.75–6.1 steps while fixed-policy
execution required 10–18. The error exists before the first execution and comes
from omitted chunk boundaries, readiness transitions, and mixed sequencing.
Exact rollout ranking is not validated.

## 7. Frontier sufficiency

On a 1,000-request stress state, K=4/8/16/32 produced stable decisions.
K=8 summary median was 0.0231 ms. K=32 remained 0.0691 ms. This validates
control-plane scaling, not real-Qwen ranking equivalence across K.

## 8. Objective audit

The frozen objective remains `mean TTFT + 0.25 × mean E2E`. No weight was
changed after observing results. TPOT/goodput contributions are not part of
this scalar; consequently this stage does not claim a multi-objective serving
optimum.

## 9–10. Pairwise data and uncertainty

Measured S2.8/S2.9 fixed-policy outcomes produce six pairwise examples per
trace with a frozen 2% tie margin. Uncertainty combines service residual,
support, OOD distance, rollout uncertainty, and frontier truncation.

## 11. Robust baseline

Using development evidence only:

- `decode_first`: 0.67% mean regret, 2.32% p95/worst;
- `slo_aware`: 1.12% mean;
- `chunked_balanced`: 2.76% mean;
- `prefill_first`: 16.68% mean.

`decode_first` was frozen as the robust default.

## 12–14. Selector v4 and no-regression gate

Pairwise v4 selects the lowest score. Risk-aware v4 departs from
`decode_first` only when improvement exceeds 0.12 and uncertainty is at most
0.35. It executed the default on 102/108 measured steps, with four
prefill-first and two SLO-aware departures. No adaptive selector was added.

## 15–17. Registry, preregistration, and freeze

Trace definitions, policies, order, sample counts, objective, metrics,
thresholds, v4 hashes, K=8, and gate parameters were written and hashed before
execution. The v1 preregistration failed during its first warm-up because token
IDs exceeded vocabulary; it produced no measured result and was explicitly
invalidated. V2 changed only token IDs, retained all workload shapes and
thresholds, was rehashed, and restarted from warm-up zero.

## 18. Benchmark methodology

Four new traces, seven policies, one warm-up, and two measured repetitions were
run in forward/reverse order. Two repetitions are below the preferred sample
size, so confidence intervals are broad. Generated outputs remained equivalent
within each trace and runtime counters remained zero.

## 19–21. Final ranking, regret, and baseline comparison

| Selector | Exact | Tie-aware | Mean regret | Median | p95/max |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 static | 25% | 25% | 4.32% | 4.66% | 7.96% |
| v4 pairwise | 25% | 50% | 2.11% | 2.01% | 4.40% |
| v4 risk-aware | 0% | **100%** | **0.70%** | **0.51%** | **1.67%** |
| robust `decode_first` | — | — | 1.30% | — | 3.74% |

Risk-aware v4 beat the robust baseline on mean and p95 regret and passed all
declared regret/tie-aware thresholds. Exact-winner rate is intentionally low
because near-equivalent choices are accepted. A full pairwise-ranking-accuracy
metric for the dynamic selector was not produced, so exact ranking remains
imperfect.

## 22. Compiler-attention provenance

The focused real-Qwen serving run produced 672 compiler operator-plan attention
invocations and 672/672 outputs entering `o_proj`, with zero fallback,
mismatch, repartition, serving reroute/override, or scheduler mutation.
Generated outputs were recorded.

Worker-event count was zero because the compiler profitably selected serial
native AVX2 for every focused shape. No forced parallel override was used.
Thus compiler-plan provenance is closed, but the requested positive
multi-worker-event count is unsupported for this selected workload.

## 23. Scalability

At 1,000 requests: K=8 summary 0.0231 ms median; pairwise v4 0.0089 ms median;
risk-aware v4 0.0085 ms median / 0.0091 ms p95. The preserved 5,000-request
summary remains 0.0684 ms median. All targets pass.

## 24–25. Stress, negatives, and tests

The preserved 1,000-request/10,501-step stress remains zero-failure. New tests
cover freeze mutation, legal policies, robust fallback, uncertainty, and the
absence of adaptive state. The final regression run passed **305 focused
Python tests** and **2/2 MLIR AttentionCPU CTests**.

## 26–27. Truth boundary and maturity

```text
Serving Distributed S2.10
conservative wall-clock policy selector validated
exact policy ranking remains imperfect
```

This is single-node CPU functional serving. It is not S3, vLLM production,
multi-GPU, TP/PP, P/D disaggregation, or multi-node execution.

## 28. Resume bullets

- Preregistered and evaluated a conservative policy selector that achieved
  100% tie-aware agreement and 0.70% mean regret, outperforming a 1.30%
  robust fixed-policy baseline.
- Quantified rollout underprediction (1.75–6.1 predicted versus 10–18 actual
  steps) and contained it with an uncertainty-aware no-regression gate used on
  94.4% of measured decisions.
- Closed focused Qwen compiler-attention provenance for 672 operator plans and
  672 `o_proj` consumptions with zero fallback or candidate mismatch while
  preserving sub-0.01 ms selector overhead.

## 29. Recommendation

**Do not begin S3 yet.** Increase final repetitions, implement a faithful
prospective policy rollout, produce dynamic pairwise ranking accuracy, and
obtain a naturally selected parallel compiler-attention workload with positive
worker-event provenance.
