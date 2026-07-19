# Serving Distributed S2.6 Wall-Clock Report

## 1–4. Verdict, preserved evidence, repository state, freeze

**S2.6 wall-clock validation failed. Serving S2.5 remains calibrated only to
the functional model.**

Preserved results:

- selector v0: 0% agreement and 41.32% original mean regret;
- selector v1: 100% modeled agreement and 0% modeled regret;
- limitation: v1 and its replay oracle share the same functional model.

The frozen v1 digest, features, objective, normalization, coefficients,
full-trace horizon, terminal cost, epoch and tie-break are recorded in
`selector_v1_freeze.json`. It was not changed during final evaluation.
Repository state is recorded separately; both HEADs remained unchanged.

## 5–7. Split, environment, and methodology

Development, validation, final-test and planning-overhead IDs/seeds are
non-overlapping. The final set contains mixed-interactive and contention
traces; no final result changed either selector.

Measurements ran on an Intel i5-10210U: four physical/eight logical cores,
SMT enabled, one NUMA node, `powersave` governor, 7.0.0-27 Linux, Python
3.12.13, PyTorch 2.11.0 CPU execution, Qwen2.5-0.5B FP32. Process affinity was
0-7; neither process nor worker affinity was applied. Frequency was not
controlled. No background model server was found.

Each policy/trace used three warm-up and ten measured runs. Policy order
alternated forward/reverse and rotated by two positions. Immutable weights and
warmed code were reused; scheduler and cache state were recreated. Intervals
use paired Student-t 95% intervals. The practical-equivalence margin was frozen
at 2%.

## 8–9. Future information and selector modes

Frozen v1 uses future arrivals, prompt/output metadata and trace termination.
It is now explicitly `offline_trace`, not online.

`selector_v1_fast` supports `epoch` and `online_step`; `online_state_view`
removes unarrived requests. It may use arrived prompt lengths, declared
expected outputs, current progress/age and historical curves only.

## 10–11. Repeated wall clock and uncertainty

The objective is mean wall TTFT + 0.25×mean wall E2E.

### Mixed interactive

| Policy | Median objective ms | Mean 95% CI ms | Planning median ms |
|---|---:|---:|---:|
| decode-first | 1553.97 | [1549.05,1563.08] | 2.012 |
| prefill-first | 1878.01 | [1864.29,1910.93] | 2.064 |
| balanced | 1556.69 | [1549.24,1612.34] | 2.026 |
| SLO-aware | 1552.40 | [1546.57,1585.27] | 2.027 |
| v0 | 1740.06 | [1729.54,1775.03] | 2.046 |
| v1 frozen | 1880.41 | [1868.22,1926.21] | 16.646 |
| v1 fast | 1696.29 | [1689.29,1736.26] | 0.996 |

Decode-first, balanced and SLO-aware are practically/statistically
indistinguishable. Both selectors are conclusively worse.

### Contention

| Policy | Median objective ms | Mean 95% CI ms | Planning median ms |
|---|---:|---:|---:|
| decode-first | 1935.86 | [1928.93,1957.60] | 2.046 |
| prefill-first | 1948.29 | [1945.84,1970.61] | 2.061 |
| balanced | 1951.11 | [1945.18,1999.32] | 2.204 |
| SLO-aware | 1975.91 | [1972.79,2020.08] | 1.984 |
| v0 | 1967.44 | [1953.58,2016.44] | 2.255 |
| v1 frozen | 1959.07 | [1950.59,2006.88] | 16.431 |
| v1 fast | 1848.09 | [1839.92,1883.45] | 1.099 |

Fast v1 wins all ten paired contention runs, but this does not offset its mixed
trace failure.

## 12–13. Wall oracle and regret

The independent `wall_clock_policy_oracle` selects the lowest median measured
fixed-policy objective:

- mixed: SLO-aware, tied practically with decode-first and balanced;
- contention: decode-first, tied practically with prefill-first and balanced.

Mixed regret: v0 12.09%, frozen v1 21.13%, fast v1 9.27%. Paired-difference
intervals exclude zero for all three.

Contention regret: v0 1.63%, frozen v1 1.20%, fast v1 0%. V0/frozen v1 are
confidence-aware equivalent; fast v1 is better in all paired runs.

No selector is confidence-aware consistent on both traces.

## 14. Modeled versus measured ranking

The functional model chose prefill-first on both traces. Wall clock chose
SLO-aware and decode-first.

| Trace | Spearman | Pairwise accuracy | Top agreement |
|---|---:|---:|---|
| Mixed | -0.80 | 0% | no |
| Contention | 0.40 | 50% | no |

Mixed centered prediction error reached 244.42 objective units and reversed
the entire ordering. Thus model ranking independently failed.

## 15. Planning breakdown

Frozen v1 cProfile is dominated by deep-cloned full-policy continuations and
request-state execution. Fast-selector instrumentation over 1,000 small-state
calls found snapshot copying dominant.

Small-state fast planning: median 0.076 ms, mean 0.080 ms, maximum 0.718 ms.
This meets the preferred microbenchmark target.

## 16–21. Fast selector and optimizations

Fast v1 adds:

- arrived-only compact snapshot;
- adaptive H=1/2/4/8 risk classification;
- phase/legality and identical-plan pruning;
- analytical deferred/terminal work;
- four-step event-aware epochs;
- zero rollout clones;
- hot-path versus evidence logging.

Pruning never removes the sole legal candidate. Live-state mutation tests pass.

Evidence logging increased 1,000-call time from 86.56 to 95.76 ms and median
from 0.078 to 0.082 ms. This is logging savings, not an algorithmic claim.

Large-state stress reveals scaling limits:

- epoch: 10,501 steps, 2,626 planning calls, actual-selection median 27.40 ms,
  p95 36.81 ms, 999 switches;
- online: 10,501 calls, median 28.41 ms, p95 37.63 ms, 999 switches.

Snapshot copying across 1,000 arrived requests dominates. Neither large-state
mode meets online targets, despite the small-state microbenchmark.

## 22. Inclusive versus exclusive costs

Frozen offline-v1 planning occurs before execution and adds roughly 16.5 ms
per trace. V0/fast planning occurs on the scheduler path; corrected exclusive
and inclusive summaries are both preserved. Planning cost does not explain the
hundreds of milliseconds of mixed-policy regret.

## 23–24. Real Qwen and practical equivalence

The matrix contains 140 measured policy/trace runs plus 42 warm-ups. Across
measured runs, 59,520 attention-layer outputs entered their corresponding
`o_proj`; generated outputs matched within every policy/trace group, and all
runtime mutation counters were zero.

These repeated runs use real eager Qwen attention. The separate S2.5 policy
runs establish compiler-attention/operator-plan provenance for each policy.

Practical equivalence is CI overlap or ≤2% objective difference. This avoids
forcing unique fixed-policy winners under measurement noise.

## 25. Generalization

Wall-clock validation covers two distinct arrival/prompt/output compositions,
including the model’s high-error mixed case. It does not cover all topologies,
objective profiles or core allocations. The broader S2.5 generalization
remains modeled-only.

## 26–27. Stress, negatives, tests

Epoch and online stress each completed 1,000 requests and 10,501 steps with
zero runtime correctness failure, rebuild, override or fallback. The cost
failure is selector scalability, not execution correctness.

Negative tests cover frozen-config mutation, final-set isolation, future
leakage, offline/online misuse, invalid horizons, pruning safety, clone
isolation, required epoch reselection, fixed equivalence margin and confidence
sample size.

Test totals:

- Operator O5 90;
- Serving S1 14;
- Serving S2 19;
- Serving S2.5 16;
- Serving S2.6 11;
- focused Python total 150;
- MLIR AttentionCPU CTests 2.

## 28–29. Truth boundary and classification

```text
Operator Distributed O5
Serving Distributed S1
Serving Distributed S2
Serving Distributed S2.5 calibrated to functional model
Serving S2.6 wall-clock validation failed
```

No S3, P/D separation, KV transfer, PagedAttention, vLLM engine, TP/PP, GPU,
NCCL or multi-node claim is made.

## 30. Resume bullets

- Built a counterbalanced 140-run real-Qwen wall-clock policy study with paired
  confidence intervals and an independent measured fixed-policy oracle.
- Identified complete modeled-to-wall policy-ranking reversal despite zero
  replay regret, preventing an unsupported scheduler-profitability claim.
- Reduced small-state selector planning to 0.076 ms median with arrived-only
  snapshots and deterministic pruning, then exposed 28 ms large-state scaling
  overhead through 10,501-step stress tests.

## 31. S3 recommendation

**Do not begin S3.** Refit the service/ranking model using independent
wall-clock traces and replace O(N-request) snapshot copying with incremental
arrays or structural sharing. Revalidate mixed-interactive regret and
large-state epoch p95 before architectural expansion.
