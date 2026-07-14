# E3 Live-Compiler Same-XNNPACK Comparison

DOCUMENT STATUS: CURRENT SAME-XNNPACK LIVE-COMPILER COMPARISON RESULT

Last verified: 2026-07-14.

## Identity

Compiler commit: `ba388a93eeccd11045f8c1f2eb950ede2601bc88`.
Runtime commit: `53c80e2c11101ec7b8db2e73f978e220c054d9a1`.
ExecuTorch: tag `v1.3.1`, commit `e2f18eb23c45bd22ca332b0b8b49a81de304b472`.
XNNPACK: `1adaa7c709d4839d29e1f219cb962b01c9e6a905`.
Source manifest SHA: `7c7f7187c47285eef7f1ecb06f6bc364b8e7081730acff1695f1896e2aee45f3`.
Recursive submodule manifest SHA: `4e1842168f25264794d09cf25172eee7583fc38d00a6980d76050354490b1a06`.
Common runner SHA: `adef50a17a4aebc953583638a0ba7d573fc53df4023f9183280887a07fd17341`.
Adapter contract SHA: `115cbfe876336525a470f47f7055e1658cd51daae2159a4a6f6893ee2dc8dbbc`.

## Question Answered

Using the same Pi, same `.pte`, same runner, same ExecuTorch, same XNNPACK, same input bytes, same timing boundary, same process lifetime, and same correctness oracle, how does a live-Compiler-selected XNNPACK configuration compare with ExecuTorch default configuration?

## Candidate Discovery

Artifact: `results/executorch_e3/discovery/e3_analysis.json`.

- Records: 162.
- Workloads: 18.
- Candidates: X1, X4, and default diagnostic.
- Warmups/repeats/sessions: 5 warmups, 20 timed repeats, 3 sessions.
- Correctness failures: 0.
- Incomplete timing records: 0.
- Tie threshold: 5%.
- Winner counts: X1 16, X4 2.
- Candidate-space verdict: `XNNPACK_ONE_STATIC_WINNER`.
- Policy recommendation: `static_X1`.
- X1 max regret within XNNPACK candidate space: `0.41590327673692074%`.

## Formal E3C Result

Artifact: `results/executorch_e3/formal/e3_formal_analysis.json`.

- Records: 60.
- Correctness failures: 0.
- Incomplete timing records: 0.
- Win/tie/loss: project policy faster 2, tied 8, ExecuTorch default faster 0.
- Geomean default/project ratio: `1.0316855419881399x`.
- Median default/project ratio: `1.028692183573202x`.
- Best project speedup vs default: `1.0800578187990661x`.
- Worst project slowdown vs default: `1.0039973063469534x`.
- Session median coefficient of variation: `0.31590315870358804%` median, `3.231855098379434%` max.

Per-win effect sizes:

- `eval_small_square_80`: default/project ratio `1.0800578187990661x`.
- `eval_wide_m_384x48x96`: default/project ratio `1.068471140773699x`.

Formal comparison verdict: `PROJECT_POLICY_FASTER_WITH_SAME_XNNPACK_STACK`.

## Interpretation

This is not a complex shape-aware policy victory. The evidence found one static XNNPACK winner for this target/workload scope. The Compiler contribution was candidate exposure, feasibility/provenance validation, calibration, static X1 selection, compiler-owned contract generation, and execution through the exact same XNNPACK stack.

Do not say “beats ExecuTorch.” Do not generalize to other models, devices, energy, NPU, or external backends.
