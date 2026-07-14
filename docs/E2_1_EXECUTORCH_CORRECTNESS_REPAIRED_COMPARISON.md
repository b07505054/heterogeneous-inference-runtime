# E2.1 Correctness-Repaired Implementation-Stack Comparison

DOCUMENT STATUS: HISTORICAL VALID IMPLEMENTATION-STACK COMPARISON, NOT COMPILER-ONLY COMPARISON

Last verified: 2026-07-14.

## Verdicts

Correctness-method verdict: `PASSED_CORRECTNESS_PREDICATE_REPAIR`.

Comparison classification: `IMPLEMENTATION_STACK_COMPARISON`.

Invalid interpretation: E2.1 does not show that the ExecuTorch compiler is better than the project compiler. The project side did not invoke the live Compiler.

## Why E2.1 Is Not Compiler-Only

E2.1 repaired E2 correctness using:

```text
abs(actual - expected) <= 1e-3 + 1e-4 * abs(expected)
```

It collected a complete 324-record suite with zero correctness failures. However, the project side hardcoded `THRESHOLD = 262144` and `PROJECT_KERNEL_ID = portable_fused_matmul_bias_relu_bm32_bn128_bk32` in `evaluation/executorch_e2_1/e21_tools.py`. It launched the project scalar/native C++ binary directly. ExecuTorch used `.pte` artifacts, XNNPACK, and the ExecuTorch runner/pthreadpool stack.

Therefore Runtime, kernel implementation, thread implementation, dispatch path, and provenance differed. E2.1 compares implementation stacks, not compiler decision quality.

## Recomputed Metrics

Artifact: `results/executorch_e2_1/e21_analysis.json`.

- Expected records: 324.
- Actual records: 324.
- Correctness failures: 0.
- Project policy geomean speedup vs ExecuTorch default: `0.3800262702319555x`.
- Equivalent project slowdown: about `2.631397x`.
- Win/tie/loss: project faster 0, tied 1, ExecuTorch faster 17.
- Worst project slowdown vs ExecuTorch default: `4.314358334931901x`.

## Valid Narrow Claim

On the frozen Raspberry Pi 5 FP32 fused MatMul + Bias + ReLU suite, the project portable scalar/native execution stack was geometrically about 2.63x slower than the ExecuTorch/XNNPACK stack.

## Preserved Historical Boundary

Raw E2.1 evidence is preserved unchanged. E3 supersedes E2.1 for live-Compiler same-XNNPACK decision comparison.
