# E2.1 Correctness-Repaired ExecuTorch XNNPACK Comparison

DOCUMENT STATUS: CURRENT E2.1 FORMAL EVALUATION RESULT

Last verified: 2026-07-13

Truth boundary: E2.1 is a fresh preregistered experiment. It does not reinterpret or overwrite E2. E2 remains historically invalid because its independent relative-error gate rejected near-zero outputs despite small absolute error.

## Verdicts

Correctness-method verdict:

`PASSED_CORRECTNESS_PREDICATE_REPAIR`

Narrow performance verdict:

`EXECUTORCH_FASTER_ON_FROZEN_PROJECT_BASELINE_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY`

This means ExecuTorch XNNPACK was faster for this frozen Raspberry Pi 5 FP32 fused MatMul + Bias + ReLU suite. It is not a general ExecuTorch superiority claim.

## Starting States

- Compiler: `b67cd644568e7f53a64370f926e241e4e42ebe10`, branch `master`, clean, ahead 9.
- Runtime before E2.1: `f0a0dab34d80776973377c5d864a30f156f55b11`, branch `main`, clean, ahead 5.
- Capabilities: `aac593da0bdde7a95c38c03920fc4d00b73011db`, branch `main`.
- Pi: Raspberry Pi 5, Debian 13, aarch64 Cortex-A76, performance governor.

## E2 Failure Forensics

- E2 records: 324
- E2 failed records: 288
- E2 max absolute error: `0.00030517578125`
- E2 max relative error: `0.020180140430738364`
- E2 failures with absolute error within E2.1 `atol` but old relative gate exceeded: 288

Limitation: E2 raw records retained aggregate correctness metrics, not per-output index/value. E2.1 therefore preserves E2 unchanged and uses fresh output validation under a new manifest.

The failures were evenly distributed across systems and modes, consistent with the old predicate rather than one implementation uniquely failing.

## Correctness Predicate

E2.1 uses the standard mixed allclose form:

```text
abs(actual - expected) <= 1e-3 + 1e-4 * abs(expected)
```

Rationale:

- avoids independent relative-error amplification near zero;
- preserves a strict absolute cap for small outputs;
- remains far below semantic-scale corruptions in this workload;
- accounts for FP32 reduction-order variation without accepting wrong graph semantics.

## Negative Controls

All negative controls were rejected:

```json
{
  "injected_inf": false,
  "injected_nan": false,
  "near_zero_plus_2e_3": false,
  "one_element_plus_1e_2": false,
  "systematic_scale_1_percent": false,
  "transposed_output": false,
  "without_relu": false,
  "wrong_bias_zero_bias": false,
  "wrong_seed": false
}
```

The predicate rejected one-element `+1e-2`, wrong bias, no ReLU, transposed output, wrong seed, NaN, Inf, 1% scaling, and a near-zero `+2e-3` corruption.

## Frozen Manifest

- Comparison ID: `E2_1_RPI5_FP32_FUSED_MATMUL_BIAS_RELU_2026_07_13`
- Embedded manifest hash: `91403dc26fe0a472db7a81457f976757acce9147e15e5fd888369d103449e82d`
- Manifest file SHA256: `a1a7c7441db9799636187c51107d31d86011a5a77c0d47a87368e62dedbb3da1`
- Raw results SHA256: `87a298881b1419c4b3a884ac04390beae76244f31af214f1628e919c1e32a990`
- Analysis SHA256: `fd951943458c3b344052cd801dccd847248fc7b98952e804e9b733c9db28ba48`

Formal samples postdate this manifest. Workloads, formulas, tolerance, and modes were not changed after freeze.

## Workload Suite

| Workload | M | N | K | Split | Category | E2.1 seed | Policy |
|---|---:|---:|---:|---|---|---:|---|
| `cal_tiny_16` | 16 | 16 | 16 | calibration | tiny | 250001 | P-SERIAL |
| `cal_small_square_64` | 64 | 64 | 64 | calibration | small_square | 250002 | P-4T |
| `cal_medium_square_192` | 192 | 192 | 192 | calibration | medium_square | 250003 | P-4T |
| `cal_skinny_m_32x512x64` | 32 | 512 | 64 | calibration | skinny_m_wide_n | 250004 | P-4T |
| `cal_wide_m_512x32x64` | 512 | 32 | 64 | calibration | wide_m_skinny_n | 250005 | P-4T |
| `cal_small_k_256x256x16` | 256 | 256 | 16 | calibration | small_k | 250006 | P-4T |
| `cal_large_k_64x64x1024` | 64 | 64 | 1024 | calibration | large_k | 250007 | P-4T |
| `cal_nondivisible_m_137` | 137 | 128 | 64 | calibration | nondivisible_m | 250008 | P-4T |
| `eval_tiny_8` | 8 | 8 | 8 | held_out | tiny | 260001 | P-SERIAL |
| `eval_small_square_80` | 80 | 80 | 80 | held_out | small_square | 260002 | P-4T |
| `eval_medium_square_256` | 256 | 256 | 256 | held_out | medium_square | 260003 | P-4T |
| `eval_large_square_384` | 384 | 384 | 384 | held_out | large_square | 260004 | P-4T |
| `eval_skinny_m_48x384x96` | 48 | 384 | 96 | held_out | skinny_m_wide_n | 260005 | P-4T |
| `eval_wide_m_384x48x96` | 384 | 48 | 96 | held_out | wide_m_skinny_n | 260006 | P-4T |
| `eval_small_k_320x320x24` | 320 | 320 | 24 | held_out | small_k | 260007 | P-4T |
| `eval_large_k_96x96x768` | 96 | 96 | 768 | held_out | large_k | 260008 | P-4T |
| `eval_nondivisible_n_151` | 128 | 151 | 64 | held_out | nondivisible_n | 260009 | P-4T |
| `eval_p1c_continuity_128` | 128 | 128 | 128 | held_out | medium_square | 260010 | P-4T |

Inputs use fresh E2.1 seeds: `P1D seed + 210000`. Each workload records hashes for A, B, bias, FP64 reference, and FP32 diagnostic reference.

## Input And Semantic Identity

Both systems consumed the same raw little-endian FP32 A/B/bias files generated from the frozen manifest. The `.pte` reports show graph inputs `a`, `b`, and `bias`; the artifacts encode shape/graph, not frozen tensor values. Project and ExecuTorch outputs were both compared to the independent reference, never to each other.

## Independent References

E2.1 records both:

- FP64-accumulation reference followed by FP32 output, used for the correctness gate;
- FP32 step-rounded diagnostic reference, retained in raw records.

Fresh formal records passed against the FP64 reference under the mixed predicate.

## Timing And Process Boundary

Warm timing is the internal repeated invocation boundary:

- Project: native kernel dispatch loop inside `portable_fused_matmul_bias_relu`, after inputs are already files and output buffer exists.
- ExecuTorch: official `executor_runner` per-iteration `method->execute()` timing, after model load and input preparation per invocation.

Load and end-to-end boundaries are not treated as equivalent formal metrics in E2.1. Process lifetime remains one process per workload/mode/session with 25 invocations, 5 warmups, 20 timed repeats.

## Formal Completeness And Correctness

- Expected records: 324
- Actual records: 324
- Correctness failures: 0
- Max E2.1 absolute error vs FP64 reference: `0.000335693359375`
- Max E2.1 relative error vs FP64 reference: `0.4855107058517014`
- Throttle states: `[('throttled=0x0', 'throttled=0x0')]`

## Performance Results

Values are median of session medians for warm execution.

| Workload | P-POLICY ms | ET-DEFAULT ms | Result | Project speedup vs ET | Project regret % |
|---|---:|---:|---|---:|---:|
| `cal_tiny_16` | 0.003982 | 0.001445 | executorch_faster | 0.363x | 0.025 |
| `cal_small_square_64` | 0.089963 | 0.020852 | executorch_faster | 0.232x | -0.746 |
| `cal_medium_square_192` | 1.369970 | 0.500749 | executorch_faster | 0.366x | -0.189 |
| `cal_skinny_m_32x512x64` | 0.242287 | 0.088027 | executorch_faster | 0.363x | 0.330 |
| `cal_wide_m_512x32x64` | 0.247241 | 0.085685 | executorch_faster | 0.347x | 0.591 |
| `cal_small_k_256x256x16` | 0.254861 | 0.122574 | executorch_faster | 0.481x | 0.769 |
| `cal_large_k_64x64x1024` | 0.831166 | 0.296565 | executorch_faster | 0.357x | -0.080 |
| `cal_nondivisible_m_137` | 0.251009 | 0.087861 | executorch_faster | 0.350x | 0.236 |
| `eval_tiny_8` | 0.000926 | 0.000945 | tie | 1.021x | -1.907 |
| `eval_small_square_80` | 0.138565 | 0.039806 | executorch_faster | 0.287x | 0.938 |
| `eval_medium_square_256` | 3.175340 | 1.124591 | executorch_faster | 0.354x | -0.086 |
| `eval_large_square_384` | 10.599050 | 3.879024 | executorch_faster | 0.366x | -0.051 |
| `eval_skinny_m_48x384x96` | 0.375203 | 0.145148 | executorch_faster | 0.387x | 0.052 |
| `eval_wide_m_384x48x96` | 0.380564 | 0.149648 | executorch_faster | 0.393x | -0.600 |
| `eval_small_k_320x320x24` | 0.527241 | 0.243018 | executorch_faster | 0.461x | -0.278 |
| `eval_large_k_96x96x768` | 1.366915 | 0.484971 | executorch_faster | 0.355x | 0.097 |
| `eval_nondivisible_n_151` | 0.276778 | 0.100334 | executorch_faster | 0.363x | 0.040 |
| `eval_p1c_continuity_128` | 0.434046 | 0.148879 | executorch_faster | 0.343x | -0.266 |

Summary:

- Geomean project-policy speedup vs ET-default: `0.3800262702319555`
- Win/tie/loss: `{'executorch_faster': 17, 'project_faster': 0, 'tie': 1}`
- Worst project slowdown vs ET-default: `4.314358334931901`
- Best project speedup vs ET-default: `1.0205183585313176`

## Project Internal Decision Metrics

- Exact match rate: `0.0`
- Mean regret percent: `-0.06257480690673205`
- Median regret percent: `-0.01313697984232868`
- Max regret percent: `0.9375170730819038`

## Kernel-Versus-Policy Interpretation

E2.1 separates two facts:

1. The project policy still selects within its own serial/4-thread candidate space with low regret.
2. ExecuTorch XNNPACK delivers lower absolute latency on nearly every workload in this suite, consistent with a stronger optimized CPU implementation/delegate rather than a direct contradiction of project policy quality.

This is an implementation/kernel-quality result, not proof that the project architecture or policy concept is invalid.

## Fairness Audit

| Confound | Status | Note |
|---|---|---|
| E2 preserved | controlled | E2 files and verdict unchanged |
| Old E2 samples reused | controlled | no; E2.1 collected fresh raw samples |
| Same input bytes | controlled | shared generated files and manifest hashes |
| Same semantics | controlled | FP32 MatMul + Bias + ReLU |
| Independent oracle | controlled | FP64 reference plus FP32 diagnostic reference |
| Correctness | controlled | 324/324 pass under frozen E2.1 predicate |
| Same Pi/governor/affinity | controlled | Pi 5, performance, `taskset -c 0-3` |
| Throttling | controlled | all recorded states `throttled=0x0` |
| XNNPACK artifact validity | controlled | shape-specific `.pte`; graph inputs not constants |
| Thread behavior | disclosed | ExecuTorch thread count remains requested/partially observed, not active-worker proof |
| Timing boundary | disclosed | warm internal invocation only; load/end-to-end not equivalent |
| Project policy retuning | controlled | none |
| Project kernel in ExecuTorch | controlled | none |

## Valid Claims

- On Raspberry Pi 5, for this frozen FP32 fused MatMul + Bias + ReLU suite, ExecuTorch v1.3.1 XNNPACK default warm execution was faster than the project policy outcome by the recorded metrics.
- The project policy maintained low internal regret within its own scalar portable CPU serial/4-thread candidate space.
- The prior E2 invalidity was caused by an unsuitable independent relative-error predicate near zero, not by a semantic mismatch found in E2.1.

## Invalid Claims

- General ExecuTorch superiority.
- General project inferiority.
- Better or worse behavior across models, ARM devices, quantization, NPU, energy, or production reliability.
- A pure thread-scheduling comparison against ExecuTorch, because ExecuTorch active worker utilization remains partially observed.

## Files

- Forensics: `results/executorch_e2_1/e2_failure_forensics.json`
- Negative controls: `results/executorch_e2_1/e21_negative_controls.json`
- Frozen manifest: `results/executorch_e2_1/e21_frozen_manifest.json`
- Raw samples: `results/executorch_e2_1/e21_raw_results.json`
- Analysis: `results/executorch_e2_1/e21_analysis.json`
- Tooling: `evaluation/executorch_e2_1/e21_tools.py`

Large `.pte` files, binaries, build trees, source checkouts, wheels, and virtualenvs are not committed.
