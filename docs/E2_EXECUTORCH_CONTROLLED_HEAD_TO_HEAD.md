# E2 Controlled Raspberry Pi 5 Comparison: Project Compiler Path vs ExecuTorch XNNPACK

DOCUMENT STATUS: CURRENT E2 FORMAL EVALUATION RESULT

Last verified: 2026-07-13

Truth boundary: E2 is a formal controlled evaluation attempt on Raspberry Pi 5 for FP32 `Y = ReLU(A @ B + bias)`. The frozen correctness gate failed for most non-tiny random-input workloads, so the formal head-to-head comparison is invalidated for correctness. Latency results are retained as diagnostic evidence only and must not be used as a valid performance claim.

## Verdict

`COMPARISON_INVALID_CORRECTNESS_ON_RASPBERRY_PI5_FP32_FUSED_MATMUL_BIAS_RELU_ONLY`

Reason: 288 of 324 formal records failed the preregistered correctness predicate. The failures were small numerical differences, not execution crashes: maximum absolute error was `0.00030517578125`, and maximum relative error was `0.020180140430738364`. The frozen rule required both absolute error <= `1e-3` and relative error <= `1e-4`, which rejected many near-zero outputs. The rule was not changed after sampling.

## Starting States

- Compiler: `b67cd644568e7f53a64370f926e241e4e42ebe10`, branch `master`, clean, ahead 9 of `origin/master`.
- Runtime: `1ab411fab87f43da8c3f4540b4540534c9dbbf2b`, branch `main`, clean, ahead 4 of `origin/main` before E2.
- Capabilities: `aac593da0bdde7a95c38c03920fc4d00b73011db`, branch `main`, clean.
- Pi: `edgeaiplatform`, Raspberry Pi 5 Model B Rev 1.1, Debian 13, aarch64 Cortex-A76, four cores, performance governor.

## Frozen Manifest

- Comparison ID: `E2_RPI5_FP32_FUSED_MATMUL_BIAS_RELU_2026_07_13`
- Embedded manifest hash: `468d0ce47ddd4b702746fb056521148588bfa193c923af801a5fb9a97a53e620`
- Manifest file SHA256: `be5ec5d8be51f31913043fb3cc4f61bfb6cbc13c6504c059f1e5e926300761d1`
- Raw results SHA256: `eb66245c2ae8c1bb419713c4007f62f877af59a4d4c27a96167ec919b3594237`
- Analysis SHA256: `da6b5fd234cb03d1b0dfe035b9bd4ebfc6358e143c4d093664c0218664f1dd21`
- Formal sample count expected: 324
- Formal sample count recorded: 324

The manifest was generated and transferred before formal sampling began. Workloads, formulas, thread modes, warmup/repeat counts, and tolerance were not changed after sampling began.

## Frozen Workload Suite

| Workload | M | N | K | Category | Split | Policy | XNNPACK PTE SHA256 |
|---|---:|---:|---:|---|---|---|---|
| `cal_tiny_16` | 16 | 16 | 16 | tiny | calibration | P-SERIAL | `c36d06ca924f...` |
| `cal_small_square_64` | 64 | 64 | 64 | small_square | calibration | P-4T | `0e080a8e3943...` |
| `cal_medium_square_192` | 192 | 192 | 192 | medium_square | calibration | P-4T | `42f9a7b43705...` |
| `cal_skinny_m_32x512x64` | 32 | 512 | 64 | skinny_m_wide_n | calibration | P-4T | `297de52fc267...` |
| `cal_wide_m_512x32x64` | 512 | 32 | 64 | wide_m_skinny_n | calibration | P-4T | `467c72eb237c...` |
| `cal_small_k_256x256x16` | 256 | 256 | 16 | small_k | calibration | P-4T | `469a70245cd8...` |
| `cal_large_k_64x64x1024` | 64 | 64 | 1024 | large_k | calibration | P-4T | `af60cb152ec7...` |
| `cal_nondivisible_m_137` | 137 | 128 | 64 | nondivisible_m | calibration | P-4T | `f153881a212e...` |
| `eval_tiny_8` | 8 | 8 | 8 | tiny | held_out | P-SERIAL | `0a4c4cbfa381...` |
| `eval_small_square_80` | 80 | 80 | 80 | small_square | held_out | P-4T | `13fa875add43...` |
| `eval_medium_square_256` | 256 | 256 | 256 | medium_square | held_out | P-4T | `98ca7e01d3e6...` |
| `eval_large_square_384` | 384 | 384 | 384 | large_square | held_out | P-4T | `930802c7cfeb...` |
| `eval_skinny_m_48x384x96` | 48 | 384 | 96 | skinny_m_wide_n | held_out | P-4T | `5feb9872e513...` |
| `eval_wide_m_384x48x96` | 384 | 48 | 96 | wide_m_skinny_n | held_out | P-4T | `0969c717f524...` |
| `eval_small_k_320x320x24` | 320 | 320 | 24 | small_k | held_out | P-4T | `6127fc19d4b9...` |
| `eval_large_k_96x96x768` | 96 | 96 | 768 | large_k | held_out | P-4T | `1e7e0f229b44...` |
| `eval_nondivisible_n_151` | 128 | 151 | 64 | nondivisible_n | held_out | P-4T | `e21c5b470486...` |
| `eval_p1c_continuity_128` | 128 | 128 | 128 | medium_square | held_out | P-4T | `da675c4483cc...` |

## Input And Oracle Contract

Inputs were generated once per workload from the frozen manifest rule:

```text
python random.Random(seed).uniform(-3, 3)
A then B then bias, serialized little-endian FP32
```

The independent reference computes `Y = ReLU(A @ B + bias)` in transparent Python FP64 accumulation over the generated values, then serializes FP32 reference bytes. The project output and ExecuTorch output were both compared against the same reference. Tensor hashes are stored per workload in the manifest.

## Timing Boundary Equivalence

| Boundary | Project path | ExecuTorch path | E2 status |
|---|---|---|---|
| load_time | Not exposed by project kernel binary | ExecuTorch runner logs model load time | Not comparable as formal metric |
| cold_first_inference | First kernel repeat in same process | First `method->execute()` after load/input prep | Recorded |
| warm_execution | Kernel-internal repeat timings after 5 warmups | Runner per-iteration `method->execute()` timings after 5 warmups | Primary sampled boundary |
| end_to_end_invocation | Not equivalently exposed without a new persistent adapter | Runner process includes load and setup outside iteration timer | Not used as valid formal comparison |

Process lifetime was not perfectly equivalent: each formal mode/workload/session used one process containing 25 internal invocations. The primary comparison used the internal warm invocation boundary rather than process elapsed time.

## Thread Observability

Thread classification values observed:

`REQUESTED_THREAD_COUNT_OBSERVED_PARTIALLY, VERIFIED_BY_PROJECT_CONTRACT_SELF_REPORT`

- Project modes are classified by the project binary's explicit self-report and contract validation.
- ExecuTorch modes are classified as requested/partially observed. The runner logs threadpool resets for requested/default modes, but E2 did not prove active worker utilization by per-thread CPU-time sampling.
- Therefore E2 does not support a claim that the project has better thread scheduling than ExecuTorch.

## Environment Controls

- Affinity: `taskset -c 0-3` for all E2 modes.
- Governor: performance.
- Throttle states observed: `[('throttled=0x0', 'throttled=0x0')]`.
- No formal sample was silently removed.

## Correctness Results

- Total formal records: 324
- Correctness-passing records: 36
- Correctness-failing records: 288
- Max absolute error: `0.00030517578125`
- Max relative error: `0.020180140430738364`
- NaN/Inf mismatches: none observed in recorded summaries.

Because correctness failed under the frozen predicate, all latency analysis below is diagnostic only.

## Diagnostic Latency Summary

These values are median-of-session-medians for `warm_execution`. They are retained to diagnose the failed run, not as valid performance claims.

| Workload | P-POLICY ms | ET-DEFAULT ms | Project/ET outcome | Project speedup vs ET |
|---|---:|---:|---|---:|
| `cal_tiny_16` | 0.004018 | 0.001444 | executorch_faster | 0.359x |
| `cal_small_square_64` | 0.091389 | 0.020982 | executorch_faster | 0.230x |
| `cal_medium_square_192` | 1.373480 | 0.495027 | executorch_faster | 0.360x |
| `cal_skinny_m_32x512x64` | 0.241175 | 0.087602 | executorch_faster | 0.363x |
| `cal_wide_m_512x32x64` | 0.246907 | 0.085296 | executorch_faster | 0.345x |
| `cal_small_k_256x256x16` | 0.254741 | 0.120463 | executorch_faster | 0.473x |
| `cal_large_k_64x64x1024` | 0.834471 | 0.300230 | executorch_faster | 0.360x |
| `cal_nondivisible_m_137` | 0.250008 | 0.088046 | executorch_faster | 0.352x |
| `eval_tiny_8` | 0.000944 | 0.001037 | project_faster | 1.099x |
| `eval_small_square_80` | 0.137343 | 0.040028 | executorch_faster | 0.291x |
| `eval_medium_square_256` | 3.176670 | 1.143572 | executorch_faster | 0.360x |
| `eval_large_square_384` | 10.601850 | 3.950474 | executorch_faster | 0.373x |
| `eval_skinny_m_48x384x96` | 0.376147 | 0.145334 | executorch_faster | 0.386x |
| `eval_wide_m_384x48x96` | 0.382315 | 0.149648 | executorch_faster | 0.391x |
| `eval_small_k_320x320x24` | 0.526823 | 0.242435 | executorch_faster | 0.460x |
| `eval_large_k_96x96x768` | 1.365005 | 0.489444 | executorch_faster | 0.359x |
| `eval_nondivisible_n_151` | 0.276713 | 0.100649 | executorch_faster | 0.364x |
| `eval_p1c_continuity_128` | 0.436037 | 0.148898 | executorch_faster | 0.341x |

Diagnostic practical outcome, if correctness had passed:

- Geomean project-policy speedup vs ET-default: `0.3818301405286423`
- Win/tie/loss: `{'executorch_faster': 17, 'project_faster': 1, 'tie': 0}`
- Worst project slowdown vs ET-default: `4.355694302123299`

## Project Decision Quality Diagnostic

The project decision-quality analysis is within the project candidate space only, comparing `P-POLICY` against `P-SERIAL` and `P-4T` on the frozen suite.

- Exact match rate: `0.4444444444444444`
- Mean regret percent: `0.10810070007980596`
- Median regret percent: `0.06066625822491358`
- Max regret percent: `1.943844492440605`

Some computed regret values are slightly negative at the workload level because independently sampled mode medians have measurement noise; the preregistered formula was preserved rather than clamped after results were seen.

## Fairness And Confound Audit

| Confound | Status | Note |
|---|---|---|
| Same workload suite | controlled | 18 P1D timed workloads frozen before sampling |
| Same input bytes | controlled | shared raw tensor files generated from manifest hashes |
| Same correctness oracle | controlled | independent reference used for both systems |
| Correctness passed | comparison-invalidating | frozen relative tolerance failed for most non-tiny workloads |
| Same dtype/semantic operation | controlled | FP32 MatMul + Bias + ReLU |
| Same affinity/governor | controlled | `taskset -c 0-3`, performance governor |
| No throttling | controlled | all recorded states `throttled=0x0` |
| ExecuTorch XNNPACK delegation | controlled | all artifacts classified `FULL_REGION_DELEGATED_FUSION_UNKNOWN` |
| Internal XNNPACK fusion | unresolved | full-region delegation proven; internal fusion not proven |
| Thread activity | disclosed | requested threadpool resets observed; active worker utilization not fully proven |
| Timing boundary | measured/disclosed | warm internal invocation sampled; load/end-to-end not equivalent |
| No post-result retuning | controlled | no policy, workload, or tolerance changes after manifest freeze |
| No project kernel in ExecuTorch | controlled | ExecuTorch uses exported graph and XNNPACK delegate |

## Valid Claims

- E2 produced a frozen manifest and complete 324-record formal sample set on Raspberry Pi 5.
- ExecuTorch v1.3.1 XNNPACK and the project binary both executed the same raw FP32 inputs for the frozen workload suite.
- The formal comparison is invalidated by the preregistered correctness predicate.
- Diagnostic latency data suggests XNNPACK default was faster on most sampled workloads, but that is not a valid formal performance claim due to the correctness gate failure.

## Unsupported Claims

- The project beats ExecuTorch generally.
- ExecuTorch beats the project generally.
- Either system is superior across models, ARM devices, quantization, energy, or NPU paths.
- The project has better thread scheduling than ExecuTorch.
- Internal XNNPACK fusion occurred.

## Files

- Frozen manifest: `results/executorch_e2/e2_frozen_manifest.json`
- Raw samples: `results/executorch_e2/e2_raw_results.json`
- Analysis: `results/executorch_e2/e2_analysis.json`
- E2 scripts: `evaluation/executorch_e2/`
- Export reports: `results/executorch_e2/export_reports/`

Large generated `.pte` files, ExecuTorch source, build trees, runner binaries, and virtual environments are not committed.

## E2.1 Follow-Up

E2 remains historically invalid. E2.1 repairs the floating-point correctness predicate and reruns a fresh preregistered comparison in `docs/E2_1_EXECUTORCH_CORRECTNESS_REPAIRED_COMPARISON.md`.
