# Paged-KV page-major release validation

> Closure status: reordered contiguous and page-major paged are production; original contiguous and token-major paged are historical executable baselines. The reordered-control symbol is benchmark-compatibility-only and never compiler-selected. These are single-request CPU measurements, not production serving.

The prior representation-level conclusion is overturned. Page-major still beats the original contiguous implementation on O2–O5, but the experimental contiguous reordered control is faster than page-major on every workload and target. The advantage is primarily loop ordering/auto-vectorization, not paged representation.

| target | workload | contiguous p95 ms | control p95 ms | token-major p95 ms | page-major p95 ms | page latency reduction vs contiguous | control speedup vs contiguous | page overhead vs control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| host | O1 | 0.006779 | 0.005594 | 0.027101 | 0.006557 | +3.27% | +21.19% | +17.23% |
| host | O2 | 0.031872 | 0.020144 | 0.190730 | 0.021990 | +31.01% | +58.22% | +9.16% |
| host | O3 | 0.123189 | 0.069547 | 0.727222 | 0.073927 | +39.99% | +77.13% | +6.30% |
| host | O4 | 0.276388 | 0.143196 | 1.512460 | 0.155168 | +43.86% | +93.01% | +8.36% |
| host | O5 | 0.260836 | 0.136500 | 1.471708 | 0.162196 | +37.82% | +91.09% | +18.83% |
| raspberry_pi | O1 | 0.007946 | 0.006738 | 0.017655 | 0.008036 | -1.14% | +17.92% | +19.26% |
| raspberry_pi | O2 | 0.033929 | 0.024007 | 0.106737 | 0.026205 | +22.77% | +41.33% | +9.15% |
| raspberry_pi | O3 | 0.123318 | 0.076316 | 0.406539 | 0.081189 | +34.16% | +61.59% | +6.38% |
| raspberry_pi | O4 | 0.363667 | 0.150494 | 0.878583 | 0.198826 | +45.33% | +141.65% | +32.12% |
| raspberry_pi | O5 | 0.361899 | 0.150373 | 0.832764 | 0.167303 | +53.77% | +140.67% | +11.26% |

All candidates perform identical analytical arithmetic: `4 × heads × valid_tokens × head_dim` FLOPs and `heads × head_dim` outputs. All corrected-run checksums agree.

Host PMU counters are unavailable because `perf_event_paranoid=4`; Pi has no `perf` executable. No permissions or packages were changed. Static assembly and compiler vectorization reports support the loop-order explanation.

Corrected kernel-only p95 production winners are: host — O1 page_major, O2 page_major, O3 page_major, O4 page_major, O5 page_major; Raspberry Pi — O1 contiguous, O2 page_major, O3 page_major, O4 page_major, O5 page_major. The reordered control is benchmark-only and is not a compiler candidate. Append+decode retains contiguous for O1 on both targets.

**Next action: C. fix contiguous loop ordering first.** Explicit SIMD is not yet justified until the production contiguous loop incorporates and revalidates the demonstrated scalar/compiler-vectorized ordering improvement.
