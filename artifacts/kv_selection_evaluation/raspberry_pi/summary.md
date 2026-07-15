# Raspberry Pi KV-selection evaluation

> Closure status: reordered contiguous and page-major paged are production; original contiguous and token-major paged are historical executable baselines. The reordered-control symbol is benchmark-compatibility-only and never compiler-selected. These are single-request CPU measurements, not production serving.

Exact-target measured-profile evaluation; single-request native FP32 CPU execution.

## Latency

| Workload | Candidate | Page | Median ms | p95 ms | p99 ms | Full loop median ms | Slowdown vs best | Samples | Correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| W1 | contiguous | - | 0.026556 | 0.026759 | 0.026963 | 0.409286 | 0.00% | 100 | PASS |
| W1 | paged | 8 | 0.041805 | 0.042166 | 0.042260 | 0.618683 | 57.58% | 100 | PASS |
| W1 | paged | 16 | 0.041704 | 0.042259 | 0.042666 | 0.615859 | 57.92% | 100 | PASS |
| W1 | paged | 32 | 0.041667 | 0.041962 | 0.042111 | 0.615795 | 56.81% | 100 | PASS |
| W2 | contiguous | - | 0.027426 | 0.027741 | 0.027888 | 0.421082 | 0.00% | 100 | PASS |
| W2 | paged | 8 | 0.046528 | 0.046870 | 0.047000 | 0.691812 | 68.96% | 100 | PASS |
| W2 | paged | 16 | 0.043444 | 0.043926 | 0.047648 | 0.645183 | 58.34% | 100 | PASS |
| W2 | paged | 32 | 0.042797 | 0.043092 | 0.043185 | 0.633007 | 55.34% | 100 | PASS |
| W3 | contiguous | - | 0.054528 | 0.054981 | 0.061055 | 2.983821 | 0.00% | 100 | PASS |
| W3 | paged | 8 | 0.135888 | 0.136703 | 0.138407 | 7.001898 | 148.64% | 100 | PASS |
| W3 | paged | 16 | 0.133601 | 0.134055 | 0.137647 | 6.897306 | 143.82% | 100 | PASS |
| W3 | paged | 32 | 0.133037 | 0.134685 | 0.136685 | 6.856325 | 144.97% | 100 | PASS |
| W5 | contiguous | - | 0.385119 | 0.391721 | 0.403739 | 86.659889 | 0.00% | 100 | PASS |
| W5 | paged | 8 | 0.967496 | 0.985404 | 0.994292 | 218.025176 | 151.56% | 100 | PASS |
| W5 | paged | 16 | 0.930228 | 0.939829 | 0.948663 | 214.284367 | 139.92% | 100 | PASS |
| W5 | paged | 32 | 0.858274 | 0.862237 | 0.869978 | 207.589274 | 120.12% | 100 | PASS |
| W6 | contiguous | - | 0.063296 | 0.063889 | 0.066537 | 3.146571 | 0.00% | 100 | PASS |
| W6 | paged | 8 | 0.163804 | 0.164907 | 0.167351 | 7.290934 | 158.11% | 100 | PASS |
| W6 | paged | 16 | 0.161360 | 0.162277 | 0.165037 | 7.130991 | 154.00% | 100 | PASS |
| W6 | paged | 32 | 0.158573 | 0.160667 | 0.161888 | 7.023481 | 151.48% | 100 | PASS |
| W8 | contiguous | - | 0.056352 | 0.056981 | 0.058407 | 1.555354 | 0.00% | 100 | PASS |
| W8 | paged | 8 | 0.169167 | 0.170055 | 0.172018 | 4.596288 | 198.44% | 100 | PASS |
| W8 | paged | 16 | 0.150314 | 0.151518 | 0.154610 | 4.003123 | 165.91% | 100 | PASS |
| W8 | paged | 32 | 0.140916 | 0.141240 | 0.141981 | 3.714670 | 147.87% | 100 | PASS |

## Memory

| Workload | Candidate | Capacity tokens | Valid tokens | Logical bytes | Request-owned bytes | Reserved process/pool bytes | Unused owned bytes | Utilization | Fragmentation | Allocated pages/objects | Page |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | contiguous | 32 | 32 | 16384 | 16384 | 16384 | 0 | 1.000000 | 0.000000 | 1 | - |
| W1 | paged | 32 | 32 | 16384 | 16384 | 16384 | 0 | 1.000000 | 0.000000 | 4 | 8 |
| W1 | paged | 32 | 32 | 16384 | 16384 | 16384 | 0 | 1.000000 | 0.000000 | 2 | 16 |
| W1 | paged | 32 | 32 | 16384 | 16384 | 16384 | 0 | 1.000000 | 0.000000 | 1 | 32 |
| W2 | contiguous | 512 | 32 | 16384 | 262144 | 262144 | 245760 | 0.062500 | 0.000000 | 1 | - |
| W2 | paged | 512 | 32 | 16384 | 16384 | 262144 | 0 | 1.000000 | 0.000000 | 4 | 8 |
| W2 | paged | 512 | 32 | 16384 | 16384 | 262144 | 0 | 1.000000 | 0.000000 | 2 | 16 |
| W2 | paged | 512 | 32 | 16384 | 16384 | 262144 | 0 | 1.000000 | 0.000000 | 1 | 32 |
| W3 | contiguous | 256 | 128 | 131072 | 262144 | 262144 | 131072 | 0.500000 | 0.000000 | 1 | - |
| W3 | paged | 256 | 128 | 131072 | 131072 | 262144 | 0 | 1.000000 | 0.000000 | 16 | 8 |
| W3 | paged | 256 | 128 | 131072 | 131072 | 262144 | 0 | 1.000000 | 0.000000 | 8 | 16 |
| W3 | paged | 256 | 128 | 131072 | 131072 | 262144 | 0 | 1.000000 | 0.000000 | 4 | 32 |
| W5 | contiguous | 1024 | 512 | 1048576 | 2097152 | 2097152 | 1048576 | 0.500000 | 0.000000 | 1 | - |
| W5 | paged | 1024 | 512 | 1048576 | 1048576 | 2097152 | 0 | 1.000000 | 0.000000 | 64 | 8 |
| W5 | paged | 1024 | 512 | 1048576 | 1048576 | 2097152 | 0 | 1.000000 | 0.000000 | 32 | 16 |
| W5 | paged | 1024 | 512 | 1048576 | 1048576 | 2097152 | 0 | 1.000000 | 0.000000 | 16 | 32 |
| W6 | contiguous | 512 | 80 | 163840 | 1048576 | 1048576 | 884736 | 0.156250 | 0.000000 | 1 | - |
| W6 | paged | 512 | 80 | 163840 | 163840 | 1048576 | 0 | 1.000000 | 0.000000 | 10 | 8 |
| W6 | paged | 512 | 80 | 163840 | 163840 | 1048576 | 0 | 1.000000 | 0.000000 | 5 | 16 |
| W6 | paged | 512 | 80 | 163840 | 196608 | 1048576 | 32768 | 0.833333 | 0.166667 | 3 | 32 |
| W8 | contiguous | 4096 | 64 | 131072 | 8388608 | 8388608 | 8257536 | 0.015625 | 0.000000 | 1 | - |
| W8 | paged | 4096 | 64 | 131072 | 131072 | 8388608 | 0 | 1.000000 | 0.000000 | 8 | 8 |
| W8 | paged | 4096 | 64 | 131072 | 131072 | 8388608 | 0 | 1.000000 | 0.000000 | 4 | 16 |
| W8 | paged | 4096 | 64 | 131072 | 131072 | 8388608 | 0 | 1.000000 | 0.000000 | 2 | 32 |

## Compiler selections

| Workload | Objective | Best legal | Selected | Executed | Reason | Selected p95 | Best p95 | Latency regret ms | Memory regret bytes | Score regret |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| W1 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.026759 | 0.026759 | 0.000000 | 0 | 0.000000 |
| W1 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.026759 | 0.026759 | 0.000000 | 0 | 0.000000 |
| W1 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.026759 | 0.026759 | 0.000000 | 0 | 0.000000 |
| W2 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.027741 | 0.027741 | 0.000000 | 0 | 0.000000 |
| W2 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.027741 | 0.027741 | 0.000000 | 0 | 0.000000 |
| W2 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.027741 | 0.027741 | 0.000000 | 0 | 0.000000 |
| W3 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.054981 | 0.054981 | 0.000000 | 0 | 0.000000 |
| W3 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.054981 | 0.054981 | 0.000000 | 0 | 0.000000 |
| W3 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.054981 | 0.054981 | 0.000000 | 0 | 0.000000 |
| W5 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.391721 | 0.391721 | 0.000000 | 0 | 0.000000 |
| W5 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.391721 | 0.391721 | 0.000000 | 0 | 0.000000 |
| W5 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.391721 | 0.391721 | 0.000000 | 0 | 0.000000 |
| W6 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.063889 | 0.063889 | 0.000000 | 0 | 0.000000 |
| W6 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.063889 | 0.063889 | 0.000000 | 0 | 0.000000 |
| W6 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.063889 | 0.063889 | 0.000000 | 0 | 0.000000 |
| W8 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.056981 | 0.056981 | 0.000000 | 0 | 0.000000 |
| W8 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.056981 | 0.056981 | 0.000000 | 0 | 0.000000 |
| W8 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.056981 | 0.056981 | 0.000000 | 0 | 0.000000 |

## Paged page-size results

| Workload | Best p95 page | Best owned-memory page | Best balanced-score page |
|---|---|---|---|
| W1 | 32 | 8,16,32 | 32 |
| W2 | 32 | 8,16,32 | 32 |
| W3 | 16 | 8,16,32 | 16 |
| W5 | 32 | 8,16,32 | 32 |
| W6 | 32 | 8,16 | 16 |
| W8 | 32 | 8,16,32 | 32 |

## Formula-based admission analysis, not real concurrent serving

| Workload | Distribution | MiB | Contiguous | Paged | Improvement | Relative | Capacity | Page | Byte formula |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| W1 | D1 | 16 | 1024 | 538 | -486 | -47.46% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D1 | 32 | 2048 | 1077 | -971 | -47.41% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D1 | 64 | 4096 | 2155 | -1941 | -47.39% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D1 | 128 | 8192 | 4311 | -3881 | -47.38% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D2 | 16 | 1024 | 141 | -883 | -86.23% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D2 | 32 | 2048 | 282 | -1766 | -86.23% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D2 | 64 | 4096 | 564 | -3532 | -86.23% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D2 | 128 | 8192 | 1129 | -7063 | -86.22% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D3 | 16 | 1024 | 71 | -953 | -93.07% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D3 | 32 | 2048 | 142 | -1906 | -93.07% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D3 | 64 | 4096 | 284 | -3812 | -93.07% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W1 | D3 | 128 | 8192 | 568 | -7624 | -93.07% | 32 | 16 | contiguous=32×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D1 | 16 | 64 | 538 | 474 | 740.62% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D1 | 32 | 128 | 1077 | 949 | 741.41% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D1 | 64 | 256 | 2155 | 1899 | 741.80% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D1 | 128 | 512 | 4311 | 3799 | 741.99% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D2 | 16 | 64 | 141 | 77 | 120.31% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D2 | 32 | 128 | 282 | 154 | 120.31% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D2 | 64 | 256 | 564 | 308 | 120.31% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D2 | 128 | 512 | 1129 | 617 | 120.51% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D3 | 16 | 64 | 71 | 7 | 10.94% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D3 | 32 | 128 | 142 | 14 | 10.94% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D3 | 64 | 256 | 284 | 28 | 10.94% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W2 | D3 | 128 | 512 | 568 | 56 | 10.94% | 512 | 16 | contiguous=512×512; paged=E[ceil(tokens/16)×16×512] |
| W3 | D1 | 16 | 64 | 269 | 205 | 320.31% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D1 | 32 | 128 | 538 | 410 | 320.31% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D1 | 64 | 256 | 1077 | 821 | 320.70% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D1 | 128 | 512 | 2155 | 1643 | 320.90% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D2 | 16 | 64 | 70 | 6 | 9.38% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D2 | 32 | 128 | 141 | 13 | 10.16% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D2 | 64 | 256 | 282 | 26 | 10.16% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D2 | 128 | 512 | 564 | 52 | 10.16% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D3 | 16 | 64 | 35 | -29 | -45.31% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D3 | 32 | 128 | 71 | -57 | -44.53% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D3 | 64 | 256 | 142 | -114 | -44.53% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W3 | D3 | 128 | 512 | 284 | -228 | -44.53% | 256 | 16 | contiguous=256×1024; paged=E[ceil(tokens/16)×16×1024] |
| W5 | D1 | 16 | 8 | 134 | 126 | 1575.00% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D1 | 32 | 16 | 269 | 253 | 1581.25% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D1 | 64 | 32 | 538 | 506 | 1581.25% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D1 | 128 | 64 | 1077 | 1013 | 1582.81% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D2 | 16 | 8 | 35 | 27 | 337.50% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D2 | 32 | 16 | 70 | 54 | 337.50% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D2 | 64 | 32 | 141 | 109 | 340.62% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D2 | 128 | 64 | 282 | 218 | 340.62% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D3 | 16 | 8 | 17 | 9 | 112.50% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D3 | 32 | 16 | 35 | 19 | 118.75% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D3 | 64 | 32 | 71 | 39 | 121.88% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W5 | D3 | 128 | 64 | 142 | 78 | 121.88% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D1 | 16 | 16 | 134 | 118 | 737.50% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D1 | 32 | 32 | 269 | 237 | 740.62% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D1 | 64 | 64 | 538 | 474 | 740.62% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D1 | 128 | 128 | 1077 | 949 | 741.41% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D2 | 16 | 16 | 35 | 19 | 118.75% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D2 | 32 | 32 | 70 | 38 | 118.75% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D2 | 64 | 64 | 141 | 77 | 120.31% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D2 | 128 | 128 | 282 | 154 | 120.31% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D3 | 16 | 16 | 17 | 1 | 6.25% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D3 | 32 | 32 | 35 | 3 | 9.38% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D3 | 64 | 64 | 71 | 7 | 10.94% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W6 | D3 | 128 | 128 | 142 | 14 | 10.94% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D1 | 16 | 2 | 134 | 132 | 6600.00% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D1 | 32 | 4 | 269 | 265 | 6625.00% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D1 | 64 | 8 | 538 | 530 | 6625.00% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D1 | 128 | 16 | 1077 | 1061 | 6631.25% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D2 | 16 | 2 | 35 | 33 | 1650.00% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D2 | 32 | 4 | 70 | 66 | 1650.00% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D2 | 64 | 8 | 141 | 133 | 1662.50% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D2 | 128 | 16 | 282 | 266 | 1662.50% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D3 | 16 | 2 | 17 | 15 | 750.00% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D3 | 32 | 4 | 35 | 31 | 775.00% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D3 | 64 | 8 | 71 | 63 | 787.50% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
| W8 | D3 | 128 | 16 | 142 | 126 | 787.50% | 4096 | 16 | contiguous=4096×2048; paged=E[ceil(tokens/16)×16×2048] |
