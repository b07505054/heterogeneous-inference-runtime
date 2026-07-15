# Host KV-selection evaluation

> Closure status: reordered contiguous and page-major paged are production; original contiguous and token-major paged are historical executable baselines. The reordered-control symbol is benchmark-compatibility-only and never compiler-selected. These are single-request CPU measurements, not production serving.

Exact-target measured-profile evaluation; single-request native FP32 CPU execution.

## Latency

| Workload | Candidate | Page | Median ms | p95 ms | p99 ms | Full loop median ms | Slowdown vs best | Samples | Correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| W1 | contiguous | - | 0.017770 | 0.019056 | 0.021138 | 0.273100 | 0.00% | 100 | PASS |
| W1 | paged | 8 | 0.037939 | 0.039026 | 0.039507 | 0.529451 | 104.80% | 100 | PASS |
| W1 | paged | 16 | 0.039152 | 0.042159 | 0.056745 | 0.541695 | 121.24% | 100 | PASS |
| W1 | paged | 32 | 0.037853 | 0.042639 | 0.056738 | 0.526887 | 123.76% | 100 | PASS |
| W2 | contiguous | - | 0.016680 | 0.017446 | 0.020323 | 0.253133 | 0.00% | 100 | PASS |
| W2 | paged | 8 | 0.043946 | 0.050452 | 0.069128 | 0.616381 | 189.19% | 100 | PASS |
| W2 | paged | 16 | 0.039256 | 0.042031 | 0.043617 | 0.552174 | 140.92% | 100 | PASS |
| W2 | paged | 32 | 0.040367 | 0.044112 | 0.050085 | 0.561355 | 152.85% | 100 | PASS |
| W3 | contiguous | - | 0.039221 | 0.042456 | 0.045900 | 2.076495 | 0.00% | 100 | PASS |
| W3 | paged | 8 | 0.182656 | 0.201505 | 0.227265 | 9.193429 | 374.62% | 100 | PASS |
| W3 | paged | 16 | 0.180954 | 0.192618 | 0.253701 | 9.087478 | 353.69% | 100 | PASS |
| W3 | paged | 32 | 0.180547 | 0.200468 | 0.243392 | 9.077901 | 372.18% | 100 | PASS |
| W4 | contiguous | - | 0.125525 | 0.152406 | 0.157063 | 15.915564 | 0.00% | 100 | PASS |
| W4 | paged | 8 | 0.672029 | 0.753843 | 0.827291 | 82.717901 | 394.63% | 100 | PASS |
| W4 | paged | 16 | 0.660168 | 0.734733 | 0.766583 | 81.617818 | 382.09% | 100 | PASS |
| W4 | paged | 32 | 0.661594 | 0.794267 | 0.922952 | 82.196822 | 421.15% | 100 | PASS |
| W5 | contiguous | - | 0.239711 | 0.300877 | 0.334399 | 59.620744 | 0.00% | 100 | PASS |
| W5 | paged | 8 | 1.383213 | 1.605167 | 2.166679 | 331.266778 | 433.50% | 100 | PASS |
| W5 | paged | 16 | 1.330771 | 1.452060 | 1.526125 | 323.622708 | 382.61% | 100 | PASS |
| W5 | paged | 32 | 1.309547 | 1.445211 | 1.816211 | 320.412003 | 380.33% | 100 | PASS |
| W6 | contiguous | - | 0.045517 | 0.050706 | 0.056882 | 2.151827 | 0.00% | 100 | PASS |
| W6 | paged | 8 | 0.219004 | 0.254937 | 0.274198 | 9.353281 | 402.78% | 100 | PASS |
| W6 | paged | 16 | 0.220434 | 0.249621 | 0.283993 | 9.229182 | 392.29% | 100 | PASS |
| W6 | paged | 32 | 0.219249 | 0.235242 | 0.251650 | 9.085150 | 363.93% | 100 | PASS |
| W7 | contiguous | - | 0.038931 | 0.042381 | 0.044339 | 1.446181 | 0.00% | 100 | PASS |
| W7 | paged | 8 | 0.181776 | 0.201050 | 0.222677 | 5.971946 | 374.39% | 100 | PASS |
| W7 | paged | 16 | 0.177711 | 0.200686 | 0.250572 | 5.833278 | 373.53% | 100 | PASS |
| W7 | paged | 32 | 0.177724 | 0.223600 | 0.252507 | 5.800314 | 427.59% | 100 | PASS |
| W8 | contiguous | - | 0.038577 | 0.044343 | 0.064695 | 1.060234 | 0.00% | 100 | PASS |
| W8 | paged | 8 | 0.202222 | 0.243557 | 0.268153 | 5.243337 | 449.26% | 100 | PASS |
| W8 | paged | 16 | 0.189743 | 0.206630 | 0.249898 | 4.852531 | 365.98% | 100 | PASS |
| W8 | paged | 32 | 0.183989 | 0.213874 | 0.232137 | 4.684102 | 382.32% | 100 | PASS |

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
| W4 | contiguous | 1024 | 256 | 524288 | 2097152 | 2097152 | 1572864 | 0.250000 | 0.000000 | 1 | - |
| W4 | paged | 1024 | 256 | 524288 | 524288 | 2097152 | 0 | 1.000000 | 0.000000 | 32 | 8 |
| W4 | paged | 1024 | 256 | 524288 | 524288 | 2097152 | 0 | 1.000000 | 0.000000 | 16 | 16 |
| W4 | paged | 1024 | 256 | 524288 | 524288 | 2097152 | 0 | 1.000000 | 0.000000 | 8 | 32 |
| W5 | contiguous | 1024 | 512 | 1048576 | 2097152 | 2097152 | 1048576 | 0.500000 | 0.000000 | 1 | - |
| W5 | paged | 1024 | 512 | 1048576 | 1048576 | 2097152 | 0 | 1.000000 | 0.000000 | 64 | 8 |
| W5 | paged | 1024 | 512 | 1048576 | 1048576 | 2097152 | 0 | 1.000000 | 0.000000 | 32 | 16 |
| W5 | paged | 1024 | 512 | 1048576 | 1048576 | 2097152 | 0 | 1.000000 | 0.000000 | 16 | 32 |
| W6 | contiguous | 512 | 80 | 163840 | 1048576 | 1048576 | 884736 | 0.156250 | 0.000000 | 1 | - |
| W6 | paged | 512 | 80 | 163840 | 163840 | 1048576 | 0 | 1.000000 | 0.000000 | 10 | 8 |
| W6 | paged | 512 | 80 | 163840 | 163840 | 1048576 | 0 | 1.000000 | 0.000000 | 5 | 16 |
| W6 | paged | 512 | 80 | 163840 | 196608 | 1048576 | 32768 | 0.833333 | 0.166667 | 3 | 32 |
| W7 | contiguous | 512 | 64 | 131072 | 1048576 | 1048576 | 917504 | 0.125000 | 0.000000 | 1 | - |
| W7 | paged | 512 | 64 | 131072 | 131072 | 1048576 | 0 | 1.000000 | 0.000000 | 8 | 8 |
| W7 | paged | 512 | 64 | 131072 | 131072 | 1048576 | 0 | 1.000000 | 0.000000 | 4 | 16 |
| W7 | paged | 512 | 64 | 131072 | 131072 | 1048576 | 0 | 1.000000 | 0.000000 | 2 | 32 |
| W8 | contiguous | 4096 | 64 | 131072 | 8388608 | 8388608 | 8257536 | 0.015625 | 0.000000 | 1 | - |
| W8 | paged | 4096 | 64 | 131072 | 131072 | 8388608 | 0 | 1.000000 | 0.000000 | 8 | 8 |
| W8 | paged | 4096 | 64 | 131072 | 131072 | 8388608 | 0 | 1.000000 | 0.000000 | 4 | 16 |
| W8 | paged | 4096 | 64 | 131072 | 131072 | 8388608 | 0 | 1.000000 | 0.000000 | 2 | 32 |

## Compiler selections

| Workload | Objective | Best legal | Selected | Executed | Reason | Selected p95 | Best p95 | Latency regret ms | Memory regret bytes | Score regret |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| W1 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.019056 | 0.019056 | 0.000000 | 0 | 0.000000 |
| W1 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.019056 | 0.019056 | 0.000000 | 0 | 0.000000 |
| W1 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.019056 | 0.019056 | 0.000000 | 0 | 0.000000 |
| W2 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.017446 | 0.017446 | 0.000000 | 0 | 0.000000 |
| W2 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.017446 | 0.017446 | 0.000000 | 0 | 0.000000 |
| W2 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.017446 | 0.017446 | 0.000000 | 0 | 0.000000 |
| W3 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.042456 | 0.042456 | 0.000000 | 0 | 0.000000 |
| W3 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.042456 | 0.042456 | 0.000000 | 0 | 0.000000 |
| W3 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.042456 | 0.042456 | 0.000000 | 0 | 0.000000 |
| W4 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.152406 | 0.152406 | 0.000000 | 0 | 0.000000 |
| W4 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.152406 | 0.152406 | 0.000000 | 0 | 0.000000 |
| W4 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.152406 | 0.152406 | 0.000000 | 0 | 0.000000 |
| W5 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.300877 | 0.300877 | 0.000000 | 0 | 0.000000 |
| W5 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.300877 | 0.300877 | 0.000000 | 0 | 0.000000 |
| W5 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.300877 | 0.300877 | 0.000000 | 0 | 0.000000 |
| W6 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.050706 | 0.050706 | 0.000000 | 0 | 0.000000 |
| W6 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.050706 | 0.050706 | 0.000000 | 0 | 0.000000 |
| W6 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.050706 | 0.050706 | 0.000000 | 0 | 0.000000 |
| W7 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.042381 | 0.042381 | 0.000000 | 0 | 0.000000 |
| W7 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.042381 | 0.042381 | 0.000000 | 0 | 0.000000 |
| W7 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.042381 | 0.042381 | 0.000000 | 0 | 0.000000 |
| W8 | latency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.044343 | 0.044343 | 0.000000 | 0 | 0.000000 |
| W8 | memory_efficiency | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.044343 | 0.044343 | 0.000000 | 0 | 0.000000 |
| W8 | balanced | contiguous | contiguous | contiguous | exact_target_workload_measured_profile_minimum_objective_score | 0.044343 | 0.044343 | 0.000000 | 0 | 0.000000 |

## Paged page-size results

| Workload | Best p95 page | Best owned-memory page | Best balanced-score page |
|---|---|---|---|
| W1 | 8 | 8,16,32 | 8 |
| W2 | 16 | 8,16,32 | 16 |
| W3 | 16 | 8,16,32 | 16 |
| W4 | 16 | 8,16,32 | 16 |
| W5 | 32 | 8,16,32 | 32 |
| W6 | 32 | 8,16 | 32 |
| W7 | 16 | 8,16,32 | 16 |
| W8 | 16 | 8,16,32 | 16 |

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
| W4 | D1 | 16 | 8 | 134 | 126 | 1575.00% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D1 | 32 | 16 | 269 | 253 | 1581.25% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D1 | 64 | 32 | 538 | 506 | 1581.25% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D1 | 128 | 64 | 1077 | 1013 | 1582.81% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D2 | 16 | 8 | 35 | 27 | 337.50% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D2 | 32 | 16 | 70 | 54 | 337.50% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D2 | 64 | 32 | 141 | 109 | 340.62% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D2 | 128 | 64 | 282 | 218 | 340.62% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D3 | 16 | 8 | 17 | 9 | 112.50% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D3 | 32 | 16 | 35 | 19 | 118.75% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D3 | 64 | 32 | 71 | 39 | 121.88% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
| W4 | D3 | 128 | 64 | 142 | 78 | 121.88% | 1024 | 16 | contiguous=1024×2048; paged=E[ceil(tokens/16)×16×2048] |
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
| W7 | D1 | 16 | 16 | 134 | 118 | 737.50% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D1 | 32 | 32 | 269 | 237 | 740.62% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D1 | 64 | 64 | 538 | 474 | 740.62% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D1 | 128 | 128 | 1077 | 949 | 741.41% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D2 | 16 | 16 | 35 | 19 | 118.75% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D2 | 32 | 32 | 70 | 38 | 118.75% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D2 | 64 | 64 | 141 | 77 | 120.31% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D2 | 128 | 128 | 282 | 154 | 120.31% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D3 | 16 | 16 | 17 | 1 | 6.25% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D3 | 32 | 32 | 35 | 3 | 9.38% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D3 | 64 | 64 | 71 | 7 | 10.94% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
| W7 | D3 | 128 | 128 | 142 | 14 | 10.94% | 512 | 16 | contiguous=512×2048; paged=E[ceil(tokens/16)×16×2048] |
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
