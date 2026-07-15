# Cross-target KV-selection comparison

> Closure status: reordered contiguous and page-major paged are production; original contiguous and token-major paged are historical executable baselines. The reordered-control symbol is benchmark-compatibility-only and never compiler-selected. These are single-request CPU measurements, not production serving.

Host and Raspberry Pi remain separate measured lookup identities.

| Workload | Host winner | Pi winner | Host best paged page | Pi best paged page | Host paged overhead | Pi paged overhead |
|---|---|---|---:|---:|---:|---:|
| W1 | contiguous | contiguous | 8 | 32 | 104.80% | 56.81% |
| W2 | contiguous | contiguous | 16 | 32 | 140.92% | 55.34% |
| W3 | contiguous | contiguous | 16 | 16 | 353.69% | 143.82% |
| W5 | contiguous | contiguous | 32 | 32 | 380.33% | 120.12% |
| W6 | contiguous | contiguous | 32 | 32 | 363.93% | 151.48% |
| W8 | contiguous | contiguous | 16 | 32 | 365.98% | 147.87% |

Required lookup key: target identity, CPU identity, workload identity, candidate identity, page size, and objective.
