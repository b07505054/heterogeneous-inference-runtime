# Contiguous loop-order optimization

> Closure status: reordered contiguous and page-major paged are production; original contiguous and token-major paged are historical executable baselines. The reordered-control symbol is benchmark-compatibility-only and never compiler-selected. These are single-request CPU measurements, not production serving.

The benchmark-only control is now a distinct compiler-selectable and runtime-dispatched production candidate. It retains the original contiguous KV representation and changes only V accumulation from dimension→token to token→dimension.

| Target | Workload | Original p95 ms | Reordered p95 ms | Page-major p95 ms | Reordered speedup | Page overhead vs reordered |
|---|---|---:|---:|---:|---:|---:|
| host | O1 | 0.006497 | 0.005302 | 0.006363 | 22.54% | 20.01% |
| host | O2 | 0.031217 | 0.019105 | 0.021061 | 63.40% | 10.24% |
| host | O3 | 0.124053 | 0.066866 | 0.072808 | 85.53% | 8.89% |
| host | O4 | 0.259263 | 0.137390 | 0.150905 | 88.71% | 9.84% |
| host | O5 | 0.279189 | 0.142804 | 0.157384 | 95.51% | 10.21% |
| raspberry_pi | O1 | 0.007908 | 0.006697 | 0.008095 | 18.09% | 20.88% |
| raspberry_pi | O2 | 0.034004 | 0.024007 | 0.026059 | 41.64% | 8.55% |
| raspberry_pi | O3 | 0.125312 | 0.076280 | 0.081648 | 64.28% | 7.04% |
| raspberry_pi | O4 | 0.370430 | 0.153296 | 0.199740 | 141.64% | 30.30% |
| raspberry_pi | O5 | 0.365258 | 0.150203 | 0.178351 | 143.18% | 18.74% |

Kernel-only reordered contiguous beats original contiguous and page-major on every O1–O5 workload on both targets. Append+decode selection differs on host O3/O4, where page-major is faster; Pi selects reordered contiguous throughout.

No explicit SIMD, prefetch, continuous batching, GPU code, or new KV representation was added.
