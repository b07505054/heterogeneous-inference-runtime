# Paged KV vs Contiguous KV on Raspberry Pi 5

- Plan hash: `dec84059d0697393ec4c048302e33059977870cffcbf4f2a403232624c03e05d`
- Native artifact hash: `9a884b32ed05c70866cacbfa9aa31ecb95227ce4e8162ca62abe1da414554eaa`
- Shape: heads=2, head_dim=8, page_tokens=8, capacity=64

## Decode Median Comparison
| Tokens | Contiguous ms | Paged ms | Result | Percent |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.011589 | 0.023343 | overhead | 101.43% |
| 7 | 0.011914 | 0.023441 | overhead | 96.75% |
| 8 | 0.011870 | 0.023518 | overhead | 98.12% |
| 9 | 0.012077 | 0.024020 | overhead | 98.89% |
| 16 | 0.012276 | 0.024247 | overhead | 97.52% |
| 32 | 0.012933 | 0.025529 | overhead | 97.40% |
| 64 | 0.014236 | 0.028584 | overhead | 100.79% |

## Lifecycle
| Active requests | Path | Median ms | Requests/s | Decode tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | contiguous | 0.657147 | 1521.73 | 13695.57 |
| 1 | paged | 1.184479 | 844.25 | 7598.28 |
| 2 | contiguous | 1.303821 | 1533.95 | 13805.58 |
| 2 | paged | 2.152051 | 929.35 | 8364.11 |
| 4 | contiguous | 2.569549 | 1556.69 | 14010.24 |
| 4 | paged | 4.281130 | 934.33 | 8408.99 |

## Memory Definitions
Native KV payload bytes are computed from compiler contract bytes/token and actual allocation policy. RSS is `/proc/self/status` VmRSS in KiB; peak RSS is `resource.getrusage(...).ru_maxrss`.

## Softmax Optimization Result

Attempted optimization: replace per-V-contribution `w[token] / sum` with one scalar reciprocal per head and `w[token] * inv_sum` inside `hir_cpu_attention_decode_paged_kv_page_major_fp32`.

Final source status: reverted after Pi validation because the 32-token and 64-token improvements were within measurement noise. Profiler/test additions and this negative-result evidence were retained.

| Tokens | Contiguous ms | Token-major paged ms | Original page-major ms | Reciprocal-fused page-major ms | vs original page-major | vs token-major | remaining overhead vs contiguous |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.011589 | 0.022012 | 0.023178 | 0.023343 | -0.71% | -6.04% | 101.43% |
| 9 | 0.012077 | 0.023154 | 0.023975 | 0.024020 | -0.19% | -3.74% | 98.89% |
| 16 | 0.012276 | 0.023994 | 0.024266 | 0.024247 | 0.07% | -1.06% | 97.52% |
| 32 | 0.012933 | 0.026608 | 0.025609 | 0.025529 | 0.31% | 4.06% | 97.40% |
| 64 | 0.014236 | 0.032151 | 0.028587 | 0.028584 | 0.01% | 11.10% | 100.79% |

Classification: `NEGATIVE_RESULT`. The dominant measured scalar softmax cost remains `expf`/exp-sum, not reciprocal/division.
