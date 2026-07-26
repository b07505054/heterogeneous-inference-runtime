# Paged KV vs Contiguous KV on Raspberry Pi 5

- Plan hash: `ef606073133d3356f8b6e86db6977c9c0700c1018f39a08c6b89373645fce03e`
- Native artifact hash: `5c31f4102c8a5c8b7263fb70952bf522b9e2eaea9960f9e1405483f25563ed8c`
- Shape: heads=2, head_dim=8, page_tokens=8, capacity=64

## Decode Median Comparison
| Tokens | Contiguous ms | Paged ms | Result | Percent |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.011698 | 0.023178 | overhead | 98.13% |
| 7 | 0.011900 | 0.023456 | overhead | 97.12% |
| 8 | 0.011923 | 0.023523 | overhead | 97.30% |
| 9 | 0.011998 | 0.023975 | overhead | 99.81% |
| 16 | 0.012302 | 0.024266 | overhead | 97.25% |
| 32 | 0.012886 | 0.025609 | overhead | 98.74% |
| 64 | 0.014269 | 0.028587 | overhead | 100.34% |

## Lifecycle
| Active requests | Path | Median ms | Requests/s | Decode tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | contiguous | 0.652415 | 1532.77 | 13794.89 |
| 1 | paged | 1.188757 | 841.22 | 7570.94 |
| 2 | contiguous | 1.288627 | 1552.04 | 13968.36 |
| 2 | paged | 2.120005 | 943.39 | 8490.55 |
| 4 | contiguous | 2.555726 | 1565.11 | 14086.02 |
| 4 | paged | 4.255140 | 940.04 | 8460.36 |

## Memory Definitions
Native KV payload bytes are computed from compiler contract bytes/token and actual allocation policy. RSS is `/proc/self/status` VmRSS in KiB; peak RSS is `resource.getrusage(...).ru_maxrss`.
