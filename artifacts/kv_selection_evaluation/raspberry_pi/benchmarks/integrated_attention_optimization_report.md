# Paged KV vs Contiguous KV on Raspberry Pi 5

- Plan hash: `ef606073133d3356f8b6e86db6977c9c0700c1018f39a08c6b89373645fce03e`
- Native artifact hash: `5c31f4102c8a5c8b7263fb70952bf522b9e2eaea9960f9e1405483f25563ed8c`
- Shape: heads=2, head_dim=8, page_tokens=8, capacity=64

## Decode Median Comparison
| Tokens | Contiguous ms | Paged ms | Result | Percent |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.011744 | 0.018840 | overhead | 60.42% |
| 7 | 0.011867 | 0.019034 | overhead | 60.40% |
| 8 | 0.011975 | 0.019057 | overhead | 59.14% |
| 9 | 0.012014 | 0.019406 | overhead | 61.54% |
| 16 | 0.012262 | 0.019669 | overhead | 60.40% |
| 32 | 0.012938 | 0.020714 | overhead | 60.11% |
| 64 | 0.014254 | 0.022683 | overhead | 59.13% |

## Lifecycle
| Active requests | Path | Median ms | Requests/s | Decode tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | contiguous | 0.663296 | 1507.62 | 13568.60 |
| 1 | paged | 1.150222 | 869.40 | 7824.58 |
| 2 | contiguous | 1.300971 | 1537.31 | 13835.82 |
| 2 | paged | 2.053285 | 974.05 | 8766.44 |
| 4 | contiguous | 2.608443 | 1533.48 | 13801.34 |
| 4 | paged | 3.962719 | 1009.41 | 9084.67 |

## Memory Definitions
Native KV payload bytes are computed from compiler contract bytes/token and actual allocation policy. RSS is `/proc/self/status` VmRSS in KiB; peak RSS is `resource.getrusage(...).ru_maxrss`.
