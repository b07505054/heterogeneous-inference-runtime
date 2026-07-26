# Paged KV vs Contiguous KV on Raspberry Pi 5

- Plan hash: `523ba15f0d8323f05c51f934e3b38fe044f328bed2ec2d4f5cfd25e6a938e97a`
- Native artifact hash: `5c31f4102c8a5c8b7263fb70952bf522b9e2eaea9960f9e1405483f25563ed8c`
- Shape: heads=2, head_dim=8, page_tokens=8, capacity=64

## Decode Median Comparison
| Tokens | Contiguous ms | Paged ms | Result | Percent |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.011612 | 0.021716 | overhead | 87.02% |
| 7 | 0.012023 | 0.022208 | overhead | 84.71% |
| 8 | 0.011903 | 0.022426 | overhead | 88.41% |
| 9 | 0.011976 | 0.023046 | overhead | 92.43% |
| 16 | 0.012239 | 0.023757 | overhead | 94.11% |
| 32 | 0.012898 | 0.026424 | overhead | 104.86% |
| 64 | 0.014319 | 0.031957 | overhead | 123.18% |

## Lifecycle
| Active requests | Path | Median ms | Requests/s | Decode tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | contiguous | 0.652137 | 1533.42 | 13800.78 |
| 1 | paged | 1.182294 | 845.81 | 7612.32 |
| 2 | contiguous | 1.283980 | 1557.66 | 14018.92 |
| 2 | paged | 2.120774 | 943.05 | 8487.47 |
| 4 | contiguous | 2.564282 | 1559.89 | 14039.02 |
| 4 | paged | 4.229714 | 945.69 | 8511.21 |

## Memory Definitions
Native KV payload bytes are computed from compiler contract bytes/token and actual allocation policy. RSS is `/proc/self/status` VmRSS in KiB; peak RSS is `resource.getrusage(...).ru_maxrss`.
