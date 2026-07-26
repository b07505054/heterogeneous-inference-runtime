# Paged KV vs Contiguous KV on Raspberry Pi 5

- Plan hash: `523ba15f0d8323f05c51f934e3b38fe044f328bed2ec2d4f5cfd25e6a938e97a`
- Native artifact hash: `5c31f4102c8a5c8b7263fb70952bf522b9e2eaea9960f9e1405483f25563ed8c`
- Shape: heads=2, head_dim=8, page_tokens=8, capacity=64

## Decode Median Comparison
| Tokens | Contiguous ms | Paged ms | Result | Percent |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.011667 | 0.022012 | overhead | 88.68% |
| 7 | 0.012018 | 0.022398 | overhead | 86.36% |
| 8 | 0.012111 | 0.022648 | overhead | 87.00% |
| 9 | 0.012028 | 0.023154 | overhead | 92.50% |
| 16 | 0.012432 | 0.023994 | overhead | 93.00% |
| 32 | 0.013099 | 0.026608 | overhead | 103.13% |
| 64 | 0.014537 | 0.032151 | overhead | 121.17% |

## Lifecycle
| Active requests | Path | Median ms | Requests/s | Decode tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | contiguous | 0.659416 | 1516.49 | 13648.45 |
| 1 | paged | 1.173591 | 852.09 | 7668.77 |
| 2 | contiguous | 1.286989 | 1554.01 | 13986.13 |
| 2 | paged | 2.134330 | 937.06 | 8433.56 |
| 4 | contiguous | 2.578543 | 1551.26 | 13961.38 |
| 4 | paged | 4.223485 | 947.09 | 8523.77 |

## Memory Definitions
Native KV payload bytes are computed from compiler contract bytes/token and actual allocation policy. RSS is `/proc/self/status` VmRSS in KiB; peak RSS is `resource.getrusage(...).ru_maxrss`.
