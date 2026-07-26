# Paged KV vs Contiguous KV on Raspberry Pi 5

- Plan hash: `ef606073133d3356f8b6e86db6977c9c0700c1018f39a08c6b89373645fce03e`
- Native artifact hash: `5c31f4102c8a5c8b7263fb70952bf522b9e2eaea9960f9e1405483f25563ed8c`
- Shape: heads=2, head_dim=8, page_tokens=8, capacity=64

## Differential Finding

Direct C++ exported-kernel profiling did not reproduce the runtime-integrated 2x decode gap. At 64 tokens, direct native timing measured contiguous at 0.002522840 ms and page-major at 0.002347030 ms, so page-major was 0.000175810 ms faster inside the exported native functions. The integrated Python/runtime benchmark still measured contiguous at 0.014292710 ms and page-major at 0.028610791 ms.

Largest positive native page-major exclusive delta at 64 tokens: `qk_score_generation_plus_max`, +0.000043240 ms. This is offset by faster page-major `v_accumulation`, -0.000204328 ms. Because the native differential is negative overall and the integrated gap is not visible inside measured native stages, no native optimization was attempted.

## Decode Median Comparison
| Tokens | Contiguous ms | Paged ms | Result | Percent |
| ---: | ---: | ---: | --- | ---: |
| 1 | 0.011641 | 0.023220 | overhead | 99.46% |
| 7 | 0.011849 | 0.023513 | overhead | 98.43% |
| 8 | 0.011888 | 0.023483 | overhead | 97.53% |
| 9 | 0.011955 | 0.023988 | overhead | 100.64% |
| 16 | 0.012318 | 0.024164 | overhead | 96.17% |
| 32 | 0.012838 | 0.025623 | overhead | 99.60% |
| 64 | 0.014293 | 0.028611 | overhead | 100.18% |

## Lifecycle
| Active requests | Path | Median ms | Requests/s | Decode tokens/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | contiguous | 0.660089 | 1514.95 | 13634.52 |
| 1 | paged | 1.189384 | 840.77 | 7566.95 |
| 2 | contiguous | 1.310503 | 1526.13 | 13735.18 |
| 2 | paged | 2.146916 | 931.57 | 8384.12 |
| 4 | contiguous | 2.585413 | 1547.14 | 13924.27 |
| 4 | paged | 4.305563 | 929.03 | 8361.28 |

## Memory Definitions
Native KV payload bytes are computed from compiler contract bytes/token and actual allocation policy. RSS is `/proc/self/status` VmRSS in KiB; peak RSS is `resource.getrusage(...).ru_maxrss`.
