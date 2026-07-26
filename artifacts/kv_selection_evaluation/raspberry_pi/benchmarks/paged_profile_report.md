# Paged KV Optimization Profile

Original plan: `523ba15f0d8323f05c51f934e3b38fe044f328bed2ec2d4f5cfd25e6a938e97a`
Optimized plan: `ef606073133d3356f8b6e86db6977c9c0700c1018f39a08c6b89373645fce03e`
Native artifact: `5c31f4102c8a5c8b7263fb70952bf522b9e2eaea9960f9e1405483f25563ed8c`

Unstable rows: original historical 53/168, stabilized original 79/280, optimized 90/280.

## Bottleneck

hir_cpu_attention_decode_paged_kv_fp32 calls paged_addr once per token for K scores and once per token per output dimension for V accumulation; paged_addr performs token/page division, token modulo, block-table load, and validation. For h=2,d=8,valid=64 this is 1152 paged_addr calls per decode.

hir_cpu_attention_decode_paged_kv_page_major_fp32 validates and caches physical page IDs once per logical page, then iterates contiguous tokens inside each page with pointer increments; for valid=64 this is 8 block-table loads per decode.

## Focus Metrics

| Operation | Tokens | Contiguous ms | Original paged ms | Optimized paged ms | Opt vs original | Opt vs contiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_attention | 1 | 0.011698 | 0.021716 | 0.023178 | -6.31% | 98.13% |
| decode_attention | 9 | 0.011998 | 0.023046 | 0.023975 | -3.87% | 99.81% |
| decode_attention | 64 | 0.014269 | 0.031957 | 0.028587 | 11.79% | 100.34% |
| append_boundary | 8 | 0.011657 | 0.034648 | 0.034685 | -0.11% | 197.53% |
| append_inside | 7 | 0.011676 | 0.026065 | 0.026130 | -0.25% | 123.79% |

## Lifecycle

| Active requests | Contiguous ms | Original paged ms | Optimized paged ms | Opt vs original | Opt vs contiguous |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.652137 | 1.182294 | 1.188757 | -0.54% | 82.29% |
| 2 | 1.283980 | 2.120774 | 2.120005 | 0.04% | 65.11% |
| 4 | 2.564282 | 4.229714 | 4.255140 | -0.60% | 65.94% |
