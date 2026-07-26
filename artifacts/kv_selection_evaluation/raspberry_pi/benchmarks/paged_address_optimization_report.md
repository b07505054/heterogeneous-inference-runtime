# Paged KV Address-Generation Optimization Result

Result: **negative**. The scalar pointer-walking variant was Pi-validated but not retained because improvements did not clear measurement noise and short contexts regressed slightly.

- Optimized artifact SHA-256: `24af6d7886b61daa492a643d4d079d6cc622bcaff1beefc0746f4ef8a480b65b`
- Optimized plan SHA-256: `d1647065053f8e0171c3c7ab2eef611cbc00139d33c3e73cc93534ebff8540fd`
- Result JSON SHA-256: `94c7e75c3d6a00665c93bec28b85b8e429fedd67d8149edab490fc719f359aae`
- Stage breakdown JSON SHA-256: `87978015df443f85854a7c057c09635da348fe00c542a8ff077af70238212796`
- Pi: Linux-6.18.34+rpt-rpi-2712-aarch64-with-glibc2.41, core 3, affinity [3]
- Temperature: {'/sys/class/thermal/thermal_zone0/temp': '45750'} -> {'/sys/class/thermal/thermal_zone0/temp': '49600'}; throttling throttled=0x0

## Decode Comparison
| Tokens | Contiguous ms | Token-major archived ms | Page-major archived ms | Pointer-walk ms | Vs page-major | Vs token-major | Vs contiguous | p95 ms | CV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.011634 | 0.021716 | 0.023178 | 0.023315 | -0.59% | -6.86% | 100.41% overhead | 0.023436 | 0.0028 |
| 9 | 0.011952 | 0.023046 | 0.023975 | 0.024118 | -0.59% | -4.45% | 101.80% overhead | 0.024206 | 0.0023 |
| 32 | 0.012837 |  |  | 0.025600 |  |  | 99.43% overhead | 0.025722 | 0.0040 |
| 64 | 0.014200 | 0.031957 | 0.028587 | 0.028492 | 0.33% | 12.16% | 100.64% overhead | 0.028597 | 0.0023 |

## Native Stage Breakdown
| Tokens | Total ms | Setup % | Page % | QK % | Softmax exp/sum % | Normalize % | V % | Remainder % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.000197720 | 18.71 | 18.71 | 4.73 | 17.84 | 6.42 | 5.57 | 9.51 |
| 9 | 0.000405935 | 9.11 | 9.11 | 13.71 | 37.27 | 8.24 | 9.01 | 4.51 |
| 32 | 0.001036940 | 3.57 | 5.30 | 17.01 | 53.41 | 6.91 | 10.21 | 0.06 |
| 64 | 0.001898370 | 1.95 | 1.95 | 18.37 | 55.66 | 8.17 | 10.98 | 0.99 |

## Memory Check
At 9 valid tokens, contiguous allocated 8192 bytes and paged allocated 2048 bytes, preserving the 75% native KV allocation reduction. At 64 tokens both allocated 8192 bytes.

## Decision
The pointer-walking source change was reverted. The retained next step should target scalar softmax/normalization or a page_tokens=8 specialization before NEON/vectorization.
