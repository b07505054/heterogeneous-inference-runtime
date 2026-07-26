# Contiguous vs Page-Major Assembly Report

| Metric | Contiguous | Page-major |
| --- | ---: | ---: |
| branch_count | 40 | 61 |
| division_count | 2 | 3 |
| expf_calls | 1 | 1 |
| fma_count | 4 | 8 |
| instruction_bytes_approx | 1056 | 1636 |
| instruction_count | 264 | 409 |
| integer_multiply_count | 7 | 9 |
| load_count | 60 | 101 |
| neon_or_vector_fp_count | 0 | 0 |
| scalar_fp_count | 16 | 20 |
| sqrtf_calls | 2 | 1 |
| stack_pair_traffic | 37 | 27 |
| store_count | 23 | 25 |

- Contiguous contains NEON/vector FP operands: no
- Page-major contains NEON/vector FP operands: no
- expf call-site references: contiguous=1, page-major=1
- Approximate function-size delta: 580 bytes
