# Runtime-owned contiguous FP32 KV validation

This is an operator-level, single-process CPU result. The compiler fixes a
`[batch, heads, capacity_tokens, head_dim]` FP32 layout, strides, capacity,
byte sizes, ABI, entry points, and compatible attention kernels. The runtime
owns one allocation and the live valid-token count.

Decode semantics are append-then-attend: append the current token K/V, advance
`valid_tokens`, then attend over the complete valid prefix. Native output was
checked against a full-history reference with maximum absolute error below
`7e-7`, relative L2 below `4e-7`, cosine similarity above
`0.9999999999999`, and no NaNs or infinities.

| Case | Cache bytes | Bytes/token | Host append median ms | Host final-context decode median ms | Pi append median ms | Pi final-context decode median ms |
|---|---:|---:|---:|---:|---:|---:|
| A: h2 d32, 16 + 16, cap32 | 16,384 | 512 | 0.00738 | 0.01024 | 0.01052 | 0.01519 |
| B: h4 d32, 64 + 64, cap128 | 131,072 | 1,024 | 0.00756 | 0.03224 | 0.01093 | 0.04099 |
| C: h4 d64, 128 + 128, cap256 | 524,288 | 2,048 | 0.00754 | 0.11295 | 0.01106 | 0.12805 |
| D: h4 d64, 128 + 384, cap512 | 1,048,576 | 2,048 | 0.00798 | 0.23179 | 0.01131 | 0.35742 |

Every case records one artifact load, one cache allocation, zero layout
reselection, and zero kernel reselection. This does not implement paged KV,
block tables, eviction, prefix sharing, continuous batching, GQA/MQA, GPU,
distributed KV, or full-model serving.
