# Native Fused Online-Softmax Attention Report

## 1. Verdict

**Native Level 5 — measured backend value.** A bounded-memory native scalar
kernel and an x86 AVX2/FMA kernel are compiler-visible, survive ExecutionPlan
v2 serialization, and execute real Qwen attention. The AVX2 implementation
beats the measured dense operator in 12/17 calibration workloads and the
compiler selects it for the short-prompt prefill and decode domains. This is
Level 5 model-forward integration, not vLLM serving.

## 2. Repository state

| Repository | Start branch / HEAD | End branch / HEAD |
|---|---|---|
| heterogeneous-inference-runtime | `main` / `5b56607cf84d8acda2691f02762f50d30332a8d1` | unchanged |
| ml-graph-compiler-runtime | `master` / `0d200c3c7463f21cda97e77f4ad0e912bbad329f` | unchanged |

Both worktrees were dirty at entry and remain dirty. Existing modified and
untracked work was preserved. No reset, clean, stash, rebase, commit, push, or
deletion occurred.

## 3. Previous limitation and reused runtime

The prior fused implementation used a Python key-tile loop and multiple
PyTorch operations per tile. This stage reuses the repository's established
fail-closed `ctypes.CDLL` pattern: a shared object is resolved by exact path,
verified by SHA-256 and ABI version, and invoked by an ExecutionPlan-carried
symbol. It does not introduce a second general runtime.

## 4. Native ABI and scalar architecture

[`native/fused_online_attention.h`](../native/fused_online_attention.h) defines
explicit Q/K/V/output pointers, four-dimensional element strides, batch,
query/context lengths, local and total query heads, KV heads, head dimension,
scale, causal position offset, tiles, worker count, and head offset. Validation
rejects null pointers, non-positive dimensions, invalid GQA, invalid head
ranges, unsupported strides, non-causal use, invalid scale/tiles/workers, and
causal ranges outside the context.

The scalar symbol `hir_fused_online_attention_scalar` makes one bounded scratch
allocation per invocation: one `key_tile` score buffer plus one
`head_dimension` output accumulator. It performs the verified online recurrence
and skips fully masked tiles. It does not allocate Q×K scores or probabilities,
call PyTorch, or call the dense helper.

For Qwen D=64 and K-tile=32:

| Metric | Native fused |
|---|---:|
| score tile | 128 B |
| output accumulator | 256 B |
| total scratch | 384 B |
| allocations/invocation | 1 |
| allocations/query row | 0 |

## 5. Correctness and allocation proof

The native scalar and AVX2 matrix contains 66 rows: 15 decode contexts, 13
square prefills, and five uneven Q/K pairs for both implementations. Against
PyTorch attention, the worst absolute error was
`1.3113021850585938e-6`; mismatch, NaN, and Inf counts were zero. Focused tests
also cover fail-closed GQA and artifact-hash behavior.

The bounded 384-byte scratch is independent of Q×K. At Q=K=256 the planner
estimates 9,175,040 bytes for dense score/probability plus GQA temporaries,
versus 384 bytes for native serial fused (1,536 bytes for four workers).

## 6. Code generation and SIMD

Compilation used:

```text
clang++ -O3 -std=c++17 -fPIC -shared -fno-vectorize -fno-slp-vectorize \
  -mavx2 -mfma native/fused_online_attention.cpp -o libfused_online_attention.so
clang++ ... -S -emit-llvm ... -o fused_online_attention.ll
clang++ ... -S ... -o fused_online_attention.s
```

Host: Intel Core i5-10210U, x86_64, 4 physical / 8 logical cores, one NUMA
node, AVX2 and FMA, 128 KiB aggregate L1D, 1 MiB aggregate L2, 6 MiB L3.
The shared object SHA-256 is
`27cd4b60b16c62f8b6dfc561e534a0ac90b5ce6eb1137d3b3d60b8f4b3845274`.
The object has 8,490 text bytes; the shared object has 16,061 text bytes.

Assembly contains AVX/FMA dot products and V accumulation (`vfmadd*`), vector
rescaling/normalization (`vmulps`), and scalar `expf` calls. Scalar was compiled
with LLVM loop and SLP vectorization disabled; the separate AVX2 symbol uses
intrinsics. Scalar exponentiation remains the principal non-vectorized inner
operation. `perf` hardware counters were not available/collected, so IPC,
cache-miss, and branch-miss claims are not made.

## 7. Runtime, workers, and provenance

[`deployment/native_fused_attention.py`](../deployment/native_fused_attention.py)
marshals PyTorch CPU FP32 tensors without per-tile Python calls.
[`deployment/attention_runtime.py`](../deployment/attention_runtime.py) loads
the artifact once per phase runtime and records implementation and exact native
symbol. Split-head uses the existing persistent pinned worker pool and passes a
global head offset, preserving Qwen's 14:2 GQA mapping without K/V expansion.
There is no numerical reduction between workers.

ExecutionPlan fields distinguish algorithm, implementation, strategy, workers,
tiles, target ISA, ABI version, artifact hash/path, and symbol. The normal
Qwen run proved selected = serialized = deserialized = executed candidate and
symbol, with zero fallback and mismatch.

## 8. Performance

Single-thread PyTorch and serial native measurements used two warmups and seven
trials per candidate. Across 17 calibration workloads:

| Comparison | Median speedup | Range |
|---|---:|---:|
| Python fused → native scalar | 6.26× | 3.57–22.34× |
| native scalar → AVX2 | 2.94× | 1.21–3.79× |
| dense / AVX2 | 1.37× | 0.66–3.13× |

AVX2 beat dense in all ten sampled decode contexts (K=4…2048) and at prefill
Q=4, 8, and 16. Dense won at prefill Q=32…256. Representative medians:

| Domain | Dense ms | Python fused ms | Scalar ms | AVX2 ms |
|---|---:|---:|---:|---:|
| decode K=128 | 0.078 | 0.682 | 0.160 | 0.055 |
| decode K=1024 | 0.423 | 5.990 | 1.214 | 0.326 |
| prefill Q=16 | 0.085 | 3.065 | 0.181 | 0.063 |
| prefill Q=64 | 0.439 | 23.229 | 2.452 | 0.606 |
| prefill Q=256 | 7.245 | 342.244 | 36.204 | 9.460 |

Thus Python/PyTorch orchestration was a major cause of the prior regression,
SIMD is material, and dense BLAS remains superior for medium/large prefill.
The compiler cost model encodes this measured phase/length crossover rather
than assuming fused always wins.

## 9. Memory-aware selection

Normal selection minimizes calibrated latency. Optional
`memory_constrained` selection filters candidates before scoring. For Q=K=256
and a 1 MiB temporary budget, dense candidates are rejected at 9.175 MiB and
the compiler selects AVX2 fused split-head/4 at 1,536 estimated scratch bytes.
The budget is meaningful: it is below the measured dense intermediate demand
but far above the bounded fused scratch, not an artificial one-byte constraint.

## 10. Qwen model-forward integration

For the four-token prompt and eight greedy decode steps, the normal compiler
selected `cpu_fused_online_native_avx2_fp32_q1_k32_serial_w1_v1` for both
phases. All 192 calls (24 prefill, 168 decode) executed the exact AVX2 symbol,
all 192 outputs entered `o_proj`, and fallback, mismatch, and manual-plan counts
were zero. Baseline and compiler token IDs were identical:

```text
[576, 5567, 18404, 264, 501, 5486, 311, 279]
```

Maximum per-step logit difference was `3.2901763916015625e-5`. The test-only
`+5.0` perturbation remained before `o_proj` and changed the generated sequence
to eight copies of token `84565`, proving the newly selected native output is
causal. Forced scalar also completed 192 native calls with identical baseline
tokens and maximum logit difference `2.2649765014648438e-5`.

At real-model prompt length 64, the compiler correctly selected dense
split-head/2 for prefill and native AVX2 for the subsequent decode domain.
The first token matched (`14252`), maximum logit difference was
`1.1444091796875e-5`, and all 24 prefill outputs entered `o_proj`. This is
model-forward prefill latency, not serving TTFT.

## 11. Selector and fixed-policy interpretation

Held-out lengths are 24, 48, 96, 192, 384, 768, 1536 for decode and 11, 24,
48, 73, 96, 192, 384 for prefill. They are separate from calibration lengths.
The serialized held-out trace records 13 generated candidates, legality, score,
algorithm, implementation, and winner. A full independently repeated held-out
latency sweep and regret distribution were not completed in this stage, so no
new exact-match or regret percentage is claimed. Fixed-policy timing shows why
phase-aware selection matters: always fused loses on large prefill, while
always dense misses every measured decode win.

## 12. MLIR/native classification and limitations

Classification is **C: compiler-selected external native C++ artifact**. The
existing MLIR Qwen GQA contract defines attention semantics, and the planner
controls candidate legality, scoring, ExecutionPlan identity, and dispatch.
The executed loops are not generated by MLIR, and no MLIR-to-LLVM claim is
made.

Unsupported: FP16/BF16, non-contiguous cache policies beyond explicit strides,
paged KV addressing, sliding window, block sparsity, vector exponential,
CUDA/Triton/GPU, vLLM CPU serving, PagedAttention, multi-node/multi-device
execution, and Level 6. Model-forward timings are not serving TTFT/TPOT.

## 13. Tests

```text
.venv/bin/python -m pytest -q \
  tests/test_native_fused_attention.py tests/test_attention_runtime.py \
  tests/test_execution_plan_loader.py
# 65 passed

ctest --test-dir build-mlir -R 'Shard|Attention|attention' --output-on-failure
# 2 passed
```

An initial `ctest --test-dir build` attempt could not create its log because
that stale directory is owned by `nobody`; the valid `build-mlir` tree passed.

## 14. Resume bullets

- Replaced Python-orchestrated tiled attention with bounded-memory scalar and
  AVX2/FMA CPU kernels, achieving median 6.26× Python-to-scalar and 2.94×
  scalar-to-SIMD speedups with ≤1.32e-6 absolute error.
- Integrated exact artifact hash, ABI, ISA, tile, implementation, and symbol
  provenance through compiler selection and ExecutionPlan v2; verified 192
  real Qwen attention outputs reached `o_proj` with zero fallback/mismatch.
- Characterized dense/fused crossover and memory trade-offs: AVX2 fused won
  12/17 calibration workloads, while bounded 384-byte scratch enabled a
  justified memory-budget selection where dense required 9.175 MiB.

## 15. Recommended next stage

Vectorize or approximate exponential under an explicit error budget, then run
an independently repeated held-out latency/regret sweep. After that, prototype
the fixed-D=64 inner row as an MLIR SCF/Vector-to-LLVM artifact without changing
the proven ABI.
