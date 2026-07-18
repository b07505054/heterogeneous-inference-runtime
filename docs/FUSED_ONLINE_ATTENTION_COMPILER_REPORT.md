# Fused tiled online-softmax attention compiler report

## 1. Verdict and achieved level

**Level D — measured compiler value, with a negative latency result.**

Levels A–C are also satisfied:

- A: exact tiled online softmax executes without full score/probability
  materialization.
- B: dense and fused algorithms are compiler-visible, scored, serialized, and
  dispatched through ExecutionPlan v2.
- C: real Qwen Q/K/V execute fused attention in an explicitly labeled
  `forced_test_override`; all outputs enter `o_proj`, logits remain equivalent,
  and eight greedy tokens match.
- D: fused intermediate memory is materially lower for medium/large prefill,
  and the measured-profitability-aware compiler correctly avoids the slower
  fused implementation on all tested normal workloads.

No fused latency or end-to-end speedup is claimed.

## 2. Starting and ending repository state

Starting state:

| Repository | Branch | HEAD | State |
|---|---|---|---|
| `heterogeneous-inference-runtime` | `main` | `5b56607cf84d8acda2691f02762f50d30332a8d1` | Dirty with existing Level 5, ExecutionPlan, CPU-sharding, AArch64, tests, and evidence |
| `ml-graph-compiler-runtime` | `master` | `0d200c3c7463f21cda97e77f4ad0e912bbad329f` | Dirty with existing HIR, attention contract, sharding, and AArch64 work |

Ending HEADs are unchanged. This work did not reset, clean, stash, rebase,
commit, push, delete, or overwrite prior evidence. New evidence is confined to
`results/runtime_paths/fused_online_attention/`.

## 3. Existing dense attention memory behavior

The dense helper is `deployment/attention_runtime.py:266-294`:

```text
scores = Q @ K^T
scores *= scale
scores += mask
row_max = max(scores)
probabilities = exp(scores - row_max)
probabilities /= sum(probabilities)
output = probabilities @ V
```

`scores` and `probabilities` are distinct live `[B,Hq,Q,K]` FP32 tensors. The
subtraction expression feeding `exp` can also create an untracked transient
full-size tensor. The structural audit conservatively counts the two named
full matrices:

```text
score bytes       = B * Hq * Q * K * 4
probability bytes = B * Hq * Q * K * 4
```

All dense serial, split-head, and split-query candidates call this same helper.
Worker partitioning changes each worker's local matrix shape, not the
algorithm. Split-head and split-query concatenate disjoint outputs; neither
performs a numerical reduction.

The runtime also uses `repeat_interleave` to expand each GQA K and V tensor
from two to 14 heads before either dense or fused execution. That common
materialized cost is reported separately and included in total temporary-byte
comparisons.

## 4. Online-softmax mathematical recurrence

For each query tile and successive key/value tiles, the implementation keeps
running maximum `m`, denominator `l`, and unnormalized output accumulator `o`:

```text
S_j = Q_i @ K_j^T * scale + causal_mask
m_new = max(m_old, row_max(S_j))
alpha = exp(m_old - m_new)
P_j = exp(S_j - m_new)
l_new = alpha * l_old + row_sum(P_j)
o_new = alpha * o_old + P_j @ V_j
O_i = o_final / l_final
```

Prior terms were normalized relative to `m_old`; when a new tile increases the
maximum, both prior denominator and accumulator must be multiplied by
`exp(m_old - m_new)`. Source comments document this invariant.

Initial `m=-inf`, `l=0`, and `o=0` are handled explicitly. Non-finite masked
scores become zero probability. A fully masked tile leaves prior state
unchanged, and final division clamps only the denominator floor.

## 5. Fused implementation architecture

`_fused_online_attention_chunk` at
`deployment/attention_runtime.py:297-382` loops over query tiles and key tiles.
Each tile performs PyTorch CPU `matmul`, `amax`, exact `exp`, reduction, and
probability–V `matmul`. It stores only:

- one score tile;
- one probability tile;
- running max and denominator per query row;
- the running output accumulator;
- the final output tensor.

The Q=1 variant is the deterministic scalar-query reference candidate.
Q=4/Q=8 candidates vectorize within a query tile using PyTorch tensor
primitives. This is not called FlashAttention and does not claim a handwritten
SIMD kernel.

## 6. Supported and unsupported semantics

Supported:

- CPU FP32;
- causal attention;
- Qwen GQA, 14 query heads / two KV heads;
- head dimension 64;
- rank-4 B,H,S,D tensors;
- prefill and one-token decode;
- contiguous Transformers KV history;
- uneven query/context lengths and final tiles;
- serial and split-head/2 or split-head/4 fused execution.

Unsupported:

- BF16/FP16;
- paged KV addressing;
- sliding-window or sparse attention;
- fused split-query;
- CUDA/Triton/tensor cores;
- quantized KV;
- vLLM serving;
- multi-device or multi-node execution.

## 7. Compiler candidate representation

ExecutionPlan candidates explicitly include:

```text
algorithm = fused_tiled_online_softmax
score_materialization = false
probability_materialization = false
online_softmax = true
query_tile / key_tile
worker_count / split_dimension
causal_supported / gqa_supported
dtype / head dimension / workload domain
implementation = torch_tiled_online_softmax_exact_v1
selection provenance / target profile
temporary-memory semantics
```

Algorithm identity is therefore compiler-visible and not inferred from the
runtime function name.

## 8. Legality rules

The planner rejects unsupported dtype/head dimension/layout/KV layout,
non-causal mode, incompatible GQA, unavailable workers, useless head/query
partitions, non-positive tiles, fused split-query, and fused decode candidates
whose query tile is not one. Runtime validation independently checks algorithm,
tiles, materialization flags, kernel ID, strategy/split dimension, and actual
Q/K/V shapes. Illegal fused candidates raise; they do not redirect to dense.

## 9. Tile candidate space

The compiler evaluates 13 candidates:

- seven existing dense serial/split-head/split-query candidates;
- fused serial Q1/K32, Q4/K32, Q8/K64;
- fused split-head/2 Q1/K32 and Q4/K32;
- fused split-head/4 Q8/K64.

Decode legality retains only Q1 fused candidates. This intentionally small
space covers a scalar-query reference, two prefill query tiles, two key tiles,
and 1/2/4-worker execution.

## 10. Temporary-memory proof

Instrumentation records full-matrix flags, named allocation sizes, largest
temporary, allocation count, tile buffers, running state, common GQA expansion,
and total temporary bytes.

Guard tests replace the dense helper, `torch.softmax`, and SDPA with failing
functions; fused execution still succeeds. Across 94 measured fused
candidate/workload rows:

```text
full_score_materialized       = false
full_probability_materialized = false
score_bytes                   = 0
probability_bytes             = 0
```

The fused kernel still creates tile-sized score and probability tensors. It is
fused with respect to complete Q×K materialization, not allocation-free.

## 11. Native lowering/codegen path

The HIR attention operation remains a verified semantic contract. The compiler
controls candidate generation, legality, cost, ExecutionPlan serialization,
and dispatch. Numerical fused execution is Python control flow invoking
PyTorch CPU tensor primitives.

There is no MLIR-to-LLVM lowering, standalone native object, or custom C++
kernel in this slice. `native_codegen_evidence.json` records this boundary.

## 12. LLVM/assembly evidence

No project-owned LLVM IR or assembly was emitted, so no vector instruction,
spill, stack, or register claims are made. The host is x86-64,
Intel Core i5-10210U, four physical/eight logical cores, one NUMA node, with
AVX2/FMA in its reported ISA. PyTorch may use vectorized library kernels
internally, but that is not attributed to this compiler. Exact `torch.exp` is
called once per score tile and is a measured performance limitation.

## 13. Calibration methodology

Calibration used real Qwen dimensions Hq=14, Hkv=2, D=64, FP32:

- prefill Q=K: 4, 8, 16, 32, 64, 128, 256;
- decode Q=1, K: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

Each legal candidate ran three warmups and 20 measured calls with
`torch.set_num_threads(1)`; persistent worker pools were used for parallel
candidates. Every row records median, p95, variance, dispatch, barrier
(zero—no separate barrier primitive), algorithm, tiles, memory, and complete
tensor correctness.

The calibration artifact contains 144 rows, including 60 fused rows, with no
correctness failures.

## 14. Held-out selector quality

Held-out workloads were:

- prefill Q=K: 11, 37, 73, 129;
- decode Q=1, K: 17, 63, 127, 263, 511.

The 82-row held-out artifact includes 34 fused rows and no correctness failure.

```text
exact winner rate: 88.89%
mean regret:         0.156%
median regret:       0%
p95 regret:          0.844%
maximum regret:      1.406%
fused selection:     0%
dense selection:   100%
fallback:            0%
```

Calibration exact-winner rate was 93.75%, with 0.216% mean and 3.461% maximum
regret. This is an analytical static model evaluated on separate shape sets,
not a fitted statistical model.

## 15. Dense/fused crossover domains

No latency crossover was observed: a dense candidate won every calibration
and held-out workload.

Memory crossover did occur. For serial Q8/K64 fused versus dense:

| Prefill Q=K | Total temporary reduction, including GQA expansion | Attention-intermediate reduction | Fused slowdown |
|---:|---:|---:|---:|
| 32 | 16.41% | 49.22% | 4.52× |
| 64 | 40.53% | 81.05% | 4.07× |
| 128 | 63.51% | 95.26% | 4.87× |
| 256 | 79.05% | 98.82% | 3.21× |

For Q≤16, fused running state exceeds the tiny dense matrix, so fused also
loses memory. The honest decision table is therefore: dense for all measured
latency objectives; fused only as a memory-oriented experimental candidate for
Q≥32, not selected by the current latency selector.

## 16. ExecutionPlan provenance

Normal plans serialize algorithm, tile sizes, materialization flags, candidate
ID, strategy, workers, domains, selector version, and selection mode. The
adapter records:

```text
generated algorithm
-> selected algorithm
-> serialized algorithm
-> deserialized algorithm
-> executed algorithm
```

The short and long normal plans selected dense. The fused model diagnostic
serialized and executed fused Q4/K32 prefill and Q1/K32 decode under
`selection_mode=forced_test_override`. All IDs and algorithms matched; runtime
reselection, fallback, and mismatch counts were zero.

## 17. Qwen short-prompt result

The normal four-token/eight-generation path generated fused candidates but
selected dense serial for prefill and decode, consistent with measurements:

```text
192 plan-driven calls
24 prefill / 168 decode
192 outputs entered o_proj
fallback = 0
mismatch = 0
manual plan calls = 0
tokens equal = true
max logit difference = 1.621246337890625e-5
```

The compiler correctly avoided slower fused execution.

## 18. Qwen long-prefill result

A deterministic 128-token real-model prompt tensor executed one full Qwen
prefill. The compiler selected dense split-head/4, not fused:

```text
selected candidate: dense split-head/4
attention calls: 24
outputs entering o_proj: 24
compiler summed attention time: 52.09 ms
baseline model-forward prefill: 433.34 ms
compiler model-forward prefill: 602.23 ms
max logit difference: 9.20296e-5
first token: 14252 on both paths
fallback/mismatch: 0/0
```

The prompt is an exact-length deterministic repeated tokenizer ID used to
exercise the real model; Q/K/V are produced by Qwen projections. These are
model-forward prefill timings, not serving TTFT.

## 19. Logit and token equivalence

The forced fused short-prompt run produced the same eight IDs as baseline:

```text
[576, 5567, 18404, 264, 501, 5486, 311, 279]
```

Its maximum per-step logit difference was
`2.3365020751953125e-5`, within the established FP32 tolerance and with no
token change. The standalone fused matrix maximum absolute output error was
`1.1920928955078125e-6`; no NaN or Inf was observed.

## 20. Fused causal dependency proof

Because no measured workload justified compiler selection of fused, the causal
test uses an explicitly labeled forced override. It does not claim
compiler-selected profitability.

```text
forced fused candidate
-> ExecutionPlan JSON
-> normal loader
-> plan-only adapter
-> 192 fused Qwen attention calls
-> 192 o_proj inputs
-> logits/tokens
```

Adding `+5.0` to fused output before `o_proj` changed every token to 84565 and
changed per-step logits by 18.23–25.51. This proves the new fused algorithm
causally controls model output when dispatched.

## 21. Operator latency

Representative serial medians:

| Workload | Dense | Best fused | Result |
|---|---:|---:|---|
| prefill 4×4 | 0.132 ms | 0.280 ms | dense 2.13× faster |
| prefill 64×64 | 0.612 ms | 2.490 ms | dense 4.07× faster |
| prefill 128×128 | 1.879 ms | 9.149 ms | dense 4.87× faster |
| prefill 256×256 | 11.066 ms | 35.473 ms | dense 3.21× faster |
| decode K=512 | 0.512 ms | 3.164 ms | dense 6.19× faster |
| decode K=1024 | 1.207 ms | 6.568 ms | dense 5.44× faster |

## 22. Model-forward latency

Short prompt, single recorded trial:

```text
baseline generation:       969.47 ms
forced fused generation:  1071.18 ms
fused attention sum:       111.49 ms
```

Normal compiler-selected dense trial:

```text
baseline generation:      1018.53 ms
compiler generation:      1001.83 ms
attention sum:              41.30 ms
```

These full-model measurements are single-run evidence and are not presented as
statistically significant speedups. Repeated operator measurements establish
that this fused implementation is slower.

## 23. Memory reduction

At prefill 256×256:

```text
dense named score+probability: 7,340,032 bytes
fused tile+running state:         86,912 bytes
intermediate reduction:           98.82%

dense total incl. GQA expansion: 9,175,040 bytes
fused total incl. GQA expansion: 1,921,920 bytes
total reduction:                  79.05%
```

The common GQA expansion is now the dominant fused temporary at large Q. A
future implementation should map query heads to KV heads without materializing
expanded K/V.

## 24. Fixed-policy comparison

Held-out mean regret:

| Policy | Mean regret | Maximum regret | Fallback |
|---|---:|---:|---:|
| compiler selector | 0.156% | 1.406% | 0% |
| always dense serial | 9.44% | 67.22% | 0% |
| always dense split-head/2 | 178.85% | 372.16% | 0% |
| always fused serial | 461.37% | 883.98% | 0% |
| always fused split-head/2 | 2151.06% | 5185.52% | 0% |
| always fused split-head/4 | 1012.28% | 3068.90% | 55.56% |

The compiler's value is profitability-aware avoidance plus memory semantics,
not fused speed.

## 25. Honest limitations

- Numerical execution is Python/PyTorch, not an MLIR-generated native kernel.
- Python tile loops and exact tile-wise `torch.exp` dominate latency.
- GQA K/V are expanded to all query heads before both algorithms.
- Allocation counts are structural tracked allocations, not a system allocator
  trace; PyTorch may create additional internal temporaries.
- No latency domain selected fused.
- Long-prompt model testing used one 128-token deterministic prompt and one
  forward, not repeated serving trials.
- No BF16, paged KV, sliding window, CUDA, Triton, vLLM serving, multi-GPU,
  multi-node, PagedAttention, or Level 6 claim.

## 26. Resume bullets

- Implemented exact fused tiled online-softmax causal GQA attention that avoids
  full score/probability matrices and matches dense FP32 output within
  `1.20e-6` across 94 fused candidate/workload measurements.
- Integrated dense/fused algorithm identity, tile parameters, memory semantics,
  legality, analytical cost, ExecutionPlan v2 provenance, and exact runtime
  dispatch into the verified Qwen Level 5 path.
- Demonstrated up to 98.82% attention-intermediate and 79.05% total temporary
  reduction at 256-token prefill while honestly measuring a 3.21× slowdown and
  showing the compiler avoids unprofitable fused candidates with 0.156% held-out
  mean regret.

## 27. Recommended next stage

Move only the fused inner loops to an isolated C++/MLIR-native kernel, eliminate
materialized GQA expansion, and vectorize QK/PV plus online recurrence for
AVX2/FMA. Re-run the same calibration and held-out suite before changing the
selector. Native codegen should be adopted only if it creates a measured
latency crossover while retaining the existing allocation and causal proofs.
