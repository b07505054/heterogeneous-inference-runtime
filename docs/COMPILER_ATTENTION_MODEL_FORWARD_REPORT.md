# Compiler-selected attention in a real Qwen model-forward path

Date: 2026-07-17. Exact achieved level: **Level 5**. Real Q/K/V tensors from
Qwen2.5-0.5B flow through an ExecutionPlan-controlled attention implementation;
the returned attention tensor is consumed by `o_proj`, all remaining layers,
the LM head, logits, and eight greedy token decisions. This is not Level 6:
the installed CUDA vLLM wheel cannot initialize its CPU platform, so no vLLM
serving request executed this backend.

## Repository and environment

Starting commits:

- `heterogeneous-inference-runtime`: `main`,
  `5b56607cf84d8acda2691f02762f50d30332a8d1`.
- `ml-graph-compiler-runtime`: `master`,
  `0d200c3c7463f21cda97e77f4ad0e912bbad329f`.

Both repositories started dirty with preserved AArch64, ExecutionPlan, and
single-node CPU-sharding work. No reset, clean, stash, commit, push, rebase,
deletion, or overwrite of a prior result artifact occurred.

Ending commits are unchanged at the same hashes. This slice adds the attention
runtime, tests, two benchmark/proof scripts, three retained model-forward
diagnostic/evidence artifacts, the candidate-evaluation artifact, this report,
the GQA MLIR fixture, and the focused GQA verifier edit. All other modified and
untracked AArch64/CPU-sharding files remain preserved.

Environment: Python 3.12.13, PyTorch 2.11.0+cu130, Transformers 5.13.0, vLLM
0.24.0 CUDA wheel, LLVM/MLIR 21.1.8, Intel i5-10210U with four physical/eight
logical cores, AVX2/FMA, one NUMA node.

## Installed vLLM attention path

The exact installed call chain is:

```text
request scheduler
-> V1 model runner builds per-layer metadata
-> set_forward_context
-> Qwen2ForCausalLM.forward
-> Qwen2Model.forward
-> Qwen2DecoderLayer.forward
-> Qwen2Attention.forward
-> QKVParallelLinear -> split -> RoPE
-> Attention.forward
-> unified_kv_cache_update custom op
-> unified_attention_with_output custom op
-> AttentionImpl.forward
-> output -> RowParallelLinear o_proj
-> residual + MLP + remaining layers
-> final norm -> LM head -> logits -> sampler
```

For the local model, Q is `[tokens,14,64]`, K/V are `[tokens,2,64]`.
`CPUAttentionBackendImpl` requires HND paged storage with logical cache shape
`[blocks,2,block_size,128]`, viewed as separate K/V
`[blocks,2,block_size,64]`. Metadata carries actual/query/sequence lengths,
query start offsets, block table, slot mapping, scheduler metadata, and causal
mode.

`AttentionBackendEnum.CUSTOM` is registerable out of tree. However, this exact
CPU platform selector always returns `CPU_ATTN`, and the installed CUDA wheel
does not activate `CpuPlatform`. No installed platform plugin exists.

Non-invasive hooks on the real local weights proved all 24 layers execute for
prefill `[1,4,896]` and again for decode `[1,1,896]` with cache length four.

## Integration boundary

The implemented boundary is Transformers 5.13's supported
`ALL_ATTENTION_FUNCTIONS` registry, paired with its supported attention-mask
registry. It receives post-RoPE real Q/K/V and returns the tensor immediately
consumed by Qwen `o_proj`. No installed package source is modified and the
model runner is not monkey-patched.

The first diagnostic artifact,
`qwen_compiler_attention_model_forward.json`, is retained but rejected: it
revealed that a custom interface must also register eager causal-mask semantics.
The corrected and final evidence is
`qwen_compiler_attention_model_forward_final.json`.

## HIR and ExecutionPlan

The compiler already had real `hir.attention` and `hir.cpu_attention` ops,
contiguous/paged KV ops, selection/lowering, and runtime adapter infrastructure.
Its verifier incorrectly required equal query/KV head counts. It now permits
GQA when query heads are divisible by KV heads and verifies K batch, KV-head,
context, and head-dimension shapes. The real-model fixture is
`mlir/attention_qwen_gqa_contract.mlir` with Q `[1,14,11,64]` and K/V
`[1,2,11,64]`; it parses and verifies with the rebuilt dialect plugin.

ExecutionPlan v2 now accepts optional
`global_decisions.attention_execution`. It records phase, exact strategy,
worker count, split dimension, Q/KV heads, head dimension, FP32, BHSD Q/K/V
layout, contiguous cache layout, causal mode, direct assembly, fallback,
provenance, kernel ID, and `runtime_no_redecision=true`. Old plans parse with
an empty decision. Runtime traces require selected ID = executed ID and state
that the traced candidate produced the returned tensor.

## Numerical implementation and candidates

The unfused FP32 path is:

```text
Q @ K^T -> scale -> additive causal mask
-> row max -> subtract -> exp -> row sum -> normalize
-> probabilities @ V
```

It uses PyTorch's native CPU tensor primitives. It is not a bespoke generated
LLVM object or fused kernel. Candidates are:

- serial;
- split-head 2/4/8: disjoint query heads, read-only GQA-expanded K/V, direct
  output assembly;
- split-query 2/4/8: disjoint query rows, shared read-only K/V, direct output
  assembly.

Workers are persistent and pinned to logical CPUs. Neither strategy uses a
reduction, barrier collective, network communication, or distributed process
group. Uneven heads/query lengths use balanced-remainder partitions.
Split-query is illegal for decode, worker count may not exceed useful
partitions, unsupported FP32/head-64/GQA/layout contracts fail closed.

## KV cache

The standalone cache layout is:

```text
[layer, sequence, kv_head, position, head_dimension]
```

It supports prompt append, consecutive one-token decode appends, separate
sequence lengths, capacity checks, and contiguous valid-range views. Tests
cover multiple decode steps and mixed cache reuse. This is not paged attention.

The real Qwen proof uses Transformers `DynamicCache`: prefill writes each
layer's prompt K/V; each decode call appends one position and passes the complete
contiguous K/V context to the registered attention implementation.

## Correctness

- 67 focused attention, cache, ExecutionPlan, existing attention-adapter,
  CPU-sharding, and vLLM-adapter tests pass.
- Serial, split-head 2/4/8, and split-query 2/4/8 compare complete tensors
  against PyTorch SDPA for divisible and uneven prefill shapes.
- Serial and split-head 2/4/8 decode compare complete tensors at several
  context lengths.
- NaN/Inf checks pass.
- 1,000 consecutive split-query invocations pass with the exact selected
  candidate ID and finite complete output.
- All 27 compiler CTests pass and the new 14:2 GQA MLIR contract verifies.
- Full runtime suite: 818 passed, 16 skipped, 39 failed. The same 39
  pre-existing/environmental groups remain: missing cross-repository
  capability/native artifacts, an import-order isolation assertion, and
  sandbox-denied localhost socket tests. No new attention test fails.

## Logits, tokens, and causal proof

Prompt: `Compiler attention proof.` (four tokens), FP32, greedy decode, eight
steps, fixed seed.

Baseline and compiler tokens are identical:

```text
[576, 5567, 18404, 264, 501, 5486, 311, 279]
```

Maximum absolute logit difference over the eight steps is `1.62125e-5`.
The compiler implementation executed 24 prefill invocations and 168 decode
invocations, with no fallback and selected ID equal to executed ID.

Test-only perturbation adds `5.0` to the attention tensor before `o_proj`.
Perturbed tokens become:

```text
[84565, 84565, 84565, 84565, 84565, 84565, 84565, 84565]
```

Per-step logit differences are approximately 18.2–25.5. The perturbation is
disabled after the proof. This demonstrates causal dependency of logits and
generated tokens on the compiler attention result.

## Performance

Attention measurements use five warmups and 30 calls per candidate. The JSON
contains median, p95, variance, dispatch, QK, softmax, P×V, and assembly time.

Representative measured winners:

| Domain | Shape Q/C | Winner | Winner median | Serial median | Speedup |
|---|---:|---|---:|---:|---:|
| prefill calibration | 8/8 | serial | 0.220 ms | 0.220 ms | 1.00x |
| prefill calibration | 32/32 | serial | 0.383 ms | 0.383 ms | 1.00x |
| prefill calibration | 128/128 | split-head 4 | 1.893 ms | 3.208 ms | 1.70x |
| prefill held-out | 16/16 | serial | 0.165 ms | 0.165 ms | 1.00x |
| prefill held-out | 64/64 | serial | 0.702 ms | 0.702 ms | 1.00x |
| prefill held-out | 192/192 | split-head 4 | 2.898 ms | 5.225 ms | 1.80x |
| decode held-out | context 16/128/384 | serial | 0.120/0.205/0.689 ms | same | 1.00x |

Eight workers never win. Decode consistently prefers serial.

One final eight-token model run measured:

- baseline TTFT 132.87 ms; compiler TTFT 135.39 ms;
- baseline TPOT median/p95 115.69/116.42 ms;
- compiler TPOT median/p95 119.09/121.04 ms;
- total generation 938.47 ms baseline, 966.28 ms compiler.

This is a single deterministic end-to-end trial, not a confidence interval.
CPU utilization, context switches, and peak RSS were not captured; those remain
limitations rather than inferred values.

## Selector and fixed policies

Calibration and held-out shapes are disjoint. On six held-out shapes:

- legal candidates: seven for prefill, four for decode;
- exact-match rate: 66.67%;
- mean/median/p95/max regret: 5.97% / 0% / 24.42% / 30.94%;
- fallback rate: 0%.

The selector misses Q=64 by choosing split-head 2 and incurs 30.94% regret; it
correctly keeps all decode cases serial. Fixed-policy mean regret:

- always serial: 13.39%;
- always split-head 4: 201.93%;
- always split-head 8: 576.82%;
- split-query 4 when legal: 64.04%, with 50% decode fallback.

Thus compiler selection is materially safer than fixed parallel policies and
improves over always serial on mean regret, but it is not yet a high-quality
general cost model.

## Unsupported and exact level

Reached: **Level 5 — generated tokens depend on compiler attention** through a
real Qwen CPU model-forward path.

Not reached: Level 6 vLLM serving, native installed vLLM CPU execution, vLLM
paged KV addressing, CUDA/GPU execution, multi-device or multi-node execution,
network collectives, NCCL, full Shardy integration, BF16 compiler attention,
batched real-model generation, a bespoke fused native object, or production
performance readiness.

## Evidence-backed achievement summary

1. Routed post-RoPE 14:2 GQA tensors from all 24 Qwen layers through a
   compiler-selected attention implementation without modifying package source.
2. Matched baseline logits within `1.63e-5` and all eight greedy tokens while
   proving causal token dependency through controlled attention perturbation.
3. Added persistent pinned serial/split-head/split-query FP32 attention with
   complete-tensor SDPA validation and 1,000-call stability.
4. Extended HIR verification and ExecutionPlan v2 for exact Qwen GQA attention
   decisions with fail-closed runtime identity tracing.
5. Measured phase-separated candidates and a held-out selector: 0% decode
   fallback, 5.97% mean regret, and up to 1.80x standalone prefill speedup.
