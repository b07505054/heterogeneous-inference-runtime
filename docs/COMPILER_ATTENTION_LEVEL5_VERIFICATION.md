# Compiler attention Level 5 verification

## 1. Verdict

**PARTIALLY VERIFIED LEVEL 5**

The numerical and causal model-forward claim is verified: real Qwen projection
tensors enter the custom attention runtime, its returned tensor is the tensor
passed to every layer's `o_proj`, the clean path reproduces the baseline logits
and eight greedy tokens, and a test-only perturbation applied before `o_proj`
changes the first and all subsequent tokens. The stricter
**compiler-selected** claim is not verified. The model-forward harness directly
constructs serial Python plans with `make_attention_plan`; it neither calls
`select_attention_plan` nor loads the optional ExecutionPlan v2
`attention_execution` decision. Labeling those plans
`cost_model_selected` is provenance text, not evidence of compiler selection.
Acceptance criterion 2 is therefore missing.

## 2. Repository state

Starting state on 2026-07-17:

| Repository | Branch | HEAD | State |
|---|---|---|---|
| `/home/allen/Desktop/Project/heterogeneous-inference-runtime` | `main` | `5b56607cf84d8acda2691f02762f50d30332a8d1` | Dirty; existing tracked ExecutionPlan edits and untracked attention, sharding, AArch64, result, test, and report files |
| `/home/allen/Desktop/Project/ml-graph-compiler-runtime` | `master` | `0d200c3c7463f21cda97e77f4ad0e912bbad329f` | Dirty; existing HIR/AArch64 tracked edits and untracked attention contract, sharding fixtures, AArch64 artifacts/tools/tests |

The only files added by this verification are this report and files under
`results/runtime_paths/attention_level5_verification/`. No reset, clean, stash,
commit, push, rebase, deletion, or overwrite of prior evidence occurred.

## 3. Claimed versus observed evidence

| Claim | Observation | Result |
|---|---|---|
| Qwen2.5-0.5B real model | Local `Qwen2ForCausalLM` snapshot `7ae557…`, 24 layers, hidden 896, 14 Q heads, 2 KV heads, head dimension 64 | Verified |
| Real Q/K/V | Registry callback receives tensors created by the current layer's `q_proj`, `k_proj`, and `v_proj`, after RoPE and cache update | Verified |
| Compiler-selected implementation | Harness hard-codes serial plans; selector and ExecutionPlan are unused | **Not verified** |
| Candidate-specific implementation executes | Serial and forced split-head/2 IDs changed and matched executed IDs | Verified for plan-driven runtime dispatch |
| No baseline fallback supplies returned tensor | Runtime has no reference fallback branch; invalid/no-active-runtime conditions raise | Verified, although there is no explicit production fallback counter |
| Attention result enters `o_proj` | All 192 compiler callbacks and `o_proj` pre-hooks had the same tensor data pointer and sum | Verified |
| Clean equivalence | Exact original per-step logit differences and all eight tokens reproduced | Verified |
| Perturbation is before `o_proj` | `output = output + 5.0` occurs in B,H,S,D attention output before transpose/head merge | Verified |
| Perturbation changes logits/tokens | Maximum per-step logit changes 18.23–25.51; every token changed to 84565 | Verified |
| Full layer/step coverage | 24 prefill calls plus 168 decode calls; eight calls per each of 24 layers | Verified |
| Level 6 | vLLM platform remains unspecified and CPU engine initialization fails before worker/backend initialization | Correctly not claimed |

### Artifact extraction and consistency

The claimed report and final JSON agree on model path, prompt, four prompt
tokens, eight generated positions, both clean token lists, 24/168 invocation
counts, perturbation `+5.0`, perturbed token IDs, and the rounded maximum logit
difference (`1.62125e-5` in prose versus
`1.621246337890625e-5` in JSON).

Extracted configuration and run values:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, local snapshot `7ae557604adf67be50417f59c2c2f167def9a775`
- Class: `Qwen2ForCausalLM`
- Configuration: 24 layers, hidden size 896, 14 query heads, 2 KV heads,
  head dimension 64, vocabulary 151936
- Weight-config dtype: BF16; test load/execution dtype: FP32
- Prompt: `Compiler attention proof.`
- Seed: `20260717`
- Decode: greedy `argmax`, eight generated token positions
- Baseline/compiler IDs:
  `[576, 5567, 18404, 264, 501, 5486, 311, 279]`
- Independently decoded baseline/compiler text:
  ` The paper presents a new approach to the`
- Perturbed IDs:
  `[84565, 84565, 84565, 84565, 84565, 84565, 84565, 84565]`
- Independently decoded perturbed text:
  `().'/().'/().'/().'/().'/().'/().'/().'/'`
- Prefill/decode plan in the model harness: serial, one worker,
  `torch_cpu_attention_fp32_serial_w1_v1`
- Clean compiler calls: 24 prefill and 168 decode; total 192
- Perturbation: scalar `+5.0` broadcast over every returned attention element,
  in all layers and forwards, before transpose/head merge and `o_proj`

The following requested values are **not recorded** in the original report or
final JSON: decoded texts, seed as a JSON field, explicit decoding-mode field,
maximum elementwise attention-output difference, maximum hidden-state
difference, per-layer/per-token counts, and a fallback counter. The report
derives the seed and mode from source. The candidate evaluation JSON is a
standalone synthetic-shape benchmark, not evidence that its selector drove the
Qwen model-forward run.

## 4. Exact model-forward dataflow

The executed integration is Hugging Face Transformers, not vLLM:

```text
run_qwen_compiler_attention.py:87-95
  current token IDs
  -> Qwen2Model.embed_tokens
  -> 24 Qwen2DecoderLayer instances
  -> Qwen2Attention.q_proj/k_proj/v_proj
  -> [B,H,S,D] view + transpose
  -> RoPE on Q/K
  -> DynamicCache.update(K,V,layer_idx)
  -> ALL_ATTENTION_FUNCTIONS["compiler_cpu_attention"]
  -> CompilerAttentionRuntime.attention
  -> transpose to [B,S,H,D]
  -> reshape [B,S,896]
  -> self.o_proj
  -> attention residual
  -> MLP residual
  -> final RMSNorm
  -> tied LM head
  -> logits[:, -1]
  -> argmax
```

Exact source boundaries:

- Harness entry and greedy selection:
  `scripts/run_qwen_compiler_attention.py:77-97`
- Runtime registration/interface:
  `deployment/attention_runtime.py:329-361`
- Runtime numerical dispatch:
  `deployment/attention_runtime.py:219-292`
- Installed Qwen projections, cache update, registry call, and `o_proj`:
  `.venv/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:206-245`
- Layer residual and MLP:
  the same file, lines 280-309
- Embedding, layer loop, and final norm:
  the same file, lines 363-413
- LM head/logits:
  the same file, lines 462-486

Observed prefill boundary shapes were Q `[1,14,4,64]`, K/V
`[1,2,4,64]`, and runtime output `[1,14,4,64]`. Decode Q was
`[1,14,1,64]`; K/V context lengths grew from 5 through 11. These are tensors
from the live projections, not fixtures, metadata-only values, or a discarded
shadow computation.

## 5. Candidate-selection proof

The Python runtime's decision mechanics are real:

```text
plan["selected_strategy"]
  -> CompilerAttentionRuntime.attention strategy branch
  -> serial / split_head / split_query implementation
  -> trace.selected_candidate_id == trace.executed_candidate_id
```

`legal_attention_candidates` is at `deployment/attention_runtime.py:100-120`,
the static selector is at lines 123-139, plan ID storage is at lines 44-63,
and dispatch is at lines 236-270. The runtime reads the plan and does not
recompute a winner.

Independent forced runs observed:

| Phase | Requested plan | Selected ID | Executed ID | Workers |
|---|---|---|---|---:|
| Prefill/decode | serial | `torch_cpu_attention_fp32_serial_w1_v1` | same | 1 |
| Prefill/decode | split-head | `torch_cpu_attention_fp32_split_head_w2_v1` | same | 2 |

Both paths produced identical tokens and the same per-step differences versus
baseline. Decode split-query/2 was rejected with
`ShardingPlanError: split_query is illegal for one-token decode`.

However, `scripts/run_qwen_compiler_attention.py:68-73` directly constructs the
two serial plans. It never calls `select_attention_plan` and never parses an
ExecutionPlan. Thus the model-forward run proves **plan-controlled runtime
dispatch**, but not that a compiler output selected the plan.

## 6. Fallback audit

For both prefill and decode, dtype/shape/phase violations raise
`ShardingPlanError`; absence of an active runtime also raises. The serial,
split-head, and split-query branches all produce their own numerical output.
There is no `try/except` reference-attention fallback and no call to PyTorch
SDPA/eager attention in `CompilerAttentionRuntime.attention`.

The plan contains a serialized `fallback` field, but this runtime does not
execute it. Therefore the observed fallback count is inferentially zero:
all 192 registered compiler calls completed and reached their trace/return.
The original artifact's claim of zero fallback is weaker than an instrumented
counter because no production fallback counter exists. Baseline eager attention
was computed in a separate run and was never substituted into a compiler run.

## 7. Invocation audit

The harness performs eight model forwards. Forward 0 consumes the four-token
prompt and produces generated token 1. Forwards 1–7 each consume one prior
generated token.

| Scope | Expected | Observed |
|---|---:|---:|
| Prefill | 24 layers × 1 forward = 24 | 24 |
| Decode | 24 layers × 7 forwards = 168 | 168 |
| Total compiler calls | 192 | 192 |
| Per layer | 8 | 8 for every layer 0–23 |
| Per generated token | token 1: 24 prefill; tokens 2–8: 24 decode each | matched |

The often-stated `8 × L` decode expectation would require eight decode forwards
after prefill and would generate nine tokens total in this loop structure. The
reported 24/168 counts are correct for eight generated tokens.

## 8. `o_proj` dependency proof

Source-level flow:

```text
runtime output [B,H,S,D]
-> transpose(1,2).contiguous() [B,S,H,D]
-> Qwen reshape(*input_shape,-1).contiguous() [B,S,896]
-> self.o_proj(attn_output)
```

The verification registered a pre-hook on all 24 `o_proj` modules. Across each
of the serial, split-head/2, and perturbed runs, all 192 returned attention
tensors had the same data pointer and exactly the same FP64 sum as the next
`o_proj` input (`max_sum_difference = 0.0`). There is no reassignment to a
baseline tensor.

## 9. Equivalence reproduction

Fresh evidence:
`results/runtime_paths/attention_level5_verification/independent_path_isolation.json`.

Per-step baseline-versus-compiler maximum logit differences:

```text
[1.2874603271484375e-05,
 1.33514404296875e-05,
 1.621246337890625e-05,
 1.1205673217773438e-05,
 9.5367431640625e-06,
 9.775161743164062e-06,
 8.106231689453125e-06,
 9.059906005859375e-06]
```

The reported `1.62e-5` was reproduced **exactly at full precision**. Baseline,
serial, and forced split-head/2 all generated
`[576, 5567, 18404, 264, 501, 5486, 311, 279]`.

The independent diagnostic retained output summaries, not full baseline and
compiler attention tensors; its maximum absolute difference between per-call
attention sums was `9.1368e-5`. Complete elementwise attention correctness is
covered separately by the focused candidate tests with `rtol=2e-5`,
`atol=2e-6`.

## 10. Perturbation audit

The perturbation is at `deployment/attention_runtime.py:271-273`, inside
`CompilerAttentionRuntime.attention`:

```python
if self.perturbation:
    output = output + self.perturbation
```

Activation is set only by the test harness at
`scripts/run_qwen_compiler_attention.py:84-85`. Default is `0.0`. The magnitude
is `+5.0`, broadcast across the complete B,H,S,D attention output for every
layer. It occurs after candidate computation/assembly, before the returned
transpose, before head merge, and before `o_proj`. It does not touch hidden
states after `o_proj`, logits, token IDs, or tokenizer data. This is valid
causal evidence.

## 11. Causal reproduction

| Path | Token IDs | Decoded text |
|---|---|---|
| A: baseline eager | `[576,5567,18404,264,501,5486,311,279]` | ` The paper presents a new approach to the` |
| B: compiler serial | `[576,5567,18404,264,501,5486,311,279]` | same |
| C: compiler serial +5 | `[84565,84565,84565,84565,84565,84565,84565,84565]` | `().'/().'/().'/().'/().'/().'/().'/().'/'` |

B versus C per-step maximum logit differences were
`[19.3958, 18.2256, 23.2404, 20.1371, 20.6730, 20.3343, 25.5111, 20.5304]`.
The attention tensor changes before the hooked `o_proj` input and the logits
and greedy token change afterward.

## 12. Temporal causality

The first generated token is directly affected by perturbation in the prefill
forward. That changed token becomes the input to decode forward 1. Every later
forward also receives a direct attention perturbation, but it additionally
operates on an already-diverged autoregressive prefix. Therefore tokens 2–8
are downstream consequences of both the direct per-forward perturbation and
prior-token divergence; this experiment does not isolate eight independent
direct causal effects.

## 13. KV-cache verification

Transformers creates a `DynamicCache` when the first forward receives
`past_key_values=None`. Each Qwen attention layer calls
`past_key_values.update(key_states, value_states, layer_idx)` before the
registered compiler interface. The observed layer-0 K/V context lengths were
`[4,5,6,7,8,9,10,11]`, matching returned cache lengths and proving that cache
state was not reset between generated tokens. Baseline and compiler runs each
start a fresh equivalent cache.

This is the Transformers in-memory dynamic/contiguous tensor cache path. It is
not vLLM's paged KV cache, uses no block tables, and is not PagedAttention. The
standalone `ContiguousKVCache` class in `attention_runtime.py` is tested but is
not the cache used by this model-forward integration.

## 14. MLIR contract role

`mlir/attention_qwen_gqa_contract.mlir` correctly records the model's GQA
relationship (14 Q heads, 2 KV heads), head dimension 64, causal FP32
B,H,S,D input/output, and a legal prefill shape `[1,14,11,64]` with K/V
`[1,2,11,64]`. Its custom verifier checks FP32 rank-4 types, static declared
shapes, Q-head divisibility, phase legality, causality, and layouts.

It is not the exact executed prompt shape (the proof prompt length is 4), has no
decode function, does not encode scale, KV-cache lifecycle/layout, candidate
ID, or dispatch metadata, and is not consumed by the Python model harness.
It serves as a **verified contract only**; it neither directly lowers to the
executed implementation nor drives this run's ExecutionPlan.

The project compiler parsed the fixture and passed custom op verification, then
failed later in the serving pass with the expected missing module artifact
metadata diagnostic. Generic upstream `mlir-opt` could not parse the custom
assembly because that binary does not register HIR. The three existing MLIR
attention/Qwen CTests passed. There are no contract-specific before/after
fixtures associated with this new file.

## 15. Compiler contribution

| Component | Responsibility | Observed implementation |
|---|---|---|
| Compiler/HIR | Attention semantics | Implemented verifier contract; not consumed by model run |
| Compiler legality | Valid candidates | Implemented in Python validation/generation |
| Candidate generation | Serial, split-head, split-query | Implemented and independently exercised |
| Compiler selection | Choose phase/strategy/workers | Static selector exists, but model harness does not call it |
| ExecutionPlan | Serialize decision | Optional v2 field validates/round-trips, but model harness does not load it |
| Runtime dispatch | Execute chosen plan without redecision | Implemented and verified for serial and split-head/2 |
| Attention kernel | Stable FP32 causal GQA | Implemented with PyTorch CPU matmul/exp primitives |
| Model adapter | Route live Q/K/V and result | Transformers attention registry implementation |
| Qwen model | Remaining layers, logits, tokens | Real 24-layer HF model |

The implemented selector can decide strategy, worker count, split dimension,
phase-specific path, KV layout label, and kernel ID. In the claimed end-to-end
run those fields are manually fixed to serial/one-worker; no tile choice exists.

## 16. Level 6 limitation

Installed versions are Python 3.12.13, PyTorch `2.11.0+cu130`, Transformers
`5.13.0`, and vLLM `0.24.0`. vLLM imports from the local virtual environment,
but reports `UnspecifiedPlatform`, no device type, no active Triton driver, and
missing `vllm._C`. A corrected CPU engine probe reached engine configuration
and failed with `RuntimeError: Device string must not be empty`, before model
worker, attention backend, rank, or serving request initialization.

Installed vLLM's Qwen path is
`Qwen2Attention.qkv_proj -> split Q/K/V -> RoPE -> Attention.forward ->
unified_attention_with_output -> o_proj` at
`.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py:209-236`
and
`.venv/lib/python3.12/site-packages/vllm/model_executor/layers/attention/attention.py:452-523`.
None of it executed in the Level 5 test. Classification remains: generated
tokens depend on the custom attention in a real HF model-forward path; no vLLM
serving request uses it.

## 17. Test results

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q tests/test_attention_runtime.py` | 22 passed |
| `.venv/bin/python -m pytest -q tests/test_execution_plan_loader.py tests/test_execution_path_builder.py` | 24 passed |
| `ctest --test-dir build-mlir -R 'AttentionCPUContractTest\|AttentionCPUFailClosedTest\|QwenToServingMlirTest' --output-on-failure` | 3 passed |
| Independent baseline/serial/split-head/invalid/perturbed model diagnostic | completed; all assertions supported by recorded traces |
| `compile-for-target` on Qwen contract | parsed/verified; later serving pass failed because module artifact ref/SHA/version are intentionally absent |
| Corrected vLLM CPU engine probe | failed before engine initialization: empty device from unspecified platform |

One initial test command used the nonexistent filename
`tests/test_execution_plan_path_builder.py`; pytest exited 4 with no tests
collected. It was corrected to `tests/test_execution_path_builder.py`, where all
24 tests passed. This diagnostic mistake is retained in the verification logs.

## 18. Final defensible claim

The implementation causally integrates a plan-driven, candidate-specific FP32
causal GQA attention runtime into a real 24-layer Hugging Face
Qwen2.5-0.5B model forward: all live Q/K/V tensors traverse the runtime, its
returned context enters every layer's `o_proj`, clean logits match eager
attention within `1.621246337890625e-5`, and eight greedy tokens match exactly;
a test-only pre-`o_proj` perturbation changes downstream logits and every
generated token. The run is only partially verified as
**compiler-selected Level 5**, because its serial plans are constructed
directly in Python rather than selected by compiler output or loaded from
ExecutionPlan v2, and it is not a vLLM serving execution.

## 19. Resume bullets

- Causally integrated candidate-specific FP32 GQA attention into a real
  24-layer Qwen model-forward path, matching eight greedy tokens and logits
  within `1.63e-5`.
- Verified serial and two-worker split-head dispatch over 192 live attention
  calls, with exact returned-tensor identity at every Qwen output projection
  and fail-closed invalid-candidate handling.
- Built an auditable pre-`o_proj` perturbation experiment proving attention
  output changes propagate through transformer layers to logits and
  autoregressive token selection, explicitly scoped outside vLLM serving.
