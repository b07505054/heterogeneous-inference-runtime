# Distributed D3A: Live Qwen Tensor Capture and Serialized Rank-Local Validation

## 1. Executive result

A real activation was captured from a live, locally-cached
`Qwen/Qwen2.5-0.5B-Instruct` Transformers forward pass at
`model.layers.0.self_attn.o_proj` — the exact module the D2 compiler
plan's operator ID (`qwen_prefill::llm.o_proj::layer_0`) was fail-closed
mapped to. That real activation and the real module weight were partitioned
exactly per D2's TP=2 plan (448/448 along the hidden/contraction axis),
executed through **serialized rank-local validation** (one CPU, ranks
executed sequentially), reconstructed through D1's exact `all_reduce(sum)`
collective contract (reused unmodified), and compared against the captured
live module output: **max absolute error `1.79e-7`** (float32). A bonus
second path reusing D1's real multi-process/IPC runtime unmodified — now
fed the same real captured tensors instead of D2's synthetic-shaped ones —
matched to **`3.42e-7`**. All 14 required cross-layer provenance counters
are zero. All 17 negative tests pass. All D1/D2 regressions remain green.
Zero orphan processes, zero temporary tensor files.

**D3A acceptance criteria: all 18 satisfied.**

## 2. Repository state

| Repo | Branch | HEAD (unchanged before/after D3A) |
|---|---|---|
| `ml-graph-compiler-runtime` | `master` | `6169bd56e6ac97b839062db0a60df31c927f165c` |
| `heterogeneous-inference-runtime` | `main` | `9d954eab94796b82ce871f26f57964c3f3bee71d` |

No commits made. **The compiler repository required zero changes** — D3A
consumes D2's already-exported `real_qwen_tp2_execution_plan.json`
unmodified. `repository_state_before.json` / `_after.json` in this result
directory.

## 3. D1 and D2 preservation

`d1_d2_preservation.json`: both result directories confirmed present with
unchanged file counts (D1: 15, D2: 20) and recorded per-file SHA-256
prefixes; `git status` shows zero changes under either
`results/runtime_paths/distributed_d{1,2}_*/` path. Both reports
(`DISTRIBUTED_D1_...md`, `DISTRIBUTED_D2_...md`) untouched.
`regression_summary.json`: D1 compiler tests, D1 runtime tests, D2 compiler
tests, D2 runtime tests, and a fresh re-run of D1's deadlock negative test
all green; `all_regressions_green: true`; zero orphan processes after every
run.

## 4. Exact compiler operator contract (Part A)

`operator_contract.json`, derived programmatically (not assumed):

| Field | Value |
|---|---|
| Transformers module path | `model.layers.0.self_attn.o_proj` |
| Module class | `torch.nn.modules.linear.Linear` |
| Weight shape | `[896, 896]` (`[out_features, in_features]`) |
| Bias | absent (`bias=False`) |
| Mathematical operation | `Y = X @ W^T` (no bias) |
| hidden_size | 896 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 |
| head_dim | 64 |
| Batch / sequence / hidden dims | 0 / 1 / 2 (confirmed at capture: `(1, 5, 896)`) |

## 5. Transformers module mapping (Part B)

`operator_mapping.json`. `map_compiler_operator_to_module()` never selects
by substring: it scans `model.named_modules()` against an explicit
structural regex (`^model\.layers\.(\d+)\.self_attn\.o_proj$`), requires
exactly one match at the declared layer, requires the module class to be
`Linear`, and requires the weight's square dimension to equal the plan's
declared hidden size. All 5 checks passed
(`layer_number_matches`, `operator_kind_matches`, `weight_shape_matches_plan`,
`module_appears_exactly_once`, `structural_match_count_all_layers=24`). 5
distinct fail-closed rejection paths verified in §16.

## 6. Live Qwen execution configuration (Part C)

`live_model_execution.json`: `Qwen/Qwen2.5-0.5B-Instruct`, loaded from the
local Hugging Face cache (offline), explicit `dtype=float32` (not the
config-declared `bfloat16` — an intentional, documented choice for
reproducible CPU numerics), `model.eval()`, `torch.no_grad()`,
`torch.manual_seed(1234)`, fixed prompt `"The capital of France is"`,
`use_cache=False` (single prefill-shaped forward, no generation loop).
**Device: CPU** — no CUDA GPU exists on this development host (Apple
Silicon Mac, Metal/MPS only); MPS was available but CPU was chosen for
float32 determinism, honestly recorded rather than fabricating CUDA usage.

## 7. Hook and capture methodology (Part D)

`capture_summary.json`. A `register_forward_hook` on the exact mapped
module captured `inputs[0]` and `output`, each immediately
`.detach().clone().cpu()`'d (no retained autograd graph). Invocation count
validated via `_require_single_invocation()`: fails closed on 0 (never
reached) or >1 (ambiguous) invocations — exactly 1 invocation observed,
representing a **prefill**-shaped call (whole 5-token prompt processed in
one call, not a decode/generation loop).

## 8. Captured tensor metadata (Part I safety)

`captured_tensor_metadata.json`: shapes, dtypes, checksums, L2 norms,
mean/std/min/max, an 8-element bounded sample, and NaN/Inf counts only —
**no full weight or activation tensor is stored**. Input/output activation:
`(1, 5, 896)` float32; weight: `(896, 896)` float32; bias: absent.

## 9. TP2 mathematical decomposition (Part E)

Derived from the **real** verified PyTorch weight layout
(`weight.shape = [out_features, in_features]`), not assumed:

```
X = concat(X0, X1) along dim=1 (in_features / hidden, axis 0 in plan convention)
W = concat(W0, W1) along dim=1 (in_features)
Y_partial_r = X_r @ W_r^T
Y = all_reduce_sum(Y_partial_0, Y_partial_1)
(o_proj has no bias; the documented contract still applies bias exactly
 once, after reduction, whenever a bias is present -- proven by the bias
 negative tests, §16)
```

`tp2_shard_plan.json`: rank 0 `[0,448)`, rank 1 `[448,896)`, both shard
widths 448, disjoint and complete, `x_shard`/`w_shard` shapes and checksums
recorded per rank.

## 10. Rank-local shard isolation (Part K)

Verified directly: `shard[0].x_shard.shape[-1] == 448 != 896`,
`shard[1].x_shard.shape[-1] == 448 != 896`, `shard[0].range_end ==
shard[1].range_start == 448`, union `[0,896)` exactly. The rank-local
compute function (`rank_local_partial_output`) accepts only a `RankShard`
(local shard fields) — never the full `X`/`W`. `rank_input_leakage_count`
(cross-layer provenance) computed 0; a dedicated negative test proves the
detector itself fires when a full-width shard is deliberately substituted.

## 11. Serialized rank-local execution (Part F)

`rank_local_execution.json`. One CPU, ranks executed **sequentially**: rank
0's partial computed and its shard object explicitly released
(`del shards[0]`) before rank 1 begins. Explicitly not concurrent, not
multi-GPU, not NCCL. `rank_events.jsonl` records real timestamps for
`planned_ranks` → `rank_local_compute_done` (×2).

## 12. Collective and reconstruction contract (Part G)

Two distinguished paths, both exercised, both reusing D1 code unmodified:

- **`serialized_collective_replay`** (primary, required): D1's
  `CollectiveCoordinator.run_all_reduce_sum()` fed through a synchronous
  stdlib `queue.Queue` instead of `multiprocessing.Queue` — same code, same
  contract, sequential replay in one process.
- **`multiprocess_ipc_replay`** (bonus): D1's `DistributedProcessRuntime`
  run unmodified, spawning **two real OS processes** with real
  `multiprocessing.Queue` IPC, now fed the real captured `X`/`W^T` instead
  of D2's synthetic-shaped workload.

`collective_events.jsonl` records collective ID, sequence ID, participant
ranks, per-partial shapes/checksums, bytes contributed, reduced-tensor
checksum, and status for both paths. `reconstruction_summary.json`: bias
applied exactly once, after reduction (vacuously — `o_proj` has none).

## 13. Live-output numerical comparison (Part H)

`live_reference_comparison.json`, dtype-appropriate tolerance (`float32`,
`atol=rtol=1e-4`):

| Comparison | max abs error | allclose |
|---|---|---|
| serialized reconstruction vs. live module output | `1.79e-7` | ✅ |
| multiprocess-IPC reconstruction vs. live module output | `3.42e-7` | ✅ |
| direct standalone PyTorch reference (`X@W^T`) vs. live module output | `0.0` (exact) | ✅ |
| serialized reconstruction vs. direct PyTorch reference | `1.79e-7` | ✅ |

No float64 reconstruction was used for the primary path (float32
throughout, matching the live operator's real dtype); the IPC-replay path
promotes to float64 internally (D1's runtime always does), reported
separately and never conflated with the float32 serialized result. No
NaN/Inf in any comparison.

## 14. Cross-layer provenance (Part J)

`cross_layer_provenance.json` — all 14 required counters computed from real
events/validations, **all zero**:

```
operator_mapping_mismatch_count=0   layer_mismatch_count=0
weight_shape_mismatch_count=0       activation_shape_mismatch_count=0
partition_axis_mismatch_count=0     shard_coverage_mismatch_count=0
shard_overlap_count=0               rank_input_leakage_count=0
collective_mismatch_count=0         bias_application_mismatch_count=0
reference_output_mismatch_count=0   silent_fallback_count=0
temporary_tensor_leak_count=0       orphan_process_count=0
```

## 15. function_plans bug analysis and resolution (Part L)

Root cause **precisely identified**: `LLMFrontendNormalizationPass`'s
`isSupportedPattern()` requires q/k/v to be rank-4, statically-shaped,
`f32` tensors carrying explicit `attention.causal/scale/key_transposed`
and `attention.softmax_axis` attributes — a shape/dtype/attribute contract
written against the legacy hand-authored fixture. The real
`qwen-onnx-to-serving-mlir` frontend emits rank-2, dynamic-batch, `f16`
q/k/v with **none** of those attention attributes. Both facts confirmed by
direct source citation (`LLMFrontendNormalizationPass.cpp` lines ~66-104;
`qwen-onnx-to-serving-mlir/main.cpp`'s `emitOp()`). **Not narrowly
fixable**: a real fix requires either redesigning the frontend's
intentional dynamic-shape model (a documented, unrelated regression risk)
or inventing missing attention semantics the compiler never computes — both
explicitly out of D3A's scope. **Resolution taken**: acceptance criterion
14's second branch — proven, not asserted, that the issue does not exist in
the final D3A plan: `plan.distributed` (D3A's sole source of operator
identity and shard metadata) is collected unconditionally, independent of
`plan.function_plans`, and a dedicated regression test
(`test_function_plans_bug_does_not_affect_d3a_provenance`) asserts
`plan.function_plans == ()` while `plan.distributed` remains fully
populated and correctly used throughout. Full detail:
`function_plans_bug_analysis.json`.

## 16. Negative tests (Part M)

`negative_tests.json` — **17/17 pass**, no silent fallback anywhere:
operator ID → no module; ambiguous mapping; wrong layer; weight-shape
mismatch; captured-hidden-dim mismatch (two sub-cases); hook never fires;
hook fires unexpected count; shard overlap; shard coverage gap; rank
receives full tensor (detector proof); bias applied twice; bias omitted;
collective participant missing; collective sequence mismatch; reconstructed
output exceeds tolerance; temporary tensor file remains (detector proof);
TP2 live-tensor plan rejected by the untouched real-vLLM adapter path.

## 17. D1/D2 regressions (Part N)

`regression_summary.json`: D1 compiler (`DistributedPlanningTest`), D2
compiler (`DistributedStrategyPlanningTest` +
`DistributedStrategyPlanningPipelineTest`), D1 runtime
(`test_distributed_tp_process_runtime.py`), D2 runtime
(`test_distributed_d2_qwen_pipeline.py`) all green. Fresh D1 deadlock
negative-test re-run: real timeout, `missing_ranks=[1]`, zero orphans.
`no_d1_d2_d3a_rank_processes_remain: true`.

## 18. Measurements (Part O)

`performance_measurements.json`, 5 repetitions (structural/correctness
only, no speedup claim), CPU:

| Metric | Median | p95 |
|---|---|---|
| model load time | 0.529 s (single load) | — |
| tokenization time | 90.7 µs | 109.6 µs |
| forward time | 63.8 ms | 110.9 ms |
| hook overhead | 9.3 µs | 46.1 µs |
| capture copy time | 152.5 µs | 162.0 µs |
| rank 0 compute time | 171.0 µs | 189.9 µs |
| rank 1 compute time | 114.0 µs | 150.0 µs |
| serialized total rank compute time | 276.7 µs | 340.0 µs |
| collective/reconstruction time | 66.4 µs | 120.2 µs |
| reference comparison time | 42.5 µs | 79.4 µs |

Peak CPU memory: 3197 MB. Peak CUDA memory: not applicable (no CUDA GPU).

## 19. Test totals

- D3A test file (`test_distributed_d3a_live_qwen_tensor.py`): **24/24
  pass** (7 positive end-to-end + 17 negative).
- Full repo suite: 898 passed, 52 skipped, 3 failed, 9 errors — the
  identical pre-existing, D3A-unrelated failure/error set confirmed present
  with or without D3A's test file (`--deselect` comparison run).
- `test_summary.json`.

## 20. Known limitations

- Only `o_proj`, layer 0, is validated — matches D2's narrow scope, not
  extended in D3A.
- CPU only; no CUDA GPU exists on this development host (MPS available,
  not used, for float32-determinism reasons).
- "Serialized rank-local" and "multiprocess IPC replay" are both
  single-host localhost validations — neither is concurrent multi-GPU
  execution.
- The captured activation reflects one fixed prompt/seed; broader
  prompt/seed coverage was not exhaustively swept.
- `plan.function_plans` remains empty (pre-existing, unrelated,
  unfixed-in-scope issue, §15) — D3A's correctness does not depend on it,
  but a future stage needing real per-op decisions for the live-per-layer
  graph will still need this addressed.

## 21. Truth boundary

D3A's only allowed claim (verbatim, and true): *"A real activation captured
from a live Qwen2.5-0.5B-Instruct Transformers forward pass was matched to
the exact compiler-selected o_proj work item, partitioned according to the
D2 TP=2 plan, executed through serialized rank-local validation,
reconstructed through the D1 collective contract, and verified against the
original live model tensor with complete cross-layer provenance."* Not
claimed: real concurrent tensor parallelism, real multi-GPU execution,
NCCL, vLLM tensor parallelism, GPU-to-GPU communication, distributed
speedup, serving profitability. `truth_boundary.json`.

## 22. Recommended D3B dependency

D3A succeeded with complete, zero-mismatch provenance against a real
captured tensor — the operator/shard contract is now genuinely mature
enough to inform a launch specification. Recommended next stage:

**D3B: vLLM Distributed Launch-Spec Materialization and Fail-Closed
Validation** — generate and validate a real vLLM-compatible distributed
launch specification (`--tensor-parallel-size 2` config, real model
artifact references, real port/rank layout) derived from the same D2 plan
D3A validated, **without** claiming successful multi-GPU execution. D3B is
not implemented in this stage.
