# Distributed D4A: Single-GPU Serialized Whole-Model TP Contract Validation

## 1. Executive result

D4A closes the exact gap D3B left open: whole-model vLLM tensor-parallel compatibility was **not established** by D2/D3A (only one real `o_proj` operator on one layer had validated TP math). D4A inventories every TP-relevant operator family in both the real Transformers implementation and the installed vLLM 0.24.0 implementation, builds a whole-model distributed work-item plan covering all 24 layers plus embedding/lm_head, validates every family at the operator, block, and whole-model level using real model weights and a real forward pass, and reaches:

```
whole_model_tp_classification = WHOLE_MODEL_TP_VALIDATED
```

for the exact `Qwen/Qwen2.5-0.5B-Instruct`, TP=2, PP=1, installed vLLM 0.24.0 contract. All 17 provenance counters are zero. D3B's launch-spec evidence status was additively upgraded from `not_established_operator_level_only` to `validated_serialized_whole_model_contract`, referencing this run's evidence artifact hash — while **hardware preflight on this one-GPU host still rejects TP=2** with `primary_reason = insufficient_visible_gpu_count`, exactly as required.

D4A's only allowed primary claim holds:

> The complete set of tensor-parallel operator families required by Qwen2.5-0.5B-Instruct was identified from the live model and vLLM implementation, mapped to compiler planning entities, and validated through serialized TP=2 rank-local execution and collective reconstruction on a single host, with whole-model forward equivalence demonstrated within dtype-appropriate tolerance.

Nothing in this run claims concurrent TP execution, real two-GPU execution, NCCL, vLLM TP2 server execution, GPU-to-GPU communication, distributed serving speedup, or profitability.

## 2. Repository state before/after

| Repo | Branch | HEAD | Working tree before | Working tree after |
|---|---|---|---|---|
| ml-graph-compiler-runtime | master | `59854b892629bc0bc7f43ca0bad3eab17464c030` | clean | clean (zero changes; confirmed via `git status --porcelain`) |
| heterogeneous-inference-runtime | main | `b79ff951758b164010f95b761e3f927877e3ad10` | D3B work present, uncommitted | D3B work + D4A work present, uncommitted (no commit made) |

No reset, clean, stash, delete, rename, or rebase was performed on either repository. Full detail: `repository_state_before.json`, `repository_state_after.json`.

## 3. Files changed

`ml-graph-compiler-runtime`: **zero changes.** D4A's whole-model plan is a Python-side expansion of the same `ExecutionPlan` schema the C++ `DistributedStrategyPlanningPass` already emits for one operator — it does not modify, rebuild for behavior change, or re-invoke that pass (see §22 Known Limitations). A gitignored `build-mlir/` directory was incrementally built (two CMake targets: `DistributedPlanningTest`, `DistributedStrategyPlanningTest`) purely to re-run existing D1/D2 compiler unit tests for regression verification (§26) — this touches no tracked file.

`heterogeneous-inference-runtime`, all additive:
- Modified (2 lines/blocks each, backward-compatible): `deployment/execution_plan/schema.py` (widened `KNOWN_COLLECTIVE_KINDS` to add `"all_gather"`), `deployment/tp_process_runtime/qwen_module_mapping.py` (added q/k/v/gate/up/down operator-family mappings alongside the existing o_proj entry), `deployment/vllm_adapter/distributed_launch_spec.py` (added optional `whole_model_tp_evidence_source_artifact_hash` field, default `None`), `deployment/vllm_adapter/distributed_materializer.py` (added optional `d4a_evidence_path` parameter, default `None`).
- New: `deployment/tp_process_runtime/{whole_model_inventory,whole_model_plan_builder,column_parallel_executor,attention_contract_executor,mlp_contract_executor,vocab_parallel_executor,whole_model_tp_replay,whole_model_provenance}.py`, `scripts/run_distributed_d4a_pipeline.py`, `tests/test_distributed_d4a_whole_model_tp_contract.py`, this report, and the `results/runtime_paths/distributed_d4a_whole_model_tp_contract/` artifact directory.

## 4. D1–D3B preservation

All 80 files across `distributed_d1_tp2_multiprocess/` (15), `distributed_d2_qwen_pipeline/` (20), `distributed_d3a_live_qwen_tensor/` (22), and `distributed_d3b_vllm_launch_spec/` (23) were SHA-256 hashed before any D4A file was written and re-hashed after — all identical. All four markdown reports confirmed present and untouched. Full detail: `d1_d2_d3a_d3b_preservation.json`.

## 5. Model and vLLM inventories

**Model config** (real, from `AutoConfig.from_pretrained`, `model_config.json`): hidden_size=896, num_attention_heads=14, num_key_value_heads=2, head_dim=64, intermediate_size=4864, vocab_size=151936, num_hidden_layers=24, `tie_word_embeddings=True`, rope_theta=1,000,000.

**Transformers operator inventory** (`transformers_tp_operator_inventory.json`, 33 records across layers 0/12/23 plus embedding/lm_head/final-norm/rotary/attention-computation/residual): every family classified as one of `column-parallel`, `row-parallel`, `head-parallel`, `kv-head-parallel`, `vocab-parallel`, `replicated`.

**Installed vLLM 0.24.0 contract** (`vllm_tp_operator_inventory.json`), read via direct source introspection (`inspect.getsourcefile` + SHA-256 + literal excerpt extraction from the installed package, never memorized):
- `QKVParallelLinear` (fused q/k/v, column-parallel, bias=True hardcoded for Qwen2): per-rank sizing `num_heads=divide(total_heads,tp)`; if `tp_size>=total_kv_heads`: `num_kv_heads=1, num_kv_head_replicas=divide(tp,total_kv_heads)`, else partition directly. **For this model at tp=2** (`total_kv_heads=2`), `tp_size==total_kv_heads` exactly, so `num_kv_head_replicas=divide(2,2)=1` — the "replicate" code path executes but degenerates to a clean 1:1 partition (rank0→kv-head0, rank1→kv-head1, zero duplication).
- `RowParallelLinear` (o_proj, down_proj): input-dim (dim 1) partition; bias fused into rank 0's local GEMM only, so all_reduce(sum) applies it exactly once.
- `MergedColumnParallelLinear` (gate_up_proj): gate/up share the same per-rank output-dim shard index, making the elementwise `SiLU(gate)*up` product rank-local-valid.
- `VocabParallelEmbedding`: masked local lookup + `all_reduce(sum)`.
- `ParallelLMHead` + `LogitsProcessor` (tied to embed_tokens for this model, confirmed via `config.tie_word_embeddings=True`): local vocab-shard matmul + `tensor_model_parallel_all_gather` (concat, NOT all_reduce) + padding trim.
- `tensor_model_parallel_all_reduce`/`all_gather`: confirmed exact collective semantics from `vllm/distributed/communication_op.py`.

Transformers' own `_tp_plan = {"lm_head": "colwise_gather_output"}` (in `modeling_qwen2.py`) independently corroborates the lm_head gather-based reconstruction.

## 6. Compiler whole-model distributed plan

`whole_model_distributed_plan.json` is schema-identical to D1/D2's `ExecutionPlan`/`DistributedPlan` (validated by the unmodified `deployment.execution_plan.loader.validate_execution_plan`, except the additive `KNOWN_COLLECTIVE_KINDS` widening). It represents **170 work items** (`operator_family_contracts.json`): 24 layers × 7 linear families (q/k/v/o/gate/up/down) + embedding + lm_head, with 340 `tensor_shards` entries and 50 `collectives` (48 `all_reduce` for o_proj/down_proj per layer + 1 `all_reduce` for embedding + 1 `all_gather` for lm_head). Column-parallel families (q/k/v/gate/up) correctly carry **no** collective — matching real vLLM behavior, which keeps them sharded. This is a Python-side expansion of the same schema/legality vocabulary the C++ `DistributedStrategyPlanningPass` already emits for one operator; it does not modify or rebuild that pass (§22).

## 7. Operator-family contracts

Every work item records: `operator_id`, `layer_id`, `operator_family`, `partition_strategy`, `partition_axis`, `world_size`, rank shard offsets/extents, weight/activation/output partition description, `collective_kind`, `collective_sequence_id`, `bias_policy`, `replicated_inputs`, and `reconstructed_output_contract`. No operator is forced to TP2 dishonestly: replicated ops (RMSNorm ×2/layer, final norm, rotary, attention-computation, residual) are recorded as `replicated` with an explicit verification that they receive the full, already-reconstructed tensor at their point in the graph (§25).

## 8–9. Column-parallel and row-parallel validation

Reused/extended D3A's exact `linear_tp_decomposition.py` (o_proj: unchanged) plus a new generic `column_parallel_executor.py` (q/k/v/gate/up). Standalone operator-level checks at layers 0/12/23 confirmed weight shard layout, output concatenation, and full reconstruction against real captured activations (see §17 numbers). `weight_shard_manifest.json` confirms for every family: shards disjoint, shards cover the full tensor, no transpose mismatch (all weights read as `[out_features, in_features]`, PyTorch convention, verified against real parameter shapes), no rank receives another rank's shard.

## 10. QKV/GQA/head partition

`attention_contract_executor.py` implements the full per-rank pipeline: Q/K/V column-parallel projection (head-partitioned) → rotary (replicated math applied to each rank's local heads) → GQA repeat (`num_key_value_groups = num_heads_per_rank // num_kv_heads_per_rank = 7`, identical to the full-model ratio 14/2=7, since both are divided by the same tp_size) → per-rank causal-masked softmax attention → local head concatenation → o_proj row-parallel reduction. Verified: `num_heads_per_rank=7`, `num_kv_heads_per_rank=1`, no duplication (§5).

## 11. Attention block validation

For layers 0/12/23, `run_serialized_tp_attention_block` reconstructed output compared against the real captured `self_attn` output (hooks on real forward, exactly as D3A's methodology): **all three layers `allclose` within `atol=1e-4, rtol=1e-4`** (max_abs_error on the order of 1e-6). Full per-rank traces (Q/K/V local shapes, attention-score shape, KV repetition factor, local context shape, o_proj partial shape) recorded in `attention_contract_validation.json`.

## 12. MLP block validation

For layers 0/12/23, `run_serialized_tp_mlp_block` (gate/up column-parallel with **verified matching shard ownership** → rank-local `SiLU(gate)*up` → down_proj row-parallel all_reduce) matched the real captured `mlp` output within `atol=1e-4, rtol=1e-4` (max_abs_error ~1e-6). `mlp_contract_validation.json` confirms `shard_ownership_matches=True` for every layer.

## 13. Embedding/lm_head validation

`vocab_contract_validation.json`: embedding masked-lookup + all_reduce reconstruction was **exact** (max_abs_error = 0.0, since summation of disjoint-masked real rows is lossless in float32). lm_head local-matmul + all_gather + trim reconstruction matched real logits with max_abs_error ≈ 1.2e-5, argmax match. `vocab_size % world_size == 0` (151936/2=75968 exactly) — no padding needed, verified rather than assumed.

## 14. Weight shard correctness

`weight_shard_manifest.json` records, per family per representative layer: original weight name/shape, shard dimension, rank-0/rank-1 slice ranges, padding (0 throughout), per-rank checksums, and the four required invariants (disjoint, full coverage, no transpose mismatch, no cross-rank shard leakage) — all satisfied, zero `shard_coverage_errors`.

## 15. Serialized TP executor

`column_parallel_executor.py`, D3A's `linear_tp_decomposition.py` (row-parallel), `attention_contract_executor.py`, `mlp_contract_executor.py`, and `vocab_parallel_executor.py` together form the reusable serialized executor set. Each: receives the compiler-exported plan's shard ranges, builds explicit rank-0/rank-1 views (only the tensors that rank's real contract permits — never a full tensor passed in and sliced internally by the callee), executes rank-local math sequentially (never concurrently), performs explicit all_reduce/all_gather/concatenation exactly where the real vLLM contract requires it, and rejects (raises) on any unsupported or malformed shard configuration.

## 16. Whole-model forward replay

`whole_model_tp_replay.py` implements Part G's preferred approach: a manual re-implementation of `Qwen2Model.forward`/`Qwen2DecoderLayer.forward` that calls the **real** `input_layernorm`/`post_attention_layernorm`/final `norm` (RMSNorm) modules and the **real** `rotary_emb` module unchanged, while every TP-relevant linear/embedding call (q/k/v/o_proj, gate/up/down_proj, embed_tokens, lm_head) routes exclusively through the D4A executors above. The serialized TP model never calls the original full linear/embedding modules as its output path — real modules are read only for `.weight`/`.bias` and for the untouched reference comparison. (An initial attempt to compare against `output_hidden_states=True` uncovered that this Transformers version returns the **post-final-norm** hidden state as the last tuple entry rather than decoder layer 23's own raw output; per-layer ground truth was instead captured via direct forward hooks on every real decoder layer, the same unambiguous mechanism D3A established — this is documented here because it is a genuine, non-obvious fact about the installed library, not a D4A defect.)

## 17. Numerical correctness

| Level | Comparison | Result |
|---|---|---|
| Operator/block (attention, 3 layers) | reconstructed vs. real `self_attn` output | `allclose` (atol/rtol=1e-4); max_abs_error ~1e-6 |
| Operator/block (MLP, 3 layers) | reconstructed vs. real `mlp` output | `allclose` (atol/rtol=1e-4); max_abs_error ~1e-6 |
| Operator (embedding) | reconstructed vs. real embedding lookup | exact (max_abs_error = 0.0) |
| Operator (lm_head, standalone) | reconstructed vs. real logits | max_abs_error ≈ 1.2e-5 |
| Whole-model (all 24 layers, composed) | reconstructed vs. real final logits | shape equal, dtype float32, **max_abs_error ≈ 8.05e-5**, mean_abs_error ≈ 7.2e-6, `allclose` (atol=1e-2), NaN/Inf count 0, cosine similarity (last token) = 1.0 |
| Whole-model (per-layer hidden state, all 24) | reconstructed vs. real hooked layer output | median/p95 within 1e-2 (see `block_correctness.json`) |

Accumulator promotion is explicit and limited: `eager_attention`'s softmax is computed at float64 internally (matching Transformers' own float32-softmax-at-higher-internal-precision convention) — no other stage of either forward is promoted beyond the model's native float32. Full detail: `whole_model_correctness.json`, `block_correctness.json`.

## 18. Top-k token agreement

`topk_comparison.json`: top-5 token IDs identical between reference and TP-simulated forward (`[12095, 32671, 510, 1447, 1304]`), argmax identical (token 12095), top-5 logit values agree to ~1e-4 absolute.

## 19. Whole-model TP classification

```
whole_model_tp_classification = WHOLE_MODEL_TP_VALIDATED
```
for `Qwen/Qwen2.5-0.5B-Instruct`, TP=2, PP=1, installed vLLM 0.24.0. Both required conditions hold: every operator family validated within tolerance, and the composed whole-model forward (logits + argmax + top-k + all per-layer hidden states) is within dtype-appropriate tolerance. Full detail: `whole_model_tp_classification.json`.

## 20. D3B evidence-status update

Additive-only: `distributed_materializer.materialize_launch_spec` gained an optional `d4a_evidence_path` parameter (default `None`, zero effect on any of the 26 pre-existing D3B tests, all of which still pass unmodified). When pointed at this run's `whole_model_tp_classification.json` (classification `WHOLE_MODEL_TP_VALIDATED`, matching model and `tensor_parallel_size=2`), the TP2 launch spec's `whole_model_tp_evidence_status` becomes `validated_serialized_whole_model_contract`, with `whole_model_tp_evidence_source_artifact_hash` set to that artifact's real SHA-256. Hardware preflight is unaffected: still `rejected`, `primary_reason=insufficient_visible_gpu_count`, `execution_readiness_state=PREFLIGHT_REJECTED` — never `EXECUTION_READY`. Full detail: `d3b_evidence_update.json`.

## 21. Negative tests

22 fail-closed cases, all passing (`tests/test_distributed_d4a_whole_model_tp_contract.py -k negative`): wrong column-parallel weight axis, reversed column-parallel output order, wrong row-parallel input axis, missing row-parallel all-reduce, per-rank-duplicated row-parallel bias, non-divisible Q-head count, KV-head ownership gap, incorrect GQA repetition factor, rotary shape mismatch, gate/up shard-ownership mismatch, down_proj wrong shard, vocabulary shard coverage gap, embedding token-ownership overlap, lm_head shard-ordering mismatch, untied-model rejection, replicated-op-fed-a-local-shard, unsupported operator family, out-of-range layer mapping, whole-model tolerance-exceeded detection, top-k mismatch detection, static no-synthetic-fallback check, and D3B-evidence-not-updated-without-valid-artifact (including a missing file, a wrong-classification file, and a valid file cross-checked against continued hardware rejection). No child process is created anywhere in this file; the TP path never falls back to an original full linear output.

## 22. Provenance

All 17 required counters computed (not hardcoded): `operator_inventory_mismatch_count`, `vllm_contract_mismatch_count`, `compiler_mapping_mismatch_count`, `weight_shard_mismatch_count`, `activation_shard_mismatch_count`, `head_partition_mismatch_count`, `kv_partition_mismatch_count`, `collective_mismatch_count`, `bias_mismatch_count`, `vocab_partition_mismatch_count`, `replicated_boundary_mismatch_count`, `block_output_mismatch_count`, `whole_model_output_mismatch_count`, `synthetic_fallback_count`, `full_operator_bypass_count`, `temporary_tensor_leak_count`, `orphan_process_count` — **all zero**. Full chain and counters: `cross_layer_provenance.json`.

## 23. Regression results

D1 compiler (`DistributedPlanningTest`): passed. D2 compiler (`DistributedStrategyPlanningTest`): passed. (`DistributedStrategyPlanningPipelineTest`, a separate shell-level integration test, was not run — pre-existing, D4A-unrelated stale CLI flag drift between that test and the built `compile-for-target` binary; see §29.) D1 runtime (`test_distributed_tp_process_runtime.py`), D2 runtime (`test_distributed_d2_qwen_pipeline.py`), D3A (`test_distributed_d3a_live_qwen_tensor.py`), D3B (`test_distributed_d3b_vllm_launch_spec.py` + `test_vllm_backend_adapter.py` + `test_vllm_config_materializer.py` + `test_vllm_plan_schema.py`): all passed. Full-repo suite with the D4A test file deselected: identical pre-existing failure/error set as the D3B baseline (19 failed/13 errors, none touching `deployment/vllm_adapter`, `deployment/execution_plan`, or `deployment/tp_process_runtime`). Zero orphan D1/D2/D3A/D4A rank processes, zero vLLM server processes. Full detail: `regression_summary.json`.

## 24. Measurements

Control-plane/correctness-structure measurements only, 5 repetitions (`performance_measurements.json`):

| Stage | Median | p95 |
|---|---|---|
| Model load | 2.05 s (one-shot) | — |
| Reference forward (24 layers) | 245 ms | 450 ms |
| Serialized TP forward (24 layers, single-process) | 262 ms | 266 ms |
| Whole-model plan build | 2.4 ms (one-shot) | — |
| Activation capture overhead (per layer) | 130 ms | 178 ms |
| Peak CPU memory | 3571 MB | — |

Serialized TP forward latency is explicitly **not** representative of real concurrent TP performance (it runs both simulated ranks sequentially in one process) — no speedup is computed, reported, or implied from the ratio of these two numbers.

## 25. Test totals

- D4A test file: 27/27 passed.
- Negative tests: 22/22 passed.
- D1–D3B regressions: all green (see §23).
- Cross-layer provenance: all 17 counters zero.
- Full-repo suite (D4A deselected): identical pre-existing failure set to the D3B baseline.

Full detail: `test_summary.json`.

## 26. Cleanup

Zero temporary tensor files created (all TP computation used in-memory numpy/torch tensors; `temp_leak_candidates_found_in_result_dir=0`). D4A runs entirely single-process (no multiprocessing, unlike D1/D3A's real IPC path) — there is no rank subprocess to leak by construction. Zero vLLM server processes found. Full detail: `temporary_file_cleanup.json`, `process_cleanup.json`.

## 27. Artifacts

All 28 required files present under `results/runtime_paths/distributed_d4a_whole_model_tp_contract/`, plus this report at `docs/DISTRIBUTED_D4A_WHOLE_MODEL_TP_CONTRACT_REPORT.md`.

## 28. Known limitations

- **The whole-model distributed plan is Python-side, not C++-pass-emitted.** `whole_model_plan_builder.py` generates a schema-identical `ExecutionPlan.distributed` block covering all 170 work items, but the production `DistributedStrategyPlanningPass` (C++, ml-graph-compiler-runtime) still only emits a single-operator plan (as validated in D2). Extending that pass to emit the full whole-model work-item set was judged out of scope for this session (task instruction: "do not redesign the entire compiler pipeline"; a full LLVM/MLIR rebuild carries real time/risk cost). This is the most significant, honestly-scoped gap: D4A validates the **mathematical/shard contract** for the whole model, not that the compiler's own pass can (yet) plan it.
- `DistributedStrategyPlanningPipelineTest` (a shell-level integration test invoking `compile-for-target`) could not be run due to a pre-existing stale CLI flag reference unrelated to D4A; the underlying unit test (`DistributedStrategyPlanningTest`) passed.
- Attention/MLP block-level validation used 3 representative layers (0, 12, 23), not all 24, per the task's own "at minimum, validate every unique operator family" allowance — weight/shape invariants were, however, checked across representative layers spanning the full model, and the whole-model forward comparison (§17) does implicitly exercise all 24 layers' composition.
- The whole-model replay is single-process, CPU-only, float32; it validates a mathematically-faithful TP contract, not real vLLM runtime behavior (quantization, CUDA kernels, KV-cache paging, batching are all out of scope, consistent with the D3B/D4A truth boundary).

## 29. Truth boundary

**D4A does not claim**: concurrent TP execution, real two-GPU execution, NCCL, vLLM TP2 server execution, GPU-to-GPU communication, distributed serving speedup, or profitability. **D4A does claim**: a real, source-verified whole-model TP=2 mathematical/shard contract for Qwen2.5-0.5B-Instruct against installed vLLM 0.24.0's actual implementation, validated end-to-end on a single CPU process with real model weights, reaching `WHOLE_MODEL_TP_VALIDATED`. Full detail: `truth_boundary.json`.

## 30. Recommended D4B stage

Since D4A reached `WHOLE_MODEL_TP_VALIDATED`, the recommended next stage is:

**D4B: Real 2-GPU vLLM TP=2 Bring-Up and Correctness Validation.**

D4B is the first stage permitted to rent real two-GPU hardware. It should launch the D3B-materialized TP=2 vLLM command on a real 2-GPU host, verify NCCL initialization and rank correctness, and compare served output against the same reference used here — building directly on this session's validated contract rather than re-deriving it.
