# Distributed D2: Real Qwen Pipeline Distributed Strategy Planning

## 1. Executive result

`DistributedStrategyPlanningPass` is now registered inside the real production
`compile-for-target` pass pipeline (28 passes total, up from the exact,
source-verified 27 — not the "16" implied by the task wording). Running the
real per-layer Qwen ONNX-imported graph (`qwen2.5-0.5b`, `hidden_size=896`,
24 layers) through this pipeline with an explicit opt-in target profile
causes the pass to find the first real `llm.o_proj` operator instance,
generate TP1/TP2 candidates, evaluate Qwen-derived legality and an explicit
cost estimate, deterministically select TP2, and populate the same
`DistributedPlan` fields `ExecutionPlanBuilder`/`ExecutionPlanExporter`
already serialize (D1's schema, unmodified). The D1 multi-process runtime —
unmodified except one new fail-closed validation rule — consumed this real
plan directly, launched two OS processes, executed a Qwen-shaped
(`sequence_length=8 × hidden_size=896`) sharded matmul + `all_reduce(sum)`,
and matched a serial reference to `7.8e-14` max absolute error, with
complete cross-layer provenance (0/9 mismatches) and zero orphan processes.

**D2 acceptance criteria: all 20 satisfied** (see §18 test totals and the
final classification at the end of this report).

## 2. Repository state

Both repos began D2 in the exact state D1 left them (nothing committed).
HEADs unchanged throughout D2 as well. Full detail:
`repository_state_before.json` / `_after.json` in this result directory. D1's
result directory (15 files) and report were verified present and untouched.

| Repo | Branch | HEAD (unchanged before/after D1 and D2) |
|---|---|---|
| `ml-graph-compiler-runtime` | `master` | `dbf7329392bd2c70fa6ef25e359b277d171b3082` |
| `heterogeneous-inference-runtime` | `main` | `34aee51fef08dc447a6a52d938b4867d60eeef70` |

## 3. Verified production pipeline inventory (Part A)

Traced directly from `mlir_passes/tools/compile-for-target/main.cpp` and
`mlir_passes/include/FusionPasses.td` — not assumed from the task wording.
The in-source comment above the `PassManager` construction itself says
"serving passes (15-pass pipeline)"; that comment is stale. The actual count,
counted from the real `pm.addNestedPass`/`pm.addPass` call list, is **27
passes before D2, 28 after**. A separate `registerServingOptimizationPipeline()`
in `lib/serving/ServingPipeline.cpp` registers a 19-invocation ad-hoc
`mlir-opt` pipeline for standalone testing; its own comment states
"compile-for-target uses its own PM" — confirming it is not the production
path. D2 extends the real 27/28-pass `compile-for-target` `PassManager`.

Full 28-entry table (pass, source file, scope): `pipeline_inventory.json`.
`DistributedStrategyPlanningPass` is inserted after `QuantizationCoDesignPass`
and before `AlternativeLoweringPlanningPass` (i.e. after every op-level
backend/dtype/kernel/weight-classification decision is settled).

**Honest pre-existing finding, unrelated to D2, documented not fixed**:
`LLMFrontendNormalizationPass` does not fully consume the raw per-layer
attention pattern from the real ONNX-imported Qwen graph (0/48
`llm.attention_prefill`/`llm.attention_decode` emitted, 48/48 residual
`llm.attention_scores`), so `ServingPhaseAnalysisPass`'s `serving.policy` gate
never fires and `plan.function_plans` is empty for this input — independently
confirmed by the pre-existing `QwenOnnxServingPlanExportTest` ctest failure.
This does not block D2: `plan.distributed` is collected unconditionally at
module scope (like `global_decisions`/`cv_extension`), independent of that
per-function gate.

## 4. DistributedStrategyPlanningPass design (Part B)

Module-scoped (`Pass<"distributed-strategy-planning", "mlir::ModuleOp">`),
mirroring the codebase's existing `QuantizationPlanningPass` precedent for a
single whole-model decision (as opposed to the 27 other, per-`func::FuncOp`
passes). New files: `mlir_passes/lib/serving/DistributedStrategyPlanningPass.cpp`,
TableGen entry in `include/FusionPasses.td`, declaration in
`include/FusionPasses.h`. Reads `llm.hidden_size`/`llm.num_attention_heads`/
`llm.num_key_value_heads` and the explicit `distributed.opt_in` flag from the
module, walks for the first supported operator instance, generates/evaluates/
selects a candidate, and emits `distributed.*` module attrs consumed by
`ExecutionPlanBuilder::collectDistributedPlan()`. Fails closed
(`signalPassFailure()`) if a selected candidate is later found structurally
illegal or `buildDistributedPlan`/`validateDistributedPlan` disagree — this
must never happen and is defensive, not reachable in normal operation.

## 5. Qwen operator selected for D2 (Part D)

**`llm.o_proj`** — the attention output projection, a square
`hidden_size × hidden_size` matmul (`896 × 896` for `qwen2.5-0.5b`). Chosen
because: (a) it is a single, unambiguous 2-D projection with clean
row-parallel + `all_reduce(sum)` TP semantics — mathematically identical to
real Megatron/vLLM tensor-parallel `o_proj`/`down_proj` sharding; (b) unlike
QKV projections, it needs no per-head bookkeeping (partitioning happens
directly on the hidden/contraction dimension, not across attention heads);
(c) `llm.mlp` and other projections remain entirely unpartitioned/TP1 —
D2 deliberately does not pretend the whole model is tensor-parallel. Exactly
**one** operator instance (`qwen_prefill::llm.o_proj::layer_0`, the first of
48 real per-layer occurrences) is ever partitioned; the other 47 stay
untouched, matching "do not pretend the whole model is tensor-parallel."

## 6. TP1/TP2 candidate representation (Part C)

Reuses D1's `DistributedCandidate`/`generateDistributedCandidates()`
unchanged (always exactly `{tp1: ws=1, tp2: ws=2}`). Each candidate's
evidence dict (module attr `distributed.candidates`, exported verbatim by
`exportDistributedEvidenceReport`) carries every required field: `candidate_id`,
`strategy`, `world_size`, `tensor_parallel_size`, `pipeline_parallel_size`,
`partitioned_operator_ids`, `partition_axis`, `shard_count`,
`required_collectives`, `estimated_communication_bytes`,
`estimated_rank_local_compute`, `legality_status`, `rejection_reasons`,
`selection_score`, `truth_boundary`. See `qwen_distributed_candidates.json`.

## 7. Legality rules and rejection evidence (Part D)

`checkQwenCandidateLegality()` (new, `DistributedPlanning.h/.cpp`) evaluates
9 named rules per candidate, each with an explicit `pass`/`fail`/`not_applicable`
status (never silently skipped): `supported_operator_type`,
`static_shape_availability`, `tensor_hidden_dimension_divisibility`,
`head_count_divisibility` (**`not_applicable`** for `o_proj` — it partitions
`hidden_size` directly, not per-head), `rank_count_consistency`,
`partition_axis_support`, `required_collective_support`,
`tensor_shape_availability`, `static_vs_dynamic_shape_handling`,
`runtime_capability_availability`. TP1 bypasses all of these (always legal,
per "preserve TP1 behavior for unsupported graphs"). Negative-case evidence
(non-divisible, unsupported operator, missing static shape, no opt-in): all
four proven both as pure-C++ unit tests
(`DistributedStrategyPlanningTest.cpp`) and structurally in
`qwen_distributed_legality.json`. See §15 for the exact non-divisible-hidden-
size and unsupported-operator rejection runs.

## 8. Cost model and truth boundary (Part E)

`estimateDistributedCost()`: rank-local compute bytes (K-slice proxy, FLOPs
not estimated since sequence length is dynamic), communication bytes
(`world_size × hidden × dtype_bytes × 2`, contribute+broadcast, D1's
central-coordinator topology), collective count, process-launch-overhead
penalty (**calibrated from D1's measured `ipc_benchmark.json`
`world_size=2` `process_startup_s` median, ~3.1 ms/rank**), unsupported-
operation and fallback penalties. `total_score` is an explicit sum (lower is
better), never hidden inside a black-box formula. `truth_boundary`:
`analytical_and_d1_local_ipc_calibrated_not_gpu_measured_not_nccl_calibrated_not_multi_gpu_latency_predictor`.
Observed for the real graph: TP1 score `1792`, TP2 score `6209064` (TP2's
overhead dominates by construction — selection does not use this score to
pick TP2; see §9). `qwen_distributed_costs.json`.

## 9. Selection policy (Part F)

Deterministic, `policy_id = "d2_explicit_opt_in_v1"`:

| Condition | Result | `selection_reason` |
|---|---|---|
| TP2 legal + profile opt-in | TP2 selected | `legal_tp2_explicit_opt_in_profile` |
| TP2 legal, no opt-in | TP1 selected | `tp2_legal_but_opt_in_not_set` |
| TP2 illegal, only capability missing | TP1 selected | `no_distributed_capability` |
| TP2 illegal, structural/legality failure | TP1 selected | `tp2_illegal_candidate_rejected` |

Observed on the real graph: `nvidia_gtx1650_maxq.json` (no opt-in) →
`selected_candidate_id=tp1`, `no_distributed_capability`.
`nvidia_gtx1650_maxq_d2_distributed_opt_in.json` (opt-in) →
`selected_candidate_id=tp2`, `legal_tp2_explicit_opt_in_profile`. TP2 is
**never** globally forced. `qwen_distributed_selection.json`.

## 10. ExecutionPlan integration (Part G)

`ExecutionPlanBuilder::collectDistributedPlan(module)` (new, unconditional,
same tier as `collectGlobalDecisions`/`collectCVPlanExtension`) reads the
pass's `distributed.*` module attrs directly back into D1's `DistributedPlan`
struct — no post-export JSON patching anywhere. `real_qwen_tp1_execution_plan.json`
carries **no** `"distributed"` key at all (verified backward compatible,
byte-shape-identical to a legacy plan). `real_qwen_tp2_execution_plan.json`
carries the full block, produced by the **normal** `compile-for-target`
run → `ExecutionPlanBuilder::build()` → `ExecutionPlanExporter::exportToFile()`
call chain, the same chain every other Qwen plan goes through.

## 11. Exported Qwen TP1/TP2 plans

`real_qwen_tp1_execution_plan.json`: `model_identity.hidden_size=896`,
`num_layers=24`, no `distributed` key. `real_qwen_tp2_execution_plan.json`
`distributed` block (excerpt):

```json
"distributed": {
  "strategy": "tensor_parallel", "world_size": 2, "tensor_parallel_size": 2,
  "ranks": [{"rank_id":0,...},{"rank_id":1,...}],
  "tensor_shards": [
    {"tensor_id":"qwen_prefill::llm.o_proj::layer_0","range_start":0,"range_end":448},
    {"tensor_id":"qwen_prefill::llm.o_proj::layer_0","range_start":448,"range_end":896}
  ],
  "collectives": [{"collective_id":"all_reduce_0","kind":"all_reduce",
                   "participants":[0,1],"tensor_id":"qwen_prefill::llm.o_proj::layer_0"}]
}
```

`448 + 448 = 896` — the real, split, `hidden_size`. Produced by:
`qwen-onnx-to-serving-mlir --graph-facts configs/models/qwen_0_5b_onnx_graph_facts.json`
→ `compile-for-target --device-profile=<opt-in profile> --mlir=<raw> --out=<json>`.

## 12. Runtime materialization (Part H)

New `deployment/tp_process_runtime/qwen_workload.py`:
`build_qwen_derived_workload(plan)` derives `hidden_dim=max(shard.range_end)`
from the **plan itself** (never hardcoded) and builds a
`(sequence_length=8, 896) × (896, 896)` matmul — `sequence_length` is a
runtime-chosen deterministic constant (documented: real serving sequence
length is inherently dynamic, not part of the static plan). This is fed
**unmodified** into D1's `DistributedProcessRuntime.run()` — no new
distributed-execution code was written; D1's runtime, tested only against a
synthetic `K=16` problem before, now runs correctly against a real
Qwen-derived `K=896` problem. `runtime_materialization.json`.

## 13. Cross-layer provenance (Part I)

`deployment/tp_process_runtime/cross_layer_provenance.py`,
`verify_cross_layer_provenance()`: 9 explicit checks, each comparing a
compiler-declared value against a runtime-observed value (never hardcoded
`True`). Observed result: **0/9 mismatches, `all_match=True`**.

| Check | Result |
|---|---|
| selected operator ID == executed operator ID | ✅ `qwen_prefill::llm.o_proj::layer_0` both sides |
| planned world_size == launched rank count | ✅ 2 == 2 |
| planned rank IDs == executed rank IDs | ✅ `{0,1}` == `{0,1}` |
| planned shard widths == executed shard widths | ✅ `{0:448, 1:448}` both sides |
| planned collective_id == executed collective_id | ✅ `all_reduce_0` |
| planned sequence_id == executed sequence_id | ✅ `0` |
| planned participant set == executed participant set | ✅ `{0,1}` |
| no silent TP2→TP1 downgrade | ✅ `world_size=2` throughout |
| no fallback to synthetic default dimensions | ✅ `896` used, not D1's synthetic `16` |

`cross_layer_provenance.json`.

## 14. Correctness result

`correctness_summary.json`: `distributed_result_matches_serial_reference=true`,
`max_abs_error=7.82e-14`, `max_rel_error=2.20e-12` (float64, `rtol=atol=1e-9`
tolerance), shape/dtype match, `all_ranks_completed`/`all_collectives_completed=true`,
over the real `sequence_length=8 × hidden_size=896` Qwen-derived workload.
All 10 provenance counters (`rank_mismatch_count` … `orphan_process_count`)
zero on this run.

## 15. Negative tests (Part J)

`negative_tests.json` — compiler-side (ctest) and runtime-side (pytest), all
passing:

- Qwen dimension not divisible by TP2 (`hidden_dim=897`) → rejected.
- Unsupported operator selected (`llm.mlp`) → rejected.
- Missing required shape metadata (dynamic hidden dim) → rejected.
- Distributed capability unavailable (no opt-in) → TP1 selected, evidence recorded.
- Distributed pass "disabled" (non-opt-in profile) → TP1 plan has no `distributed` key at all.
- Duplicate rank ID → `DistributedRuntimeError` (fail closed).
- Collective references unknown operator/tensor → `DistributedRuntimeError` (new fail-closed rule added to `runtime.py`).
- Runtime receives dimensions differing from the compiler plan (`K=100` vs. plan's `896`) → `DistributedRuntimeError: shard coverage ...`.
- Cross-layer check catches a simulated operator-ID/dimension bookkeeping bug (`workload_hidden_dim=16` vs. real `896`) → `all_match=False`.
- Real-Qwen TP2 plan sent to the real-vLLM adapter path (`plan_schema.py`) → still rejected (`tensor_parallel_size must be == 1`), path untouched.

No silent fallback observed in any case.

## 16. D1 regression (Part K)

`d1_regression_summary.json`: D1 compiler ctest (`DistributedPlanningTest`,
8/8) — pass. D1 runtime pytest (`test_distributed_tp_process_runtime.py`,
16/16) — pass. D1 deadlock negative test re-run fresh: real timeout,
`missing_ranks=[1]`, zero orphans. `all_d1_regressions_green=true`. No
`d1-rank` or D2 rank processes remained after any test run (verified via
`os.kill(pid, 0)` → `ProcessLookupError` for every process, every run).

## 17. Measurements (Part M)

`performance_measurements.json`, 5 repetitions (structural execution only,
not a profitability/GPU claim):

| Metric | Median | p95 |
|---|---|---|
| compiler pipeline wall time (TP1 profile) | 32.5 ms | 63.9 ms |
| compiler pipeline wall time (TP2 opt-in profile) | 31.4 ms | 31.5 ms |
| legality+cost evaluation latency (in-process, per TP1+TP2 pair) | 3.68 µs | — |
| candidate count | 2 (constant) | — |
| runtime plan-load latency | 46.5 µs | 62.0 µs |
| process startup latency | 5.12 ms | 5.54 ms |
| rank-local compute latency | 0.10 ms | 0.12 ms |
| collective latency | 84.7 ms | 86.9 ms |
| end-to-end simulated execution latency | 304.5 ms | 307.7 ms |

Compiler latency is whole-pipeline wall-clock (28 passes), not isolated to
the new pass alone — MLIR pass-timing instrumentation was not wired into
compile-for-target in D2 (known limitation, §19). No speedup/slowdown claim
is made in either direction.

## 18. Test totals

- Compiler: `DistributedStrategyPlanningTest` 9/9 (pure C++ unit),
  `DistributedStrategyPlanningPipelineTest` 1/1 (real end-to-end integration,
  both profiles). Full ctest suite: 36/39 (3 pre-existing, D2-unrelated
  failures, identical to D1's baseline: `ImplementationCandidateTest`,
  `LLMFrontendNormalizationTest`, `QwenOnnxServingPlanExportTest`).
- Runtime: `test_distributed_d2_qwen_pipeline.py` 12/12. Full pytest suite:
  874/883 non-skip (846 D1-baseline-pre-existing-pass + 16 D1 + 12 D2 new =
  874 passed; 3 pre-existing failures + 9 pre-existing errors, identical to
  D1's baseline, confirmed unrelated).
- `test_summary.json`.

## 19. Known limitations

- `DistributedStrategyPlanningPass` is not wired into the legacy
  `qwen-to-serving-mlir` ModelSpec path (that path has no `o_proj` op at
  all) — only the real per-layer ONNX-imported path is exercised.
- `plan.function_plans` is empty for the real-Qwen exports in this stage,
  because of the pre-existing, unrelated `LLMFrontendNormalizationPass` gap
  described in §3 — `plan.distributed` is unaffected and correctly populated.
- Compiler-side latency measurement is whole-pipeline, not pass-isolated.
- Only one operator instance (`o_proj`, layer 0) is ever partitioned; the
  other 47 real per-layer occurrences and every other operator type remain
  TP1/unpartitioned by design.
- The executed workload is Qwen-**shaped** (real `hidden_size=896`), not a
  captured live tensor from a running Transformers/vLLM model.
- Selection score is not a validated profitability signal (explicitly out of
  scope, matches selector-v4 exclusion).

## 20. Truth boundary

D2's only allowed claim (verbatim, and true): *"The real Qwen compiler
serving pipeline generated, evaluated, selected, and exported TP=1/TP=2
distributed strategy candidates as part of the normal ExecutionPlan
construction path, and the runtime consumed the resulting real-Qwen plan
through the D1 multi-process simulator with complete plan-to-execution
provenance."* Not claimed: real Qwen tensor parallel execution, real vLLM
distributed execution, real GPU tensor parallelism, NCCL, multi-GPU speedup,
distributed serving profitability. `truth_boundary.json`.

## Recommended D3 boundary

D2's operator/shard contract (`o_proj`, real `hidden_size`, real per-layer
identity) is now genuinely plan-to-execution provenance-complete and
correctness-verified against real Qwen dimensions — but the executed tensor
is still synthetic-Qwen-shaped, not a captured live value. The natural next
dependency is therefore:

**D3A: Live Qwen Tensor Capture and Serialized Rank-Local Validation** —
capture a real intermediate activation tensor from one `o_proj` forward pass
(e.g. via a PyTorch/Transformers forward hook on the real `qwen2.5-0.5b`
checkpoint), serialize it, shard it exactly as D2's plan declares, and
re-verify D1's collective + reconstruction against that real captured value
instead of a Qwen-*shaped* synthetic one. This is the correct next step
*before* D3B (vLLM Distributed Launch Specification Materialization),
because materializing a vLLM `--tensor-parallel-size 2` launch spec from a
plan whose operator/shard contract has never touched a real tensor would
outrun the evidence D2 actually has.

D3 is not implemented in this stage.
