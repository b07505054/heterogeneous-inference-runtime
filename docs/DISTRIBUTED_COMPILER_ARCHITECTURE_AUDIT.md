# End-to-End Architecture Audit: Compiler Ownership of Distributed vLLM Execution

Scope: `ml-graph-compiler-runtime` (compiler) and `heterogeneous-inference-runtime`
(runtime), covering the full D1→D2→D3A→D3B→D4A→D4B→D5 chain. Read-only audit;
no source was modified to produce this document.

**Update (D6): §1's central finding — that the compiler computed cost
evidence but never compared it to select a winner — has been resolved by
implementation, not by further audit alone. See "D6 re-audit" at the
bottom of this document for the post-implementation trace and updated
answers to the five closing questions. §1–§10 below are preserved
unmodified as the historical, pre-D6 record; do not read them as
describing current behavior without also reading the D6 section.**

---

## 1. Compiler ownership

**Claim under test**: the TP decision originates in
`DistributedStrategyPlanningPass` / the D1 candidate-generation functions,
not a later hard-coded value.

### 1.1 Candidate generation (real, exhaustive, not a search)

`mlir_passes/lib/serving/DistributedPlanning.cpp` (declared in
`mlir_passes/include/serving/DistributedPlanning.h:37`):

```cpp
std::vector<DistributedCandidate> generateDistributedCandidates();
```

Always produces exactly `{tp1, tp2}` (`DistributedPlanning.h:34-37`). This
is explicit by design — the header states "D1 requires generation to be
explicit, not implicit in a selector."

### 1.2 Legality (real, per-operator, fail-closed)

`checkQwenCandidateLegality(candidate, ctx)` (`DistributedPlanning.h:123-125`,
implemented in the corresponding `.cpp`) evaluates real per-operator facts:
`hidden_dim` divisibility by `tensor_parallel_size`, operator-type allow-list
membership, head-count divisibility, rank-count consistency, static-shape
availability, and distributed-capability availability. `hidden_dim` and
`hidden_dim_is_static` are read directly from the op's own MLIR result
tensor type (`DistributedStrategyPlanningPass.cpp:209-219`), never assumed.

### 1.3 Cost estimate (real, computed, but — critical finding — NOT the selector)

`estimateDistributedCost(candidate, ctx)` (`DistributedPlanning.h:143-145`)
is called for **both** TP1 and TP2 candidates
(`DistributedStrategyPlanningPass.cpp:245`, `tp1Cost`/`tp2Cost`), and its
`total_score` is recorded into `distributed.candidates[*].selection_score`
(`DistributedStrategyPlanningPass.cpp:88`) for every candidate.

**However**, tracing the actual selection branch
(`DistributedStrategyPlanningPass.cpp:260-277`):

```cpp
const DistributedCandidate *selected = tp1;
if (tp2Legality.legal && opCtx.distributed_capability_available) {
  selected = tp2;
  selectionReason = "legal_tp2_explicit_opt_in_profile";
} else if (tp2Legality.legal && !opCtx.distributed_capability_available) {
  selected = tp1;
  selectionReason = "tp2_legal_but_opt_in_not_set";
} else {
  selected = tp1;
  ...
}
```

`tp1Cost.total_score` and `tp2Cost.total_score` are **never referenced** in
this branch. Selection is a pure function of (a) TP2 legality and (b) a
boolean module attribute `distributed.opt_in`
(`DistributedStrategyPlanningPass.cpp:175-176`), which is set by whichever
target-profile JSON file the pipeline is invoked against — concretely,
`configs/target_profiles/nvidia_gtx1650_maxq.json` (no opt-in → TP1) vs.
`nvidia_gtx1650_maxq_d2_distributed_opt_in.json` (opt-in → TP2), per
`heterogeneous-inference-runtime/scripts/run_distributed_d2_pipeline.py:40-41`.

This is not hidden — the header comment in `DistributedPlanning.h:9-10`
states outright: *"There is no profitability selector here."* The policy ID
recorded alongside the decision is literally `"d2_explicit_opt_in_v1"`
(`DistributedStrategyPlanningPass.cpp:262`), and its truth-boundary string
is `"...not_distributed_profitability_claim"`
(`DistributedStrategyPlanningPass.cpp:284-288`).

**Finding (important, not a bug)**: the compiler's TP1-vs-TP2 choice is
legality-gated and opt-in-gated, not cost/profitability-gated. The cost
estimate is real, computed, and exported as evidence, but is not the thing
that decides. Anyone reading the D2 report's "legality/cost analysis, TP2
selection" framing should understand "cost analysis" as *evidence
generation*, not *decision input*, for this stage. See §10 for how this
affects the D4B/D5 truthfulness question.

### 1.4 TP degree is not hard-coded downstream of this pass

- `encodeSelectedDistributedPlan` (`DistributedStrategyPlanningPass.cpp:104-157`)
  writes `distributed.tensor_parallel_size` etc. onto the module purely from
  the already-`selected` candidate's fields — no separate literal.
- `ExecutionPlanExporter::serializeDistributedPlan`
  (`ExecutionPlanExporter.cpp:905-910`) writes `tensor_parallel_size` from
  the in-memory `DistributedPlan` struct field, not a literal.
- Every runtime-side consumer (§2–§4) reads this same field back out of the
  JSON; none of them re-decide or override it.

---

## 2. ExecutionPlan ownership — field-by-field

| Field | Producer | Consumer | Serialization | Deserialization |
|---|---|---|---|---|
| `distributed.strategy` | `DistributedStrategyPlanningPass.cpp:106` (from `buildDistributedPlan`) | `distributed_materializer._effective_distributed_fields` (raises `UnknownDistributedStrategyError` if not `"tensor_parallel"`) | `ExecutionPlanExporter.cpp:907` | `schema.py:511,523` (`DistributedPlan.strategy`) |
| `distributed.world_size` | `DistributedStrategyPlanningPass.cpp:107-108` | `_effective_distributed_fields`; `ServerLaunchController` process-count expectation | `ExecutionPlanExporter.cpp:908` | `schema.py:512,524` |
| `distributed.tensor_parallel_size` | `DistributedStrategyPlanningPass.cpp:109-110` | `distributed_cli.build_cli` → `--tensor-parallel-size` argv | `ExecutionPlanExporter.cpp:909` | `schema.py:513,525` |
| `distributed.pipeline_parallel_size` | same pass | `build_cli` → `--pipeline-parallel-size` | `ExecutionPlanExporter.cpp:910` | `schema.py:514,526` |
| `distributed.ranks[].rank_id` / `.logical_device` | `DistributedStrategyPlanningPass.cpp:114-123` | `build_rank_placement` (physical GPU assignment) | `ExecutionPlanExporter.cpp:912-919` | `schema.py:515,527-531` |
| `distributed.tensor_shards[]` | same pass, from `buildDistributedPlan` | consumed by `tp_process_runtime` (D1/D2 simulated runtime) and D4A's whole-model replay; **not** consumed by the vLLM materializer (vLLM's own installed TP implementation shards weights internally — the plan only proves compiler-side shard-legality, never re-implements vLLM's sharding) | `ExecutionPlanExporter.cpp:921-932` | `schema.py:516,532-536` |
| `distributed.collectives[]` | same pass | same as tensor_shards — proves the compiler's collective-legality reasoning, not consumed for real NCCL configuration (vLLM configures its own NCCL collectives) | `ExecutionPlanExporter.cpp:934-945` | `schema.py:517,537-541` |
| `distributed.truth_boundary` | same pass | preserved verbatim end-to-end, never rewritten by the runtime (confirmed: the field observed in `real_qwen_tp2_execution_plan.json` is `"d1_simulated_localhost_multiprocess_ipc_not_real_gpu_not_nccl_not_measured_gpu_performance"` — this is D1's own truth-boundary string, carried through D2/D3B/D4B/D5 unchanged) | `ExecutionPlanExporter.cpp:945` (via the wrapping object) | `schema.py:518,541` |

**No field is ignored.** `tensor_shards` and `collectives` are the two
fields that are *not* consumed by the real vLLM launch path — this is
correct and intentional, not a gap: vLLM's installed TP implementation
does its own weight sharding and collective configuration; the compiler's
shard/collective plan exists to prove the compiler's *legality reasoning*
was sound (divisibility, non-overlapping coverage, contiguous ranks — see
D2's `validateDistributedPlan`), not to hand-drive vLLM's internals. This
distinction is explicitly documented in D4B's truth boundary ("not
evidence that vLLM executed the compiler's work items individually rather
than its own installed TP implementation").

**Standing architectural note — carried over from §1.3**: the
`distributed.truth_boundary` string physically present in
`real_qwen_tp2_execution_plan.json` is D1's simulated-runtime truth
boundary, not a D2/D3B-specific one. This is consistent with the fact
that `DistributedStrategyPlanningPass` calls the *same*
`buildDistributedPlan` that D1 defined and never overwrites its
truth-boundary string with a D2-specific one. Not a bug — but a precise
reader should note that the *plan schema itself* still self-identifies as
"D1 simulated," even when reused three stages later by the real-hardware
D4B/D5 chain. The stages built their own, additional truth-boundary
artifacts (`truth_boundary.json` per stage) to correct for this rather
than editing the embedded string.

---

## 3. Materializer audit

### 3.1 TP degree comes from ExecutionPlan

`distributed_materializer.py:104-120`, `_effective_distributed_fields`:

```python
def _effective_distributed_fields(plan):
    if plan.distributed is None:
        return 1, 1, 1, (0,)
    d = plan.distributed
    if d.strategy not in KNOWN_DISTRIBUTED_STRATEGIES:
        raise UnknownDistributedStrategyError(...)
    return d.tensor_parallel_size, d.pipeline_parallel_size, d.world_size, rank_ids
```

Called at `distributed_materializer.py:219`, and its output (`tp`) flows,
unmodified, into the `tensor_parallel_size = prov(...)` provenance-wrapped
field (`distributed_materializer.py:268`), into `fields_for_cli`
(`distributed_materializer.py:316`), into `build_cli`
(`distributed_cli.py:109-111`), into `argv`. No intermediate reassignment.

### 3.2 Model path comes from plan/config

`REAL_HF_MODEL_ID` / `KNOWN_MODEL_ID_MAP` (`distributed_materializer.py:60-75`,
added in D5) maps the compiler's abbreviated `plan.model_identity.model_id`
(e.g. `"qwen2.5-0.5b"`) to a real HF checkpoint id. This mapping table is
the one place a human decision enters — D3A independently validated the
0.5B mapping by comparing compiler-declared metadata (`hidden_size=896`,
`num_layers=24`) against the real cached checkpoint's own config
(documented in `results/runtime_paths/distributed_d3a_live_qwen_tensor/`).
The D5 addition (`qwen2.5-7b` → `Qwen/Qwen2.5-7B-Instruct`) was verified
the same way in this session (real `AutoConfig.from_pretrained` fetch
matched against the plan's declared `hidden_size=3584`,
`num_attention_heads=28`, `num_kv_heads=4`, `num_layers=28` before the
plan file was written).

### 3.3 Launch arguments come from compiler output

Every CLI argument in `build_cli` (`distributed_cli.py:92-145`) is sourced
from the `fields` dict passed by `materialize_launch_spec`, which is
itself sourced from either (a) `plan.distributed.*` (compiler-plan-sourced,
tagged `FieldSource.COMPILER_PLAN`), (b) real runtime discovery (GPU count,
CUDA availability — tagged `FieldSource.RUNTIME_DISCOVERY`), or (c) an
explicit, documented D3B default for fields the compiler plan does not
declare (host, port, `max_model_len`, `max_num_seqs`,
`gpu_memory_utilization`, etc. — tagged `FieldSource.EXPLICIT_D3B_DEFAULT`,
each with an inline comment explaining why). Every field's source is
recorded in `field_provenance` and exported in the launch spec JSON
(`whole_model_evidence_hash` etc.) — this is inspectable per-field, not
asserted.

### 3.4 No hidden constants, no manual TP override, no env-var override

- `build_cli` contains no `tensor_parallel_size = N` literal anywhere.
- No environment variable is read for `tensor_parallel_size`
  (`_environ_conflicts`, `distributed_materializer.py:186-195`, only
  *detects conflicts* between planned env vars and pre-existing ones —
  it never lets a pre-existing env var win).
- `ServerLaunchController` (`distributed_launch_controller.py`) has no
  `force_launch`/`ignore_preflight`/`allow_unsupported` parameter anywhere
  in its dataclass or methods (confirmed by direct reading of the full
  file) — a rejected preflight cannot be bypassed by any parameter this
  class exposes.
- `VLLMDistributedAdapter` (`backend_adapter.py:73-90`) — the class that
  wraps the materializer for the general `ExecutionPath` framework — is
  an 18-line pass-through with a docstring stating exactly this: *"No
  `force`, `ignore_preflight`, or hidden bypass parameter exists anywhere
  on this class."*

### 3.5 Full-repository literal search (as requested)

```
grep -rn "tensor_parallel_size" --include="*.py" deployment/ scripts/   → 69 occurrences
grep -rn "tensor_parallel_size\s*=\s*2\|world_size\s*=\s*2" (excluding tests) → 2 occurrences
grep -rn "\-\-tensor-parallel-size"                                      → 6 occurrences
```

Every occurrence, documented:

| Location | What it is | Hardcode? |
|---|---|---|
| `deployment/tp_process_runtime/whole_model_tp_replay.py:211` | D4A's CPU-only numerical validation of the whole-model TP contract against real HF Transformers weights. `world_size=2` is hardcoded because D4A's *entire documented scope* is validating the one strategy (TP=2) the compiler chain has ever produced end-to-end. | Yes, but self-contained — feeds only `whole_model_tp_classification.json`, an **advisory** artifact (see 3.6) |
| `deployment/tp_process_runtime/whole_model_plan_builder.py:35` | Same D4A component, builds the whole-model (170-work-item) `DistributedPlan` shape in Python, explicitly documented in its own docstring as *"a Python-side expansion... it does not modify or rebuild that C++ pass"* — i.e., an honestly-labeled extension, not a silent bypass. | Yes, same scope as above |
| `deployment/vllm_adapter/policy_executor.py:24`, `deployment/vllm_adapter/backend_adapter.py:116`, `scripts/run_vllm_max_num_seqs_diagnostic.py:23` | Direct CLI-arg construction, but `tensor_parallel_size` is *read from* a `fixed`/`config` dict passed in by the caller, never assigned a literal. `policy_executor.py` is a separate, unrelated compiler-guided policy (vLLM's `max_num_seqs`, schema `vllm.max_num_seqs.policy.v1`) with its own independent fail-closed validation (`validate_policy`, `policy_executor.py:9-22`) — not part of the TP1/TP2 decision chain. | No |
| `deployment/vllm_adapter/gpu_evidence.py:5`, `scripts/run_distributed_d3b_pipeline.py:235` | Prose comments explaining the module's evidentiary philosophy | No (not code) |
| `deployment/vllm_adapter/distributed_environment.py:111` | Prose comment | No (not code) |

**No occurrence of a hardcoded literal `tensor_parallel_size` (or
equivalent) was found anywhere in the real D3B/D4B/D5 vLLM launch path**
(`distributed_materializer.py`, `distributed_cli.py`,
`distributed_launch_controller.py`, `backend_adapter.py`'s
`VLLMDistributedAdapter`).

### 3.6 The advisory-evidence exception, precisely

`_resolve_whole_model_tp_evidence` (`distributed_materializer.py:178-204`)
reads D4A's `whole_model_tp_classification.json` (produced by the §3.5
hardcoded-TP=2 CPU validation) and, if it matches
(`classification == "WHOLE_MODEL_TP_VALIDATED"` and `model` matches and
`tensor_parallel_size == 2`), sets
`whole_model_tp_evidence_status = "validated_serialized_whole_model_contract"`.
This status is *never* read by anything that decides `tensor_parallel_size`
— it is a separate spec field, output-only, consumed only by test
assertions and reports. Grep confirms: no `if ... whole_model_tp_evidence`
branch anywhere sets or modifies `tensor_parallel_size`.

---

## 4. Runtime ownership — the call chain, traced

```
ExecutionPlan JSON (file path, chosen by caller — see §7)
    │  deployment.execution_plan.loader.load_execution_plan()
    ▼
distributed_materializer.materialize_launch_spec()
    │  reads plan.distributed.* (or its absence) → tp, pp, world_size
    │  reads plan.model_identity.model_id → real_hf_model_id (§3.2)
    │  builds field_provenance dict, calls run_preflight() (fail-closed gate)
    ▼
distributed_cli.build_cli()
    │  emits argv purely from the fields dict (§3.3); checks each arg
    │  against the installed vLLM registry (never silently drops/renames)
    ▼
distributed_launch_controller.ServerLaunchController
    │  argv passed to subprocess.Popen(list(argv), ..., shell=False)  ← no
    │  string interpolation, no shell parsing
    ▼
real subprocess: .venv/bin/python -m vllm.entrypoints.openai.api_server ...
    ▼
vLLM's own distributed bootstrap (multiproc_executor.py, parallel_state.py)
    ▼
real NCCL init, real GPU workers (verified §5)
```

Confirmed no manual editing step exists between any two arrows:
`materialize_launch_spec` returns a frozen dataclass (`VLLMDistributedLaunchSpec`,
`@dataclass(frozen=True)`, `distributed_launch_spec.py`); `build_cli`
returns a frozen `CLIRepresentation`; `ServerLaunchController.start()`
(`distributed_launch_controller.py:88-92`) passes `list(self.argv)` — the
exact tuple built above — directly to `subprocess.Popen`, with
`start_new_session=True` for process-group tracking, never a shell string.

---

## 5. vLLM distributed verification (from real D4B logs)

Source: `results/runtime_paths/distributed_d4b_real_2gpu_vllm_tp2/nccl_initialization.json`
and `rank_gpu_mapping.json`, both produced by direct regex extraction from
raw server stdout (`logs/tp2_server.log`), not inferred from CLI flags.

- `tensor_parallel_size` equals compiler output: `d4b_tp2_launch_spec.json`'s
  `tensor_parallel_size: 2` traces via SHA-256 hash cross-reference
  (`source_artifact_hashes.json` equivalent) back to
  `real_qwen_tp2_execution_plan.json`'s `distributed.tensor_parallel_size: 2`.
- `world_size` matches: raw log line —
  `"world_size=2 rank=0 local_rank=0 ... backend=nccl"` and
  `"world_size=2 rank=1 local_rank=1 ... backend=nccl"` (two distinct
  `Worker` PIDs, 20528 and 20539).
- NCCL initialized: `"NCCL version 2.28.9+cuda13.0"` appears twice (once
  per worker), `backend_mentions: {"nccl": 195, "gloo": 0}`.
- Two workers launched, each bound to a different GPU:
  `rank_gpu_mapping.json` — PID 20528 → `GPU-3e930a03-...`, PID 20539 →
  `GPU-c8703d8b-...`, `two_distinct_gpus_used: true`,
  `duplicate_assignment: false`, cross-checked against
  `expected_rank_placements` (`rank_placement_agrees_with_d3b_launch_spec: true`).
- Model loaded in distributed mode: same log evidence shows both workers
  independently completing `ncclCommInitRank` and weight loading before
  the API server reports readiness (health check 200 OK).

---

## 6. Negative-path audit

| Mutation | Where it's tested | Result |
|---|---|---|
| Compiler outputs TP1 → runtime launches TP1 | D4B pipeline's own TP1 launch (`d4b_tp1_launch_spec.json`, `tensor_parallel_size: 1`) + every D5 TP1 sweep cell | Confirmed — `visible_devices=(0,)`, single process, no NCCL rank>1 log lines |
| Compiler outputs TP2 → runtime launches TP2 | D4B's TP2 launch | Confirmed — §5 |
| Compiler outputs illegal TP (dimension not divisible) | Compiler-side: `DistributedStrategyPlanningTest` case `qwen_dimension_not_divisible_by_tp2` (run live via `ctest` inside `run_distributed_d2_pipeline.py:261,267-279`, not just trusted from a stale log) | Fails legality, `selected = tp1` |
| Compiler outputs illegal TP (unsupported operator) | `unsupported_operator_selected_for_partitioning` (same ctest run) | Fails legality |
| Runtime receives TP2 on a 1-GPU host | `test_tp2_materializes_and_preflight_rejects_on_one_gpu_host` (`test_distributed_d3b_vllm_launch_spec.py:100`) | `preflight.passed == False`, `insufficient_visible_gpu_count` in `rejection_reasons` |
| Attempted launch despite rejected preflight | `test_no_subprocess_launched_for_rejected_tp2_spec` (`:151`), `test_negative_attempted_launch_while_preflight_rejected_raises_provenance_bypass` (`:390`) | No subprocess started; explicit provenance-bypass exception raised |
| Malformed plan: `tensor_parallel_size` set but `world_size` inconsistent | `test_negative_tp_pp_mismatch_malformed_plan_rejected_by_loader` (`:187`) | `ExecutionPlanError` raised by the loader before the materializer ever runs |
| Malformed plan: gap in tensor-shard coverage | `test_negative_malformed_distributed_plan_gap_in_shards` (`:294`) | Rejected by `validateDistributedPlan`'s structural check (same function used for real compiler-built plans) |
| Two TP ranks mapped to the same physical GPU | `test_negative_two_tp_ranks_never_mapped_to_one_gpu` (`:216`) | `no_duplicate_physical_device == True`; unmappable rank left `None`, never fabricated |
| `--tensor-parallel-size` unsupported by installed vLLM version | `test_negative_unsupported_cli_flag_is_never_silently_emitted` (`:226`) | Flag omitted from argv entirely, `all_arguments_supported == False` — never silently dropped without being reported |
| A stray `tensor_parallel_size: 2` reaches the *non-distributed* vLLM schema/adapter | `test_reject_tp2_real_qwen_plan_on_real_vllm_adapter_path` (`test_distributed_d2_qwen_pipeline.py:205`) + `VLLMBackendAdapter.validate()` (`backend_adapter.py:35-36`, `"tensor_parallel_size_must_be_1"`) | Two independent fail-closed layers (schema validator + adapter validator) both reject it |
| D5: TP1 illegal on memory grounds, TP2 legal | `TPCostModel.decide()`'s `is_feasible` branch (`tp_cost_model.py`), unit-tested in `test_cost_model_forces_tp2_when_tp1_infeasible` | Forces TP2, `reason: "capacity_forced"` — never fired in the real D5 measurement set (§10 caveat) |

All failures occur at the layer closest to the invalid input (loader
rejects malformed JSON before the materializer runs; preflight rejects
hardware-infeasible requests before the launch controller runs; the
launch controller never starts a subprocess for a rejected preflight) —
i.e., genuinely fail-closed, not fail-open-then-caught.

---

## 7. Bypass audit

| Path | Production or test? | Bypasses compiler? | Disposition |
|---|---|---|---|
| Manual CLI (typing vLLM flags by hand) | N/A — no such wrapper script exists in this repo that accepts free-form CLI flags for the distributed path | N/A | Nothing to remove; not present |
| Manual JSON (hand-editing an ExecutionPlan file) | Possible in principle (it's a JSON file on disk) | Yes, in principle — anyone could hand-edit `real_qwen_tp2_execution_plan.json` | **Real residual risk.** Nothing computes a fresh hash of the ExecutionPlan against a known-good compiler-output hash before materializing it in the general case; D4B/D5's *scripts* do record `source_execution_plan_sha256`/`source_artifact_hashes.json` as an audit trail, but `materialize_launch_spec()` itself does not refuse to load a plan whose hash doesn't match a compiler-signed value, because no such signature exists. Should be documented as a known trust boundary: "the materializer trusts the ExecutionPlan file's contents; provenance is established by pipeline discipline (hash-recording scripts) and code review, not by cryptographic verification." |
| Environment variables | Checked in §3.4 | No override path found | Documented as safe |
| Test-only path | `tests/test_distributed_d3b_vllm_launch_spec.py` and siblings construct `fields` dicts by hand to test `build_cli`/`run_preflight` in isolation (e.g. `test_negative_unsupported_cli_flag_is_never_silently_emitted:232-249`) | These are unit tests exercising the function directly with synthetic inputs — this is normal, correct unit-testing practice, not a bypass a real launch could take, since `materialize_launch_spec` is what real scripts call, and it is not test-only code | Fine as-is |
| Benchmark-only path | `tp_benchmark_harness.py` (D5) sends real HTTP requests to an *already-materialized-and-launched* server; it does not construct or alter launch args | Not a bypass — measurement-only, downstream of materialization | Fine as-is |
| Legacy launch script | `deployment/vllm_adapter/policy_executor.py` + `scripts/run_vllm_max_num_seqs_diagnostic.py` — a **separate, real, production** code path that launches a real vLLM server directly from a JSON `--fixed` config, entirely independent of `distributed_materializer.py` | Production (has its own fail-closed schema validation, `policy_executor.validate_policy`) | Does **not** bypass the TP1/TP2 decision — it is scoped to a different policy axis (`max_num_seqs`) and, in every artifact found, is used with `tensor_parallel_size: 1` (single-GPU). **However**: it is architecturally capable of launching vLLM with any `tensor_parallel_size` its input JSON declares, through a code path that never touches `distributed_materializer`'s preflight/rank-placement logic. **Recommendation**: document this explicitly as "a second, independent real-launch path for a different compiler decision; if ever pointed at a distributed config, it has no rank-placement or GPU-count preflight of its own." Medium-priority documentation gap, not a live bypass in current use. |
| Direct vLLM invocation (a developer just running `vllm serve ...` by hand on the host) | Always possible on any host with vLLM installed | Trivially yes, as with any CLI tool | Out of scope for a software-level audit — no repo code enables or hides this; it is the same category of risk as "a developer could `rm -rf` the repo." Not a finding. |

**Summary**: no *hidden* bypass was found. One documented-but-unenforced
trust boundary exists (hand-editable ExecutionPlan JSON, §7 row 2), and one
architecturally-separate-but-real production path exists for a different
policy dimension (`policy_executor.py`) that is not itself a bypass in
current usage but should be labeled as out-of-scope-for-TP-decisions in
project docs.

---

## 8. Whole-system dependency graph

```
┌─────────────────────────────────────────────────────────────────────┐
│ COMPILER (ml-graph-compiler-runtime)                                 │
│                                                                       │
│  Qwen ONNX graph facts                                                │
│        │                                                              │
│        ▼                                                              │
│  DistributedStrategyPlanningPass                                     │
│    ├─ generateDistributedCandidates()      [always {tp1,tp2}]        │
│    ├─ checkQwenCandidateLegality()         [real, per-op, fail-closed]│
│    ├─ estimateDistributedCost()            [real, EVIDENCE ONLY]     │
│    └─ selection: legality ∧ distributed.opt_in  ══════════ OWNERSHIP │
│         (opt_in comes from the target-profile JSON chosen by whoever  │
│          invokes compile-for-target — a human/pipeline-script choice, │
│          made ONCE per profile file, not re-decided per workload)     │
│        │                                                              │
│        ▼                                                              │
│  ExecutionPlanExporter::serializeDistributedPlan()  [field-exact]     │
│        │                                                              │
└────────┼──────────────────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ RUNTIME (heterogeneous-inference-runtime)                             │
│                                                                       │
│  real_qwen_tp{1,2}_execution_plan.json  ══════════════════ OWNERSHIP │
│  (which FILE PATH gets loaded is chosen by the calling script/test —  │
│   D4B loads both, deliberately, to compare; D5's cost model chooses   │
│   between the two pre-existing files per workload — see §10)         │
│        │  deployment.execution_plan.loader                           │
│        ▼                                                              │
│  distributed_materializer.materialize_launch_spec()  [field-faithful] │
│        │  distributed_preflight.run_preflight()      [fail-closed]    │
│        ▼                                                              │
│  distributed_cli.build_cli()                          [argv, no shell]│
│        ▼                                                              │
│  distributed_launch_controller.ServerLaunchController [no force-flag] │
│        ▼                                                              │
│  subprocess: vllm.entrypoints.openai.api_server                       │
│        ▼                                                              │
│  vLLM's own distributed bootstrap (multiproc_executor, parallel_state)│
│        ▼                                                              │
│  real NCCL (ncclCommInitRank, world_size=2, distinct cudaDev per rank)│
│        ▼                                                              │
│  real GPU workers, real weight loading, real inference                │
└─────────────────────────────────────────────────────────────────────┘
```

**Ownership boundaries, marked explicitly**:

1. **Compiler owns**: legality, candidate evidence, and the
   legality+opt-in-gated selection rule. Does **not** own: profitability
   comparison (no cost-based selector exists).
2. **The profile-file choice / script-level plan-path choice owns**: which
   of `{TP1, TP2}` gets materialized for a given run. This is a human or
   pipeline-script decision made when authoring/choosing a target profile
   (compiler side) or a `TP1_PLAN_PATH`/`TP2_PLAN_PATH` constant / D5 cost
   model call (runtime side) — not a runtime re-invocation of the compiler.
3. **Materializer owns**: faithful, field-exact translation of whatever
   plan it's given into a real, launchable vLLM command — with no
   authority to second-guess or override the TP degree.
4. **vLLM owns**: everything after the subprocess boundary (its own
   sharding, its own NCCL configuration) — the compiler's `tensor_shards`/
   `collectives` fields prove compiler-side legality reasoning, not
   evidence that vLLM executed them literally (§2).

---

## 9. Evidence discipline

Every claim above cites an exact file and, where the file has stable
enough content, an exact line number or function name. Where I could not
verify a claim with source-level evidence (e.g., whether a developer might
hand-invoke vLLM outside any repo tooling), I said so explicitly (§7, last
row) rather than asserting a boundary that isn't enforced.

---

## 10. Final conclusion

### 1. Does the compiler genuinely determine distributed execution?

**Partially, and precisely so**: the compiler genuinely determines
**legality** (a real, per-operator, fail-closed check that a TP2 strategy
is even valid for this model/shape) and genuinely determines the
**content** of whichever plan is selected (shard ranges, collective
structure, rank count — all real, all field-traceable to compiler output,
never re-derived downstream). It does **not** genuinely determine
**profitability** — TP1-vs-TP2 selection at the compiler layer is gated by
legality plus an external opt-in flag, not by comparing the two candidates'
computed costs, despite both costs being computed and exported. This is
openly documented in the compiler's own source (`DistributedPlanning.h:9-10`),
not something this audit uncovered as hidden.

Downstream of the compiler, every runtime layer (materializer → CLI →
launch controller → vLLM) is genuinely field-faithful: no layer between
the ExecutionPlan file and the real vLLM subprocess re-decides, overrides,
or hardcodes the TP degree.

### 2. Does any hidden manual path bypass the compiler?

**No hidden bypass was found.** One real, always-possible trust boundary
exists and should be documented rather than treated as closed: the
materializer does not cryptographically verify that an ExecutionPlan JSON
file actually came from a compiler run (§7). One architecturally-separate,
already-production, already-fail-closed path exists for a *different*
compiler decision (`policy_executor.py`, `max_num_seqs`) that is not a TP
bypass in any observed usage but has no TP-specific preflight of its own
and should be labeled as out-of-scope for the TP question in project docs.

### 3. Can the reported D4B/D5 achievements truthfully be described as compiler-guided distributed inference?

**D4B: yes, precisely as scoped.** The exact TP2 `ExecutionPlan` artifact
that the real compiler pipeline produced (legality-checked, opt-in-selected,
field-exact) was carried, unedited, through the materializer/launch
controller/vLLM chain to real 2-GPU execution. This is a correctly-labeled
claim already in D4B's own truth boundary.

**D5: yes, but requires one precision correction.** D5's per-workload
TP1/TP2 decision is made by `tp_cost_model.py::TPCostModel`, a **new,
Python, runtime-repo-only** linear regression fit on measured throughput
data — architecturally distinct from `DistributedStrategyPlanningPass`
(the C++/MLIR compiler pass audited in §1). D5 never re-invokes the actual
MLIR compiler per workload; it chooses between two already-existing,
already-compiled `ExecutionPlan` files. Calling this "a compiler cost
model" (as D5's own report does) is defensible only under a broad reading
of "compiler" that includes this offline-fit policy layer — a precise
reader should understand D5's achievement as: *"a policy model, built on
top of the same real compiler→materializer→vLLM execution chain D4B
validated, correctly predicts which of two compiler-produced plans to use"* —
not *"the MLIR compiler recomputed a fresh cost-based decision for each
workload."* The execution chain each decision is carried through remains
the genuine, audited, compiler-produced one (§4); only the *decision-making
component itself* in D5 is new and separate from the C++ pass.

### 4. Architectural weaknesses remaining before production quality, ranked

1. **(High)** The compiler's TP1/TP2 selector has no profitability model —
   selection is legality + external opt-in flag only. A production
   compiler/runtime interface should either implement a real cost-based
   selector at the MLIR layer, or stop describing the D2 stage's language
   ("legality/cost analysis, TP2 selection") in a way that implies the cost
   estimate is load-bearing.
2. **(High)** D5's `TPCostModel` and the compiler's `DistributedStrategyPlanningPass`
   are two independent systems solving related but distinct problems
   (compile-time legality+opt-in vs. runtime-benchmark-fit throughput
   prediction), sharing only the word "cost model" and the word "compiler"
   in prose. A production system should either integrate D5's regression
   into the actual compiler pipeline (so a fresh compile can consult
   measured hardware history) or rename D5's component to avoid implying
   that connection (e.g. "runtime TP policy model").
3. **(Medium)** No cryptographic or hash-pinned provenance check exists
   between "this ExecutionPlan file" and "a specific compiler invocation
   that produced it" at materialization time — current provenance is
   pipeline discipline (scripts recording SHA-256 hashes into evidence
   JSON) rather than an enforced check inside `materialize_launch_spec`
   itself.
4. **(Medium)** `policy_executor.py`'s direct-launch path (`max_num_seqs`
   policy) is a second, real, production vLLM-launching code path with no
   shared preflight/rank-placement logic with the TP materializer. Not
   currently a TP bypass, but its existence as an independent launch path
   should be documented so a future change to one path is not assumed to
   apply to the other.
5. **(Low)** The `distributed.truth_boundary` string embedded in the
   ExecutionPlan schema itself still self-identifies as D1's
   simulated-runtime boundary even when the same schema is reused for real
   D4B/D5 hardware execution three stages later — each stage compensates
   with its own separate truth-boundary artifact rather than the schema
   being self-consistent at the field level.

---

## D6 re-audit: compiler-owned profitability selection

Traced fresh, after implementation, with the same rigor as §1–§9 above:
exact file/function/line citations, no claim without a source-level check.

### D6.1 The traced chain

```
--model-profile JSON (real weight_footprint_mb)  ─┐
--workload-profile JSON (real input/output/conc.)  ├─▶ module attrs
target profile's distributedProfitability block ───┘   (compile-for-target/main.cpp:1142-1233)
        │
        ▼
DistributedStrategyPlanningPass::runOnOperation()
  readModelProfile() / readWorkloadProfile() / readCalibration()
  (DistributedStrategyPlanningPass.cpp:84-177)
        │
        ▼
estimateDistributedProfitability(tp1, ...) and (tp2, ...)
  -- real memory-feasibility gate, then calibrated linear throughput
  prediction, in tokens/s (DistributedPlanning.cpp, new section)
        │
        ▼
Candidate comparison (DistributedStrategyPlanningPass.cpp:432-479):
  optIn gates candidate space; among legal+feasible candidates, the
  higher predicted_throughput_tokens_per_s wins; epsilon tie-break
  prefers TP1
        │
        ▼
module->setAttr("distributed.selected_candidate_id", ...)  [ONE candidate]
        │
        ▼
buildDistributedPlan() → ExecutionPlanExporter::serializeDistributedPlan()
  [unchanged D1/D2 serialization, §2 above]
        │
        ▼
real_qwen*_execution_plan.json (fresh, one per compile invocation)
        │
        ▼
deployment.execution_plan.loader.load_execution_plan()  [unchanged]
        │
        ▼
distributed_materializer.materialize_launch_spec()  [UNCHANGED — zero
  lines touched for D6; verified by direct execution, not just diff,
  in Part F: results/runtime_paths/distributed_d6_compiler_owned_tp_selection/
  part_f_materializer_verification.json]
        │
        ▼
distributed_cli.build_cli() → --tensor-parallel-size <N>  [unchanged]
        │
        ▼
ServerLaunchController → real vLLM subprocess → real NCCL/GPU workers
  [verified on real 2xRTX4090 hardware for both a compiler-selected TP1
  and a compiler-selected TP2 plan — Part K, part_k_*_real_verification.json]
```

### D6.2 What changed, cited exactly

- `mlir_passes/include/serving/DistributedPlanning.h`: added
  `DistributedModelProfile`, `DistributedWorkloadProfile`,
  `DistributedThroughputCoefficients`, `DistributedProfitabilityCalibration`,
  `DistributedProfitabilityEstimate`, `estimateDistributedProfitability`.
  Updated the module header comment that previously stated "There is no
  profitability selector here" to point at the new one, without altering
  D1's own still-true scope note (D1's `generateDistributedCandidates`/
  `buildDistributedPlan` remain profitability-free by design).
- `mlir_passes/lib/serving/DistributedPlanning.cpp`: implemented
  `distributedKvCacheBytesPerTokenPerGpu`, `distributedPerGpuWeightMb`,
  `estimateDistributedProfitability` — a direct, numerically-verified
  (bit-for-bit, §D6.3) C++ port of
  `heterogeneous-inference-runtime/deployment/vllm_adapter/tp_cost_model.py`.
- `mlir_passes/lib/serving/DistributedStrategyPlanningPass.cpp`: replaced
  the `legal && distributed.opt_in => tp2` branch with a real comparison
  (lines ~432–479); `distributed.opt_in` now only widens candidate space
  (`consideredForProfitability`, same file, ~line 397); every candidate,
  including an excluded or illegal TP2, still gets legality/cost evidence
  recorded (`excluded_from_consideration` field) — never silently dropped.
- `mlir_passes/lib/serving/ExecutionPlanExporter.cpp`: `attrToJSON` gained
  a `FloatAttr` case (previously silently returned `null` for any
  floating-point evidence field — a real bug caught during D6's own
  smoke test of the evidence report, fixed before any decision was
  trusted).
- `mlir_passes/tools/compile-for-target/main.cpp`: new `--model-profile`
  and `--workload-profile` CLI flags; new `distributedProfitability`
  block parsing on the target profile.
- New, real, non-fixture configs: `configs/target_profiles/
  nvidia_rtx4090_d6_distributed_profitability.json` (real 2xRTX4090 facts
  + real D5-calibrated coefficients), `configs/models/
  qwen_{0_5b,7b}_model_profile.json`, `configs/models/
  qwen_7b_onnx_graph_facts.json` (real HF `AutoConfig` values, verified
  live on 2026-07-19), `configs/workloads/*.json` (48 files, one per D5
  workload cell).
- New `tools/generate_distributed_profitability_profile.py`: reads real
  D5 calibration-split rows only (never held-out rows), refits via
  `heterogeneous-inference-runtime`'s own tested `fit_linear_regression`,
  and cross-checks its output against the published `cost_model_fitted.json`
  — confirmed `0.00e+00` max coefficient difference.
- `heterogeneous-inference-runtime/deployment/vllm_adapter/tp_cost_model.py`:
  docstring reclassified; zero behavioral changes.
- `heterogeneous-inference-runtime/deployment/vllm_adapter/distributed_materializer.py`,
  `distributed_cli.py`, `distributed_launch_controller.py`,
  `backend_adapter.py`: **zero changes for D6** (confirmed via `git diff`
  showing no post-D6 delta, and via direct execution in Part F).

### D6.3 Numerical verification, not just code inspection

`estimateDistributedProfitability`'s C++ output was compared against the
Python `tp_cost_model.py` prediction for the identical inputs
(0.5B, in32/out32/c1): **558.1676359962089 vs. 558.1676359962089 (TP1)**
and **499.733014181306 vs. 499.733014181306 (TP2)** — exact match to the
full double-precision value printed, not merely "close." This is not a
coincidence-prone unit test; it is the same regression evaluated by two
independent implementations of the same formula.

### D6.4 Real fresh-compilation reproduction, not a static-plan re-selection

21 separate `compile-for-target` invocations (one per D5 held-out
workload cell, 17 for 0.5B + 4 for 7B), each with its own
`--workload-profile` file, produced 21 separate, freshly-generated
`execution_plan.json` files. The compiler's decision matched the real D5
oracle in 21/21 cells (100%, 0% mean/p95/worst-case regret — see
`held_out_evaluation_d6.json`). This is categorically different from "the
runtime chooses between two precompiled plan files": every plan here was
compiled once, for that specific workload, by the real MLIR pipeline.

### D6.5 Re-run of the full grep sweep (§3.5's method, repeated)

```
grep "tensor_parallel_size\s*=\s*2\|world_size\s*=\s*2" in new D6 files → 0 hardcodes (1 comment, historical D1 reference)
grep "selected = tp2" → 2 occurrences, both inside the profitability/capacity-forced
                          comparison branches (DistributedStrategyPlanningPass.cpp:463,475)
grep "getenv\|std::env" in compiler-side D6 files → 0 occurrences
git diff on distributed_materializer.py/distributed_cli.py/distributed_launch_controller.py/
  backend_adapter.py since before D6 → no changes
```

No new hardcode, no new environment-variable override, no new
runtime-side plan-selection logic was introduced anywhere in D6.

### D6.6 Answers to the five closing questions

**1. Does the C++/MLIR compiler now make the profitability decision?**
Yes. Traced at the source level (§D6.1–§D6.2) and confirmed by 21 fresh,
separate compiler invocations whose decisions match the real measured
oracle exactly (§D6.4). The comparison (`predicted_throughput_tokens_per_s`
for TP1 vs. TP2) happens inside `DistributedStrategyPlanningPass.cpp`,
inside the compiler process, not in any Python code, not at runtime.

**2. Is the Python runtime selector absent from the production path?**
Yes. `tp_cost_model.py`'s only production-adjacent consumer
(`scripts/run_d5_fit_and_evaluate_cost_model.py`) is an offline
analysis/evaluation script, never invoked by any server-launch path.
Five new tests (`tests/test_distributed_d6_compiler_owned_tp_selection.py`)
prove this by AST-scanning every production module for the import and by
independently materializing all 21 fresh plans with the module's presence
never checked or required.

**3. Does one compiler-selected plan directly control vLLM TP execution?**
Yes, proven twice: (a) Part F shows the unmodified materializer consuming
a fresh compiler-selected TP2 plan and correctly rejecting it on a 1-GPU
host (fail-closed, unchanged from D3B/D4B) and correctly materializing
`--tensor-parallel-size 2`; (b) Part K shows the same plan launched for
real on the 2xRTX4090 host, with real NCCL `world_size=2` and two
distinct physical GPU UUIDs in the process-to-device mapping — and a
compiler-selected TP1 plan launched with no NCCL initialization at all
and a single GPU UUID.

**4. Are D4B/D5/D6 claims now accurately described as compiler-selected
distributed inference?** Yes, with the precision the original audit
called for: D4B's claim (execution chain is genuine) stands unchanged.
D5's claim needed a correction (the decision-maker was a separate Python
system, not literally "the compiler") — D6 resolves that correction by
moving the decision into the actual `DistributedStrategyPlanningPass`,
so this same criticism no longer applies to D6 itself.

**5. What limitations remain?**
- The linear regression can extrapolate to physically nonsensical
  (negative) absolute predicted-throughput values outside the exact
  calibrated range (observed for the 7B candidates: e.g. -254.5 and
  -173.3 predicted tokens/s at one cell) — the *relative ranking* between
  TP1 and TP2 remains correct and matches the measured oracle in every
  tested cell, but the model's absolute outputs should never be read as
  physically meaningful numbers, only as a ranking signal. This is a
  cost-model-expressiveness limitation, not a wiring or serialization bug.
- Calibration is specific to this exact 2xRTX4090 host
  (`calibration_hardware_identity` is recorded and the pass has no
  mechanism to warn if a profile is applied to different hardware beyond
  the honesty of that recorded string) — using this profile on different
  hardware would silently produce wrong predictions, correctly gated only
  by human review of the profile's declared identity, not by an automated
  hardware fingerprint check.
- No cryptographic provenance check exists between an `ExecutionPlan`
  file and the specific compiler invocation that produced it (unchanged
  from the original audit's §7 finding) — the new calibration profile
  adds a `calibrationCompilerCommit`/`calibrationRuntimeCommit` pair for
  human-auditable provenance, but nothing enforces it is checked before
  use.
- The tie-break epsilon (`1e-6` tokens/s) is a reasonable but arbitrary
  choice; with real calibrated data, ties are not observed in practice
  (0/21 held-out cells), so this path is exercised only by the synthetic
  unit test (`testEqualPredictedThroughputIsReachableAndExactlyEqual`),
  not by any real compilation.
