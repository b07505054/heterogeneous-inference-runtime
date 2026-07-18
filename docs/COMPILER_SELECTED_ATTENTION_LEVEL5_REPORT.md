# Compiler-selected attention Level 5 report

## 1. Verdict

**VERIFIED LEVEL 5**

The previously missing compiler-ownership requirement is closed. The normal
Qwen model-forward path now invokes the attention compiler/planner with real
prompt/model metadata, evaluates candidate legality and static cost scores,
emits the selected prefill and decode decisions into ExecutionPlan v2, writes
the plan to JSON, reloads it through the normal loader, and gives only the
deserialized plan to the model adapter. All 192 runtime invocations match the
compiler-selected candidate, have zero fallback and zero mismatch, and enter
`o_proj`. This remains Hugging Face model-forward integration, not vLLM
serving.

## 2. Starting and ending repository state

Starting state:

| Repository | Branch | HEAD | State |
|---|---|---|---|
| `heterogeneous-inference-runtime` | `main` | `5b56607cf84d8acda2691f02762f50d30332a8d1` | Dirty with pre-existing ExecutionPlan edits and untracked attention, sharding, AArch64, tests, reports, and results |
| `ml-graph-compiler-runtime` | `master` | `0d200c3c7463f21cda97e77f4ad0e912bbad329f` | Dirty with pre-existing HIR/AArch64 edits and untracked attention/sharding/AArch64 work |

Ending HEADs are unchanged. This stage changed only the attention planner,
runtime, loader, Qwen harness, attention tests, a new selector-evaluation
script, this report, and the new result directory. No reset, clean, stash,
commit, push, rebase, deletion, or prior-artifact overwrite occurred.

## 3. Previous gap and exact fix

Previously, `run_qwen_compiler_attention.py` directly called
`make_attention_plan` twice with serial/one-worker. The plans carried
`cost_model_selected` text but were not selected by the compiler and were not
loaded from ExecutionPlan.

The normal path is now:

```text
actual prompt/model workload metadata
-> AttentionWorkload validation
-> generate seven supported candidates
-> legality reasons
-> static cost score for every legal candidate
-> deterministic winner
-> phase-specific ExecutionPlan v2 table
-> JSON file
-> load_execution_plan
-> ExecutionPlanAttentionAdapter
-> exact plan candidate dispatch
-> live Qwen Q/K/V
-> attention result
-> o_proj
-> logits
-> argmax token
```

## 4. Manual plan-construction inventory

| Location | Role | Affects normal Qwen path? |
|---|---|---|
| `deployment/attention_runtime.py:29`, `make_attention_plan` | Low-level candidate constructor | Only through compiler candidate generation |
| `deployment/attention_runtime.py`, legacy `legal_attention_candidates` and legacy selector | Existing compatibility API | No |
| `deployment/attention_planner.py`, `select_attention_plan` | Normal compiler candidate construction after legality/score evaluation | Yes, compiler-owned |
| `tests/test_attention_runtime.py` | Forced candidate correctness and negative tests | Test only |
| `scripts/evaluate_compiler_attention_selector.py` | Explicit fixed-policy measurements | Diagnostic only; records forced policy |

The former offending calls in `scripts/run_qwen_compiler_attention.py` were
removed. The file no longer imports or calls `make_attention_plan`.

## 5. Compiler selector architecture

`deployment/attention_planner.py` is the compiler/planning boundary:

- `AttentionWorkload`: canonical input and validation.
- `_legality`: worker/partition/phase legality.
- `_score`: deterministic compute, memory, dispatch, and assembly cost terms.
- `select_attention_plan`: generate, reject, score, tie-break, and select.
- `emit_execution_plan`: emit ExecutionPlan v2.

The selector is a **rule-based static cost selector**, version
`attention_static_cost_selector_v1`. It is not described as calibrated from the
candidate-evaluation JSON. Tie-breaking is minimum score then lexicographic
candidate ID. No runtime rescoring exists.

## 6. Real attention workload descriptor

The descriptor includes phase, batch, query/context lengths, Q/KV heads, head
dimension, dtype, causal flag, Q/K/V/output layouts, KV layout, available
logical workers, and target CPU profile.

Planning uses the actual four-token prompt and loaded Qwen configuration:

```text
prefill: b1 q4 kv4 qh14 kvh2 d64 fp32
decode:  b1 q1 kv5..11 qh14 kvh2 d64 fp32
```

At every registered attention callback,
`AttentionWorkload.from_tensors(query,key,value)` re-derives and validates the
actual shapes. The adapter rejects a tensor workload outside the serialized
domain.

## 7. Candidate generation and legality

Generated candidates are serial/1, split-head/2/4/8, and
split-query/2/4/8. Runtime support and planner candidates are identical.

For Q=4 prefill, six of seven are legal; split-query/8 is rejected because
workers exceed query-token partitions. For decode, four are legal; all
split-query candidates are rejected with
`split_query_illegal_for_decode`. Worker counts also cannot exceed available
logical workers or query heads. GQA requires 14 query heads divisible by two KV
heads. Uneven split-head partitions remain legal because the runtime uses
balanced uneven ranges.

## 8. Scoring and selection

Every legal candidate records:

- estimated attention compute units;
- estimated tensor/cache bytes;
- dispatch cost;
- assembly cost;
- total score.

The real Q=4 prefill and KV=5 decode planning events both selected
`torch_cpu_attention_fp32_serial_w1_v1`. This is an evaluated decision, not a
serial default. Larger matrix workloads selected split-head/2 at Q=64 and
split-head/4 at Q=128.

## 9. ExecutionPlan schema and emitted example

The existing `global_decisions.attention_execution` field now accepts a
backward-compatible phase table:

```json
{
  "decision_kind": "cpu_attention_plan_table_v1",
  "operator_kind": "attention",
  "operator_id": "qwen.layers.*.self_attn",
  "plan_kind": "phase_specific_exact_prefill_decode_context_range",
  "selector_version": "attention_static_cost_selector_v1",
  "selection_mode": "compiler_selected",
  "phase_decisions": {
    "prefill": {
      "native_kernel_id": "torch_cpu_attention_fp32_serial_w1_v1",
      "workload_domain": {
        "batch": 1,
        "query_length_min": 4,
        "query_length_max": 4,
        "context_length_min": 4,
        "context_length_max": 4
      }
    },
    "decode": {
      "native_kernel_id": "torch_cpu_attention_fp32_serial_w1_v1",
      "workload_domain": {
        "batch": 1,
        "query_length_min": 1,
        "query_length_max": 1,
        "context_length_min": 5,
        "context_length_max": 11
      }
    }
  },
  "fallback": {"policy": "hard_failure", "count": 0},
  "runtime_no_redecision": true
}
```

Validation covers IDs, strategy/worker consistency, split dimension, phases,
head compatibility, dtype/layout, domains, provenance, and zero-count
hard-failure fallback.

## 10. Serialization/deserialization proof

The selector result is inserted into an ExecutionPlan object-shaped payload,
written to `execution_plan_roundtrip.json`, and loaded using
`deployment.execution_plan.loader.load_execution_plan`. The adapter receives
only that returned typed `ExecutionPlan`.

For both phases:

```text
selector candidate
== serialized candidate
== deserialized candidate
== runtime executed candidate
```

`execution_plan_roundtrip_proof.json` records
`all_four_stage_ids_match: true` and `in_memory_override: false`.

## 11. Model-forward plan-consumption proof

`ExecutionPlanAttentionAdapter` requires a deserialized plan table. It creates
phase runtimes only from `phase_decisions`, matches actual tensors against the
serialized workload domain, and dispatches the named candidate. It never calls
the selector and has no default serial path. Missing table, missing candidate,
domain mismatch, or candidate mismatch raises `ShardingPlanError`.

A unit-test spy replaces the selector with a failure function during adapter
execution; plan consumption still succeeds, proving there is no runtime
reselection.

## 12. Selected-versus-executed proof

The clean run reports:

```text
selector invocations:       2
ExecutionPlan round-trip:   exact
runtime attention calls:    192
candidate mismatches:       0
fallbacks:                  0
manual-plan calls:          0
```

All calls execute `torch_cpu_attention_fp32_serial_w1_v1`, matching both
phase decisions.

## 13. Invocation provenance

Every record contains plan ID, operator/layer ID, phase, decode step, actual
workload signature, selector version, selection mode, selected/serialized/
executed candidate IDs, worker count, fallback/mismatch flags, attention timing,
output sum, returned tensor pointer, and `o_proj` input pointer.

Observed counts:

- total: 192;
- prefill: 24;
- decode: 168;
- per layer: eight for every layer 0–23;
- token 1: 24 prefill calls;
- tokens 2–8: 24 decode calls each;
- candidate histogram: serial/1 = 192;
- `output_entered_o_proj`: 192.

## 14. Fallback and mismatch audit

The primary and perturbed runs each have fallback count zero and mismatch count
zero. Invalid plans fail during loader validation. Missing plans fail adapter
construction. Actual workload/domain mismatch fails before numerical dispatch.
An injected executed-candidate mismatch raises explicitly. No silent fallback
or serial construction exists in the normal adapter.

## 15. Baseline/compiler numerical equivalence

Per-step maximum logit differences:

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

The maximum remains exactly `1.621246337890625e-5`.

## 16. Eight-token equality

Baseline and compiler-selected plan-driven paths both generated:

```text
[576, 5567, 18404, 264, 501, 5486, 311, 279]
```

Decoding was deterministic greedy argmax.

## 17. Perturbation causal proof

The same compiler selection, serialization, loading, and adapter path was run
with the existing test-only `+5.0` perturbation on the candidate output before
transpose/head merge and `o_proj`. It generated eight `84565` tokens. Per-step
logit changes ranged from 18.23 to 25.51. Token 1 is directly changed during
prefill; later tokens are both directly perturbed and autoregressively
downstream from earlier divergence.

## 18. KV-cache context behavior

The live Transformers DynamicCache context grew from four prompt positions
through 11 positions. Prefill uses the exact KV=4 plan. Decode performs
predicate matching against the compiler-emitted KV=5..11 range; it does not
rescore. This remains a Transformers in-memory contiguous/dynamic cache, not
vLLM paged KV storage.

## 19. Compiler policy versus fixed-policy performance

Thirty measured calls after five warmups were collected for eight standalone
Qwen-compatible attention workloads. Complete tensors matched serial within
`rtol=2e-5`, `atol=2e-6`.

| Policy | Mean regret | Maximum regret |
|---|---:|---:|
| compiler selector | 3.18% | 24.88% |
| always serial | 5.53% | 44.25% |
| always split-head/2 | 246.70% | 506.26% |
| always split-head/4 | 292.48% | 509.09% |
| always split-head/8 | 676.48% | 1139.81% |

The compiler selected serial for all decode workloads and avoided large
over-parallelization regressions. At Q=128 prefill it selected split-head/4:
1.430 ms versus 1.423 ms for the measured split-query/4 winner and slower
serial. At Q=64 it incorrectly selected split-head/2, producing the maximum
24.88% regret. This negative result is retained.

The eight-token model run measured 38.09 ms summed custom-attention time and
980.07 ms full compiler generation versus 922.08 ms baseline generation in
that trial. No end-to-end speedup is claimed; single-trial full-model wall
timing is informational, while the repeated attention matrix is the policy
comparison.

## 20. Selector quality and regret

| Domain | Workloads | Exact match | Mean regret | Median | p95 | Maximum | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 8 | 75% | 3.18% | 0% | 16.36% | 24.88% | 0% |
| prefill | 4 | 50% | 6.35% | 0.26% | 21.23% | 24.88% | 0% |
| decode | 4 | 100% | 0% | 0% | 0% | 0% | 0% |

The matrix is an evaluation of a rule-based static selector, not a
train/calibration claim. The older calibration/held-out artifact was preserved
and not silently reopened by the runtime.

## 21. Focused test results

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q tests/test_attention_runtime.py tests/test_execution_plan_loader.py tests/test_execution_path_builder.py` | 52 passed |
| `ctest --test-dir build-mlir -R 'AttentionCPUContractTest\|AttentionCPUFailClosedTest\|QwenToServingMlirTest' --output-on-failure` | 3 passed |
| `py_compile` on planner/runtime/Qwen/evaluator scripts | passed |
| `git diff --check` | passed |
| compiler-selected Qwen causal harness | completed; eight clean tokens equal, perturbed tokens changed |

Negative tests cover missing plan table, missing candidate, runtime selector
recomputation, invalid decode split-query, and selected/executed mismatch.

## 22. Level 6 limitation

Installed vLLM remains 0.24.0 with a CUDA-oriented environment,
`UnspecifiedPlatform`, and CPU engine failure before engine/worker
initialization. No vLLM request scheduler, worker, attention backend, paged KV
cache, tensor parallelism, or serving engine used this path. This is Level 5
model-forward integration, not Level 6.

## 23. Exact defensible achievement statement

Implemented and independently verified a compiler-selected,
ExecutionPlan-driven attention path for real Qwen model-forward execution. The
compiler generated, legality-filtered, and scored supported attention
candidates, serialized the selected phase strategies through ExecutionPlan v2,
and the runtime executed the same candidate for every one of 192 live model
attention invocations with zero fallback or mismatch. Custom attention outputs
entered every `o_proj`, reproduced eight greedy tokens with logits within
`1.621246337890625e-5`, and a pre-`o_proj` perturbation changed downstream
logits and generated tokens, establishing causal dependency. This does not
claim vLLM serving integration.

## 24. Resume bullets

- Implemented a rule-based attention compiler selector that legality-filters
  and scores seven CPU candidates, serializes phase decisions through
  ExecutionPlan v2, and dispatches without runtime reselection.
- Verified 192 plan-driven real-Qwen attention calls with zero fallback or
  candidate mismatch, exact `o_proj` dataflow, equivalent logits, and identical
  eight-token greedy generation.
- Evaluated compiler policy across eight Qwen-compatible workloads, achieving
  75% winner matching and 3.18% mean regret while avoiding the severe decode
  regressions of fixed 2/4/8-worker policies.
