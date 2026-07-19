# Distributed D3B: vLLM Distributed Launch-Spec Materialization and Fail-Closed Validation

## 1. Executive result

D3B closes exactly the one gap it was scoped to close: the compiler-selected,
D3A-validated TP=2 plan for the real Qwen2.5-0.5B-Instruct model was
deterministically materialized into a typed, version-aware vLLM distributed
launch specification. Every field in that specification records its
provenance. The specification, its generated CLI, and its environment
contract were validated fail-closed against the actually-installed vLLM
0.24.0 package and the actual current host.

On this real single-GPU development host:

```
launch_spec_generation = success
execution_preflight     = rejected
primary_reason          = insufficient_visible_gpu_count
```

This is the expected, successful D3B result. TP=2 was never silently
downgraded to TP=1. No subprocess was ever launched for the rejected TP=2
spec. The existing real single-GPU (TP=1) vLLM path remains valid and
reaches its best available D3B state (`DRY_RUN_VALIDATED`).

D3B's only allowed primary claim holds:

> A compiler-selected TP=2 plan for a real Qwen model was deterministically
> materialized into a version-aware vLLM distributed launch specification,
> with model, tensor-parallel, pipeline-parallel, dtype, memory, batching,
> tokenizer, network, process, and environment contracts validated
> fail-closed against the current host and vLLM installation.

## 2. Repository state

Captured via `git status --porcelain` / `git rev-parse HEAD` on both
repositories immediately before any D3B file was touched, and again after
all D3B work completed.

| Repo | Branch | HEAD before | HEAD after | Working tree |
|---|---|---|---|---|
| ml-graph-compiler-runtime | master | `59854b892629bc0bc7f43ca0bad3eab17464c030` | unchanged | clean (D3B made zero changes here) |
| heterogeneous-inference-runtime | main | `b79ff951758b164010f95b761e3f927877e3ad10` | unchanged (no commit made) | 2 files modified (additive edits only), 10 new files |

Modified files (both purely additive):
- `deployment/vllm_adapter/__init__.py` — new exports added, nothing removed.
- `deployment/vllm_adapter/backend_adapter.py` — new `VLLMDistributedAdapter` class added; the existing `VLLMBackendAdapter` class and `_server_command` helper are unchanged.

New files: `deployment/vllm_adapter/distributed_{capability_inventory,launch_spec,materializer,rank_placement,environment,cli,argument_registry,preflight,dry_run,provenance}.py`, `tests/test_distributed_d3b_vllm_launch_spec.py`, `scripts/run_distributed_d3b_pipeline.py`, this report, and the `results/runtime_paths/distributed_d3b_vllm_launch_spec/` artifact directory.

No reset, clean, stash, delete, rename, rebase, commit, or push was performed on either repository.

Full detail: `results/runtime_paths/distributed_d3b_vllm_launch_spec/repository_state_before.json` and `repository_state_after.json`.

## 3. D1/D2/D3A preservation

Every file in `results/runtime_paths/distributed_d1_tp2_multiprocess/` (15 files), `distributed_d2_qwen_pipeline/` (20 files), and `distributed_d3a_live_qwen_tensor/` (22 files) was SHA-256 hashed before D3B work began and re-hashed after. All 57 hashes are byte-identical before and after — see `d1_d2_d3a_preservation.json`. The D1/D2/D3A markdown reports are all confirmed still present and untouched. `ml-graph-compiler-runtime` required zero changes for D3B: it only consumes the D2-exported `real_qwen_tp{1,2}_execution_plan.json` artifacts, unmodified.

## 4. Installed vLLM environment

Discovered by directly importing and introspecting the installed packages on this host (`deployment/vllm_adapter/distributed_capability_inventory.py:discover_environment`), not assumed from memory:

| Field | Value |
|---|---|
| vLLM version | 0.24.0 |
| Python version | 3.12.13 |
| PyTorch version | 2.11.0+cu130 |
| Transformers version | 5.13.0 |
| CUDA available | True (CUDA 13.0) |
| Visible GPU count | 1 |
| GPU | NVIDIA GeForce GTX 1650 with Max-Q Design, 3717.94 MB, compute capability 7.5 |
| bf16 hardware-reported support | True |
| NCCL available (torch) | True |
| Gloo available (torch) | True |
| Ray importable | False |

This is exactly the single-GPU host the D3B task requires — no hardware was rented, no second GPU exists. Full detail: `vllm_environment_inventory.json`.

## 5. Supported argument inventory

The CLI argument registry was built by calling `vllm.entrypoints.openai.cli_args.make_arg_parser()` on the installed package and introspecting its `argparse.ArgumentParser` actions directly (dest, option strings, type, default, choices) — never scraped from `--help` text or read from documentation. 267 arguments were discovered. Fields D3B depends on were all present, including `tensor_parallel_size` (`-tp`/`--tensor-parallel-size`), `pipeline_parallel_size`, `distributed_executor_backend` (choices: `external_launcher`, `mp`, `ray`, `uni`), `data_parallel_size`, `dtype` (choices: `auto`, `bfloat16`, `float`, `float16`, `float32`, `half`), `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `enable_prefix_caching`/`enable_chunked_prefill` (boolean flag pairs with `--no-` negatives), `host`, `port`, `master_addr`, `master_port`, `revision`, `served_model_name`, `seed`, `trust_remote_code`.

One notable discovery: the installed vLLM 0.24.0 registry has **no `--swap-space` argument** — it was removed from this vLLM version. The pre-existing (D3B-unrelated) `deployment/vllm_adapter/config_materializer.py`/`backend_adapter.py` TP1 path still references `swap_space` and would emit an unsupported `--swap-space` flag if invoked with a non-`None` swap-space value. D3B's new distributed materializer never emits this flag (the D2 distributed plan schema has no `swap_space` concept), so this pre-existing incompatibility is out of D3B's scope to fix, but is recorded here as a genuine, version-aware finding (see §23).

Full detail: `vllm_argument_registry.json`.

## 6. Source compiler plan

Source of truth: `results/runtime_paths/distributed_d2_qwen_pipeline/real_qwen_tp2_execution_plan.json` (plan_id `nvidia-gtx1650-maxq-d2-distributed-opt-in_serving_plan`) and `real_qwen_tp1_execution_plan.json` (plan_id `nvidia-gtx1650-maxq_serving_plan`), loaded via the existing `deployment.execution_plan.loader.load_execution_plan` (unmodified). The TP2 plan declares `distributed.strategy = "tensor_parallel"`, `tensor_parallel_size = 2`, `pipeline_parallel_size = 1`, `world_size = 2`, ranks `{0, 1}`, one `all_reduce(sum)` collective over `qwen_prefill::llm.o_proj::layer_0`, and 448/448 tensor shards. The TP1 plan carries no `distributed` block at all — its absence is itself the compiler's TP1 declaration (per the schema's documented convention), which D3B's materializer treats identically (`tensor_parallel_size=1, pipeline_parallel_size=1, world_size=1, ranks={0}`). Neither JSON file was patched or hand-edited; both are copied verbatim into `source_tp1_execution_plan.json` / `source_tp2_execution_plan.json`.

The real Qwen model identity (`Qwen/Qwen2.5-0.5B-Instruct`) is not present in the D2 plan's abbreviated `model_identity.model_id` (`"qwen2.5-0.5b"`) — it is the D3A-validated linkage (`distributed_d3a_live_qwen_tensor/live_model_execution.json`) that ties the compiler's abbreviated identifier to the real, locally-cached HF checkpoint. D3B treats this D2→D3A linkage as a single `compiler_plan` provenance chain, documented explicitly in the launch spec's `field_provenance` (§8).

## 7. Launch-spec schema

`deployment.vllm_adapter.distributed_launch_spec.VLLMDistributedLaunchSpec` (schema_version `1.0.0`) is a frozen dataclass with every field required by the task spec (`source_execution_plan_id`, `source_candidate_id`, `source_operator_ids`, `model`, `tokenizer`, `served_model_name`, `revision`, `dtype`, `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `distributed_executor_backend`, `world_size`, `rank_count`, `rank_placements`, `visible_devices`, `device_type`, `host`, `port`, `master_address`, `master_port`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `enable_prefix_caching`, `enable_chunked_prefill`, `trust_remote_code`, `seed`, `environment`, `cli_arguments`, `preflight_status`, `rejection_reasons`, `truth_boundary`) plus two D3B-specific additions (`whole_model_tp_evidence_status`, `d3b_mode`) and a `field_provenance` map. `to_dict()` produces a fully JSON-serializable structure — verified by an explicit test and by every artifact write in this report. Full schemas: `tp1_launch_spec.json`, `tp2_launch_spec.json`.

## 8. Compiler-plan to vLLM mapping

Every field's source is tagged with one of four provenance categories (`FieldSource` enum) and a human-readable reason string, verified non-empty by test:

| Category | Meaning | Example fields |
|---|---|---|
| `compiler_plan` | Read from the D2 ExecutionPlan / D3A-validated linkage | `tensor_parallel_size`, `pipeline_parallel_size`, `world_size`, `model`, `tokenizer`, `seed` (reused from D3A's `1234`) |
| `capability_profile` | Matches the installed vLLM registry's own declared default | `data_parallel_size` (1), `port` (8000), `master_address`/`master_port`, `trust_remote_code` (False), `revision` (None) |
| `runtime_discovery` | Probed live on this host at materialization time | `visible_devices`, `device_type` |
| `explicit_D3B_default` | Neither the plan nor the registry decides it; D3B picks it explicitly and documents why | `dtype` (float16, pinned for determinism over `auto`), `host` (127.0.0.1), `max_model_len`/`max_num_seqs`/`max_num_batched_tokens`/`gpu_memory_utilization` (conservative values for a 4GB-class GPU), `enable_prefix_caching`/`enable_chunked_prefill` (True), `distributed_executor_backend` (`mp`), `served_model_name` |

`tensor_parallel_size → --tensor-parallel-size`, `pipeline_parallel_size → --pipeline-parallel-size`, `world_size → expected rank/process count`, model identity `→ --model`/`--tokenizer`, dtype decision `→ --dtype`, memory policy `→ --max-model-len`/`--gpu-memory-utilization`, batch policy `→ --max-num-seqs`/`--max-num-batched-tokens`, prefix policy `→ --enable-prefix-caching`, chunk policy `→ --enable-chunked-prefill` — all mapped explicitly in `deployment/vllm_adapter/distributed_materializer.py` and verified in the generated CLI (§13).

## 9. Whole-model TP evidence status

`whole_model_tp_evidence_status = "not_established_operator_level_only"` on every materialized spec, hardcoded by the materializer (`whole_model_tp_evidence_established = False`, never set True by any code path — enforced by test `test_negative_operator_level_evidence_never_marked_whole_model_ready`). D2/D3A prove operator-level TP correctness for exactly one real `o_proj` operator on layer 0 (serialized rank-local computation + D1 collective reconstruction, `max_abs_error ≈ 1.79e-7`/`3.42e-7` against a live captured activation). They do **not** establish whole-model vLLM TP legality across all 24 layers, all operator types, or vLLM's own internal TP/all-reduce implementation. `d3b_mode = "planning_only"` on every spec; execution readiness never advances past `PREFLIGHT_REJECTED`/`DRY_RUN_VALIDATED` regardless of this status. Full detail: `whole_model_tp_evidence_gap.json`.

## 10. Rank placement

D2's TP2 plan declares ranks `{0, 1}` (`compiler_plan`-sourced). D3B's explicit placement policy (`explicit_D3B_default`, since D2's ranks are simulated CPU processes, not GPU-indexed) maps rank *i* → logical GPU *i*, contiguously. On this one-GPU host, rank 0 resolves to physical device 0; rank 1 resolves `physical_device_index = None` — it is never fabricated onto physical GPU 0 alongside rank 0 (verified by `test_negative_two_tp_ranks_never_mapped_to_one_gpu`). Validated: rank IDs contiguous, one placement per rank, no duplicate physical-device assignment, placement count equals world size, logical→physical mapping always explicit (including the `None` case). Full detail: `rank_placement.json`.

## 11. Environment materialization

Only variables D3B's own launch actually needs are included, each tagged with scope (`global_launch`/`per_rank`/`optional_diagnostic`), source, and justification: `CUDA_VISIBLE_DEVICES` (runtime_discovery, restricts to the resolved placement), `MASTER_ADDR`/`MASTER_PORT` (capability_profile, match registry defaults), `WORLD_SIZE` (compiler_plan, descriptive), `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `TOKENIZERS_PARALLELISM=false` (explicit_D3B_default, launch hygiene), `NCCL_DEBUG=INFO` (optional_diagnostic). `RANK`/`LOCAL_RANK` are explicitly **excluded** with a documented reason: vLLM's own `mp`/`ray` executor assigns per-worker rank internally — fabricating a single top-level value would misrepresent a variable vLLM manages itself. `NCCL_SOCKET_IFNAME` and `HF_HOME` are also explicitly excluded with reasons (single-node scope; existing cache already resolves the model). Full detail embedded in `tp1_launch_spec.json`/`tp2_launch_spec.json`'s `environment` field.

## 12. CLI generation

Deterministic argv array, never executed. TP2:

```
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --no-trust-remote-code --dtype float16 --seed 1234 \
  --served-model-name qwen2.5-0.5b-instruct-tp2-planning-only \
  --host 127.0.0.1 --port 8000 --master-addr 127.0.0.1 --master-port 29501 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --distributed-executor-backend mp --max-model-len 2048 --max-num-seqs 4 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --enable-chunked-prefill
```

TP1 is identical except `--tensor-parallel-size 1` and the served-name suffix. Both argv arrays, their `shlex.join` shell-string form, environment map, working directory, expected process count (2 for TP2, 1 for TP1), and expected GPU assignment (`{"0": 0, "1": null}` for TP2) are recorded in `tp1_cli.json`/`tp2_cli.json`. Every emitted argument passed the installed-registry compatibility check first (`all_arguments_supported = true`, `unsupported_arguments = []` for both).

## 13. Version-aware validation

Every candidate CLI field is checked by dest against the live-introspected registry before being emitted (`distributed_argument_registry.check_argument`); an unsupported dest is recorded but never silently included in argv (proven by `test_negative_unsupported_cli_flag_is_never_silently_emitted`, which mocks a registry missing `tensor_parallel_size` and confirms `--tensor-parallel-size` never appears in the resulting argv). `distributed_argument_registry.build_mock_registry_without` builds a simulated incompatible/older vLLM registry for tests only — never called on the real materialization path. Test `test_negative_incompatible_installed_vllm_version_via_mock_registry` proves three core distributed dests are correctly flagged unsupported against such a mock.

## 14. TP1 preflight result

All 21 applicable checks passed: model locally resolvable (real HF cache hit), vLLM installed, all CLI arguments supported, TP/PP ≥ 1, world_size = TP×PP = 1, CUDA available, visible GPU count (1) ≥ requested (1), per-rank memory plausible (3346 MB budget vs. ~1454 MB estimated footprint — measured from the real 988 MB cached `.safetensors` file, not a param-count guess), dtype supported, `max_model_len` (2048) ≤ real model `max_position_embeddings` (32768), port valid and free, master address valid, rank placement complete/contiguous/non-duplicated, no environment conflicts, executor backend supported. Result: **passed**, readiness state `DRY_RUN_VALIDATED`. Full detail: `tp1_preflight.json`.

## 15. TP2 preflight result

Every check passes **except** `sufficient_visible_gpu_count` (visible=1, requested=2) — the sole rejection. `primary_reason = "insufficient_visible_gpu_count"`, exactly as the D3B task specifies as the expected successful outcome. The advisory (non-blocking, always-recorded-for-TP>1) `whole_model_tp_evidence_not_established` check also fires, listed separately from the primary hardware reason. Result: **rejected**, readiness state `PREFLIGHT_REJECTED`. Full detail: `tp2_preflight.json`.

## 16. Dry-run validation

For TP1 (the only spec whose preflight passed, so the only one dry-run is meaningful for advancing state): CLI argument parsing via the real installed `argparse` parser (267 fields parsed, no `SystemExit`), launch-spec JSON schema (all required keys present, `json.dumps` succeeds), environment map (7 string-valued variables), port field well-formed, rank count matches world size, `CUDA_VISIBLE_DEVICES` declares the expected device count, and command reproducibility (two independent re-materializations from the same inputs yield byte-identical argv). All passed → `DRY_RUN_VALIDATED`. TP2's dry-run also runs (informationally) but cannot advance the readiness state past `PREFLIGHT_REJECTED`. Explicitly documented as NOT validated: real model weight loading, real KV-cache allocation, real NCCL process-group formation, real request serving, actual multi-GPU memory pressure, whole-model TP numerical correctness. Full detail: `dry_run_validation.json`.

## 17. Adapter integration

`deployment/vllm_adapter/backend_adapter.py` gained a new `VLLMDistributedAdapter` class alongside the pre-existing `VLLMBackendAdapter` (untouched) — the same real-vLLM adapter module, not a disconnected script. `VLLMDistributedAdapter.materialize_from_execution_plan()` is its only method; it has no `force`, `ignore_preflight`, or hidden bypass parameter (verified by `test_no_force_or_bypass_parameter_exists_anywhere_on_the_adapter`, which inspects the actual method signature). `deployment/vllm_adapter/__init__.py` gained additive exports only. `scripts/run_distributed_d3b_pipeline.py` drives this same adapter/materializer path to produce every artifact — it is not an independent implementation.

## 18. Negative tests

19 fail-closed cases, all passing (`tests/test_distributed_d3b_vllm_launch_spec.py -k negative`): TP2-with-one-GPU, world-size mismatch, TP×PP mismatch (malformed plan rejected by the existing D1 loader validator), missing rank placement, duplicate rank placement, unsupported CLI flag never silently emitted, unsupported dtype, invalid model identifier, invalid port, port already occupied (real socket bind test), missing vLLM installation, malformed distributed plan (shard gap), unknown distributed strategy (new `UnknownDistributedStrategyError` guard added to the materializer, mirroring the loader's existing `KNOWN_COLLECTIVE_KINDS` fail-closed pattern for the top-level `strategy` field, which the existing loader did not itself cover), operator-level evidence never marked whole-model-ready, TP2 never silently downgraded, two TP ranks never mapped to one GPU, unsupported executor backend, incompatible installed vLLM version (mocked registry), and attempted launch while preflight rejected (provenance-bypass counter proven to fire for a hypothetically forced state, while the real pipeline never produces that state). No child process is created for any rejected spec in these tests. Full detail: `negative_tests.json`.

## 19. Provenance

Cross-layer chain: compiler candidate ID → selected TP plan → model identity → distributed plan fields → vLLM capability inventory → materialized launch fields → CLI arguments → environment → rank placements → preflight validations → final readiness classification. All 13 required counters computed (not hardcoded) for both TP1 and TP2 specs: `source_plan_mismatch_count`, `candidate_mismatch_count`, `model_mismatch_count`, `tp_mismatch_count`, `pp_mismatch_count`, `world_size_mismatch_count`, `rank_placement_mismatch_count`, `unsupported_argument_count`, `silent_default_count`, `silent_downgrade_count`, `preflight_bypass_count`, `unexpected_process_launch_count`, `orphan_process_count` — **all zero** for both. A preflight rejection (TP2's `insufficient_visible_gpu_count`) is correctly not counted as a provenance mismatch. Full detail: `cross_layer_provenance.json`.

## 20. Measurements

Control-plane latencies only, 7 repetitions, median/p95 reported (full detail: `performance_measurements.json`):

| Stage | Median | p95 |
|---|---|---|
| Capability discovery | 35.5 ms | 41.7 ms |
| Plan loading | 0.44 ms | 0.53 ms |
| TP1 materialization (full: rank placement + env + CLI + preflight + dry-run) | 154 ms | 171 ms |
| TP2 materialization (same, rejected at preflight) | 173 ms | 189 ms |
| Argument validation (single dest check) | 7.1 µs | 8.5 µs |
| TP1 command generation | 0.26 ms | 0.31 ms |
| TP2 command generation | 0.20 ms | 0.24 ms |

No serving performance, TTFT/TPOT/throughput, or projected TP speedup is measured or claimed anywhere in this report or its artifacts.

## 21. Test totals

- D3B test file: 26/26 passed (`tests/test_distributed_d3b_vllm_launch_spec.py`).
- Negative tests: 19/19 passed.
- Existing `vllm_adapter` regression tests (`test_vllm_backend_adapter.py`, `test_vllm_config_materializer.py`, `test_vllm_plan_schema.py`): 14/14 passed, unaffected by the new adapter class.
- Full repo suite with D3B deselected: 19 failed / 13 errored / 1044 passed / 3 skipped — confirmed **identical** failure/error set with and without D3B's test file present (pre-existing, unrelated: `test_deployment_planner.py`, `test_model_adapter_registry.py`, `test_p1b`/`test_p1c`/`test_p1d` cross-repo contract tests, `test_rmsnorm_cuda_correctness.py` — none touch `deployment/vllm_adapter` or `deployment/execution_plan`).
- Cross-layer provenance counters: all zero for both TP1 and TP2.

Full detail: `test_summary.json`.

## 22. Process cleanup

D3B never calls `subprocess.Popen`/`os.exec*` with the materialized vLLM server argv — the only subprocess calls made anywhere in the D3B pipeline are read-only (`nvidia-smi -L` for capability discovery, `git status`/`rev-parse` for repository-state recording, and `pytest` for running the test suites themselves). No EngineCore was ever constructed; no GPU worker was ever allocated. Zero tracked D3B PIDs; zero orphan D3B processes. Verified explicitly by `test_no_subprocess_launched_for_rejected_tp2_spec`, which patches `subprocess.Popen` to raise if the vLLM server entry module ever appears in an invoked command, then runs the full rejected-TP2 materialization path successfully. Full detail: `process_cleanup.json`.

## 23. Known limitations

- The pre-existing (D3B-unrelated) TP1 `config_materializer.py`/`backend_adapter.py` path still references a `swap_space` CLI field that no longer exists in installed vLLM 0.24.0 (`--swap-space` was removed from the registry). This is a real, version-aware finding surfaced by D3B's registry introspection but is out of scope to fix here, since it belongs to the pre-existing single-GPU serving path, not the new distributed launch-spec materializer (which never emits this flag). Recommended as a small, separate follow-up fix.
- `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization` are `explicit_D3B_default` values (conservative, documented) because neither D2 nor D3A declares a serving/batch policy for the distributed plan — a future compiler pass could make these `compiler_plan`-sourced instead.
- `distributed_executor_backend="mp"` is a D3B policy choice among four registry-valid options; D3B does not evaluate `ray`/`external_launcher`/`uni` tradeoffs.
- Preflight's per-rank GPU memory plausibility check uses each GPU's *total* memory, not currently-free memory; on a host with other GPU memory consumers this could be optimistic (not relevant to the current rejection, which fires on GPU *count* first).

## 24. Truth boundary

**D3B does not claim**: successful TP=2 vLLM execution, successful multi-GPU launch, NCCL execution, real distributed Qwen serving, GPU-to-GPU data transfer, or distributed performance benefit. **D3B does claim**: deterministic, version-aware, fail-closed materialization of a compiler-selected TP=2 plan into a vLLM distributed launch specification, validated against the real installed vLLM 0.24.0 and the real single-GPU host. Execution readiness reached: TP1 → `DRY_RUN_VALIDATED`, TP2 → `PREFLIGHT_REJECTED`. Neither ever reached `EXECUTION_READY` or `EXECUTION_STARTED`. Full detail: `truth_boundary.json`.

## 25. Recommended D4 dependency

**D4A: Single-GPU Serialized Whole-Model TP Contract Validation.**

Not D4B, because none of D4B's three preconditions hold yet: the launch specification is complete, but (1) whole-model TP compatibility is explicitly **not** established (§9) — D2/D3A proved correctness for one operator on one layer only, not vLLM's own whole-model TP sharding across all 24 layers and operator types; and (2) while TP2's current rejection is purely hardware (`insufficient_visible_gpu_count`), that is not sufficient on its own — D4B additionally requires whole-model TP compatibility to be established first, which it is not. D4A should extend D2/D3A's operator-level validation to the full Qwen model graph (all attention and MLP projections, all 24 layers) on this same single-GPU host before any real 2-GPU bring-up is attempted.
