# Distributed D4B: Real 2-GPU vLLM TP=2 Bring-Up and Correctness Validation

## 1. Executive result

D4B closes the final gap D3B left open: the compiler-selected TP=2 plan for `Qwen/Qwen2.5-0.5B-Instruct` was materialized through the existing, unmodified D3B adapter and **successfully executed by a real vLLM 0.24.0 server on two physical NVIDIA RTX 4090 GPUs**, with real distributed workers, real NCCL communicator initialization (`NCCL version 2.28.9+cuda13.0`, `world_size=2`, `rank=0`/`rank=1` on distinct `cudaDev`), and inference output verified identical to a same-host TP=1 reference across token IDs, text, finish reasons, and logprobs. All 23 provenance counters are zero. All required negative tests and OOM-safety checks pass. Zero orphan processes, zero stale ports, GPU memory returned to idle baseline.

D4B's only allowed primary claim holds:

> A compiler-selected TP=2 plan for Qwen2.5-0.5B-Instruct was materialized through the existing D3B vLLM adapter and successfully executed by vLLM 0.24.0 on two physical GPUs with real distributed workers and NCCL initialization, producing inference results consistent with a TP=1 reference under deterministic correctness workloads.

No claim of speedup, better TTFT/TPOT, higher throughput, profitable distributed selection, general multi-node support, or compiler-controlled per-operator vLLM execution is made anywhere in this report or its artifacts.

**Critical clarification honored throughout:** D4B validates that the compiler selects and materializes the whole-model TP strategy, and that vLLM's installed, source-verified whole-model TP implementation executes that strategy — it does **not** claim vLLM executed D4A's 170 Python-side work items individually.

## 2. Repository state before/after

| Repo | Branch | HEAD | Working tree |
|---|---|---|---|
| ml-graph-compiler-runtime | master | `59854b892629bc0bc7f43ca0bad3eab17464c030` | clean, zero changes throughout D4B |
| heterogeneous-inference-runtime | main | `f89adabf85c747ae99fc50dafa9a3f4a326593bb` | matched local/GitHub exactly at clone time; additive D4B changes only, no commit made on the remote host |

All work was performed on a freshly cloned copy on the 2-GPU host (`ssh -p 42356 root@137.175.76.24`, a Vast.ai rented instance, hostname `d4cc1c785bea`). Both repositories were cloned directly from `https://github.com/b07505054/{ml-graph-compiler-runtime,heterogeneous-inference-runtime}` — no stale local clone was reused. Full detail: `repository_state_before.json`, `repository_state_after.json`.

## 3. Files changed

All additive, mirroring the D3B/D4A convention:
- New: `deployment/vllm_adapter/{distributed_launch_controller,gpu_evidence,correctness_workload}.py`, `scripts/run_distributed_d4b_pipeline.py`, `scripts/run_distributed_d4b_pipeline_part2.py`, `tests/test_distributed_d4b_real_2gpu_vllm_tp2.py`, this report, and the `results/runtime_paths/distributed_d4b_real_2gpu_vllm_tp2/` artifact directory (46 files including bounded server logs).
- Modified, small and backward-compatible: `deployment/vllm_adapter/distributed_materializer.py` (moved D4A evidence resolution earlier so the already-existing, always-non-blocking `whole_model_tp_evidence_established` advisory check reflects a genuinely linked D4A artifact instead of always reading `False`; fixed `_estimate_model_footprint_mb` to resolve the real HF cache root via `huggingface_hub.constants.HF_HUB_CACHE` instead of a hardcoded path), `deployment/vllm_adapter/distributed_preflight.py` (same HF-cache-root fix for `check_model_locally_resolvable`), `deployment/vllm_adapter/distributed_launch_controller.py` (preserve a `FAILED`/`TIMED_OUT` terminal state through `stop()` instead of always overwriting it to `STOPPED` — a real bug caught by D4B's own negative testing, see §23).

All 53 pre-existing D3B+D4A tests were re-verified passing on the original dev machine after each fix before syncing to the 2-GPU host.

## 4. Cloud host and GPU inventory

Rented cloud marketplace instance (Vast.ai; hostname `d4cc1c785bea`). AMD EPYC 7452 (32-core/64-thread ×2 sockets = 128 threads), 377GB RAM, Ubuntu 24.04.4 LTS, driver 580.159.03, CUDA 13.0.

| GPU | Name | UUID | PCI bus | Memory | Compute cap. |
|---|---|---|---|---|---|
| 0 | NVIDIA GeForce RTX 4090 | `GPU-3e930a03-3101-6841-b2c0-6d7af909067d` | `00000000:41:00.0` | 24564 MiB | 8.9 |
| 1 | NVIDIA GeForce RTX 4090 | `GPU-c8703d8b-0918-b881-caf3-7d8343a9af35` | `00000000:61:00.0` | 24564 MiB | 8.9 |

Two distinct physical GPUs confirmed by distinct UUIDs **and** distinct PCI bus IDs (not MIG partitions, not logical aliases). Full detail: `cloud_host_inventory.json`.

## 5. Software environment

| Component | Version | Matches D3B/D4A baseline |
|---|---|---|
| Python | 3.12.3 | 3.12.13 on dev host — same minor version, patch differs (documented, not a version drift of concern) |
| PyTorch | 2.11.0+cu130 | **exact match** |
| vLLM | 0.24.0 | **exact match, no silent upgrade** |
| NCCL | 2.28.9 (via torch) | available |
| CUDA runtime (torch) | 13.0 | — |
| Live vLLM CLI argument registry | 267/267 arguments, **zero mismatches** vs. the committed D3B registry | confirmed identical |

vLLM 0.24.0 was installed via `uv pip install vllm==0.24.0` in a fresh Python 3.12 venv — the same installation method used on the original dev host — which naturally resolved to the identical pinned torch build (`2.11.0+cu130`), confirming no silent version drift. Full detail: `software_environment.json`.

## 6. Source compiler plan

Source of truth: `results/runtime_paths/distributed_d2_qwen_pipeline/real_qwen_tp2_execution_plan.json` (unchanged, byte-identical to the local dev machine and D2's original export). `source_execution_plan.json` is a verbatim copy. Full detail: `source_execution_plan.json`, `source_hashes` in `cross_layer_provenance.json`.

## 7. D4A evidence linkage

`results/runtime_paths/distributed_d4a_whole_model_tp_contract/whole_model_tp_classification.json` (classification `WHOLE_MODEL_TP_VALIDATED`) was hash-verified identical between the local dev machine and this 2-GPU host (`sha256: 375a08cd9a3f...`), then passed as `d4a_evidence_path` into the unmodified `materialize_launch_spec()` call. Result: `tp2_bundle.spec.whole_model_tp_evidence_status == "validated_serialized_whole_model_contract"`, with `whole_model_tp_evidence_source_artifact_hash` matching exactly. Full detail: `source_d4a_evidence.json`, `d4a_consistency` in `cross_layer_provenance.json`.

## 8. D3B launch-spec materialization result

Both TP1 and TP2 launch specs were generated by the same, unmodified `deployment.vllm_adapter.distributed_materializer.materialize_launch_spec()` used throughout D3B/D4A. No manually authored vLLM command was used. Full detail: `d4b_tp1_launch_spec.json`, `d4b_tp2_launch_spec.json`, `source_d3b_launch_spec.json`.

## 9. Two-GPU preflight

| Check | 1-GPU host (D3B) | 2-GPU host (D4B) |
|---|---|---|
| `sufficient_visible_gpu_count` | fail (`visible_gpu_count=1`) | **pass** (`visible_gpu_count=2`) |
| `model_identifier_resolvable` | pass | pass (after fixing the HF-cache-root bug, §23) |
| `no_duplicate_physical_device_assignment` | n/a (only 1 physical device resolvable) | pass |
| `whole_model_tp_evidence_established` (advisory) | n/a | pass (was always non-blocking; now correctly reflects the linked D4A evidence) |
| **Overall** | `PREFLIGHT_REJECTED`, `primary_reason=insufficient_visible_gpu_count` | **`preflight.passed = True`, zero rejection reasons** |

Rank placements resolved to real, distinct physical GPUs: rank 0 → physical device 0 (`GPU-3e930a03...`), rank 1 → physical device 1 (`GPU-c8703d8b...`). TP2 was never downgraded (`tensor_parallel_size` remained 2 throughout). Full detail: `tp1_preflight.json`, `tp2_preflight.json`.

## 10. TP1 launch

Real single-GPU vLLM 0.24.0 server, `--tensor-parallel-size 1`, `CUDA_VISIBLE_DEVICES=0`. Ready in **69.1s**. A real `/v1/completions` request against the full 10-prompt correctness corpus (plus later reuse) succeeded (HTTP 200) on every call. Graceful shutdown in 1.01s, zero remaining descendant processes, GPU memory returned to idle. Full detail: `tp1_cli.json`, `tp1_server_lifecycle.json`, `process_tree_tp1.json`.

## 11. TP2 launch

Real 2-GPU vLLM 0.24.0 server, `--tensor-parallel-size 2 --pipeline-parallel-size 1`, `CUDA_VISIBLE_DEVICES=0,1`. Ready in **81.1s** (`pid=20106`, real descendants: `EngineCore`, two `VLLM::Worker_TP{0,1}` processes). Graceful shutdown in 1.01s, zero remaining descendants, GPU memory returned to idle. Full detail: `tp2_cli.json`, `tp2_server_lifecycle.json`, `process_tree_tp2.json`.

## 12. Worker and rank topology

Tracked process tree during TP2: API server (pid 20106) → EngineCore (pid 20320/20321 pair) → `VLLM::Worker_TP0` (pid 20528) and `VLLM::Worker_TP1` (pid 20539) — exactly 2 rank workers, matching `world_size=2`. Full detail: `rank_worker_inventory.json`.

## 13. Physical GPU assignment

Directly queried via `nvidia-smi --query-compute-apps` while TP2 was active:

| Process | PID | GPU UUID | GPU memory used |
|---|---|---|---|
| `VLLM::Worker_TP0` | 20528 | `GPU-3e930a03-...` (index 0) | 22394 MiB |
| `VLLM::Worker_TP1` | 20539 | `GPU-c8703d8b-...` (index 1) | 22394 MiB |

Two **distinct** physical GPU UUIDs used, no duplicate assignment, rank placement agrees exactly with the D3B launch spec (rank 0 → GPU 0, rank 1 → GPU 1). This is **not** inferred solely from `--tensor-parallel-size 2` — it is a direct, live process-to-GPU query. Full detail: `rank_gpu_mapping.json`, `gpu_inventory_during_tp2.json`.

## 14. NCCL initialization evidence

Direct log evidence (not inferred from server readiness alone):

```
(Worker pid=20528) INFO [parallel_state.py:1588] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:54565 backend=nccl
(Worker pid=20539) INFO [parallel_state.py:1588] world_size=2 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:54565 backend=nccl
d4cc1c785bea:20528:20528 [0] NCCL INFO NCCL version 2.28.9+cuda13.0
d4cc1c785bea:20528:20528 [0] NCCL INFO ncclCommInitRank comm ... rank 0 nranks 2 cudaDev 0 ... Init COMPLETE
d4cc1c785bea:20539:20539 [1] NCCL INFO ncclCommInitRank comm ... rank 1 nranks 2 cudaDev 1 ... Init COMPLETE
```

195 NCCL log lines total; `backend=nccl` explicit for both ranks; `world_size=2`; distinct `cudaDev` (0 and 1) per rank. This is real NCCL communicator initialization for world_size=2 — not a different backend, not a claim made from server readiness alone. Full detail: `nccl_initialization.json`.

## 15. Model/tokenizer identity

`Qwen/Qwen2.5-0.5B-Instruct`: hidden_size=896, 24 layers, 14 attention heads, 2 KV heads, vocab_size=151936, `tie_word_embeddings=True` — identical config used by both TP1 and TP2 (same local HF cache snapshot, same config hash). Full detail: `model_tokenizer_identity.json`.

## 16. Correctness workload

10 deterministic prompts spanning very-short, medium, longer-prefill, code, numeric-reasoning, and repeated-prefix categories; `temperature=0`, `top_p=1`, fixed `max_tokens=24`, fixed `seed=1234`, `logprobs=5`. Full detail: `correctness_prompt_corpus.json`.

## 17. TP1/TP2 token comparison

Token IDs were derived deterministically from vLLM's own `logprobs.tokens` field (the literal emitted tokens in decoded surface form) via a reverse lookup against the real tokenizer vocabulary, with a round-trip decode integrity check — not by re-tokenizing surface text. **All token IDs matched between TP1 and TP2 for every prompt**, with zero undetermined extractions. Full detail: `token_comparison.json`.

## 18. Text and finish-reason comparison

**All generated text matched exactly** between TP1 and TP2; **all finish reasons matched**. Full detail: `text_comparison.json`.

## 19. Logprob comparison

Selected token IDs matched for every prompt; mean top-5 logprob-ID agreement rate **99.7%**; max absolute logprob error **0.047** (small floating-point-level variance consistent with different rank-local reduction order, not a correctness defect — no unexplained large divergence). Full detail: `logprob_comparison.json`.

## 20. Repeated and mixed-shape validation

20 repetitions of the same prompt against TP2: all succeeded, exactly 1 distinct output text (fully stable, no rank desynchronization). Mixed-length sequential requests (short/long/code/medium/numeric interleaved): all succeeded, no cross-contamination between prompts. Full detail: `repeated_request_validation.json`, `mixed_shape_validation.json`.

## 21. Minimal concurrency validation

Concurrency 2 and concurrency 4 (bounded, not a throughput benchmark): all requests completed successfully at both levels, no collective deadlock, no rank timeout. Full detail: `concurrency_correctness.json`.

## 22. Negative tests

16 required fail-closed cases + 3 additional structural checks, all passing: TP2-with-CUDA_VISIBLE_DEVICES-exposing-one-GPU, duplicate physical GPU placement, invalid GPU index, occupied API port, invalid master port, model resolution failure, unsupported CLI flag, startup timeout (bounded, real process, clean kill), premature server exit (real process, `FAILED` state correctly detected), request timeout, worker/rank exit (smallest controlled test: real TP1 server, killed the actual `EngineCore` PID identified from its own log line, confirmed the subsequent request failed and cleanup was still complete), malformed launch spec, D4A evidence hash mismatch, whole-model evidence missing, attempted TP2-to-TP1 downgrade (never occurs), attempted launch after rejected preflight. No child process was created for any rejected spec. Full detail: `negative_tests.json`.

## 23. OOM safety

`--gpu-memory-utilization 0.01` (intentionally unsafe for KV cache) was launched under a bounded timeout: the server process exited cleanly with a non-zero exit code (fail-closed, no host-wide OOM), and cleanup verified zero remaining descendants. This test caught a **real bug**: `ServerLaunchController.stop()` was unconditionally overwriting an already-`FAILED` terminal state with `STOPPED`, obscuring the true failure reason. Fixed to preserve the prior `FAILED`/`TIMED_OUT` classification through `stop()`'s cleanup duty. A second real bug was caught by the worker-rank-exit negative test (an earlier version picked an arbitrary low-numbered descendant PID to kill rather than the actual `EngineCore`, silently not exercising a real worker failure) — fixed to parse the real `EngineCore` PID from the server's own log line. Full detail: `oom_safety_validation.json`.

## 24. Process, port, and GPU cleanup

Zero orphan descendants after both TP1 and TP2 shutdown (verified via `psutil` descendant enumeration). No stale port listeners on either allocated port. No Ray processes introduced. GPU memory returned to idle baseline (1 MiB per GPU) within the bounded polling window on both runs — nvidia-smi lags a few seconds behind actual process exit for CUDA memory reclamation, which the cleanup verification explicitly polls for rather than checking once immediately. Full detail: `process_cleanup.json`, `port_cleanup.json`, `gpu_memory_cleanup.json`.

## 25. Cross-layer provenance

All 23 required counters computed from real events, **all zero**: `source_plan_mismatch_count`, `candidate_mismatch_count`, `d4a_evidence_mismatch_count`, `vllm_version_mismatch_count`, `model_identity_mismatch_count`, `tp_mismatch_count`, `pp_mismatch_count`, `world_size_mismatch_count`, `rank_count_mismatch_count`, `physical_gpu_mismatch_count`, `duplicate_gpu_assignment_count`, `nccl_initialization_mismatch_count`, `worker_count_mismatch_count`, `request_failure_count`, `token_output_mismatch_count`, `text_output_mismatch_count`, `finish_reason_mismatch_count`, `silent_downgrade_count`, `preflight_bypass_count`, `unexpected_backend_count`, `unexpected_process_launch_count`, `orphan_process_count`, `stale_port_count`, `gpu_memory_cleanup_mismatch_count`. Full detail: `cross_layer_provenance.json`.

## 26. Regression results

D1 (`test_distributed_tp_process_runtime.py`), D2 (`test_distributed_d2_qwen_pipeline.py`), D3A (`test_distributed_d3a_live_qwen_tensor.py`): all pass with zero exceptions on this 2-GPU host. D3B (`test_distributed_d3b_vllm_launch_spec.py`): 23/26 pass. D4A (`test_distributed_d4a_whole_model_tp_contract.py`): 26/27 pass.

**The 4 non-passing tests are not a functional regression.** They are 3 D3B tests and 1 D4A test whose entire purpose, by name and by design, is to assert that TP2 preflight *rejects* due to `insufficient_visible_gpu_count` — the only possible outcome on the original single-GPU D3B/D4A development host. On this genuine 2-GPU D4B host, preflight correctly *passes* instead, because 2 GPUs really are visible — exactly the behavior the fail-closed preflight design exists to produce once real hardware supports it. This is confirmation the same code adapts correctly across environments, not a defect. The D3B/D4A test files were not modified; they remain accurate evidence of the original single-GPU baseline. All 49 other D3B/D4A tests (every negative test, provenance check, and whole-model correctness assertion not specific to the 1-GPU rejection outcome) pass unchanged. Full detail: `regression_summary.json`.

## 27. Structural measurements

Bring-up and correctness diagnostics only — explicitly **not** a controlled performance benchmark:

| Metric | TP1 | TP2 |
|---|---|---|
| Startup/readiness latency | 69.1 s | 81.1 s |
| Request latency (median / p95, n=10) | 0.073 s / 0.408 s | 0.076 s / 0.476 s |
| Shutdown latency | 1.01 s | 1.01 s |
| GPU memory per rank | 22094 MiB (1 GPU) | 22394 MiB × 2 GPUs |

No speedup, TTFT, TPOT, or throughput comparison is computed or implied from these numbers. Full detail: `performance_measurements.json`.

## 28. Test totals

- D4B structural tests: 2/2 passed.
- D4B negative tests: 19/19 passed (16 required + 3 additional).
- D4B OOM safety: 1/1 passed.
- D1/D2/D3A regressions: all green, zero exceptions.
- D3B/D4A regressions: 49/53 pass; 4 non-passing are explained 1-GPU-host-assumption tests (§26), not defects.
- Cross-layer provenance: 23/23 counters zero.
- TP1/TP2 correctness: 100% HTTP 200, 100% token/text/finish-reason match.

Full detail: `test_summary.json`.

## 29. Artifacts

All 46 required files present under `results/runtime_paths/distributed_d4b_real_2gpu_vllm_tp2/` (including a bounded `logs/` subdirectory with the two real server logs, 20KB and 48KB — small enough to commit directly, no massive raw logs), plus this report.

## 30. Known limitations

- The pipeline's real-hardware execution spanned two SSH sessions: a transient network interruption on the rented cloud instance dropped the connection during the (lightweight, non-GPU) regression-suite step, *after* all real TP1/TP2 launches, correctness testing, negative tests, and cleanup had already completed and been verified clean. A resume script (`run_distributed_d4b_pipeline_part2.py`) completed the remaining bookkeeping (regressions, provenance, measurements, test summary) by reading back the already-written, verified artifacts as source of truth, after confirming on reconnect that zero orphan processes and idle GPU memory remained — no server was re-launched.
- `gpu_memory_per_rank_mb_during_tp2` in `performance_measurements.json` could not be recovered post-hoc after the interruption; the equivalent real measurement is preserved in `gpu_inventory_during_tp2.json`'s live `compute_apps` snapshot instead.
- Python patch version differs from the original D3B/D4A dev host (3.12.3 vs. 3.12.13) — same minor version, not a functional concern, and not vLLM/torch (which matched exactly).
- Consistent with D4A's own known limitation: the whole-model distributed plan remains a Python-side schema expansion, not emitted by the C++ `DistributedStrategyPlanningPass` itself.

## 31. Truth boundary

**D4B does not claim**: TP=2 speedup, better TTFT, better TPOT, higher throughput, profitable distributed selection, general multi-node support, or compiler-controlled per-operator vLLM execution. **D4B does claim**: real vLLM 0.24.0 TP=2 execution on two physical GPUs with real NCCL initialization, correctness-consistent with a TP=1 reference, using the compiler-selected strategy materialized through the unmodified D3B adapter. Full detail: `truth_boundary.json`.

## 32. Recommended D5

Since D4B succeeded on every acceptance criterion, the recommended next stage is:

**Distributed D5: Real Multi-GPU Profiling and Compiler Cost-Model Calibration.**

D5 should measure TP1 vs. TP2 across controlled workload dimensions (batch size, sequence length, concurrency) on this same real 2-GPU class of hardware, replacing the current CPU/IPC-derived distributed cost assumptions in the compiler's cost model with real GPU and NCCL measurements — building directly on D4B's now-proven real execution path rather than re-deriving bring-up correctness.
