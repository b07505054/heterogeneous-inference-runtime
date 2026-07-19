# D4B Achievement Report: Real 2-GPU vLLM TP=2 Bring-Up and Correctness Validation

**Status: COMPLETE.** This is the final, consolidated technical reference for D4B — the stage that took a compiler-selected tensor-parallel strategy all the way to real, verified execution on physical multi-GPU hardware. It is written to stand alone: a reader with no prior context should be able to understand what was built, what was proven, and how to check the evidence themselves.

For the turn-by-turn process narrative (including the mid-run SSH interruption and recovery), see [`DISTRIBUTED_D4B_REAL_2GPU_VLLM_TP2_REPORT.md`](DISTRIBUTED_D4B_REAL_2GPU_VLLM_TP2_REPORT.md). For a portfolio-length summary, see [`DISTRIBUTED_D4B_ACHIEVEMENT_SUMMARY.md`](DISTRIBUTED_D4B_ACHIEVEMENT_SUMMARY.md).

All raw evidence referenced below lives in `heterogeneous-inference-runtime/results/runtime_paths/distributed_d4b_real_2gpu_vllm_tp2/` — see [`EVIDENCE_INDEX.md`](../results/runtime_paths/distributed_d4b_real_2gpu_vllm_tp2/EVIDENCE_INDEX.md) in that directory for a categorized map of every file.

---

## 1. Architecture

D4B sits at the bottom of a six-stage evidence chain. Each stage is a separate, independently-verifiable milestone; D4B is the first to touch real multi-GPU hardware.

```
D1  Compiler-planned TP=2, simulated multi-process (localhost IPC, no GPU)
D2  Real Qwen2.5-0.5B-Instruct graph → compiler TP1/TP2 candidate generation,
    legality + cost analysis, TP2 selection under explicit opt-in
D3A Real live Qwen activation capture; serialized rank-local o_proj math
    validated against the real model (max_abs_error ~1.8e-7)
D3B Version-aware vLLM 0.24.0 launch-spec materialization; fail-closed
    preflight (rejects TP=2 correctly on a 1-GPU host)
D4A Whole-model (170-work-item) serialized TP contract validation against
    the real Transformers + installed vLLM implementation
    (WHOLE_MODEL_TP_VALIDATED)
D4B Real 2×RTX-4090 execution of the D3B-materialized TP=2 launch spec:
    real vLLM server, real NCCL, real distributed workers, output
    verified against a TP=1 reference          <-- this report
```

### Component architecture (D4B additions)

D4B added three new modules to the existing `deployment/vllm_adapter/` package, plumbed into the unmodified D3B materializer:

```
deployment/vllm_adapter/
├── distributed_materializer.py      (D3B, existing — unmodified call path,
│                                      2 small bug fixes, see §5)
├── distributed_preflight.py         (D3B, existing — 1 small bug fix, see §5)
├── distributed_launch_controller.py (NEW — bounded process lifecycle)
├── gpu_evidence.py                  (NEW — nvidia-smi parsing, NCCL log
│                                      extraction, cleanup verification)
└── correctness_workload.py          (NEW — deterministic prompt corpus,
                                       TP1-vs-TP2 comparison logic)
```

`ServerLaunchController` (in `distributed_launch_controller.py`) is the key new primitive: it starts a materialized launch spec's argv as a real subprocess (never `shell=True`), polls the OpenAI-compatible `/health` endpoint under a bounded timeout, tracks the full descendant process tree via `psutil`, and guarantees termination (graceful `SIGTERM` → escalating `SIGKILL`) with a verified zero-descendant post-condition. It exposes no `force_launch`, `ignore_preflight`, or `allow_unsupported` parameter — a rejected preflight result can never be pushed into a running process.

## 2. Execution flow

```
 1. Load the exact D2 ExecutionPlan (real_qwen_tp2_execution_plan.json)
 2. materialize_launch_spec(plan, d4a_evidence_path=...)          [D3B, unmodified]
        -> preflight.passed?  --- NO --> PREFLIGHT_REJECTED, STOP (no process ever started)
                               --- YES ↓
 3. ServerLaunchController.start(argv, env)      [real subprocess.Popen, no shell]
 4. wait_for_readiness(timeout_s=300)            [poll /health; detect premature exit]
        -> ready? --- NO --> FAILED / TIMED_OUT, stop(), STOP
                   --- YES ↓
 5. Prove 2 physical GPUs used     [nvidia-smi --query-compute-apps, live]
 6. Prove real NCCL init           [regex-scanned server log: ncclCommInitRank ... COMPLETE]
 7. Run correctness workload against TP2, then against a same-host TP1 reference
 8. Compare: token IDs / text / finish_reason / logprobs
 9. Repeated / mixed-shape / bounded-concurrency validation
10. stop()  [SIGTERM -> poll -> SIGKILL if needed -> verify zero descendants]
11. Verify GPU memory returns to idle baseline (bounded poll, not a single snapshot)
12. Compute all 23 provenance counters; assert all zero
```

Every arrow in this diagram is backed by an artifact — see §9-§11 below for the specific evidence at each step.

## 3. Hardware inventory

Rented cloud GPU instance (Vast.ai marketplace; container hostname `d4cc1c785bea`), reached via SSH — never a local machine (the development laptop used for D1–D4A has exactly one GPU and cannot run this stage).

| Property | Value |
|---|---|
| CPU | AMD EPYC 7452, 2 sockets × 32 cores × 2 threads = 128 threads |
| RAM | 377 GB |
| OS | Ubuntu 24.04.4 LTS |
| Driver | 580.159.03 |
| CUDA (driver-reported) | 13.0 |
| GPU 0 | NVIDIA GeForce RTX 4090, UUID `GPU-3e930a03-3101-6841-b2c0-6d7af909067d`, PCI `00000000:41:00.0`, 24564 MiB, compute capability 8.9 |
| GPU 1 | NVIDIA GeForce RTX 4090, UUID `GPU-c8703d8b-0918-b881-caf3-7d8343a9af35`, PCI `00000000:61:00.0`, 24564 MiB, compute capability 8.9 |

Two **physically distinct** GPUs, confirmed by distinct UUIDs *and* distinct PCI bus IDs — not a MIG partition, not a logical alias, not two ranks folded onto one device. Source: `cloud_host_inventory.json`, `gpu_inventory_before.json`.

## 4. Software versions

| Component | D4B (2-GPU host) | D3B/D4A baseline (1-GPU dev host) | Match |
|---|---|---|---|
| Python | 3.12.3 | 3.12.13 | same minor version |
| PyTorch | `2.11.0+cu130` | `2.11.0+cu130` | **exact** |
| vLLM | `0.24.0` | `0.24.0` | **exact — no silent upgrade** |
| NCCL | 2.28.9 (bundled via torch) | — | new evidence this stage |
| CUDA runtime (torch) | 13.0 | 13.0 | exact |
| vLLM CLI argument registry | 267 arguments | 267 arguments | **0 mismatches** |

vLLM 0.24.0 was installed with `uv pip install vllm==0.24.0` — the identical installation method used originally — and independently resolved to the identical pinned torch build, which is itself evidence against silent dependency drift. Source: `software_environment.json`.

## 5. Compiler → runtime boundary

This is the exact seam D4B validates, and the exact claim it does **not** make.

```
COMPILER SIDE (ml-graph-compiler-runtime, C++)     ── unmodified, 0 changes ──
  DistributedStrategyPlanningPass selects TP=2 for one real Qwen o_proj
  operator instance → exports real_qwen_tp2_execution_plan.json

              │  (D2 ExecutionPlan JSON, byte-identical, hash-verified)
              ▼
RUNTIME SIDE (heterogeneous-inference-runtime, Python)
  materialize_launch_spec()  [D3B]
      -> typed VLLMDistributedLaunchSpec, version-aware CLI argv
      -> fail-closed preflight (hardware, model, registry checks)

              │  (real argv + env, e.g. --tensor-parallel-size 2)
              ▼
vLLM 0.24.0 (installed package, unmodified)
      -> vLLM's OWN whole-model TP implementation (QKVParallelLinear,
         RowParallelLinear, MergedColumnParallelLinear, VocabParallelEmbedding
         — all validated by D4A against this exact installed version)
      -> real NCCL process group, real distributed workers, real GPUs
```

**The precise, load-bearing claim:** the compiler selects and materializes the whole-model TP strategy; vLLM's installed, source-verified whole-model TP implementation executes that strategy. D4B does **not** claim that vLLM's real execution consumed D4A's 170 Python-side work items individually — D4A validated the *mathematical contract* those work items describe against vLLM's real source; D4B validates that the *real vLLM binary*, launched with the *compiler's chosen TP degree*, produces *correct output* on *real hardware*. These are two different, complementary proofs, not one continuous causal chain through vLLM's internals.

Two small, backward-compatible bug fixes were made to the D3B code during D4B (both caught by real-hardware testing, both re-verified against the full 53-test D3B/D4A suite before and after):

1. `distributed_preflight.check_model_locally_resolvable` (and `distributed_materializer._estimate_model_footprint_mb`) hardcoded `~/.cache/huggingface/hub` as the model cache root. On a host with a custom `HF_HOME` (this instance uses `/workspace/.hf_home`), that silently produced a false "model not resolvable" rejection. Fixed to resolve via `huggingface_hub.constants.HF_HUB_CACHE`, which is exactly what the library itself uses.
2. `materialize_launch_spec` computed the (always non-blocking, advisory-only) `whole_model_tp_evidence_established` preflight flag *before* resolving the D4A evidence link, so it always read `False` even when valid D4A evidence was supplied. Fixed by reordering so the flag reflects the real resolved status.

Neither fix changes any D3B/D4A test expectation; both were re-verified against the pre-existing 53-test suite.

## 6. Launch-spec materialization

Both TP1 and TP2 launch specs were produced by the same unmodified `deployment.vllm_adapter.distributed_materializer.materialize_launch_spec()` call used throughout D3B/D4A — **no manually authored vLLM command was used anywhere in D4B.**

Representative TP2 argv (from `tp2_cli.json`):

```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --no-trust-remote-code --dtype float16 --seed 1234 \
  --served-model-name qwen2.5-0.5b-instruct-tp2-planning-only \
  --host 127.0.0.1 --port <bound-free-port> \
  --master-addr 127.0.0.1 --master-port 29501 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --data-parallel-size 1 \
  --distributed-executor-backend mp \
  --max-model-len 2048 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.9 --enable-prefix-caching --enable-chunked-prefill
```

Preflight transition, the central before/after of the whole D3B→D4B story:

| Check | 1-GPU host (D3B, Nov session) | 2-GPU host (D4B, this stage) |
|---|---|---|
| `sufficient_visible_gpu_count` | **fail** — `visible_gpu_count=1` | **pass** — `visible_gpu_count=2` |
| Overall preflight | `PREFLIGHT_REJECTED`, `primary_reason=insufficient_visible_gpu_count` | `passed=True`, zero rejection reasons |
| `tensor_parallel_size` | 2 (spec generated, execution blocked) | 2 (spec generated **and executed**) |
| Rank → physical GPU | unresolved (only 1 device) | rank 0 → GPU 0, rank 1 → GPU 1 |

TP=2 was never downgraded to TP=1 at any point (`silent_downgrade_count = 0` in provenance). Source: `d4b_tp1_launch_spec.json`, `d4b_tp2_launch_spec.json`, `tp1_preflight.json`, `tp2_preflight.json`, `tp1_cli.json`, `tp2_cli.json`.

## 7. NCCL evidence

Direct log evidence — not inferred from `--tensor-parallel-size 2` or from server readiness alone:

```
(Worker pid=20528) INFO [parallel_state.py:1588] world_size=2 rank=0 local_rank=0 \
    distributed_init_method=tcp://127.0.0.1:54565 backend=nccl
(Worker pid=20539) INFO [parallel_state.py:1588] world_size=2 rank=1 local_rank=1 \
    distributed_init_method=tcp://127.0.0.1:54565 backend=nccl
d4cc1c785bea:20528:20528 [0] NCCL INFO NCCL version 2.28.9+cuda13.0
d4cc1c785bea:20528:20528 [0] NCCL INFO ncclCommInitRank comm 0x... rank 0 nranks 2 cudaDev 0 ... Init COMPLETE
d4cc1c785bea:20539:20539 [1] NCCL INFO ncclCommInitRank comm 0x... rank 1 nranks 2 cudaDev 1 ... Init COMPLETE
```

195 total NCCL log lines captured. `backend=nccl` explicit for both ranks. `world_size=2` explicit. Distinct `cudaDev` (0 and 1) per rank — i.e. the two NCCL ranks are bound to the two different physical GPUs, not to the same device twice. Source: `nccl_initialization.json` (includes the full regex-scanned line set), `logs/tp2_server.log` (complete raw log, 48 KB).

## 8. GPU assignment evidence

Queried live via `nvidia-smi --query-compute-apps` while the TP=2 server was actively serving — a direct, real-time process-to-device query, not an inference from configuration:

| Process | PID | GPU UUID | GPU index | GPU memory used |
|---|---|---|---|---|
| `VLLM::Worker_TP0` | 20528 | `GPU-3e930a03-...` | 0 | 22394 MiB |
| `VLLM::Worker_TP1` | 20539 | `GPU-c8703d8b-...` | 1 | 22394 MiB |

Two distinct GPU UUIDs, zero duplicate assignment, rank placement agrees exactly with the D3B launch spec's declared mapping (rank 0 → GPU 0, rank 1 → GPU 1). Source: `rank_gpu_mapping.json`, `gpu_inventory_during_tp2.json`, `rank_worker_inventory.json`.

## 9. Correctness evidence

Deterministic workload: 10 prompts (very-short ×3, medium ×2, longer-prefill ×1, code ×1, numeric-reasoning ×2, repeated-prefix ×1), `temperature=0`, `top_p=1`, fixed `max_tokens=24`, fixed `seed=1234`, `logprobs=5` requested. Run against a same-host TP1 reference server and the TP2 server, in separate (non-concurrent) sessions.

| Signal | Result |
|---|---|
| HTTP status | 200/200 for every request, both TP1 and TP2 |
| Generated token IDs (derived from vLLM's own `logprobs.tokens` via reverse-vocabulary lookup + round-trip integrity check, not re-tokenized text) | **100% match**, 0 undetermined |
| Generated text | **100% exact match** |
| `finish_reason` | **100% match** |
| Selected-token logprob agreement | **100% match** |
| Top-5 logprob-ID agreement rate | **99.7%** |
| Max absolute logprob error | 0.047 (small floating-point-level variance from a different rank-local reduction order — not a correctness defect) |
| Repeated requests (20×, same prompt, TP2) | 1 distinct output text — fully stable, no rank desync |
| Mixed-length sequential requests | all succeeded, no cross-contamination between prompts |
| Bounded concurrency (2, 4 simultaneous requests) | all succeeded, no deadlock, no rank timeout |

Source: `token_comparison.json`, `text_comparison.json`, `logprob_comparison.json`, `repeated_request_validation.json`, `mixed_shape_validation.json`, `concurrency_correctness.json`, `tp1_outputs.jsonl`, `tp2_outputs.jsonl`.

## 10. Negative tests

19 fail-closed cases, all passing — 16 required plus 3 additional structural checks:

| Category | Cases |
|---|---|
| Hardware/config | CUDA_VISIBLE_DEVICES exposing 1 GPU, duplicate physical GPU placement, invalid GPU index |
| Networking | occupied API port, invalid master port |
| Model/registry | model resolution failure, unsupported CLI flag |
| Process lifecycle | startup timeout, premature server exit, request timeout, worker/rank exit (real `EngineCore` process killed mid-run) |
| Plan integrity | malformed launch spec, D4A evidence hash mismatch, whole-model evidence missing |
| Fail-closed guarantees | attempted TP2→TP1 downgrade (never occurs), attempted launch after rejected preflight (no subprocess created) |

**Two real bugs were caught and fixed by these tests** (not merely test-writing exercises — genuine defects found through fail-closed adversarial testing):
- `ServerLaunchController.stop()` was unconditionally overwriting an already-`FAILED` terminal state with `STOPPED`, obscuring the true failure reason after a crashed process was cleaned up. Fixed to preserve the prior terminal classification.
- The worker-kill negative test originally targeted an arbitrary low-PID descendant process rather than the real `EngineCore`, so it wasn't actually exercising a worker failure. Fixed to parse the real `EngineCore` PID from the server's own log line before killing it.

No child process was created for any rejected launch spec, in any test. OOM safety (`--gpu-memory-utilization 0.01`) failed closed with a clean, bounded, non-zero-exit-code process termination — never a host-wide OOM. Source: `negative_tests.json`, `oom_safety_validation.json`.

## 11. Provenance

All 23 required counters, computed from real events (not asserted), **all zero**:

`source_plan_mismatch_count`, `candidate_mismatch_count`, `d4a_evidence_mismatch_count`, `vllm_version_mismatch_count`, `model_identity_mismatch_count`, `tp_mismatch_count`, `pp_mismatch_count`, `world_size_mismatch_count`, `rank_count_mismatch_count`, `physical_gpu_mismatch_count`, `duplicate_gpu_assignment_count`, `nccl_initialization_mismatch_count`, `worker_count_mismatch_count`, `request_failure_count`, `token_output_mismatch_count`, `text_output_mismatch_count`, `finish_reason_mismatch_count`, `silent_downgrade_count`, `preflight_bypass_count`, `unexpected_backend_count`, `unexpected_process_launch_count`, `orphan_process_count`, `stale_port_count`, `gpu_memory_cleanup_mismatch_count`.

The provenance chain traced end-to-end: `compiler candidate ID → selected TP=2 ExecutionPlan → D4A whole-model evidence hash → D3B vLLM launch specification → live 2-GPU hardware inventory → preflight → generated argv/environment → server process → distributed workers → physical GPU placements → NCCL initialization → TP=2 requests → TP=1 reference requests → output comparison → shutdown and cleanup`. Source: `cross_layer_provenance.json`.

## 12. Limitations

- **Regression tests, not a regression.** 4 of the pre-existing D3B/D4A tests "fail" on this 2-GPU host — but by design: their entire purpose is to assert the 1-GPU hardware-rejection outcome. On real 2-GPU hardware, preflight correctly passes instead of rejecting, which is exactly what the fail-closed design exists to do once real hardware supports it. The D3B/D4A test files were not modified. All other 49 tests pass unchanged. See `regression_summary.json` for the itemized explanation.
- **Session interruption, fully recovered.** A transient SSH disconnect on the rented instance interrupted the pipeline during the (GPU-idle) regression-suite step — *after* every real-hardware step (TP1/TP2 launch, correctness suite, negative tests, OOM safety, cleanup) had already completed and been verified clean. A resume script completed the remaining bookkeeping from the already-written, verified artifacts; no server was re-launched, and zero orphan processes / idle GPU memory were confirmed on reconnect before proceeding.
- **`gpu_memory_per_rank_mb_during_tp2` in `performance_measurements.json`** could not be recovered after the interruption; the equivalent live measurement is preserved in `gpu_inventory_during_tp2.json`.
- **The compiler still plans one operator, not the whole model.** D4A's whole-model plan is a Python-side schema expansion; the C++ `DistributedStrategyPlanningPass` itself has not been extended to emit a whole-model work-item set. This is an explicit, carried-forward limitation from D4A, not something D4B changes.
- **No performance claim exists or was measured.** Structural timings (startup ~69-81s, request latency ~0.07s median) are bring-up diagnostics only — see `performance_measurements.json`'s own `no_speedup_claim: true` field.

## 13. Truth boundary

**D4B does not claim:** TP=2 speedup, better TTFT, better TPOT, higher throughput, profitable distributed selection, general multi-node support, or compiler-controlled per-operator vLLM execution.

**D4B does claim:** a compiler-selected TP=2 plan for Qwen2.5-0.5B-Instruct was materialized through the existing D3B vLLM adapter and successfully executed by vLLM 0.24.0 on two physical GPUs with real distributed workers and NCCL initialization, producing inference results consistent with a TP=1 reference under deterministic correctness workloads.

This is the first stage in the D1→D4B chain to touch real multi-GPU hardware, and the last stage whose scope is correctness and bring-up rather than performance. Source: `truth_boundary.json`.
