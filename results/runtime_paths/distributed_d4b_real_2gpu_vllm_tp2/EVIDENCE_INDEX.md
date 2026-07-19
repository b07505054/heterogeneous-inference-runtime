# D4B Evidence Index

This directory contains every raw artifact produced by the D4B pipeline
(`scripts/run_distributed_d4b_pipeline.py` + `run_distributed_d4b_pipeline_part2.py`),
run against a real 2× RTX 4090 host. Files are kept **flat** (matching the
D1–D4A convention elsewhere in this repository, and matching exactly what
the pipeline scripts write), so this index exists to give a new reader a
categorized map instead of a directory tree to dig through.

Start here if you want the narrative: [`../../docs/DISTRIBUTED_D4B_ACHIEVEMENT_REPORT.md`](../../../docs/DISTRIBUTED_D4B_ACHIEVEMENT_REPORT.md).
This index exists for readers who want to go straight to a specific piece of evidence.

## 1. Repository and evidence-chain preservation

| File | What it proves |
|---|---|
| `repository_state_before.json` | Git HEAD/branch/working-tree state of both repos, captured before any D4B file was written |
| `repository_state_after.json` | Same, captured after D4B completed — confirms no uncommitted stray changes |
| `d1_d2_d3a_d3b_d4a_preservation.json` | SHA-256 of every file in the D1/D2/D3A/D3B/D4A result directories — confirms D4B never modified prior-stage evidence |

## 2. Hardware and software environment

| File | What it proves |
|---|---|
| `cloud_host_inventory.json` | Full machine inventory: hostname, CPU, RAM, OS, driver, and per-GPU name/UUID/PCI-bus/memory/compute-capability for both GPUs |
| `software_environment.json` | Exact Python/PyTorch/vLLM/NCCL/CUDA versions, plus a 267-argument vLLM CLI registry diff against the committed D3B baseline (0 mismatches) |
| `gpu_inventory_before.json` | `nvidia-smi` snapshot before any server was launched (idle baseline) |
| `gpu_inventory_during_tp1.json` | Live snapshot while the TP=1 server was actively serving |
| `gpu_inventory_during_tp2.json` | Live snapshot while the TP=2 server was actively serving — **this is the file with the two-distinct-GPU compute-apps proof** |
| `gpu_inventory_after.json` | Snapshot after both servers were shut down (back to idle) |

## 3. Source compiler plan and cross-stage evidence linkage

| File | What it proves |
|---|---|
| `source_execution_plan.json` | Verbatim copy of the D2 compiler-exported `real_qwen_tp2_execution_plan.json` — the actual source of truth, hash-verified identical to the original |
| `source_d3b_launch_spec.json` | Verbatim copy of the committed D3B `tp2_launch_spec.json`, for hash cross-checking |
| `source_d4a_evidence.json` | Verbatim copy of the D4A `whole_model_tp_classification.json` (`WHOLE_MODEL_TP_VALIDATED`), the artifact D4B links into the launch spec |

## 4. Launch-spec materialization and preflight

| File | What it proves |
|---|---|
| `d4b_tp1_launch_spec.json` / `d4b_tp2_launch_spec.json` | The full typed `VLLMDistributedLaunchSpec` produced by the unmodified D3B materializer on this host |
| `tp1_preflight.json` / `tp2_preflight.json` | Every individual preflight check result — **`tp2_preflight.json` is where the 1-GPU-host `insufficient_visible_gpu_count` rejection is replaced by a full pass** |
| `tp1_cli.json` / `tp2_cli.json` | The deterministic argv array actually used to launch each server, plus version-aware argument-support annotations |

## 5. Server lifecycle and process topology

| File | What it proves |
|---|---|
| `tp1_server_lifecycle.json` / `tp2_server_lifecycle.json` | Full lifecycle event log (start → ready → stop), latencies, exit codes, stop-result cleanup summary |
| `process_tree_tp1.json` / `process_tree_tp2.json` | Process tree snapshots before/during/after each launch |
| `rank_worker_inventory.json` | The tracked descendant PIDs during TP=2 (API server, EngineCore, 2× Worker) |
| `logs/tp1_server.log` (20 KB) / `logs/tp2_server.log` (48 KB) | Complete raw server stdout/stderr — the primary source for the NCCL evidence below |

## 6. GPU assignment and NCCL evidence (the core D4B proof)

| File | What it proves |
|---|---|
| `rank_gpu_mapping.json` | Direct `nvidia-smi --query-compute-apps` cross-reference: `VLLM::Worker_TP0`→GPU 0, `VLLM::Worker_TP1`→GPU 1, two distinct UUIDs, zero duplicate assignment |
| `nccl_initialization.json` | Regex-extracted NCCL log evidence: `NCCL version 2.28.9+cuda13.0`, `world_size=2`, `rank=0`/`rank=1`, `backend=nccl`, distinct `cudaDev` per rank, `ncclCommInitRank ... Init COMPLETE` for both ranks |

## 7. Model/tokenizer identity

| File | What it proves |
|---|---|
| `model_tokenizer_identity.json` | Config hash, architecture, and weight-file manifest confirming TP1 and TP2 used the identical model checkpoint |

## 8. Correctness evidence

| File | What it proves |
|---|---|
| `correctness_prompt_corpus.json` | The 10 deterministic prompts (with hashes) used for every comparison below |
| `tp1_outputs.jsonl` / `tp2_outputs.jsonl` | Raw per-prompt HTTP responses (status, latency, full OpenAI-compatible JSON body) from each server |
| `token_comparison.json` | Token-ID-level TP1-vs-TP2 comparison (derived via reverse-vocabulary lookup + round-trip check, not re-tokenized text) — 100% match |
| `text_comparison.json` | Generated-text and finish-reason comparison — 100% match |
| `logprob_comparison.json` | Selected-token and top-5 logprob comparison — 100% selected-token match, 99.7% top-5 agreement |
| `repeated_request_validation.json` | 20× same-prompt repetition against TP2 — output stability (no distributed state drift) |
| `mixed_shape_validation.json` | Interleaved short/long/code/numeric prompts against TP2 — no cross-contamination |
| `concurrency_correctness.json` | Bounded concurrency (2, 4 simultaneous requests) against TP2 — all succeed, no deadlock |

## 9. Negative tests and safety

| File | What it proves |
|---|---|
| `negative_tests.json` | Result summary + stdout tail for all 19 fail-closed negative tests (16 required + 3 additional) |
| `oom_safety_validation.json` | Result for the intentionally-unsafe-memory-configuration test — confirms fail-closed, bounded failure, never a host-wide OOM |

## 10. Provenance, regressions, and measurements

| File | What it proves |
|---|---|
| `cross_layer_provenance.json` | All 23 required provenance counters, computed from real events — all zero |
| `regression_summary.json` | D1/D2/D3A/D3B/D4A regression suite results on this host, with an explicit, itemized explanation of the 4 environment-dependent (not regressed) D3B/D4A test outcomes |
| `performance_measurements.json` | Bring-up/correctness timing diagnostics only (explicitly labeled `no_speedup_claim: true`) |

## 11. Cleanup verification

| File | What it proves |
|---|---|
| `process_cleanup.json` | Zero orphan descendant processes after both TP1 and TP2 shutdown |
| `port_cleanup.json` | No stale port listeners after shutdown |
| `gpu_memory_cleanup.json` | GPU memory returned to idle baseline within a bounded poll window (not a single immediate snapshot) |

## 12. Summary and truth boundary

| File | What it proves |
|---|---|
| `test_summary.json` | One-page pass/fail rollup of every check above |
| `truth_boundary.json` | The explicit claim / not-claim statement for this stage |

---

This evidence was produced by `scripts/run_distributed_d4b_pipeline.py` (and the
lightweight `run_distributed_d4b_pipeline_part2.py` completion step) run against a
real 2-GPU host, materializing the D3B launch spec and linking the D4A whole-model
evidence artifact — see
[`docs/DISTRIBUTED_D4B_ACHIEVEMENT_REPORT.md`](../../../docs/DISTRIBUTED_D4B_ACHIEVEMENT_REPORT.md)
for the full narrative.
