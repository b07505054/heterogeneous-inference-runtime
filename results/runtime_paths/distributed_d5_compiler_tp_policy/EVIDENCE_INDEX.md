# D5 Evidence Index

Raw artifacts for the D5 compiler TP1/TP2 policy milestone. 0.5B-model
artifacts (Qwen2.5-0.5B-Instruct, the primary/full-fidelity chain reusing
the exact D2/D3B/D4A evidence unmodified) live at the top of this
directory; the model-size-axis expansion to Qwen2.5-7B-Instruct lives
under `7b/`. Start with
[`../../../docs/DISTRIBUTED_D5_COMPILER_TP_POLICY_REPORT.md`](../../../docs/DISTRIBUTED_D5_COMPILER_TP_POLICY_REPORT.md)
for the narrative; this index is for readers who want a specific artifact.

## 1. Declared workload matrix and calibration/held-out split (before any measurement)

| File | What it proves |
|---|---|
| `calibration_holdout_split.json` | 0.5B: 36-cell workload grid, declared split rule (`sha256(workload_id)[0] % 2`), split result, weighting scheme — all written before the first benchmark ran |
| `7b/calibration_holdout_split_7b.json` | 7B: 12-cell representative grid, same declared split rule |

## 2. Real benchmark measurements

| File | What it proves |
|---|---|
| `tp1_sweep_full.json` / `tp2_sweep_full.json` | 0.5B: full launch spec, preflight, and per-workload-cell streaming-latency measurements (36 cells × 5 reps) |
| `7b/tp1_sweep_full_7b.json` / `7b/tp2_sweep_full_7b.json` | 7B: same, 12 cells × 10 reps |
| `logs/d5_tp{1,2}_server.log` / `7b/logs/d5_7b_tp{1,2}_server.log` | Raw server stdout/stderr for every calibration-sweep launch |
| `source_artifact_hashes.json` | SHA-256 of the D2 ExecutionPlan and D4A evidence artifacts consumed, confirming the unmodified upstream chain |

## 3. 7B legal-operating-range probe

| File | What it proves |
|---|---|
| `7b/legal_range_probe_results.json` | All 6 probed configs (max_model_len 2048→32768, max_num_seqs 4→16) × both TP degrees: startup success, peak GPU memory |
| `7b/legal_range_finding_summary.md` | Honest finding: no startup-level memory-capacity crossover exists in this range |
| `7b/logs/probe_tp{1,2}_*.log` | Raw server logs for all 12 probe launches |

## 4. Correctness (preserved under every policy)

| File | What it proves |
|---|---|
| `7b/tp1_correctness_outputs_7b.json` / `7b/tp2_correctness_outputs_7b.json` | Raw per-prompt HTTP responses from each 7B server |
| `7b/correctness_comparison_7b.json` | 7B TP1-vs-TP2 text/logprob comparison — 10/10 text match |
| (0.5B correctness reuses the D4B corpus/comparison unchanged — see `distributed_d4b_real_2gpu_vllm_tp2/`) | |

## 5. Cost model and held-out validation

| File | What it proves |
|---|---|
| `cost_model_fitted.json` | Frozen regression coefficients (both TP degrees), fit on 54 calibration rows only |
| `held_out_evaluation.json` | Per-cell compiler decision, decision reason, oracle choice, and regret for all 21 held-out cells (both models) |

## 6. Cleanup and GPU state

| File | What it proves |
|---|---|
| `gpu_inventory_before_sweep.json` / `gpu_inventory_after_sweep.json` | 0.5B: idle GPU state before/after the full sweep |
| `7b/gpu_inventory_before_sweep_7b.json` / `7b/gpu_inventory_after_sweep_7b.json` | 7B: same |
| Per-`tp{1,2}_sweep_full*.json`'s embedded `gpu_cleanup` field | Zero-orphan, memory-returned-to-baseline confirmation for every one of the 10 real server launches in this stage |

## 7. Manifests

| File | What it proves |
|---|---|
| `d5_calibration_sweep_manifest.json` | 0.5B sweep summary: startup latencies, cleanup results |
| `7b/d5_7b_calibration_sweep_manifest.json` | 7B sweep summary: startup latencies, cleanup results, correctness match count |

---

Code: `deployment/vllm_adapter/tp_workload_matrix.py`, `tp_benchmark_harness.py`,
`tp_cost_model.py`; orchestration: `scripts/run_d5_calibration_sweep.py`,
`scripts/run_d5_7b_legal_range_probe.py`, `scripts/run_d5_7b_calibration_sweep.py`,
`scripts/run_d5_fit_and_evaluate_cost_model.py`,
`scripts/verify_7b_download_integrity.py`; tests: `tests/test_distributed_d5_compiler_tp_policy.py`.
