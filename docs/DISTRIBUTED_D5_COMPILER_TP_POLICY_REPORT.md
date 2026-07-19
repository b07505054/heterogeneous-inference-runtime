# D5: Compiler-Guided TP1/TP2 Policy — Real Measured Optimization

## 0. Achievement statement (measured numbers only)

On the same real 2×RTX 4090 host used for D4B, across two real models
(Qwen2.5-0.5B-Instruct and Qwen2.5-7B-Instruct) served with real vLLM
0.24.0 and real NCCL TP=2 execution, a compiler cost model — fit only on
calibration-split measurements, using only pre-execution features (model
weight footprint, KV-cache bytes/token, GPU count, and workload shape) —
selects the tensor-parallel degree (TP=1 or TP=2) for each of 21 held-out
workload cells. Compared against the offline oracle (the TP degree that
actually measured higher throughput for that cell):

| Policy | Mean regret vs. oracle (held-out, 21 cells) |
|---|---:|
| **Compiler-guided (this work)** | **0.000%** (matches oracle in 21/21 cells) |
| Always TP1 (fixed) | 7.330% |
| Always TP2 (fixed) | 5.673% |

This is a real, measured compiler optimization: the compiler-guided policy
strictly dominates both fixed policies on data it never saw during
fitting. No claim beyond this statement is made in this document without
being immediately qualified (see §8, Truth boundary).

## 1. Objective and scope

D4B proved a compiler-selected TP=2 strategy could be executed correctly
on real 2-GPU hardware. It made no performance claim. D5's objective was
narrower and harder: prove that a compiler-selected policy — choosing
between TP=1 and TP=2 using only information available before execution —
is *measurably better* than either fixed policy, on real hardware, using
held-out data the policy was never fit against.

D5 preserves the entire D1→D2→D3A→D3B→D4A→D4B evidence chain unmodified.
The existing Qwen2.5-0.5B-Instruct D2 `ExecutionPlan` artifacts
(`real_qwen_tp1_execution_plan.json` / `real_qwen_tp2_execution_plan.json`)
and the D3B materializer are reused exactly as committed. The only code
change to that chain is an additive parameterization
(`distributed_materializer.KNOWN_MODEL_ID_MAP`) that lets the same,
otherwise-unmodified materializer also resolve a second real model — this
was necessary to search the model-size axis and is verified to leave the
0.5B path byte-identical (65/65 pre-existing D2/D3B/D4A tests still pass).

## 2. Method

### 2.1 Workload matrix and the calibration/held-out split

Two workload grids were declared **before any measurement was taken**:

- **0.5B grid**: 3 input lengths × 3 output lengths × 4 concurrency levels
  = 36 cells, 2 warmup + 5 measured repetitions per cell.
- **7B grid**: a smaller, declared-upfront representative subset (2 input
  lengths × 2 output lengths × 3 concurrency levels = 12 cells) so that 10
  measured repetitions per cell were affordable within the same
  real-hardware time budget.

The calibration/held-out partition is a deterministic function of
workload identity alone — `held_out iff sha256(workload_id)[0] % 2 == 0`
— computed and written to disk (`calibration_holdout_split.json`,
`7b/calibration_holdout_split_7b.json`) before the first benchmark ran,
and never recomputed or adjusted afterward. The weighting scheme is
uniform across all cells, declared for the same reason: no production
traffic trace was available to justify a non-uniform mix, and inventing
one post-hoc would risk fitting the weighting to a desired conclusion.

Split sizes: 0.5B — 19 calibration / 17 held-out (of 36). 7B — 8
calibration / 4 held-out (of 12). Combined: 54 calibration rows / 42
held-out rows (21 held-out cells × 2 TP degrees).

### 2.2 Benchmark harness

A new streaming-request harness (`tp_benchmark_harness.py`) measures TTFT,
TPOT, end-to-end latency, and aggregate throughput against a real,
already-running vLLM server via the real streaming `/v1/completions`
endpoint (`stream=True`), timing actual first-chunk arrival — never a
non-streaming approximation. Concurrency is realized by firing N real
simultaneous HTTP requests through a thread pool, never simulated.
Warmup requests are discarded before measured repetitions begin.

Every server launch reuses the exact D4B `ServerLaunchController` /
`materialize_launch_spec` chain unmodified: the same fail-closed preflight,
the same bounded process lifecycle, the same SIGTERM→SIGKILL escalation
and descendant-process sweep.

### 2.3 Legal operating range probe (7B)

Before the performance sweep, 6 configurations spanning
`max_model_len` from 2048 to 32768 (the model's real
`max_position_embeddings` ceiling) and `max_num_seqs` from 4 to 16 were
probed for both TP1 and TP2 (`run_d5_7b_legal_range_probe.py`). **Every
configuration started successfully on both TP degrees** — no
startup-level memory-capacity boundary exists in this range. Peak GPU-0
memory during TP1 actually *decreased* slightly as `max_model_len` grew
(21096 MiB at 2048 → 18060 MiB at 32768/16), which is the direct evidence
that vLLM's paged-attention KV-cache manager adaptively sizes its block
pool to whatever memory is available after weight loading, rather than
pre-reserving worst-case capacity. This rules out a *startup-legality*
capacity crossover on this hardware/model combination; it does not rule
out a runtime scheduling effect (preemption under sustained concurrent
load), which was not separately isolated. See
`7b/legal_range_finding_summary.md` and `7b/legal_range_probe_results.json`.

### 2.4 Cost model

Two linear regressions (`tp_cost_model.py`), one per TP degree, predict
aggregate throughput from six pre-execution features: per-GPU weight
footprint (MB), KV-cache bytes/token/GPU, GPU count, input length, output
length, and concurrency. Weight footprint and KV-cache-per-token are
computed from each model's real config (`hidden_size`,
`num_attention_heads`, `num_kv_heads`, `num_layers` — the same values used
to build each model's D2 `ExecutionPlan`) and each model's real measured
or Hub-reported checkpoint size. Fit via ordinary least squares on the 54
calibration rows only, then frozen — no further adjustment after seeing
held-out data.

A separate, prior hard-constraint layer (`is_feasible`) checks whether a
TP degree can legally hold its weight shard plus a worst-case
`max_num_seqs × max_model_len` KV cache within the real GPU memory budget;
if only one TP degree is feasible, that one is chosen without consulting
the regression. In every actual calibration/held-out cell measured, both
degrees were feasible, so every real decision reported below came from
the performance regression, not the feasibility shortcut — this is stated
explicitly per-row in `held_out_evaluation.json`'s `decision_reason` field.

Fit quality: TP1 regression R²=0.854 (n=27), TP2 regression R²=0.831
(n=27).

## 3. Results

### 3.1 Qwen2.5-0.5B-Instruct: TP1 wins

TP1 is faster on point estimate in **35/36 cells**; TP2 is faster on point
estimate in exactly 1 cell (`in256_out256_c4`, 0.5306s vs. 0.5310s — a
tie within noise, not a real effect). Applying a strict 2σ
non-overlap test (mean ± 2×stdev must not overlap between TP1 and TP2's
per-request distributions) to be honest about statistical significance:
**23/36 cells are statistically clean TP1 wins; 0/36 are statistically
clean TP2 wins; the remaining 13 are directionally consistent with TP1
but not clean at this bar.** Those 13 cluster almost entirely at
concurrency=8, where the launch spec's `max_num_seqs=4` default causes
real HTTP-level concurrency to exceed the server's actual scheduling
width — request queuing introduces substantial variance (stdev up to 35%
of the mean) that is a genuine property of this configuration, not a
benchmark artifact.

Mechanistically: at 0.5B parameters, per-GPU compute time per request is
small, so the fixed cost of an NCCL all-reduce over PCIe (no NVLink on
this host) dominates and TP2 never recoups it.

### 3.2 Qwen2.5-7B-Instruct: TP2 wins, decisively

TP2 wins **12/12 cells**, every one statistically clean by the same 2σ
test — separation margins range from 0.15s to 1.72s against per-request
stdevs of 0.001–0.016s, an order of magnitude beyond noise. Aggregate
throughput improves 60–70% (e.g. 62.6→105.9 tok/s at `in32_out32_c1`) and
mean end-to-end latency drops 36–43% (e.g. 4.15s→2.56s at
`in32_out256_c4`). Correctness held: 10/10 prompt outputs matched between
TP1 and TP2 (`7b/correctness_comparison_7b.json`).

At 7B parameters, per-GPU compute time is large enough that halving it
via tensor parallelism outweighs the same fixed NCCL overhead observed at
0.5B — the crossover is a genuine model-size effect, not noise, not a
configuration artifact, and not assumed in advance (D5's stage-3 mandate
was to find the real boundary without assuming TP=2 wins; the 0.5B result
alone would have supported the opposite conclusion).

### 3.3 Held-out validation

| Metric | Value |
|---|---:|
| Held-out cells evaluated | 21 (17 from 0.5B, 4 from 7B) |
| Compiler/oracle match rate | 100% (21/21) |
| Mean compiler regret | 0.000% |
| Mean always-TP1 regret | 7.330% |
| Mean always-TP2 regret | 5.673% |

Every compiler decision in the held-out set was made via
`decision_reason: "performance_regression"` — i.e. by the fitted
regression comparing predicted throughput, never by a hardcoded
per-model rule and never by consulting the held-out cell's own measured
outcome. The regression, trained on both models' calibration data
without ever being told "small model → TP1, large model → TP2" as a
rule, discovered this boundary purely from the weight-footprint and
KV-cache-per-token features varying between the two calibration sets.

Full per-cell detail: `held_out_evaluation.json`. Interactive dashboard
with per-cell bar charts: published artifact (see chat).

## 4. Correctness and cleanup, preserved under every policy

- 0.5B: 10/10 prompts token-ID/text match between TP1 and TP2
  (pre-existing D4B correctness corpus, reused unmodified).
- 7B: 10/10 prompts text match between TP1 and TP2
  (`7b/correctness_comparison_7b.json`).
- Every one of the 6 legal-range-probe launches + 2 calibration-sweep
  launches (0.5B) + 2 calibration-sweep launches (7B) = 10 real server
  launches ended with the `ServerLaunchController.stop()` /
  `wait_for_gpu_memory_baseline()` pair reporting zero orphan descendant
  processes and GPU memory returned to the pre-launch idle baseline
  (`within_tolerance: true` in every recorded `gpu_cleanup` result).
- Final state confirmed directly via `nvidia-smi` after the full D5 run:
  both GPUs at 1 MiB used, 0 vLLM-related processes running.

## 5. Infrastructure notes (not part of the achievement claim)

- The rented Vast.ai host's SSH connection dropped three separate times
  during D5 (once mid-download, twice mid-benchmark) — the same
  transient-network pattern observed during D4B. Each time, long-running
  remote work was structured to survive the disconnect (detached via
  `nohup` + output redirected to a remote file, immune to the local SSH
  session's lifecycle) and was verified, on reconnect, to have continued
  or completed correctly with zero orphaned GPU processes at any point.
  No benchmark step was re-run after a disconnect; every number in this
  report came from a single, uninterrupted measurement pass per cell.
- The Qwen2.5-7B-Instruct checkpoint (14.2 GB) was downloaded to the
  rented host and integrity-verified against the real Hugging Face Hub
  blob-size metadata (14/14 files, exact byte match) before any
  benchmark used it.

## 6. Truth boundary

**What this shows**: a compiler policy using only pre-execution features
and constants regressed from calibration data alone correctly selects the
throughput-maximizing TP degree on 100% of a held-out set spanning two
real model sizes on real hardware, beating both fixed policies by
5.7–7.3% mean regret.

**What this does not show**:

- A memory-capacity-forced crossover. None was found — vLLM's paged
  KV-cache manager adapts its block count rather than hard-failing at any
  tested context length up to 32768 tokens on this hardware. The
  compiler's feasibility layer exists and is exercised in code
  (`is_feasible`), but never fired as the deciding factor in any
  measured cell — every real decision came from the performance
  regression.
- Generalization to model sizes between 0.5B and 7B. Only two points on
  the size axis were measured; the crossover point itself was not
  isolated.
- Generalization beyond this exact 2×RTX 4090, PCIe-only, single-node
  host, or to >2-GPU or NVLink-connected topologies.
- A speedup claim relative to any other serving system, or a
  cost/dollar claim.
- That vLLM executed the compiler's per-operator work items (D4A)
  individually rather than its own installed TP implementation — the
  same distinction documented in D4B's truth boundary applies here
  unchanged.

## 7. Evidence index

All raw artifacts live under
`results/runtime_paths/distributed_d5_compiler_tp_policy/`:

| File | Content |
|---|---|
| `calibration_holdout_split.json` | 0.5B workload matrix + declared split (before measurement) |
| `tp1_sweep_full.json` / `tp2_sweep_full.json` | 0.5B raw per-request measurements, 36 cells |
| `7b/calibration_holdout_split_7b.json` | 7B workload matrix + declared split |
| `7b/legal_range_probe_results.json`, `7b/legal_range_finding_summary.md` | 7B legal-operating-range probe |
| `7b/tp1_sweep_full_7b.json` / `7b/tp2_sweep_full_7b.json` | 7B raw per-request measurements, 12 cells |
| `7b/correctness_comparison_7b.json` | 7B TP1-vs-TP2 correctness |
| `cost_model_fitted.json` | Frozen regression coefficients, both TP degrees |
| `held_out_evaluation.json` | Per-cell decisions, regret, oracle comparison |
| `7b/logs/`, `logs/` | Raw server stdout/stderr for every launch |

Code: `deployment/vllm_adapter/tp_workload_matrix.py`,
`tp_benchmark_harness.py`, `tp_cost_model.py`; `scripts/run_d5_*.py`.
