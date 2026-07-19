# D5 Achievement Summary

A short, portable summary of the D5 milestone for resumes, portfolios, and
interviews. For full technical detail see
[`DISTRIBUTED_D5_COMPILER_TP_POLICY_REPORT.md`](DISTRIBUTED_D5_COMPILER_TP_POLICY_REPORT.md).

## GitHub project description (one line)

> A compiler cost model that picks TP=1 vs TP=2 from pre-execution features alone matches the offline oracle on 100% of held-out real-hardware benchmarks, beating both fixed policies by 5.7–7.3% mean regret across two real models on 2×RTX 4090.

## Elevator pitch (30 seconds)

I built a compiler policy that decides, before running a single token,
whether a model should be served on one GPU or split across two — using
only information available in advance: model weight size, KV-cache cost
per token, and workload shape. I measured real TP1 and TP2 throughput on
a real 2-GPU vLLM host across two model sizes (0.5B and 7B parameters),
fit the policy on half the data, and tested it on the other half. It
picked the right answer in every single held-out case — including a
complete reversal in which direction is even winning between the two
model sizes — and beat both "always use TP1" and "always use TP2" by a
clear, statistically separated margin.

## Resume bullets

- Designed and measured a real TP1-vs-TP2 crossover on a rented 2×RTX 4090
  host (PCIe-only, no NVLink): a compiler policy fit on calibration data
  alone matched an offline oracle on 100% of 21 held-out real-hardware
  benchmark cells spanning two model sizes.
- Quantified the achievement in fixed-policy regret terms: the
  compiler-guided policy achieved 0.000% mean regret vs. the oracle,
  compared to 7.33% for a fixed always-TP1 policy and 5.67% for a fixed
  always-TP2 policy.
- Found and explained the real mechanism behind the crossover: at 0.5B
  parameters, fixed NCCL communication overhead dominates and TP1 wins
  35/36 measured workload cells; at 7B parameters, per-GPU compute time
  dominates and TP2 wins 12/12 cells with 60–70% higher throughput.
- Built a real streaming-latency benchmark harness (TTFT/TPOT/throughput)
  against a live vLLM OpenAI-compatible server, with warmup-then-measured
  repetitions and a pre-declared, hash-based calibration/held-out split
  fixed before any measurement was taken.
- Ran a 6-configuration legal-operating-range probe up to a 7B model's
  real 32,768-token context ceiling and reported, honestly, that no
  memory-capacity-forced crossover exists on this hardware — direct
  evidence that vLLM's paged-attention KV-cache manager adapts rather
  than hard-fails under memory pressure.
- Preserved every D1–D4B guarantee unchanged: correctness (10/10 TP1/TP2
  output match at both model sizes) and full process/GPU-memory cleanup
  verified after all 10 real server launches in this stage.

## Interview talking points

- **The result flips direction, and that's the point.** The same
  regression, trained on both models without ever being told "small
  model → TP1, big model → TP2," discovered opposite answers for the two
  model sizes purely from weight-footprint and KV-cache features. That's
  a stronger result than either fixed policy or a naive "TP2 is always
  better" assumption.
- **Rigor about statistical significance, not just point estimates.** I
  didn't stop at "TP1 wins 35/36 cells" — I checked which of those wins
  survive a 2-sigma separation test against real per-request variance,
  and reported the 13 cells that don't (concentrated at a specific
  concurrency level where a real queuing effect from the fixed
  `max_num_seqs` cap adds noise) as directionally consistent but not
  statistically clean, rather than rounding up.
- **A negative result taken seriously first.** Before finding the 7B
  crossover, I ran and fully reported a workload-shape-only sweep on the
  0.5B model that found *no* profitable TP2 region at all — I didn't
  discard that result once the 7B experiment succeeded; both are in the
  final evidence chain.
- **Infrastructure resilience under real failure.** The rented host's
  SSH connection dropped three separate times mid-run; every long
  benchmark was structured to survive the disconnect independently (via
  detached processes writing to remote log files, verified against
  process state and GPU memory on reconnect) so no measurement had to be
  discarded or re-run from a corrupted state.

## Key metrics

| Metric | Value |
|---|---|
| Models measured | Qwen2.5-0.5B-Instruct, Qwen2.5-7B-Instruct |
| Hardware | 2× real RTX 4090, PCIe-only (no NVLink) |
| Held-out cells (compiler vs. oracle match rate) | 21/21 (100%) |
| Mean regret — compiler-guided policy | 0.000% |
| Mean regret — always-TP1 fixed policy | 7.330% |
| Mean regret — always-TP2 fixed policy | 5.673% |
| 0.5B: TP1 wins (point estimate / statistically clean) | 35/36 / 23/36 |
| 7B: TP2 wins (point estimate / statistically clean) | 12/12 / 12/12 |
| 7B throughput improvement (TP2 vs TP1) | 60–70% |
| Correctness (TP1 vs TP2 text match), both models | 10/10 |
| Orphan processes after cleanup, all 10 launches | 0 |
