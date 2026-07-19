# D4B Achievement Summary

A short, portable summary of the D4B milestone for resumes, portfolios, and
interviews. For full technical detail see
[`DISTRIBUTED_D4B_ACHIEVEMENT_REPORT.md`](DISTRIBUTED_D4B_ACHIEVEMENT_REPORT.md).

## GitHub project description (one line)

> Compiler-guided vLLM tensor-parallel serving: a real Qwen2.5-0.5B-Instruct TP=2 strategy, selected by a custom compiler pass, executed end-to-end on real 2-GPU hardware with verified NCCL initialization and TP1-equivalent correctness.

## Elevator pitch (30 seconds)

I built a six-stage evidence chain that takes a tensor-parallel strategy
selected by a custom MLIR-based compiler pass and carries it, without any
manual authoring at the final step, all the way to a real vLLM server
running on two physical GPUs with real NCCL communication — then proved
correctness by comparing its output token-for-token against a single-GPU
reference. Every stage is backed by artifact evidence: hashes, process
inspection, live GPU queries, and computed provenance counters, not
assertions. The last stage (D4B) is the first to touch real multi-GPU
hardware, and I designed it to fail closed at every step rather than
silently paper over a gap — for example, on a single-GPU machine the exact
same code correctly refuses to launch TP=2, and on the real 2-GPU host it
correctly launches it.

## Resume bullets

- Designed and implemented a 6-stage compiler-to-runtime distributed
  serving pipeline (custom MLIR compiler pass → typed vLLM launch-spec
  materializer → real 2-GPU vLLM execution), closing the gap from static
  tensor-parallel planning to verified real-hardware execution.
- Built a version-aware vLLM launch-spec materializer with fail-closed
  preflight validation (267-argument live CLI registry cross-check,
  hardware/model/rank-placement checks) that correctly rejects an
  unsatisfiable TP=2 request on single-GPU hardware and correctly permits
  it on verified 2-GPU hardware, with zero silent downgrades.
- Executed and validated a real vLLM 0.24.0 TP=2 server on two physical
  RTX 4090 GPUs (rented cloud instance), proving real NCCL communicator
  initialization via direct log evidence and real dual-GPU process
  placement via live `nvidia-smi` process-to-device queries.
- Achieved 100% token-ID, text, and finish-reason agreement between the
  TP=2 server and a same-host TP=1 reference across a 10-prompt
  deterministic correctness corpus, plus 99.7% top-5 logprob agreement.
- Wrote 19 fail-closed negative tests (startup timeout, premature exit,
  killed worker process, OOM-unsafe config, duplicate GPU assignment,
  rejected-preflight launch attempts, etc.) that caught and fixed 2 real
  bugs in the process-lifecycle controller before they reached the
  "achievement" claim.
- Computed and zeroed 23 cross-layer provenance counters spanning the
  entire chain from compiler candidate ID to post-shutdown GPU-memory
  cleanup, with fully automated, bounded process/port/GPU cleanup
  verification (zero orphaned processes, zero stale ports, GPU memory
  returned to idle).

## Interview talking points

- **Fail-closed design, proven both ways.** The same preflight code that
  correctly rejects TP=2 on a 1-GPU host (because that's the honest
  answer) correctly permits it on a verified 2-GPU host — I have artifact
  evidence of both outcomes from the identical code path, which is a
  stronger proof of correctness than only ever showing the success case.
- **What "real" actually means here, precisely.** GPU usage is proven via
  `nvidia-smi --query-compute-apps` matching live process PIDs to distinct
  GPU UUIDs — not inferred from the presence of a `--tensor-parallel-size
  2` flag. NCCL is proven via direct log parsing for
  `ncclCommInitRank ... Init COMPLETE` on two distinct `cudaDev` values —
  not inferred from the server merely becoming ready.
- **Negative testing found real bugs.** A worker-kill test and an
  OOM-safety test each caught a genuine defect in the process-lifecycle
  controller (a terminal failure state being silently overwritten during
  cleanup) — concrete evidence that the negative-test suite wasn't just
  padding, it did its job.
- **Honest about what's not proven.** The project explicitly does not
  claim a speedup, does not claim vLLM ran the compiler's per-operator
  work items individually (a subtly different and unproven claim from
  "vLLM correctly executed the compiler's chosen strategy"), and
  documents this distinction directly in the truth-boundary artifact
  rather than leaving it implicit.
- **Recovered gracefully from a real infrastructure failure.** A rented
  cloud instance's network dropped mid-run; because every stage writes
  its artifacts incrementally and the process-cleanup logic is
  independent of the SSH session, I could verify on reconnect that
  nothing was left running and complete the remaining (non-GPU)
  bookkeeping from already-written evidence rather than re-running
  expensive GPU steps from scratch.

## Key metrics

| Metric | Value |
|---|---|
| GPUs used | 2× physical RTX 4090 (distinct UUIDs, distinct PCI buses) |
| vLLM version | 0.24.0 (pinned, verified, no silent upgrade) |
| Token/text agreement, TP1 vs TP2 | 100% |
| Top-5 logprob agreement | 99.7% |
| Cross-layer provenance counters computed | 23, all zero |
| Negative tests | 19, all passing (caught 2 real bugs) |
| Orphan processes after cleanup | 0 |
| Compiler code changes required | 0 (compiler repo untouched across all 6 stages) |
