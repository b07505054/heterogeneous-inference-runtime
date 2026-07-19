# D6 Achievement Summary

A short, portable summary of the D6 milestone. For full technical detail
see [`DISTRIBUTED_D6_COMPILER_OWNED_TP_SELECTION.md`](DISTRIBUTED_D6_COMPILER_OWNED_TP_SELECTION.md).

## GitHub project description (one line)

> A real C++/MLIR compiler pass now computes calibrated whole-model throughput predictions for TP1 and TP2 from pre-execution features, and its own comparison — not an opt-in flag, not a Python runtime script — selects the profitable tensor-parallel degree, reproducing a real measured 2-GPU result across 21 held-out workloads with 100% oracle agreement.

## Elevator pitch (30 seconds)

An earlier architecture audit found a real gap: my compiler computed real
cost estimates for both single-GPU and 2-GPU execution, but never actually
compared them — the decision was gated by a legality check and a boolean
flag, and the "does the workload actually run faster" question was
answered by a separate Python script, not by the compiler. I closed that
gap by adding a real profitability contract to the C++ pass itself:
calibrated from genuine, measured throughput data, comparing predicted
tokens/second for both strategies, and picking the winner inside the
compiler process. I proved it works by having the compiler freshly
compile 21 held-out workloads it had never seen before and checking its
decisions against real, previously-measured hardware results — 100%
agreement — then launched two of those exact compiler-produced plans on
real dual-GPU hardware to confirm the decision survives all the way to
real NCCL initialization and real inference.

## Resume bullets

- Closed a self-identified architecture gap: moved a tensor-parallelism
  profitability decision from an ungoverned boolean flag into a real,
  versioned cost-comparison contract inside a C++/MLIR compiler pass,
  replacing "legal AND opt-in ⇒ always TP2" with a genuine predicted-
  throughput comparison between calibrated alternatives.
- Built a real calibration pipeline: an offline script fits linear
  regression coefficients from real, previously-measured GPU throughput
  data and embeds them, with cryptographic dataset-provenance hashes and
  fail-closed version checking, into a versioned compiler target profile
  — the compiler never reads a raw benchmark file directly.
- Verified the new C++ cost model bit-for-bit against its Python
  reference implementation (exact double-precision match across multiple
  real test cases), then reproduced a real measured result via 21
  independent, fresh compiler invocations — not a runtime choice between
  precompiled artifacts — achieving 100% agreement with the real,
  previously-measured hardware oracle and 0% regret.
- Verified the compiler's decision survives unmodified through the
  entire existing execution stack (unchanged launch materializer, unchanged
  process controller) to real dual-GPU hardware: confirmed real NCCL
  initialization for a compiler-selected 2-GPU plan and its correct
  absence for a compiler-selected single-GPU plan, both with full
  correctness and complete resource cleanup.
- Wrote 15 new automated tests (10 C++ compiler unit tests, 5 Python
  integration tests) proving the old runtime-side selection script is
  fully absent from the production decision path, including an explicit
  test that materializes real compiler output with the old script's
  module never even imported.

## Interview talking points

- **I audited my own work and found a real gap, then fixed it.** The
  previous milestone's "compiler cost model" language was, on close
  reading, doing more work than the code actually did — I didn't wait for
  someone else to find that; I traced it myself, named it precisely, and
  then did the harder engineering work of actually closing it rather than
  just softening the claim.
- **Numerical verification, not just code review.** I didn't just port a
  Python formula to C++ and assume it was right — I ran both
  implementations against identical inputs and diffed the output to
  1e-6 precision, catching a real serialization bug (a missing
  floating-point case in the evidence exporter) before trusting any
  decision it reported.
- **"Fresh compilation" is a real, load-bearing distinction.** Reusing two
  precompiled plans and picking one at runtime would have been much
  cheaper to build and would have looked identical in a demo — I
  specifically verified 21 *separate compiler invocations*, each with its
  own real workload input, to make sure the decision genuinely happens at
  compile time from real features, not via a cache of two answers.
- **Honest about the model's real limits.** The linear regression
  extrapolates to physically meaningless negative numbers outside its
  calibrated range — I found this, reported it, and explained precisely
  why it doesn't undermine the result (the ranking between alternatives
  stays correct even when the absolute number doesn't).

## Key metrics

| Metric | Value |
|---|---|
| Held-out cells reproduced via fresh compilation | 21 (17 × 0.5B, 4 × 7B) |
| Compiler/oracle agreement | 100% (21/21) |
| Mean/P95/worst-case compiler regret | 0.000% / 0.000% / 0.000% |
| Mean regret, always-TP1 fixed policy | 7.330% |
| Mean regret, always-TP2 fixed policy | 5.673% |
| C++/Python numerical cross-check | exact match to 1e-6 (multiple cells) |
| Compiler test suite | 31/31 passing (10 new) |
| Runtime integration tests | 5/5 passing (new) |
| Real 2-GPU launches verified | 2 (compiler-selected TP1 and TP2) |
| Correctness on real hardware | 20/20 prompts (100%), both launches |
| Orphan processes after cleanup | 0 (both real launches) |
