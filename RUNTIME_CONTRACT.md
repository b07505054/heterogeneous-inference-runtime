# Runtime Contract

Last verified: 2026-07-14.

Runtime source/evidence HEAD during the Slice 3G documentation refresh:
`0989181d547cee57c7fd241242c53ecd60b3e9a2`; uncommitted Slice 1-3G work is
intentionally preserved.
Canonical architecture host: `../ml-graph-compiler-runtime`.

The Runtime validates and executes exact compiler contracts. The production
serialized contract is `ExecutionPlan`. Slice 3G extends that contract to
identity-validated portable CPU and ExecuTorch/XNNPACK complete candidates.

## Owns

- ExecutionPlan or evaluation-contract parsing and validation.
- Artifact resolution.
- Exact backend/kernel/runtime dispatch requested by the compiler contract.
- Memory/resource execution.
- Execution provenance, output artifacts, timing samples, and telemetry.
- Explicit failure and compiler-authorized fallback.

## Does Not Own

- Global implementation selection.
- Candidate generation.
- Compiler legality analysis.
- Online benchmarking for implementation selection.
- Silent backend, kernel, precision, layout, artifact, or thread-schedule substitution.
- Runtime repacking, recalibration, re-quantization, or backend/precision re-selection.

## Canonical Examples

- Portable CPU adapter: validates compiler-selected CPU kernel ID, target profile, tensor contract, and thread schedule before launching the native portable C++ kernel.
- E3 XNNPACK adapter: validates runner hash, `.pte` hash, ExecuTorch/XNNPACK provenance, requested threads, backend, and runner self-report before accepting timing.
- Slice 3 custom runner: validates ordered quantize/load/execute/return stages,
  calibration, packed-weight and binary identities, then loads packed weights
  once and executes without runtime repacking or redecision.
- Slice 3G XNNPACK router: validates the exact candidate, runner, `.pte`,
  workload, delegation proof, target, precision, and fixed thread count. It
  fails explicitly instead of silently selecting the portable backend.
- vLLM materialization: materializes compiler-plan-derived serving configuration, including AWQ when the plan selects that deployment path. This is an executable parallel path, not proof of canonical quantization policy.

## Evaluation Boundary

Benchmark scripts and comparison harnesses are evidence mechanisms. Slice 3F's
primary boundary compares an already-loaded canonical custom ExecutionPlan
invocation with already-loaded ExecuTorch `Method::execute`, including input
binding, execution, and output availability. Slice 3G reuses the five-session
evidence only when every identity matches, then performs a smaller routing and
latency sanity revalidation. E2.1 did not meet this standard; E3 and Slice
3F/3G do within their documented scopes.
