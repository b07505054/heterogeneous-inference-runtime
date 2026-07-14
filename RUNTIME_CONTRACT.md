# Runtime Contract

Last verified: 2026-07-14.

Runtime HEAD: `53c80e2c11101ec7b8db2e73f978e220c054d9a1`.
Canonical architecture host: `../ml-graph-compiler-runtime`.

The Runtime validates and executes exact compiler contracts. The current production serialized contract is `ExecutionPlan`; E3 also uses an additive evaluation contract named `executorch_xnnpack_runner_contract` for same-stack comparison.

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

## Canonical Examples

- Portable CPU adapter: validates compiler-selected CPU kernel ID, target profile, tensor contract, and thread schedule before launching the native portable C++ kernel.
- E3 XNNPACK adapter: validates runner hash, `.pte` hash, ExecuTorch/XNNPACK provenance, requested threads, backend, and runner self-report before accepting timing.
- vLLM materialization: materializes compiler-plan-derived serving configuration, including AWQ when the plan selects that deployment path. This is an executable parallel path, not proof of canonical quantization policy.

## Evaluation Boundary

Benchmark scripts and comparison harnesses are evidence mechanisms. A comparison claiming compiler-selected behavior must invoke the live Compiler and consume its emitted decision artifact. E2.1 did not meet that standard; E3 does.
