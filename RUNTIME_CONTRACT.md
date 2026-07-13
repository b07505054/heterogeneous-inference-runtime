# Runtime Contract

Last verified: 2026-07-13\nSource host: GPU Linux /home/allen/Desktop/Project/heterogeneous-inference-runtime\nVerified runtime HEAD: f4cc98bc93e1e8e5ecea32ffb0779b0a5c801097 (main, ahead 1 of origin/main)\nCanonical architecture host: /home/allen/Desktop/Project/ml-graph-compiler-runtime\n

The Runtime validates and executes exact compiler contracts. The current serialized contract is `ExecutionPlan`; documentation may call the broader concept an Execution Contract, but this phase does not rename production schemas.

## Owns

- ExecutionPlan parsing and validation.
- Artifact resolution.
- Exact backend/kernel/runtime dispatch.
- Memory/resource execution.
- Execution provenance and telemetry.
- Explicit failure and compiler-authorized fallback.

## Does Not Own

- Global implementation selection.
- Candidate generation.
- Compiler legality analysis.
- Online benchmarking for selection.
- Silent backend, kernel, precision, layout, artifact, or thread-schedule substitution.

## Current Strict Examples

- `deployment/execution_plan/portable_cpu_kernel_adapter.py`: validates compiler-selected CPU kernel ID, target profile, tensor contract, and thread schedule before launching the native portable C++ kernel.
- vLLM materialization path: materializes compiler-plan-derived serving configuration, including AWQ quantization when the plan selects that deployment path.

## Boundary With Evaluation Paths

Benchmark scripts, comparison backends, and simulated distributed runtime artifacts are evidence/evaluation mechanisms unless an ExecutionPlan explicitly selects them as runtime contracts. They should not be documented as compiler-owned implementation policy.
