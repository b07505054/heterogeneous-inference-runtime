# Deployment Planner

Deployment Planner v1 is the architecture layer above optimization policies.

```text
Measured Baselines
        |
        v
Capability Profiles
        |
        v
Optimization Policies
        |
        v
Deployment Planner
        |
        v
Deployment Plan
```

The planner does not optimize models, modify backends, run benchmarks, or
implement runtime scheduling. It selects a deployment configuration from
existing measured evidence and explicit capability facts.

## Responsibilities

- Load capability profiles for hardware, backends, kernels, and measured
  support.
- Load measured baseline or policy artifacts.
- Normalize runtime-specific artifacts into generic deployment candidates.
- Apply reusable deployment constraints.
- Rank eligible candidates with an objective.
- Emit `deployment_plan` artifacts with a truth boundary.

## Constraints

`constraint_solver.py` is runtime-neutral. It supports:

- maximum latency
- maximum package size
- maximum memory
- maximum numerical drift
- minimum throughput

## Objectives

`objective.py` supports:

- `latency`
- `throughput`
- `memory`
- `package_size`
- `balanced`

Balanced scoring uses normalized weighted score:

- latency: 0.4
- throughput: 0.3
- memory: 0.2
- package size: 0.1

Lower score is better. Future versions may add Pareto optimization.

## CLI

```bash
python deployment/planner/planner_cli.py \
  --profiles capabilities/profiles/ \
  --artifacts results/ \
  --runtime coreml \
  --objective latency \
  --max-p95-ms 5 \
  --output results/deployment/coreml_plan.json
```

```bash
python deployment/planner/planner_cli.py \
  --artifacts results/ \
  --runtime server \
  --objective throughput \
  --min-tokens-per-second 50 \
  --output results/deployment/server_plan.json
```

## Truth Boundary

Deployment plans are recommendations only. They do not claim runtime
optimization, model optimization, custom kernels, scheduler modification, or
backend implementation work.
