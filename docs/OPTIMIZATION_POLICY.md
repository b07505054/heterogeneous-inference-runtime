# Optimization Policy

This project uses optimization policies to select among already measured
deployment options. A policy artifact is not a benchmark result and does not
create new runtime features.

## CoreML Edge Policy

`CoreMLEdgePolicy` is implemented in `deployment/coreml_edge_policy.py` and
generated with `scripts/generate_coreml_edge_policy.py`.

The policy reads CoreML `artifact_type: "measured_baseline"` JSON files and
selects:

- `input_size`
- `compression`
- `compute_unit`

Selection is constrained by measured:

- p95 steady-state latency
- package size
- RSS delta
- numerical drift

The policy rejects simulated artifacts, non-CoreML artifacts, missing metrics,
and candidates that exceed the requested latency, package-size, or drift
constraints. When a `CapabilityProfile` is provided, a baseline must also be
present in measured support with measured evidence and `status: "ok"`.

Example:

```bash
python scripts/generate_coreml_edge_policy.py \
  --baselines results/measured_baselines/coreml_cv_fp16_all.json \
              results/measured_baselines/coreml_cv_fp16_cpu.json \
  --max-p95-ms 5.0 \
  --max-package-mb 10.0 \
  --max-drift 0.01 \
  --prefer latency \
  --output results/policies/coreml_edge_policy.json
```

## Truth Boundary

CoreML already provides compute units, model packaging, and compression support.
This repository does not reimplement CoreML kernels, CoreML compression, ANE
scheduling, or custom CoreML optimization.

The value-add is measured deployment policy selection: given existing measured
CoreML artifacts and optional capability metadata, choose the deployment option
that best satisfies explicit constraints.
