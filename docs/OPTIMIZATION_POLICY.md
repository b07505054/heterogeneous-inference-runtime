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

## Server Runtime Policy

`ServerRuntimePolicy` is implemented in `deployment/server_runtime_policy.py`
and generated with `scripts/generate_server_runtime_policy.py`.

The policy reads OpenAI-compatible server `artifact_type: "measured_baseline"`
JSON files and selects:

- `concurrency`
- `model`
- `max_model_len`, when present in the measured artifact
- `max_tokens`, when present in the measured artifact

Selection is constrained by measured:

- TTFT p95
- TPOT p95
- end-to-end latency p95
- tokens/sec
- success and error counts

The policy rejects simulated artifacts, non-OpenAI-compatible artifacts,
missing metrics, erroring candidates unless `--allow-errors` is passed, and
candidates that exceed the requested TTFT, TPOT, or end-to-end latency
constraints.

Example:

```bash
python scripts/generate_server_runtime_policy.py \
  --baselines results/measured_baselines/vllm_qwen05b_concurrency1.json \
              results/measured_baselines/vllm_qwen05b_concurrency4.json \
  --max-ttft-p95-ms 200 \
  --max-tpot-p95-ms 50 \
  --max-e2e-p95-ms 2000 \
  --prefer latency \
  --output results/policies/server_runtime_policy.json
```

vLLM already provides batching, scheduling, paged attention, and serving
internals. This repository does not reimplement vLLM, modify vLLM kernels, or
claim custom scheduler improvements.

The value-add is measured safe-concurrency and admission policy selection:
given existing OpenAI-compatible/vLLM measurements, choose a runtime setting
that satisfies explicit latency and reliability constraints.
