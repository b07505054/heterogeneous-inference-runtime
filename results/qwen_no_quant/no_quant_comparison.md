# Qwen No-Quant Benchmark Comparison

**Plan:** `nvidia-gtx1650-maxq_serving_plan`
**Model:** `qwen2.5-0.5b`
**Hardware profile:** `nvidia-gtx1650-maxq`
**Date:** 2026-07-06T17:35:12Z

## Truth Boundary

Compiler-guided no-quant Qwen uses the **original HuggingFace Qwen weights**.
Differences between paths come from **runtime policy decisions** extracted from
the compiler execution plan (KV layout, memory budget, serving topology, prefix
reuse eligibility), not from weight optimization or quantization.

The compiler plan declares:
- Quantization: **none** (no `global_decisions.quantization` emitted for this profile)
- Per-op: `fp16_fallback` for accuracy-sensitive ops (stays in fp16)
- GPU memory utilization: **0.75** (from compiler `memory_budget_fraction`)
- KV layout: **paged**

Neither path modifies model weights. Measured differences are **declared profile evidence**,
not measured silicon performance.

## Workloads

| Workload | Description |
|---|---|
| `short` | 32 requests, mixed prompts, 64 max_tokens, uniform arrival, no shared prefix |
| `shared_prefix` | 32 requests, 4 unique prompts, shared system prefix, exercises prefix cache |
| `no_shared_prefix` | 32 requests, all 6 mixed prompts, no shared prefix, no cache benefit expected |

## Server Commands

**Baseline vLLM:**
```
.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --tensor-parallel-size 1 --pipeline-parallel-size 1
```

**Compiler-guided vLLM:**
```
.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --gpu-memory-utilization 0.75 --tensor-parallel-size 1 --pipeline-parallel-size 1 --served-model-name qwen2.5-0.5b
```

## Results

Results are written to `results/qwen_no_quant/` when the benchmark runs.
Each result file is a `measured_envelope` with `evidence_type: "measured"`.

| Workload | Baseline | Compiler-guided |
|---|---|---|
| short | `baseline_short.json` | `compiler_short.json` |
| shared_prefix | `baseline_shared_prefix.json` | `compiler_shared_prefix.json` |
| no_shared_prefix | `baseline_no_shared_prefix.json` | `compiler_no_shared_prefix.json` |

Results pending measurement. Files above are populated by running this script
on a Linux host with a GTX 1650 GPU and vLLM installed.

## How to Run

```bash
# Dry-run (prints commands, no server started):
DRY_RUN=1 bash scripts/run_qwen_no_quant_benchmark.sh

# Full benchmark on Linux GPU host:
BASELINE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
COMPILER_PLAN=../ml-graph-compiler-runtime/artifacts/qwen/execution_plan.json \
bash scripts/run_qwen_no_quant_benchmark.sh
```
