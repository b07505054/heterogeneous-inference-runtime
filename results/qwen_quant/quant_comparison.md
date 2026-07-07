# Qwen A/B/C Quantization Benchmark Comparison

**Plan (B, no-quant):** `nvidia-gtx1650-maxq_serving_plan`
**Plan (C, awq):**      `nvidia-gtx1650-maxq-awq-forced-experimental_serving_plan`
**Hardware profile (C):** `nvidia-gtx1650-maxq-awq-forced-experimental`
**Date:** 2026-07-07T21:35:48Z

## Paths

| Path | Weights | Execution plan | Quantization |
|---|---|---|---|
| A: baseline | original HF Qwen | none (manual vLLM config) | none |
| B: compiler no-quant | original HF Qwen | compiler ExecutionPlan | none |
| C: compiler quant | AWQ Qwen checkpoint (`artifacts/qwen_awq`) | compiler ExecutionPlan | awq (strategy=weight_only_int4) |

## Truth Boundaries

- **C's quantization decision truth_boundary:** `experimental_forced_quant_not_native_int4_support_on_gtx1650`
- **C vs B** isolates quantized weights: both use the compiler ExecutionPlan;
  only the weights and `--quantization` flag differ.
- **C vs A** combines quantized weights AND compiler execution plan policy
  (KV layout, memory budget, serving topology) -- a C-vs-A delta cannot be
  attributed to quantization alone.
- Do not claim a speedup unless a measured result below shows one beyond
  repeatability noise (see `results/qwen_no_quant/repeatability_summary.md`
  for what "noise" looked like for B vs A: ~0.5-1.1% across 3 trials).
- Do not claim accuracy parity -- no accuracy evaluation (perplexity, task
  benchmarks) was run for C.
- Do not claim GTX 1650 has native INT4 Tensor Core support. C's compiler
  plan is an explicit experimental forced-quant override
  (`nvidia_gtx1650_maxq_awq_forced.json`); per-op kernel planning for this
  target is unchanged and still reports no native int4 kernel path.

## Server Commands

**A (baseline):**
```
.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --tensor-parallel-size 1 --pipeline-parallel-size 1
```

**B (compiler no-quant):**
```
.venv/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-0.5B-Instruct --tokenizer Qwen/Qwen2.5-0.5B-Instruct --dtype float16 --max-model-len 2048 --gpu-memory-utilization 0.75 --block-size 16 --max-num-seqs 4 --max-num-batched-tokens 2048 --tensor-parallel-size 1 --pipeline-parallel-size 1 --served-model-name qwen2.5-0.5b
```

**C (compiler awq):**
```
.venv/bin/python -m vllm.entrypoints.openai.api_server --model artifacts/qwen_awq --tokenizer artifacts/qwen_awq --dtype float16 --quantization awq --max-model-len 2048 --gpu-memory-utilization 0.75 --block-size 16 --max-num-seqs 4 --max-num-batched-tokens 2048 --tensor-parallel-size 1 --pipeline-parallel-size 1 --served-model-name qwen2.5-0.5b
```

## Results

| Workload | A (baseline) | B (compiler no-quant) | C (compiler awq) |
|---|---|---|---|
| short | `baseline_short.json` | `compiler_noquant_short.json` | `compiler_awq_short.json` |
| shared_prefix | `baseline_shared_prefix.json` | `compiler_noquant_shared_prefix.json` | `compiler_awq_shared_prefix.json` |
| no_shared_prefix | `baseline_no_shared_prefix.json` | `compiler_noquant_no_shared_prefix.json` | `compiler_awq_no_shared_prefix.json` |

C's result files are populated only when `/Users/allen/Documents/Codex/project/systems-portfolio/heterogeneous-inference-runtime/../ml-graph-compiler-runtime/artifacts/qwen_awq` exists locally
(see `compiler_awq_status.txt`) -- otherwise C is materialized-only, and
these paths are pending measurement.

## How to Run

```bash
# Dry-run (prints commands, no server started):
DRY_RUN=1 bash scripts/run_qwen_quant_benchmark.sh

# Produce the AWQ artifact first (CUDA-capable Linux host):
(cd ../ml-graph-compiler-runtime && .venv/bin/python tools/export_qwen_awq.py)

# Full benchmark on Linux GPU host:
bash scripts/run_qwen_quant_benchmark.sh
```
