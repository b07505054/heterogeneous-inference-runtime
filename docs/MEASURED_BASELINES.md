# Measured Baselines

This repo separates measured baselines from simulator and policy-ablation
artifacts.

Measured baseline JSON files must include:

```json
{
  "artifact_type": "...",
  "evidence_type": "measured",
  "benchmark_target": {},
  "hardware": {},
  "software_versions": {},
  "command": [],
  "git_commit": "...",
  "metrics": {},
  "notes": []
}
```

The measured baseline scripts live under `scripts/` and use thin helpers from
`benchmark/`.

Generated model packages and measured result artifacts are local evidence and
are ignored by git:

```text
models/coreml/
results/measured_baselines/
```

This document may summarize representative local measurements, but the source
artifacts remain local under `results/measured_baselines/` unless they are
explicitly exported elsewhere.

## OpenAI-Compatible Server

`scripts/benchmark_openai_compatible_server.py` is a client-only benchmark for
servers that expose OpenAI-compatible HTTP endpoints. It does not install,
start, stop, or manage vLLM or any other server.

Example:

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/generate_llm_request_trace.py \
  --output traces/llm_request_trace.jsonl \
  --num-requests 32 \
  --prompt-set mixed \
  --max-tokens 64 \
  --arrival-pattern burst \
  --seed 0

PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_openai_compatible_server.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --trace traces/llm_request_trace.jsonl \
  --concurrency 4 \
  --warmup 2 \
  --claimed-server vllm \
  --output results/measured_baselines/vllm_baseline.json
```

`benchmark_target.kind` remains `openai_compatible_server`. The optional
`claimed_server` field is only a user-supplied label.

Example deterministic traces are available under `traces/examples/`. These
files contain `request_id`, `prompt`, `max_tokens`, and monotonic `arrival_ms`
fields; the benchmark client currently uses the prompt and token limit.

### No-Quant Qwen Compiler-Guided vLLM Evidence

The no-quant Qwen benchmark compares a compiler-guided vLLM runtime policy
against manually conservative vLLM settings on the same GTX 1650 Max-Q 4 GB
host. Compiler-guided no-quant Qwen uses original Qwen weights. Differences
come from execution/runtime policy, not model weight optimization. Do not claim
compiler-optimized weights. Do not claim AWQ/GPTQ yet.

Default vLLM OOMs on GTX1650 Max-Q 4GB with default greedy startup/warmup. The
compiler-generated execution plan avoids default vLLM OOM by materializing a
low-memory policy: `gpu_memory_utilization=0.75`, `max_model_len=2048`,
`max_num_seqs=4`, `max_num_batched_tokens=2048`, `block_size=16`, fp16 dtype,
and single tensor/pipeline parallelism.

Compared with manually conservative vLLM config, compiler-guided no-quant
matches performance within about 1% E2E in 3-trial repeatability:

| Workload | Compiler-guided E2E delta | Interpretation |
|---|---:|---|
| `short` | +0.476% | benchmark noise |
| `shared_prefix` | +0.776% | benchmark noise |
| `no_shared_prefix` | +1.089% | benchmark noise |

Treat these deltas as benchmark noise, not speedup. The remaining command
difference is `--served-model-name qwen2.5-0.5b` on the compiler-guided path,
which changes OpenAI-compatible model routing/name matching while preserving the
same served model root: `Qwen/Qwen2.5-0.5B-Instruct`.

Artifact paths:

```text
results/qwen_no_quant/repeatability_raw.json
results/qwen_no_quant/repeatability_summary.md
results/qwen_no_quant/failed_default_baseline/
```

Per-run evidence (server commands, ready/not-ready status, `nvidia-smi`
snapshots before/after, and per-workload measured JSON) lives alongside these
files under `results/qwen_no_quant/` — e.g.
`baseline_conservative_fixed_command.txt`, `baseline_default_status.txt`
(`oom`), `compiler_guided_fixed_gpu_ready.txt`, `compiler_guided_fixed_short.json`.
`baseline_default_server.log` (default vLLM config; not committed — `*.log` is
gitignored) holds the actual `RuntimeError: Engine core initialization failed`
traceback backing the OOM claim.

**Materializer code path:** `../ml-graph-compiler-runtime/artifacts/qwen/execution_plan.json`
(compiler `ExecutionPlan`, quantization `none`/`fp16_fallback`) ->
`deployment/execution_plan/path_builder.py` (`build_execution_paths`,
`build_baseline_vllm_path`) -> `deployment/vllm_adapter/config_materializer.py`
(`materialize_vllm_cli_args_from_path`) -> concrete vLLM CLI args, invoked by
`scripts/run_qwen_no_quant_benchmark.sh`.

**Future: quantized (AWQ/GPTQ) Qwen — not implemented.** There is no AWQ/GPTQ
export tool or quantized Qwen artifact in this repo or in
`ml-graph-compiler-runtime` today. The GTX 1650 Max-Q target profile used above
declares `supportedQuantModes: ["none"]` (Turing has no native INT4 tensor
cores), so the compiler cannot yet emit a real quantization decision for this
hardware. See `ml-graph-compiler-runtime/docs/future_work.md` for the Phase C
plan (quantized export step, quant-capable target profile, materializer
`--quantization` flag, repeatability pass).

### Current vLLM Linux GPU Baseline

The first external serving baseline was collected against an already running
OpenAI-compatible vLLM server. This is a hardware-constrained GTX 1650 Max-Q
measurement, not a general statement about vLLM performance.

Environment:

- Server: external OpenAI-compatible vLLM server.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- GPU: NVIDIA GeForce GTX 1650 Max-Q, 4 GB VRAM.
- vLLM: `0.24.0`.
- PyTorch: `2.11.0+cu130`.
- Server settings: `max_model_len=512`, `gpu_memory_utilization=0.70`.

Measured summary:

| Concurrency | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Tokens/sec | Success | Error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150.928 | 12.004 | 902.447 | 71.471 | 28 | 0 |
| 4 | 322.738 | 175.184 | 11350.372 | 5.714 | 28 | 0 |

Interpretation:

- Concurrency 4 regressed heavily on this 4 GB laptop GPU.
- Do not present this as evidence that vLLM is generally slow or unsuitable.
- This result is useful as a measured, hardware-constrained baseline for
  policy experiments that avoid pathological overload on small GPUs.
- Observed server warnings included FlashAttention-2 unsupported on this GPU,
  FlashInfer sampler fallback, Triton attention fallback, and Triton JIT during
  first inference.

## Native CoreML CV

`scripts/export_coreml_mobilenetv2.py` exports a native MobileNetV2
`.mlpackage` when `coremltools`, `torch`, and `torchvision` are installed.
The default export is FP16:

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/export_coreml_mobilenetv2.py \
  --precision fp16 \
  --compression none \
  --input-size 224 \
  --output models/coreml/mobilenet_v2_fp16_224.mlpackage
```

Compressed exports are optional. Palettization uses `coremltools` optimization
APIs when they are available:

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/export_coreml_mobilenetv2.py \
  --precision fp16 \
  --compression palettize \
  --input-size 224 \
  --output models/coreml/mobilenet_v2_fp16_224_palettized.mlpackage
```

`scripts/benchmark_coreml_cv_baseline.py` compares the native CoreML package
against PyTorch CPU and, when available, PyTorch MPS. Missing optional
dependencies or missing CoreML packages are reported as `status: "partial"` or
backend-level `status: "unavailable"` rather than failing CI.

Pass `--model-precision fp16` and `--model-compression none|palettize|unknown`
so the measured artifact records which package was benchmarked. Palettization
can reduce package size and may affect latency or numerical drift. Actual
speedup depends on the model, OS, hardware, and CoreML runtime placement.

The CoreML benchmark accepts `--compute-unit`:

- `cpu`: maps to CoreML `CPU_ONLY`.
- `cpu_gpu`: maps to CoreML `CPU_AND_GPU`.
- `all`: maps to CoreML `ALL` and is the default.

`ALL` allows the CoreML runtime to use the Neural Engine when supported by the
model, OS, and hardware. The measured artifact records the requested compute
unit, but actual hardware placement is determined by CoreML. Do not claim ANE
execution unless it is separately measured or reported by runtime tooling.

To compare fixed image input sizes, export one package per shape and pass the
same `--input-size` to the benchmark:

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/export_coreml_mobilenetv2.py \
  --precision fp16 \
  --compression none \
  --input-size 224 \
  --output models/coreml/mobilenet_v2_fp16_224.mlpackage

PYTHONPATH=$PWD .venv/bin/python scripts/export_coreml_mobilenetv2.py \
  --precision fp16 \
  --compression none \
  --input-size 256 \
  --output models/coreml/mobilenet_v2_fp16_256.mlpackage

PYTHONPATH=$PWD .venv/bin/python scripts/export_coreml_mobilenetv2.py \
  --precision fp16 \
  --compression none \
  --input-size 384 \
  --output models/coreml/mobilenet_v2_fp16_384.mlpackage
```

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_coreml_cv_baseline.py \
  --coreml-model models/coreml/mobilenet_v2_fp16_224.mlpackage \
  --input-size 224 \
  --compute-unit all \
  --iterations 100 \
  --warmup 20 \
  --model-precision fp16 \
  --model-compression none \
  --output results/measured_baselines/coreml_cv_fp16_224_all.json
```

Repeat the benchmark with the matching `--coreml-model`, `--input-size`, and
output path for 256 and 384. Compare steady-state p50/p95 and cold-start
metrics with `scripts/compare_measured_baselines.py`; only metrics present in
both artifacts are compared.

### Current CoreML Mac Baseline

The current edge baseline is native CoreML MobileNetV2 exported as FP16
`.mlpackage` and compared against PyTorch CPU/MPS where available.

Coverage:

- Compute units: `CPU_ONLY`, `CPU_AND_GPU`, and `ALL`.
- Fixed input-size buckets: 224, 256, and 384.
- Compression sanity check: FP16 uncompressed vs FP16 palettized at input size
  224 with `compute_unit=all`.

Palettization summary:

| Model | Package MB | Steady Latency | Drift |
|---|---:|---|---:|
| FP16 MobileNetV2 224 | 6.713 | baseline | 0.0 |
| FP16 palettized MobileNetV2 224 | 3.445 | roughly neutral to slightly regressed | 0.0 |

The palettized package was about 48.7% smaller. This is only a sanity baseline:
it reports package size, latency, RSS, and numerical drift, but does not claim
accuracy improvement or guaranteed speedup. Runtime placement is still
determined by CoreML.

## Next Optimization Plan

Server lane:

- Start policy optimization experiments against the measured vLLM baseline.
- First candidate: a concurrency/admission policy or request scheduling
  guardrail that avoids the pathological concurrency-4 degradation observed on
  the 4 GB GTX 1650 Max-Q.

Edge lane:

- Later, use the CoreML measured baselines to drive an edge optimization policy
  over compute unit, input size, and compression choices.

## Simulator Boundary

Existing LLM runtime outputs under `results/llm_runtime_artifacts/` are
internal simulator, policy-ablation, and invariant-validation artifacts. They
may be inspired by vLLM/SGLang/TensorRT-LLM concepts, but they are not measured
external serving baselines.
