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

## Native CoreML CV

`scripts/export_coreml_mobilenetv2.py` exports a native MobileNetV2
`.mlpackage` when `coremltools`, `torch`, and `torchvision` are installed.
The default export is FP16:

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/export_coreml_mobilenetv2.py \
  --precision fp16 \
  --compression none \
  --output models/coreml/mobilenet_v2_fp16.mlpackage
```

Compressed exports are optional. Palettization uses `coremltools` optimization
APIs when they are available:

```bash
PYTHONPATH=$PWD .venv/bin/python scripts/export_coreml_mobilenetv2.py \
  --precision fp16 \
  --compression palettize \
  --output models/coreml/mobilenet_v2_fp16_palettized.mlpackage
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

## Simulator Boundary

Existing LLM runtime outputs under `results/llm_runtime_artifacts/` are
internal simulator, policy-ablation, and invariant-validation artifacts. They
may be inspired by vLLM/SGLang/TensorRT-LLM concepts, but they are not measured
external serving baselines.
