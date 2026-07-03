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
PYTHONPATH=$PWD .venv/bin/python scripts/benchmark_openai_compatible_server.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --trace traces/openai_requests.jsonl \
  --concurrency 4 \
  --warmup 2 \
  --claimed-server vllm \
  --output results/measured_baselines/vllm_baseline.json
```

`benchmark_target.kind` remains `openai_compatible_server`. The optional
`claimed_server` field is only a user-supplied label.

## Native CoreML CV

`scripts/export_coreml_mobilenetv2.py` exports a native MobileNetV2
`.mlpackage` when `coremltools`, `torch`, and `torchvision` are installed.

`scripts/benchmark_coreml_cv_baseline.py` compares the native CoreML package
against PyTorch CPU and, when available, PyTorch MPS. Missing optional
dependencies or missing CoreML packages are reported as `status: "partial"` or
backend-level `status: "unavailable"` rather than failing CI.

## Simulator Boundary

Existing LLM runtime outputs under `results/llm_runtime_artifacts/` are
internal simulator, policy-ablation, and invariant-validation artifacts. They
may be inspired by vLLM/SGLang/TensorRT-LLM concepts, but they are not measured
external serving baselines.

