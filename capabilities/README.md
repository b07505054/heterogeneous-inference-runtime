# Capability Layer

The capability layer is the explicit boundary between measured runtime evidence
and future optimization policy.

The repository flow is:

```text
Measured Baseline
        |
        v
Capability Layer
        |
        v
Optimization Policy
        |
        v
Deployment Decision
```

Capabilities describe what exists. They do not run benchmarks, select policies,
or predict performance.

## Concepts

### HardwareCapability

Physical hardware facts only:

- Apple M-series SoC, Apple GPU, Apple ANE, unified memory.
- NVIDIA GPU family, CUDA compute capability, VRAM.
- CPU family, core count, memory class.

Benchmark results do not belong in `HardwareCapability`.

### BackendCapability

Runtime/backend support only:

- CoreML, Metal, MPS, CUDA, vLLM, ONNX Runtime, TensorRT.
- Supported runtime features, precisions, compute units, and fallback backends.

Measured performance does not belong in `BackendCapability`.

### KernelLibraryCapability

Runtime or kernel implementation availability:

- MatMul, Conv, Attention, Softmax, RMSNorm, FlashAttention, PagedAttention,
  PrefixCache, Speculative.
- Availability is one of `builtin`, `opaque`, `custom`, or `unsupported`.

This is runtime/kernel availability, not compiler lowering.

### MeasuredSupport

Experimentally verified support facts only:

- FP16 benchmark completed.
- Palettization benchmark completed.
- Input size 224 measured.
- CoreML ComputeUnit ALL measured.
- vLLM TTFT measured.
- Concurrency benchmark completed.

Predictions do not belong in `MeasuredSupport`.

## Truth Boundary

Measured baselines are evidence. Capabilities describe what exists. Policies
choose among capabilities. Simulators evaluate ideas. These are separate layers
and must not be merged.

Do not claim support for FP8, NVFP4, MXFP4, EAGLE, MTP, SpecForge, LMCache,
HiCache, or KVBM unless support is measured or explicitly represented as future
capability with a non-measured evidence level.

Future policies such as `CoreMLEdgePolicy`, `QuantizationPolicy`, `KVPolicy`,
`ServerRuntimePolicy`, and `PDPolicy` should consume the capability layer rather
than inferring hardware or backend support directly from benchmark scripts.
