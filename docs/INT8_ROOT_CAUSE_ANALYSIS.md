# INT8 Quantization Root Cause Analysis

## Observation

INT8 dynamic quantization significantly increased ONNX Runtime latency compared with FP32.

| Model | Avg Latency | p95 | p99 |
|---|---:|---:|---:|
| FP32 ONNX | 5.29 ms | 7.32 ms | 7.79 ms |
| Optimized ONNX | 4.84 ms | 6.65 ms | 7.78 ms |
| INT8 ONNX | 113.83 ms | 130.78 ms | 138.12 ms |

## Model Structure Analysis

| Model | Node Count |
|---|---:|
| FP32 | 102 |
| Optimized | 100 |
| INT8 | 417 |

INT8 dynamic quantization introduced:

| Operator | Count |
|---|---:|
| Mul | 106 |
| Add | 63 |
| Reshape | 53 |
| DynamicQuantizeLinear | 53 |
| Cast | 53 |
| ConvInteger | 52 |
| Clip | 35 |
| MatMulInteger | 1 |

## Profiling Result

Over 20 runs, ONNX Runtime profiling showed:

| Operator | Total Time |
|---|---:|
| ConvInteger | 2224.89 ms |
| Mul | 95.92 ms |
| Cast | 74.36 ms |
| Add | 74.36 ms |
| DynamicQuantizeLinear | 65.44 ms |
| Clip | 37.79 ms |

## Root Cause

The INT8 model did not achieve faster inference because dynamic quantization introduced many additional operators and routed convolution layers through `ConvInteger`, which dominated runtime under `CPUExecutionProvider`.

## Conclusion

INT8 quantization reduced model precision but did not improve latency in this setup. The bottleneck was backend/operator support rather than the quantization idea itself.

Future work:
- Test static quantization with `QLinearConv`
- Compare with hardware-aware backends
- Evaluate ExecuTorch/XNNPACK quantized deployment
- Test QNN / NNAPI / OpenVINO / CoreML where available