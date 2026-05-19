import json
from pathlib import Path

from backends.pytorch_backend import PyTorchBackend
from backends.onnxruntime_backend import ONNXRuntimeBackend
from backends.executorch_backend import ExecuTorchBackend
from backends.cpp_backend import CppInferenceBackend
from backends.thread_scaling_backend import ThreadScalingBackend
from backends.tensorrt_backend import TensorRTBackend
def serialize_result(r):
    return {
        "backend": r.backend,
        "precision": r.precision,
        "device": r.device,
        "avg_latency_ms": r.avg_latency_ms,
        "p95_latency_ms": r.p95_latency_ms,
        "p99_latency_ms": r.p99_latency_ms,
        "throughput_qps": r.throughput_qps,
        "extra": r.extra,
    }


def main():
    backends = [
        ONNXRuntimeBackend(
            model_path="models/mobilenet_v2_fp32.onnx",
            precision="FP32 CUDA",
            provider="CUDAExecutionProvider",
        ),

        ONNXRuntimeBackend(
            model_path="models/mobilenet_v2_optimized.onnx",
            precision="Optimized FP32 CUDA",
            provider="CUDAExecutionProvider",
        ),

        ONNXRuntimeBackend(
            model_path="models/mobilenet_v2_int8.onnx",
            precision="INT8 CPU",
            provider="CPUExecutionProvider",
        ),

        PyTorchBackend(
            precision="FP32",
            device="cpu",
        ),
        # ExecuTorchBackend(
        #     csv_path="results/YOUR_EXECUTORCH_CSV.csv",
        #     precision="FP32",
        #     backend_name="XNNPACK",
        # ),
        CppInferenceBackend(
            csv_path="results/cpp_benchmark.csv",
            precision="FP32",
            device="CPU C++",
        ),

        ThreadScalingBackend(
            csv_path="results/mobilenet_v2_optimized_bs1_t1_benchmark.csv",
            threads=1,
        ),
        ThreadScalingBackend(
            csv_path="results/mobilenet_v2_optimized_bs1_t2_benchmark.csv",
            threads=2,
        ),
        ThreadScalingBackend(
            csv_path="results/mobilenet_v2_optimized_bs1_t4_benchmark.csv",
            threads=4,
        ),
        ThreadScalingBackend(
            csv_path="results/mobilenet_v2_optimized_bs1_t8_benchmark.csv",
            threads=8,
        ),
    ]

    results = []

    for backend in backends:
        print(f"[INFO] Running {backend.name} / {backend.precision}")
        results.append(serialize_result(backend.benchmark()))

    output_path = Path("results/backend_validation_summary.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()