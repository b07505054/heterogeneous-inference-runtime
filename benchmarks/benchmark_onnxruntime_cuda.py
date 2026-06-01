import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_PATH = "models/mobilenet_v2_fp32.onnx"

RUNS = 100
WARMUP = 10


def percentile(values, p):
    return float(np.percentile(values, p))


def benchmark_session(session, input_name):
    latencies = []

    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    for _ in range(WARMUP):
        session.run(None, {input_name: dummy_input})

    for _ in range(RUNS):
        start = time.perf_counter()

        session.run(None, {input_name: dummy_input})

        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    return {
        "avg_ms": round(float(np.mean(latencies)), 4),
        "p50_ms": round(percentile(latencies, 50), 4),
        "p95_ms": round(percentile(latencies, 95), 4),
        "p99_ms": round(percentile(latencies, 99), 4),
        "min_ms": round(float(np.min(latencies)), 4),
        "max_ms": round(float(np.max(latencies)), 4),
    }


def main():
    print("=== ONNX Runtime CUDA EP Benchmark ===")

    providers = ort.get_available_providers()

    print("Available providers:")
    print(providers)

    if "CUDAExecutionProvider" not in providers:
        print("CUDAExecutionProvider not available")
        return

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CUDAExecutionProvider"]
    )
    actual_providers = session.get_providers()
    print("Session providers:")
    print(actual_providers)

    if "CUDAExecutionProvider" not in actual_providers:
        output = {
            "runtime": "ONNX Runtime",
            "requested_provider": "CUDAExecutionProvider",
            "actual_providers": actual_providers,
            "status": "fallback_to_cpu",
            "reason": "CUDAExecutionProvider failed to load, likely missing cuDNN dependency"
        }

        output_path = Path("results/onnxruntime_cuda_ep_failed.json")
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2))
        print(f"Saved failure report to {output_path}")
        return

    input_name = session.get_inputs()[0].name

    results = benchmark_session(session, input_name)

    output = {
        "runtime": "ONNX Runtime",
        "execution_provider": "CUDAExecutionProvider",
        "model": MODEL_PATH,
        "runs": RUNS,
        **results
    }

    print(json.dumps(output, indent=2))

    output_path = Path("results/onnxruntime_cuda_ep.json")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()