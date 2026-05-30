import argparse
import os
import time
import psutil
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import onnxruntime as ort


def percentile(values, p):
    return float(np.percentile(values, p))


def benchmark_pytorch(iterations=100, warmup=20, batch_size=1):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        latencies = []
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024

        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        peak_mem = process.memory_info().rss / 1024 / 1024

    return {
        "backend": "PyTorch",
        "model_format": "FP32",
        "batch_size": batch_size,
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "throughput_qps": 1000 / np.mean(latencies) * batch_size,
        "memory_mb": peak_mem - start_mem,
        "model_size_mb": None,
    }


def benchmark_onnx(model_path, iterations=100, warmup=20, batch_size=1, threads=1, profile=False):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile:
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = "results/onnxruntime_profile"
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    x = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

    for _ in range(warmup):
        _ = session.run(None, {input_name: x})

    latencies = []
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024

    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: x})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    peak_mem = process.memory_info().rss / 1024 / 1024
    model_size = os.path.getsize(model_path) / 1024 / 1024

    profile_file = None
    if profile:
        profile_file = session.end_profiling()
        print(f"ONNX Runtime profile saved to: {profile_file}")

    return {
        "backend": "ONNX Runtime",
        "model_format": "INT8" if "int8" in model_path.lower() else "FP32",
        "batch_size": batch_size,
        "threads": threads,
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "throughput_qps": 1000 / np.mean(latencies) * batch_size,
        "memory_mb": peak_mem - start_mem,
        "model_size_mb": model_size,
        "profile_file": profile_file,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["pytorch", "onnx"], required=True)
    parser.add_argument("--model-path", type=str, default="models/mobilenet_v2_fp32.onnx")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.backend == "pytorch":
        result = benchmark_pytorch(
            iterations=args.iterations,
            batch_size=args.batch_size,
        )
    else:
        result = benchmark_onnx(
            model_path=args.model_path,
            iterations=args.iterations,
            batch_size=args.batch_size,
            threads=args.threads,
            profile=args.profile,
        )

    os.makedirs("results", exist_ok=True)
    if args.backend == "onnx":
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]
        output_path = f"results/{model_name}_bs{args.batch_size}_t{args.threads}_benchmark.csv"
    else:
        output_path = f"results/{args.backend}_bs{args.batch_size}_benchmark.csv"

    df = pd.DataFrame([result])
    df.to_csv(output_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved result to {output_path}")


if __name__ == "__main__":
    main()