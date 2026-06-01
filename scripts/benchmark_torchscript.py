import time
import numpy as np
import torch
import torchvision.models as models
import pandas as pd
import os
import psutil


def percentile(values, p):
    return float(np.percentile(values, p))


def main():
    batch_size = 1
    iterations = 100
    warmup = 20

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.eval()

    # 👉 TorchScript compile
    scripted_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    scripted_model.eval()

    x = torch.randn(batch_size, 3, 224, 224)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = scripted_model(x)

    latencies = []
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024

    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = scripted_model(x)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    peak_mem = process.memory_info().rss / 1024 / 1024

    result = {
        "backend": "TorchScript",
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

    os.makedirs("results", exist_ok=True)
    output_path = "results/torchscript_benchmark.csv"

    pd.DataFrame([result]).to_csv(output_path, index=False)

    print(pd.DataFrame([result]).to_string(index=False))
    print(f"\nSaved result to {output_path}")


if __name__ == "__main__":
    main()