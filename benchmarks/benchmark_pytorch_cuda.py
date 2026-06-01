import time
import json
from pathlib import Path

import torch
import torchvision.models as models


def benchmark(dtype=torch.float32, runs=100, warmup=20):
    device = torch.device("cuda")

    model = models.mobilenet_v2(weights=None).eval().to(device)

    if dtype == torch.float16:
        model = model.half()

    x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()

    latencies = []

    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

    latencies_t = torch.tensor(latencies)

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "device": torch.cuda.get_device_name(0),
        "runs": runs,
        "avg_ms": round(latencies_t.mean().item(), 4),
        "p50_ms": round(latencies_t.median().item(), 4),
        "p95_ms": round(torch.quantile(latencies_t, 0.95).item(), 4),
        "p99_ms": round(torch.quantile(latencies_t, 0.99).item(), 4),
        "min_ms": round(latencies_t.min().item(), 4),
        "max_ms": round(latencies_t.max().item(), 4),
        "cuda_version": torch.version.cuda,
    }


def main():
    results = [
        benchmark(torch.float32),
        benchmark(torch.float16),
    ]

    output_path = Path("results/pytorch_cuda_benchmark.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()