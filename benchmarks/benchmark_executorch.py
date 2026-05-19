import csv
import time
from pathlib import Path

import numpy as np
import torch
from executorch.runtime import Runtime


MODEL_PATH = Path("models/mobilenet_v2_xnnpack.pte")
OUTPUT_PATH = Path("results/executorch_benchmark.csv")

RUNS = 100
WARMUP = 10


def main():
    runtime = Runtime.get()

    program = runtime.load_program(str(MODEL_PATH))
    method = program.load_method("forward")

    dummy_input = torch.randn(1, 3, 224, 224)

    for _ in range(WARMUP):
        method.execute([dummy_input])

    latencies = []

    for _ in range(RUNS):
        start = time.perf_counter()
        method.execute([dummy_input])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    avg = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    qps = 1000.0 / avg

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    row = {
        "backend": "ExecuTorch",
        "model": "MobileNetV2",
        "delegate": "XNNPACK",
        "threads": "",
        "avg_latency_ms": round(avg, 4),
        "p95_latency_ms": round(p95, 4),
        "p99_latency_ms": round(p99, 4),
        "throughput_qps": round(qps, 4),
        "source": "executorch_runtime_python",
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(row)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()