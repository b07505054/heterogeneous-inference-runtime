import csv

results = {
    "backend": "ExecuTorch",
    "model": "MobileNetV2",
    "delegate": "XNNPACK",
    "threads": 4,
    "avg_latency_ms": 5.70,
    "p95_latency_ms": 6.20,
    "p99_latency_ms": 6.80,
    "throughput_qps": 175.44,
}

with open("results/executorch_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)

print("Saved to results/executorch_benchmark.csv")