import json
from pathlib import Path

from backends.base import Backend, BenchmarkResult


class TensorRTBackend(Backend):
    name = "TensorRT"

    def __init__(self, summary_path: str = "results/tensorrt_precision_comparison.json"):
        self.summary_path = Path(summary_path)

    def benchmark(self) -> list[BenchmarkResult]:
        data = json.loads(self.summary_path.read_text(encoding="utf-8"))

        results = []

        for row in data["results"]:
            results.append(
                BenchmarkResult(
                    backend="TensorRT",
                    precision=row["mode"],
                    device=data["device"],
                    avg_latency_ms=row["latency_mean_ms"],
                    p95_latency_ms=row["latency_p95_ms"],
                    p99_latency_ms=row["latency_p99_ms"],
                    throughput_qps=row["throughput_qps"],
                    extra={
                        "enqueue_mean_ms": row["enqueue_mean_ms"],
                        "h2d_mean_ms": row["h2d_mean_ms"],
                        "gpu_compute_mean_ms": row["gpu_compute_mean_ms"],
                        "d2h_mean_ms": row["d2h_mean_ms"],
                    },
                )
            )

        return results
