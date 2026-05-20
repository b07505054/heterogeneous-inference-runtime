import csv
from pathlib import Path

from backends.base import Backend, BenchmarkResult


class TensorRTBackend(Backend):
    name = "TensorRT"

    def __init__(
        self,
        csv_path: str = "results/tensorrt_benchmark.csv",
        precision: str = "FP16",
    ):
        self.csv_path = Path(csv_path)
        self.precision = precision

    def benchmark(self) -> BenchmarkResult:
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise RuntimeError(f"No rows found in {self.csv_path}")

        row = rows[0]

        return BenchmarkResult(
            backend="TensorRT",
            precision=row.get("precision", self.precision),
            device=row.get("device", "CUDA"),
            avg_latency_ms=float(row["avg_latency_ms"]),
            p95_latency_ms=float(row["p95_latency_ms"]),
            p99_latency_ms=float(row["p99_latency_ms"]),
            throughput_qps=float(row["throughput_qps"]),
            extra={
                "csv_path": str(self.csv_path),
                "source": "tensorrt_python_runtime",
            },
        )