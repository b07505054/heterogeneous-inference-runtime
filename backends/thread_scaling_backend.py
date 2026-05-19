import csv
from pathlib import Path

from backends.base import Backend, BenchmarkResult


class ThreadScalingBackend(Backend):
    name = "ThreadScaling"

    def __init__(
        self,
        csv_path: str,
        threads: int,
        precision: str = "Optimized FP32",
    ):
        self.csv_path = Path(csv_path)
        self.threads = threads
        self.precision = precision

    def benchmark(self) -> BenchmarkResult:
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise RuntimeError(f"No rows found in {self.csv_path}")

        row = rows[0]

        latency_key = None

        for key in [
            "avg_latency_ms",
            "mean_latency_ms",
            "latency_ms",
        ]:
            if key in row:
                latency_key = key
                break

        if latency_key is None:
            raise RuntimeError(
                f"No latency column found. "
                f"Columns: {list(row.keys())}"
            )

        latency = float(row[latency_key])

        return BenchmarkResult(
            backend="ThreadScaling",
            precision=self.precision,
            device=f"{self.threads} threads",
            avg_latency_ms=round(latency, 4),
            throughput_qps=round(1000.0 / latency, 4),
            extra={
                "threads": self.threads,
                "csv_path": str(self.csv_path),
            },
        )