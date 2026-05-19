import csv
from pathlib import Path

from backends.base import Backend, BenchmarkResult


class CppInferenceBackend(Backend):
    name = "CppInference"

    def __init__(
        self,
        csv_path: str,
        precision: str = "FP32",
        device: str = "CPU",
    ):
        self.csv_path = Path(csv_path)
        self.precision = precision
        self.device = device

    def benchmark(self) -> BenchmarkResult:
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise RuntimeError(f"No benchmark rows found in {self.csv_path}")

        row = rows[0]

        latency_key = None
        for key in ["avg_latency_ms", "mean_latency_ms", "latency_ms"]:
            if key in row:
                latency_key = key
                break

        if latency_key is None:
            raise RuntimeError(
                f"No latency column found in {self.csv_path}. "
                f"Available columns: {list(row.keys())}"
            )

        latency = float(row[latency_key])

        return BenchmarkResult(
            backend="CppInference",
            precision=self.precision,
            device=self.device,
            avg_latency_ms=round(latency, 4),
            throughput_qps=round(1000.0 / latency, 4),
            extra={
                "csv_path": str(self.csv_path),
                "latency_column": latency_key,
                "source": "existing C++ inference benchmark artifact",
            },
        )