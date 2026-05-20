import csv
from pathlib import Path

from backends.base import Backend, BenchmarkResult


class ExecuTorchBackend(Backend):
    name = "ExecuTorch"

    def __init__(
        self,
        csv_path: str,
        precision: str = "FP32",
        backend_name: str = "XNNPACK",
    ):
        self.csv_path = Path(csv_path)
        self.precision = precision
        self.backend_name = backend_name

    def benchmark(self) -> BenchmarkResult:
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            raise RuntimeError("No benchmark rows found")

        row = rows[0]

        latency = float(row["avg_latency_ms"])

        return BenchmarkResult(
            backend="ExecuTorch",
            precision=self.precision,
            device=self.backend_name,
            avg_latency_ms=latency,
            p95_latency_ms=float(row["p95_latency_ms"]) if row.get("p95_latency_ms") else None,
            p99_latency_ms=float(row["p99_latency_ms"]) if row.get("p99_latency_ms") else None,
            throughput_qps=float(row["throughput_qps"]) if row.get("throughput_qps") else round(1000.0 / latency, 4),
            extra={
                "csv_path": str(self.csv_path),
                "source": row.get("source"),
                "delegate": row.get("delegate"),
            },
        )