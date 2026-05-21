import csv
from pathlib import Path

from backends.base import Backend, BenchmarkResult


class ExecuTorchBackend(Backend):
    name = "ExecuTorch"

    def __init__(
        self,
        artifact_path: str = "results/executorch_benchmark.csv",
        precision: str = "FP32",
    ):
        self.artifact_path = Path(artifact_path)
        self.precision = precision

    def benchmark(self) -> BenchmarkResult:
        if not self.artifact_path.exists():
            return BenchmarkResult(
                backend="ExecuTorch",
                precision=self.precision,
                device="XNNPACK",
                avg_latency_ms=-1.0,
                p95_latency_ms=None,
                p99_latency_ms=None,
                throughput_qps=None,
                extra={
                    "status": "not_available",
                    "reason": "ExecuTorch benchmark not generated in this repo yet",
                    "expected_artifact": str(self.artifact_path),
                },
            )

        with self.artifact_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            raise RuntimeError(f"No rows found in {self.artifact_path}")

        row = rows[0]
        latency = float(row["avg_latency_ms"])

        return BenchmarkResult(
            backend="ExecuTorch",
            precision=self.precision,
            device=row.get("delegate", self.backend_name),
            avg_latency_ms=round(latency, 4),
            p95_latency_ms=float(row["p95_latency_ms"]) if row.get("p95_latency_ms") else None,
            p99_latency_ms=float(row["p99_latency_ms"]) if row.get("p99_latency_ms") else None,
            throughput_qps=float(row["throughput_qps"]) if row.get("throughput_qps") else round(1000.0 / latency, 4),
            extra={
                "csv_path": str(self.artifact_path),
                "source": row.get("source", "executorch_runtime_python"),
                "model": row.get("model"),
                "delegate": row.get("delegate"),
                "threads": row.get("threads"),
                "backend_name": self.backend_name,
            },
        )
        