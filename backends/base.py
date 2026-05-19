from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class BenchmarkResult:
    backend: str
    precision: str
    device: str
    avg_latency_ms: float
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    throughput_qps: float | None = None
    extra: Dict[str, Any] | None = None


class Backend:
    name: str = "base"

    def benchmark(self) -> BenchmarkResult:
        raise NotImplementedError
