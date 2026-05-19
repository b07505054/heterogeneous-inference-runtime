import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from backends.base import Backend, BenchmarkResult


class ONNXRuntimeBackend(Backend):
    name = "ONNXRuntime"

    def __init__(
        self,
        model_path: str,
        precision: str = "FP32",
        provider: str = "CPUExecutionProvider",
        runs: int = 100,
        warmup: int = 10,
    ):
        self.model_path = Path(model_path)
        self.precision = precision
        self.provider = provider
        self.runs = runs
        self.warmup = warmup

    def benchmark(self) -> BenchmarkResult:
        session = ort.InferenceSession(
            str(self.model_path),
            providers=[self.provider],
        )

        actual_providers = session.get_providers()
        input_name = session.get_inputs()[0].name

        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

        for _ in range(self.warmup):
            session.run(None, {input_name: dummy_input})

        latencies = []

        for _ in range(self.runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        latencies = np.array(latencies)

        return BenchmarkResult(
            backend="ONNXRuntime",
            precision=self.precision,
            device=self.provider,
            avg_latency_ms=round(float(np.mean(latencies)), 4),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 4),
            p99_latency_ms=round(float(np.percentile(latencies, 99)), 4),
            throughput_qps=round(1000.0 / float(np.mean(latencies)), 4),
            extra={
                "model_path": str(self.model_path),
                "requested_provider": self.provider,
                "actual_providers": actual_providers,
                "runs": self.runs,
                "warmup": self.warmup,
            },
        )