import time

import numpy as np
import torch
import torchvision.models as models

from backends.base import Backend, BenchmarkResult


class PyTorchBackend(Backend):
    name = "PyTorch"

    def __init__(
        self,
        precision: str = "FP32",
        device: str = "cpu",
        runs: int = 100,
        warmup: int = 10,
    ):
        self.precision = precision
        self.device = device
        self.runs = runs
        self.warmup = warmup

    def benchmark(self) -> BenchmarkResult:
        model = models.mobilenet_v2(weights=None)

        model.eval()
        model.to(self.device)

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)

        if self.precision == "FP16":
            model = model.half()
            dummy_input = dummy_input.half()

        with torch.no_grad():
            for _ in range(self.warmup):
                _ = model(dummy_input)

        latencies = []

        with torch.no_grad():
            for _ in range(self.runs):
                start = time.perf_counter()

                _ = model(dummy_input)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()

                latencies.append((end - start) * 1000)

        latencies = np.array(latencies)

        return BenchmarkResult(
            backend="PyTorch",
            precision=self.precision,
            device=self.device,
            avg_latency_ms=round(float(np.mean(latencies)), 4),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 4),
            p99_latency_ms=round(float(np.percentile(latencies, 99)), 4),
            throughput_qps=round(1000.0 / float(np.mean(latencies)), 4),
            extra={
                "runs": self.runs,
                "warmup": self.warmup,
            },
        )