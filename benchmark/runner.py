from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable


class BenchmarkRunner:
    """Small composition-based runner for measured benchmark scripts."""

    def __init__(
        self,
        *,
        measure_fn: Callable,
        finalize_fn: Callable | None = None,
        export_fn: Callable | None = None,
        sync_fn: Callable[[], None] | None = None,
        concurrency: int = 1,
        time_measurements: bool = False,
        collect_warmup: bool = False,
    ):
        self.measure_fn = measure_fn
        self.finalize_fn = finalize_fn or (lambda rows: rows)
        self.export_fn = export_fn
        self.sync_fn = sync_fn
        self.concurrency = max(1, int(concurrency))
        self.time_measurements = time_measurements
        self.collect_warmup = collect_warmup
        self.results = []
        self.warmup_results = []
        self.iterations_run = 0
        self.warmup_iterations_run = 0
        self.finalized = None

    def warmup(self, iterations) -> list:
        rows = []
        for item in _iteration_items(iterations):
            row = self.measure(item, warmup=True)
            rows.append(row)
            if self.collect_warmup:
                self.warmup_results.append(row)
            self.warmup_iterations_run += 1
        return rows

    def run(self, iterations) -> list:
        items = list(_iteration_items(iterations))
        if self.concurrency == 1:
            for item in items:
                self.results.append(self.measure(item, warmup=False))
                self.iterations_run += 1
            return self.results

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self.measure, item, False) for item in items]
            for future in as_completed(futures):
                self.results.append(future.result())
                self.iterations_run += 1
        return self.results

    def measure(self, item=None, warmup: bool = False):
        if not self.time_measurements:
            return self.measure_fn(item, warmup=warmup)
        if self.sync_fn:
            self.sync_fn()
        start = time.perf_counter()
        value = self.measure_fn(item, warmup=warmup)
        if self.sync_fn:
            self.sync_fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "elapsed_ms": round(elapsed_ms, 6),
            "value": value,
            "warmup": warmup,
        }

    def finalize(self):
        self.finalized = self.finalize_fn(self.results)
        return self.finalized

    def export(self, payload):
        if self.export_fn is None:
            raise ValueError("export_fn is not configured")
        return self.export_fn(payload)


def _iteration_items(iterations) -> Iterable:
    if isinstance(iterations, int):
        return range(iterations)
    return iterations
