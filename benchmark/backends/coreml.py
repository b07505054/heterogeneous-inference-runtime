from __future__ import annotations

import os
from pathlib import Path

import numpy as np


COMPUTE_UNIT_CHOICES = ("cpu", "cpu_gpu", "all")


def coreml_available() -> bool:
    try:
        import coremltools  # noqa: F401
    except Exception:
        return False
    return True


def resolve_compute_unit(compute_unit: str, coremltools_module=None):
    if compute_unit not in COMPUTE_UNIT_CHOICES:
        raise ValueError(f"unsupported CoreML compute unit: {compute_unit}")
    ct = coremltools_module
    if ct is None:
        import coremltools as ct
    mapping = {
        "cpu": "CPU_ONLY",
        "cpu_gpu": "CPU_AND_GPU",
        "all": "ALL",
    }
    enum_name = mapping[compute_unit]
    try:
        return getattr(ct.ComputeUnit, enum_name)
    except AttributeError as exc:
        raise RuntimeError(f"CoreML ComputeUnit.{enum_name} is not available") from exc


def package_size_mb(path: str | Path) -> float | None:
    package_path = Path(path)
    if not package_path.exists():
        return None
    if package_path.is_file():
        return round(package_path.stat().st_size / 1024 / 1024, 6)
    total = 0
    for root, _dirs, files in os.walk(package_path):
        for filename in files:
            total += (Path(root) / filename).stat().st_size
    return round(total / 1024 / 1024, 6)


def numerical_drift(reference: np.ndarray, candidate: np.ndarray) -> dict:
    diff = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "max_abs": round(float(diff.max()), 8),
        "mean_abs": round(float(diff.mean()), 8),
        "top1_match": bool(np.argmax(reference) == np.argmax(candidate)),
    }


class CoreMLMobileNetV2Backend:
    def __init__(self, model_path: str | Path, compute_unit: str = "all"):
        self.model_path = Path(model_path)
        self.compute_unit = compute_unit
        self.model = None
        self.sample = None
        self.input_name = None
        self.process = None
        self.rss_before = None
        self.rss_after_load = None

    def setup(self, sample: np.ndarray | None = None) -> dict:
        if not coreml_available():
            return {"status": "unavailable", "reason": "coremltools_not_installed", "metrics": {}}
        if not self.model_path.exists():
            return {"status": "unavailable", "reason": "mlpackage_not_found", "metrics": {}}

        import coremltools as ct
        import psutil

        self.sample = sample if sample is not None else np.random.randn(1, 3, 224, 224).astype(np.float32)
        self.process = psutil.Process(os.getpid())
        self.rss_before = self.process.memory_info().rss
        self.model = ct.models.MLModel(
            str(self.model_path),
            compute_units=resolve_compute_unit(self.compute_unit, ct),
        )
        self.rss_after_load = self.process.memory_info().rss
        self.input_name = self.model.get_spec().description.input[0].name
        return {"status": "ok"}

    def execute(self):
        return self.model.predict({self.input_name: self.sample})

    def rss_delta_mb(self) -> float:
        rss_after = self.process.memory_info().rss
        return round((rss_after - self.rss_before) / 1024 / 1024, 6)

    def rss_load_delta_mb(self) -> float:
        return round((self.rss_after_load - self.rss_before) / 1024 / 1024, 6)


def _first_array(outputs: dict) -> np.ndarray | None:
    for value in outputs.values():
        array = np.asarray(value)
        if array.size:
            return array
    return None
