#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmark.backends.coreml import CoreMLMobileNetV2Backend, _first_array, numerical_drift, package_size_mb
from benchmark.backends.pytorch import PyTorchMobileNetV2Backend, torch_available
from benchmark.exporters import measured_envelope, write_json
from benchmark.metrics import latency_summary_ms
from benchmark.runner import BenchmarkRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark native CoreML MobileNetV2 against PyTorch CPU/MPS.")
    parser.add_argument("--mlpackage", "--coreml-model", dest="mlpackage", default="models/mobilenet_v2.mlpackage")
    parser.add_argument("--runs", "--iterations", dest="runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--include-pytorch", default="cpu,mps")
    parser.add_argument("--compute-unit", choices=["cpu", "cpu_gpu", "all"], default="all")
    parser.add_argument("--model-precision", choices=["fp16"], default="fp16")
    parser.add_argument("--model-compression", choices=["none", "palettize", "unknown"], default="unknown")
    parser.add_argument("--output", default="results/measured_baselines/coreml_mobilenetv2_baseline.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    sample_np = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    metrics = {
        "model": {
            "name": "MobileNetV2",
            "precision": args.model_precision,
            "compression": args.model_compression,
            "package_size_mb": package_size_mb(args.mlpackage),
        },
        "coreml": {},
        "pytorch_cpu": {},
        "pytorch_mps": {},
    }
    status = "ok"
    notes = [
        "Native CoreML path uses .mlpackage when coremltools and the package are available.",
        "Unavailable optional backends are reported in schema and are not required by scripts/check.sh.",
    ]
    reference_output = None

    include_pytorch = {item.strip() for item in args.include_pytorch.split(",") if item.strip()}

    if torch_available():
        if "cpu" in include_pytorch:
            cpu = _run_pytorch_backend("cpu", sample_np, args.runs, args.warmup)
            metrics["pytorch_cpu"] = _strip_reference(cpu)
            reference = cpu.get("metrics", {}).get("reference_output")
            if reference is not None:
                reference_output = reference.numpy()
        else:
            metrics["pytorch_cpu"] = {"status": "skipped", "reason": "not_requested"}
        if "mps" in include_pytorch:
            mps = _run_pytorch_backend("mps", sample_np, args.runs, args.warmup)
            metrics["pytorch_mps"] = _strip_reference(mps)
        else:
            metrics["pytorch_mps"] = {"status": "skipped", "reason": "not_requested"}
    else:
        status = "partial"
        metrics["pytorch_cpu"] = {"status": "unavailable", "reason": "torch_or_torchvision_not_installed"}
        metrics["pytorch_mps"] = {"status": "unavailable", "reason": "torch_or_torchvision_not_installed"}

    coreml = _run_coreml_backend(
        args.mlpackage,
        sample_np,
        reference_output,
        args.runs,
        args.warmup,
        args.compute_unit,
    )
    if coreml["status"] != "ok":
        status = "partial"
        coreml.setdefault("metrics", {})["package_size_mb"] = package_size_mb(args.mlpackage)
    metrics["coreml"] = coreml

    payload = measured_envelope(
        artifact_type="measured_baseline",
        benchmark_target={
            "kind": "native_coreml_cv",
            "backend": "coreml",
            "model": "MobileNetV2",
            "mlpackage": args.mlpackage,
            "model_precision": args.model_precision,
            "model_compression": args.model_compression,
            "comparators": ["pytorch_cpu", "pytorch_mps"],
            "runs": args.runs,
            "warmup": args.warmup,
        },
        metrics=metrics,
        notes=notes,
        command=sys.argv,
        status=status,
        extra={"execution": {"compute_unit": args.compute_unit}},
    )
    write_json(args.output, payload)
    print(args.output)


def _strip_reference(result: dict) -> dict:
    cleaned = dict(result)
    inner = dict(cleaned.get("metrics", {}))
    inner.pop("reference_output", None)
    cleaned["metrics"] = inner
    return cleaned


def _run_pytorch_backend(device: str, sample_np: np.ndarray, runs: int, warmup: int) -> dict:
    backend = PyTorchMobileNetV2Backend(device=device)
    setup = backend.setup(sample_np)
    if setup["status"] != "ok":
        return setup

    runner = BenchmarkRunner(
        measure_fn=lambda _item, warmup=False: backend.execute(),
        sync_fn=backend.sync,
        time_measurements=True,
    )
    cold = runner.measure(warmup=False)
    runner.warmup(warmup)
    warm = runner.measure(warmup=False)
    runner.run(runs)
    return {
        "status": "ok",
        "backend": f"pytorch_{device}",
        "metrics": {
            "model_load_ms": setup["model_load_ms"],
            "cold_start_ms": cold["elapsed_ms"],
            "warm_start_ms": warm["elapsed_ms"],
            "steady_state_latency_ms": latency_summary_ms(row["elapsed_ms"] for row in runner.results),
            "reference_output": cold["value"].detach().cpu(),
        },
    }


def _run_coreml_backend(
    mlpackage: str,
    sample_np: np.ndarray,
    reference_output: np.ndarray | None,
    runs: int,
    warmup: int,
    compute_unit: str = "all",
) -> dict:
    backend = CoreMLMobileNetV2Backend(mlpackage, compute_unit=compute_unit)
    load_start = time.perf_counter()
    setup = backend.setup(sample_np)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    if setup["status"] != "ok":
        return setup

    runner = BenchmarkRunner(
        measure_fn=lambda _item, warmup=False: backend.execute(),
        time_measurements=True,
    )
    cold = runner.measure(warmup=False)
    runner.warmup(warmup)
    warm = runner.measure(warmup=False)
    runner.run(runs)

    output_array = _first_array(cold["value"])
    drift = None
    if reference_output is not None and output_array is not None:
        drift = numerical_drift(reference_output, output_array)

    return {
        "status": "ok",
        "backend": "coreml_mlpackage",
        "metrics": {
            "model_load_ms": round(load_ms, 6),
            "cold_start_ms": cold["elapsed_ms"],
            "warm_start_ms": warm["elapsed_ms"],
            "steady_state_latency_ms": latency_summary_ms(row["elapsed_ms"] for row in runner.results),
            "rss_delta_mb": backend.rss_delta_mb(),
            "rss_load_delta_mb": backend.rss_load_delta_mb(),
            "package_size_mb": package_size_mb(mlpackage),
            "numerical_drift": drift,
        },
    }


if __name__ == "__main__":
    main()
