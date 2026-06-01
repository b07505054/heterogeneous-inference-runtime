import argparse
import json
import statistics
import time
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def load_model_bytes(path: Path):
    start = time.perf_counter()
    data = path.read_bytes()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return data, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/executorch/mv2_xnnpack_fp32.pte")
    parser.add_argument("--output", default="results/executorch_benchmark_metadata.json")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"ExecuTorch model not found: {model_path}")

    # Cold load: first load from disk
    cold_data, cold_load_time_ms = load_model_bytes(model_path)

    # Warm/repeated loads
    warm_load_times = []
    for _ in range(args.runs):
        _, elapsed_ms = load_model_bytes(model_path)
        warm_load_times.append(elapsed_ms)

    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    result = {
        "runtime": "ExecuTorch",
        "backend": "XNNPACK",
        "model": "MobileNetV2",
        "format": ".pte",
        "model_path": str(model_path),
        "deployment_target": "on-device CPU inference",
        "edge_constraint_profile": {
            "batch_size": 1,
            "execution_mode": "CPU-only deployable artifact",
            "mobile_relevance": [
                "cold start latency",
                "warm load stability",
                "model artifact size",
                "filesystem loading overhead",
            ],
        },
        "artifact": {
            "model_size_mb": round(model_size_mb, 4),
            "artifact_bytes": len(cold_data),
        },
        "load_benchmark_ms": {
            "runs": args.runs,
            "cold_load_time_ms": round(cold_load_time_ms, 4),
            "warm_avg_ms": round(statistics.mean(warm_load_times), 4),
            "warm_median_ms": round(statistics.median(warm_load_times), 4),
            "warm_p95_ms": round(percentile(warm_load_times, 95), 4),
            "warm_p99_ms": round(percentile(warm_load_times, 99), 4),
            "warm_min_ms": round(min(warm_load_times), 4),
            "warm_max_ms": round(max(warm_load_times), 4),
        },
        "notes": [
            "This benchmark measures deployable ExecuTorch artifact size and load behavior.",
            "Full operator-level inference latency requires ExecuTorch C++ runtime or mobile runtime integration.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()