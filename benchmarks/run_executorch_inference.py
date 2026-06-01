import argparse
import statistics
import time
from pathlib import Path

import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def run_forward(module, inputs):
    try:
        return module.forward(inputs)
    except Exception:
        try:
            return module.run_method("forward", inputs)
        except Exception:
            return module(*inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="models/executorch/mv2_xnnpack_fp32.pte",
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("[INFO] Loading ExecuTorch model...")
    module = _load_for_executorch(str(model_path))

    # MobileNet input
    x = torch.randn(1, 3, 224, 224).contiguous()

    print("[INFO] Warmup...")
    for _ in range(args.warmup):
        run_forward(module, (x,))

    latencies = []

    print("[INFO] Running inference...")
    for _ in range(args.runs):
        start = time.perf_counter()
        run_forward(module, (x,))
        latencies.append((time.perf_counter() - start) * 1000)

    print("\n=== ExecuTorch Inference ===")
    print(f"avg: {statistics.mean(latencies):.4f} ms")
    print(f"p95: {percentile(latencies, 95):.4f} ms")
    print(f"p99: {percentile(latencies, 99):.4f} ms")


if __name__ == "__main__":
    main()