import json
from pathlib import Path

import matplotlib.pyplot as plt

INPUT_PATH = Path("results/tensorrt_precision_comparison.json")
OUTPUT_DIR = Path("results")

OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    results = data["results"]

    modes = [r["mode"] for r in results]

    throughput = [r["throughput_qps"] for r in results]
    latency = [r["latency_mean_ms"] for r in results]
    enqueue = [r["enqueue_mean_ms"] for r in results]
    gpu_compute = [r["gpu_compute_mean_ms"] for r in results]

    # Throughput plot
    plt.figure(figsize=(8, 5))
    plt.bar(modes, throughput)
    plt.ylabel("Queries Per Second")
    plt.title("TensorRT Throughput Comparison")
    plt.tight_layout()

    throughput_path = OUTPUT_DIR / "tensorrt_throughput.png"
    plt.savefig(throughput_path, dpi=300)
    plt.close()

    # Latency plot
    plt.figure(figsize=(8, 5))
    plt.bar(modes, latency)
    plt.ylabel("Mean Latency (ms)")
    plt.title("TensorRT Mean Latency Comparison")
    plt.tight_layout()

    latency_path = OUTPUT_DIR / "tensorrt_latency.png"
    plt.savefig(latency_path, dpi=300)
    plt.close()

    # Enqueue plot
    plt.figure(figsize=(8, 5))
    plt.bar(modes, enqueue)
    plt.ylabel("Enqueue Time (ms)")
    plt.title("TensorRT Enqueue Overhead")
    plt.tight_layout()

    enqueue_path = OUTPUT_DIR / "tensorrt_enqueue.png"
    plt.savefig(enqueue_path, dpi=300)
    plt.close()

    # GPU compute plot
    plt.figure(figsize=(8, 5))
    plt.bar(modes, gpu_compute)
    plt.ylabel("GPU Compute Time (ms)")
    plt.title("TensorRT GPU Compute Time")
    plt.tight_layout()

    compute_path = OUTPUT_DIR / "tensorrt_gpu_compute.png"
    plt.savefig(compute_path, dpi=300)
    plt.close()

    print("Saved:")
    print(throughput_path)
    print(latency_path)
    print(enqueue_path)
    print(compute_path)


if __name__ == "__main__":
    main()