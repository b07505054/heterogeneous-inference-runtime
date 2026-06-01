import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "results" / "runtime_comparison.json"
    output_path = base_dir / "results" / "runtime_comparison_final.png"

    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = data["runtimes"]

    labels = [r["runtime"] for r in rows]
    backends = [r["backend"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]

    plt.figure(figsize=(9, 5))
    bars = plt.barh(labels, latencies)

    plt.xlabel("Average Latency (ms)")
    plt.title("Edge Inference Runtime Comparison (MobileNetV2, CPU)")
    plt.grid(axis="x", alpha=0.3)

    for bar, latency, backend in zip(bars, latencies, backends):
        plt.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{latency:.2f} ms  |  {backend}",
            va="center",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()