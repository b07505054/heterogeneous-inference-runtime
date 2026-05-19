import json
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = Path("results/backend_validation_summary.json")
OUTPUT_PATH = Path("results/backend_validation_latency.png")


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    labels = [f'{r["backend"]}\n{r["precision"]}' for r in data]
    latencies = [r["avg_latency_ms"] for r in data]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, latencies)

    plt.ylabel("Average Latency (ms)")
    plt.title("Backend Validation Latency Comparison")
    plt.grid(axis="y", alpha=0.3)

    for bar, latency in zip(bars, latencies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{latency:.2f} ms",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()