import json
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = Path("results/backend_validation_summary.json")
OUTPUT_PATH = Path("results/backend_validation_throughput.png")


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    labels = []
    qps = []

    for r in data:

        if r["backend"] == "ONNXRuntime":
            label = f'ORT\n{r["precision"]}'

        elif r["backend"] == "ThreadScaling":
            threads = r["extra"]["threads"]
            label = f"Thread\n{threads}T"

        elif r["backend"] == "CppInference":
            label = "C++\nFP32"

        else:
            label = f'{r["backend"]}\n{r["precision"]}'

        labels.append(label)
        qps.append(r["throughput_qps"])

    plt.figure(figsize=(14, 6))

    bars = plt.bar(labels, qps)

    plt.ylabel("Queries Per Second (QPS)")
    plt.title("Backend Validation Throughput Comparison")

    plt.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=15)

    for bar, value in zip(bars, qps):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=200)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()