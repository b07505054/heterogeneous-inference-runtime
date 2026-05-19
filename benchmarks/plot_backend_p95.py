import json
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_PATH = Path("results/backend_validation_summary.json")
OUTPUT_PATH = Path("results/backend_validation_p95.png")


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    labels = []
    p95 = []

    for r in data:

        if r["backend"] == "ONNXRuntime":
            label = f"ORT\n{r['precision']}"

        elif r["backend"] == "ThreadScaling":
            threads = r["extra"]["threads"]
            label = f"Thread\n{threads}T"

        elif r["backend"] == "CppInference":
            label = "C++\nFP32"

        else:
            label = f"{r['backend']}\n{r['precision']}"

        value = r.get("p95_latency_ms")

        if value is None:
            continue

        labels.append(label)
        p95.append(value)

    plt.figure(figsize=(14, 6))

    bars = plt.bar(labels, p95)

    plt.ylabel("P95 Latency (ms)")
    plt.title("Backend Validation P95 Latency Comparison")

    plt.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=15)

    for bar, value in zip(bars, p95):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f} ms",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=200)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()