import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    input_path = Path("results/executorch_thread_scaling.json")
    output_path = Path("results/executorch_thread_scaling.png")

    data = json.loads(input_path.read_text(encoding="utf-8"))

    threads = [row["threads"] for row in data]
    latency = [row["avg_latency_ms"] for row in data]

    plt.figure()
    plt.plot(threads, latency, marker="o")

    plt.xlabel("CPU Threads")
    plt.ylabel("Average Latency (ms)")
    plt.title("ExecuTorch C++ Runtime Thread Scaling")
    plt.xticks(threads)
    plt.grid(True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()