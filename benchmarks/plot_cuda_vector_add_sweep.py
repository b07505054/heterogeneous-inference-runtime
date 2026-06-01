import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "results" / "cuda_vector_add_block_sweep.csv"
    output_path = base_dir / "results" / "cuda_vector_add_block_sweep.png"

    block_sizes = []
    latencies = []

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_sizes.append(int(row["block_size"]))
            latencies.append(float(row["kernel_avg_ms"]))

    plt.figure(figsize=(8, 5))
    plt.plot(block_sizes, latencies, marker="o")
    plt.xlabel("CUDA Block Size")
    plt.ylabel("Kernel Avg Latency (ms)")
    plt.title("CUDA Vector Add Block Size Sweep")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()