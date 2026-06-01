import json
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import os


def classify_op(name: str) -> str:
    name = name.lower()

    if "conv" in name:
        return "Conv"
    if "gemm" in name or "linear" in name or "matmul" in name:
        return "Gemm / Linear"
    if "hardtanh" in name or "relu" in name or "clip" in name:
        return "Activation"
    if "pool" in name:
        return "Pooling"
    if "add" in name or "sum" in name:
        return "Add"
    if "softmax" in name:
        return "Softmax"

    return "Other"


def main(path):
    with open(path) as f:
        data = json.load(f)

    op_time = defaultdict(float)

    for e in data:
        if e.get("cat") == "Node":
            name = e.get("name", "unknown")
            dur = e.get("dur", 0)
            op_type = classify_op(name)
            op_time[op_type] += dur

    total = sum(op_time.values())

    labels = []
    sizes = []

    for k, v in sorted(op_time.items(), key=lambda x: -x[1]):
        labels.append(k)
        sizes.append(v / total * 100)

    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title("Operator-level Latency Breakdown (ONNX Runtime)")

    os.makedirs("results", exist_ok=True)
    output_path = "results/operator_breakdown.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1])