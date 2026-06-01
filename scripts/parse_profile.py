import json
import sys
from collections import defaultdict


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

    print("Operator-level latency breakdown:")
    for k, v in sorted(op_time.items(), key=lambda x: -x[1]):
        print(f"{k}: {v / total * 100:.2f}%")

    print("\nTop raw nodes:")
    raw_time = defaultdict(float)
    for e in data:
        if e.get("cat") == "Node":
            raw_time[e.get("name", "unknown")] += e.get("dur", 0)

    for k, v in sorted(raw_time.items(), key=lambda x: -x[1])[:10]:
        print(f"{k}: {v / total * 100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1])