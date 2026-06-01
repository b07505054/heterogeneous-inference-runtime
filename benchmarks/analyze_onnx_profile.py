import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    with open(args.profile, "r", encoding="utf-8") as f:
        events = json.load(f)

    op_time = defaultdict(float)
    op_count = defaultdict(int)

    for e in events:
        args_data = e.get("args", {})
        op_name = args_data.get("op_name")
        dur = e.get("dur", 0)

        if op_name:
            op_time[op_name] += dur / 1000  # microseconds to ms
            op_count[op_name] += 1

    print("=== ONNX Runtime Operator Profile ===")
    for op, total_ms in sorted(op_time.items(), key=lambda x: x[1], reverse=True):
        print(f"{op:30s} total={total_ms:.4f} ms count={op_count[op]}")


if __name__ == "__main__":
    main()