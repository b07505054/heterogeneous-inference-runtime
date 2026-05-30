import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-json", default="results/onnx_operator_profile_summary.json")
    args = parser.parse_args()

    with open(args.profile, "r", encoding="utf-8") as f:
        events = json.load(f)

    op_time = defaultdict(float)
    op_count = defaultdict(int)

    for event in events:
        args_data = event.get("args", {})
        op_name = args_data.get("op_name")
        duration_us = event.get("dur", 0)

        if op_name:
            op_time[op_name] += duration_us / 1000.0
            op_count[op_name] += 1

    summary = []

    print("=== ONNX Runtime Operator Profile ===")

    for op, total_ms in sorted(op_time.items(), key=lambda item: item[1], reverse=True):
        avg_ms = total_ms / op_count[op]

        row = {
            "op": op,
            "total_ms": total_ms,
            "count": op_count[op],
            "avg_ms": avg_ms,
        }
        summary.append(row)

        print(
            f"{op:30s} "
            f"total={total_ms:.4f} ms "
            f"count={op_count[op]} "
            f"avg={avg_ms:.4f} ms"
        )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary to: {args.output_json}")


if __name__ == "__main__":
    main()