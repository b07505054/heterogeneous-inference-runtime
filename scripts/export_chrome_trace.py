import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--output-trace",
        default="results/chrome_trace_onnx_cpu.json",
    )
    parser.add_argument(
        "--output-summary",
        default="results/chrome_trace_onnx_cpu_summary.json",
    )
    args = parser.parse_args()

    profile_path = Path(args.profile)
    output_trace = Path(args.output_trace)
    output_summary = Path(args.output_summary)

    with profile_path.open("r", encoding="utf-8") as f:
        events = json.load(f)

    output_trace.parent.mkdir(parents=True, exist_ok=True)

    with output_trace.open("w", encoding="utf-8") as f:
        json.dump(events, f)

    op_time_ms = defaultdict(float)
    op_count = defaultdict(int)
    category_count = defaultdict(int)

    total_duration_us = 0

    for event in events:
        category = event.get("cat", "unknown")
        category_count[category] += 1

        duration_us = event.get("dur", 0)
        total_duration_us += duration_us

        args_data = event.get("args", {})
        op_name = args_data.get("op_name")

        if op_name:
            op_time_ms[op_name] += duration_us / 1000.0
            op_count[op_name] += 1

    top_ops = []

    for op, total_ms in sorted(
        op_time_ms.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        top_ops.append(
            {
                "op": op,
                "total_ms": total_ms,
                "count": op_count[op],
                "avg_ms": total_ms / op_count[op],
            }
        )

    summary = {
        "source_profile": str(profile_path),
        "chrome_trace": str(output_trace),
        "event_count": len(events),
        "total_recorded_duration_ms": total_duration_us / 1000.0,
        "categories": dict(category_count),
        "top_ops": top_ops,
    }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved Chrome Trace to: {output_trace}")
    print(f"Saved trace summary to: {output_summary}")

    print("\nTop operators:")
    for row in top_ops:
        print(
            f"{row['op']:30s} "
            f"total={row['total_ms']:.4f} ms "
            f"count={row['count']} "
            f"avg={row['avg_ms']:.4f} ms"
        )


if __name__ == "__main__":
    main()