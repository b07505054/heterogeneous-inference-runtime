#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment.planner.planner import plan_deployment, write_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deployment plan from measured evidence.")
    parser.add_argument("--profiles", nargs="*", default=[])
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--runtime", choices=["coreml", "server"], default=None)
    parser.add_argument(
        "--objective",
        choices=["latency", "throughput", "memory", "package_size", "size", "balanced"],
        default="latency",
    )
    parser.add_argument("--max-p95-ms", type=float, default=None)
    parser.add_argument("--max-latency-ms", type=float, default=None)
    parser.add_argument("--max-package-mb", type=float, default=None)
    parser.add_argument("--max-memory-mb", type=float, default=None)
    parser.add_argument("--max-rss-mb", type=float, default=None)
    parser.add_argument("--max-drift", type=float, default=None)
    parser.add_argument("--min-throughput", type=float, default=None)
    parser.add_argument("--min-tokens-per-second", type=float, default=None)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    constraints = {
        "max_p95_ms": args.max_p95_ms,
        "max_latency_ms": args.max_latency_ms,
        "max_package_mb": args.max_package_mb,
        "max_memory_mb": args.max_memory_mb,
        "max_rss_mb": args.max_rss_mb,
        "max_drift": args.max_drift,
        "min_throughput": args.min_throughput,
        "min_tokens_per_second": args.min_tokens_per_second,
    }
    plan = plan_deployment(
        profile_paths=_expand_json_paths(args.profiles),
        artifact_paths=_expand_json_paths(args.artifacts),
        runtime=args.runtime,
        constraints={key: value for key, value in constraints.items() if value is not None},
        objective=args.objective,
    )
    write_plan(args.output, plan)
    print(args.output)


def _expand_json_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*.json")))
        else:
            expanded.append(path)
    return expanded


if __name__ == "__main__":
    main()
