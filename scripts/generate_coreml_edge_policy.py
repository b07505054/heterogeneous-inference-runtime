#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment.coreml_edge_policy import (
    generate_coreml_edge_policy,
    load_capability_profile,
    write_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a CoreML edge deployment policy from measured baseline artifacts."
    )
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--capability-profile", default=None)
    parser.add_argument("--max-p95-ms", type=float, required=True)
    parser.add_argument("--max-package-mb", type=float, required=True)
    parser.add_argument("--max-drift", type=float, required=True)
    parser.add_argument("--prefer", choices=["latency", "size", "memory"], required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = generate_coreml_edge_policy(
        args.baselines,
        max_p95_ms=args.max_p95_ms,
        max_package_mb=args.max_package_mb,
        max_drift=args.max_drift,
        prefer=args.prefer,
        capability_profile=load_capability_profile(args.capability_profile),
    )
    write_policy(args.output, policy)
    print(args.output)


if __name__ == "__main__":
    main()
