#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment.server_runtime_policy import generate_server_runtime_policy, write_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a server runtime policy from measured OpenAI-compatible baseline artifacts."
    )
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--max-ttft-p95-ms", type=float, required=True)
    parser.add_argument("--max-tpot-p95-ms", type=float, required=True)
    parser.add_argument("--max-e2e-p95-ms", type=float, required=True)
    parser.add_argument("--prefer", choices=["latency", "throughput"], required=True)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = generate_server_runtime_policy(
        args.baselines,
        max_ttft_p95_ms=args.max_ttft_p95_ms,
        max_tpot_p95_ms=args.max_tpot_p95_ms,
        max_e2e_p95_ms=args.max_e2e_p95_ms,
        prefer=args.prefer,
        allow_errors=args.allow_errors,
    )
    write_policy(args.output, policy)
    print(args.output)


if __name__ == "__main__":
    main()
