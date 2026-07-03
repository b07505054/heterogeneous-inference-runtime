#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROMPT_SETS = {
    "small": [
        "Summarize the main idea of edge inference in one paragraph.",
        "List three practical reasons to benchmark model latency.",
        "Explain what TTFT means for an LLM serving system.",
        "Give a concise definition of tokens per second.",
    ],
    "coding": [
        "Write a Python function that computes p95 latency from a list of floats.",
        "Show a small JSONL parser that skips blank lines.",
        "Explain how to use a thread pool for concurrent HTTP requests.",
        "Write pseudocode for measuring cold start and warm start latency.",
    ],
    "mixed": [
        "Summarize the tradeoff between latency and throughput.",
        "Write a Python dataclass for a benchmark request.",
        "Explain why measured evidence should include hardware metadata.",
        "Draft a short incident note for a server timeout during benchmarking.",
        "Create a checklist for validating a saved benchmark artifact.",
        "Describe how compression might affect model package size and drift.",
    ],
}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic OpenAI-compatible LLM request traces.")
    parser.add_argument("--output", default="traces/llm_request_trace.jsonl")
    parser.add_argument("--num-requests", type=positive_int, default=32)
    parser.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), default="small")
    parser.add_argument("--max-tokens", type=positive_int, default=64)
    parser.add_argument("--arrival-pattern", choices=["uniform", "burst"], default="uniform")
    parser.add_argument("--seed", type=non_negative_int, default=0)
    return parser


def generate_rows(
    *,
    num_requests: int,
    prompt_set: str,
    max_tokens: int,
    arrival_pattern: str,
    seed: int,
) -> list[dict]:
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"unknown prompt set: {prompt_set}")
    rng = random.Random(seed)
    prompts = PROMPT_SETS[prompt_set]
    rows = []
    for index in range(num_requests):
        rows.append(
            {
                "request_id": f"req_{index:04d}",
                "prompt": rng.choice(prompts),
                "max_tokens": max_tokens,
                "arrival_ms": arrival_ms(index, arrival_pattern),
            }
        )
    return rows


def arrival_ms(index: int, pattern: str) -> int:
    if pattern == "uniform":
        return index * 100
    if pattern == "burst":
        burst_index = index // 8
        within_burst = index % 8
        return burst_index * 1000 + within_burst * 10
    raise ValueError(f"unknown arrival pattern: {pattern}")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    rows = generate_rows(
        num_requests=args.num_requests,
        prompt_set=args.prompt_set,
        max_tokens=args.max_tokens,
        arrival_pattern=args.arrival_pattern,
        seed=args.seed,
    )
    write_jsonl(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
