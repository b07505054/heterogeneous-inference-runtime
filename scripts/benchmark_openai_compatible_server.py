#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.backends.openai_compatible import OpenAICompatibleBackend, OpenAICompatibleConfig
from benchmark.exporters import measured_envelope, write_json
from benchmark.metrics import openai_latency_metrics
from benchmark.runner import BenchmarkRunner
from benchmark.trace import load_jsonl_trace, normalize_openai_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible server.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True, help="JSONL trace with messages or prompt rows.")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output", default="results/measured_baselines/openai_compatible_server.json")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--claimed-server", default=None, help="Optional label, e.g. vllm.")
    args = parser.parse_args()

    trace_rows = normalize_openai_trace(load_jsonl_trace(args.trace))
    config = OpenAICompatibleConfig(
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
        endpoint=args.endpoint,
        stream=not args.no_stream,
    )
    backend = OpenAICompatibleBackend(config)
    server_metadata = backend.fetch_model_metadata()
    runner = BenchmarkRunner(
        measure_fn=backend.execute,
        finalize_fn=lambda rows: {
            "metrics": openai_latency_metrics(rows),
            "request_results": rows,
        },
        export_fn=lambda payload: write_json(args.output, payload),
        concurrency=args.concurrency,
    )
    runner.warmup(trace_rows[: args.warmup])
    runner.run(trace_rows[args.warmup :])
    result = runner.finalize()

    notes = [
        "Generic OpenAI-compatible benchmark client only; it does not install, start, stop, or manage the server.",
    ]
    extra = {
        "server_metadata": server_metadata,
        "request_results": result["request_results"],
    }
    if args.claimed_server:
        extra["claimed_server"] = args.claimed_server

    payload = measured_envelope(
        artifact_type="openai_compatible_server_baseline",
        benchmark_target={
            "kind": "openai_compatible_server",
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "model": args.model,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "stream": not args.no_stream,
        },
        metrics=result["metrics"],
        notes=notes,
        command=sys.argv,
        extra=extra,
    )
    runner.export(payload)
    print(args.output)


if __name__ == "__main__":
    main()
