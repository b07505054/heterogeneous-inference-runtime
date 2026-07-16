#!/usr/bin/env python3
"""Execute each compiler-selected workload/objective plan in a fresh server."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_vllm_max_num_seqs_session.py")
    for workload in ("S1", "S2", "S3"):
        for objective in ("latency", "throughput", "balanced"):
            out = args.raw_dir / f"{workload}-{objective}.json"
            if out.exists():
                continue
            subprocess.run([
                sys.executable, str(runner),
                "--plan", str(args.plans / f"{workload}-{objective}.json"),
                "--workload-manifest", str(args.workload_manifest),
                "--workload", workload,
                "--session", "100",
                "--port", str(args.port),
                "--out", str(out),
                "--log", str(args.log_dir / f"{workload}-{objective}.log"),
                "--startup-timeout", "240",
            ], check=True)


if __name__ == "__main__":
    main()
