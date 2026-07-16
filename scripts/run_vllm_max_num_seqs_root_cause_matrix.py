#!/usr/bin/env python3
"""Resumable randomized root-cause reproduction and client-concurrency sweep."""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path


def run(args, kind, workload, value, concurrency, session, mode="runtime_equivalent"):
    output = args.raw_dir / kind / f"{workload}-m{value}-c{concurrency}-s{session}.json"
    log = args.log_dir / kind / f"{workload}-m{value}-c{concurrency}-s{session}.log"
    if output.exists():
        print(f"SKIP {output.name}", flush=True)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = [sys.executable, str(Path(__file__).with_name("run_vllm_max_num_seqs_diagnostic.py")), "--fixed", str(args.fixed), "--workload-manifest", str(args.workload_manifest), "--workload", workload, "--max-num-seqs", str(value), "--client-concurrency", str(concurrency), "--session", str(session), "--port", str(args.port), "--out", str(output), "--log", str(log), "--mode", mode]
    print(f"START {kind} {workload} max={value} client={concurrency} session={session}", flush=True)
    subprocess.run(command, check=True)
    record = {"kind": kind, "workload": workload, "max_num_seqs": value, "client_concurrency": concurrency, "session": session, "mode": mode, "started_unix": started, "elapsed_seconds": time.time() - started, "output": str(output), "log": str(log)}
    records = json.loads(args.order_log.read_text()) if args.order_log.exists() else []
    records.append(record)
    args.order_log.parent.mkdir(parents=True, exist_ok=True)
    args.order_log.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"DONE {output.name} {record['elapsed_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--order-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    seed = 20260716
    values = [1, 2, 3, 4, 8]
    for session in range(5):
        for workload, concurrency in (("S2", 4), ("S3", 8)):
            order = values[:]
            random.Random(seed + session * 10 + concurrency).shuffle(order)
            for value in order:
                run(args, "reproduction", workload, value, concurrency, session)
    sweep = [(value, concurrency) for value in values for concurrency in (1, 2, 3, 4, 6, 8) if concurrency != 8]
    random.Random(seed).shuffle(sweep)
    for value, concurrency in sweep:
        run(args, "concurrency_sweep", "S3", value, concurrency, 0)
    for value in (2, 4):
        run(args, "direct_control", "S2", value, 4, 0, mode="direct_control")


if __name__ == "__main__":
    main()
