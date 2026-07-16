#!/usr/bin/env python3
"""Run the resumable clean-server max_num_seqs measurement matrix."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ORDERS = {
    0: ("4", "default", "1", "2", "8"),
    1: ("4", "8", "default", "1", "2"),
    2: ("2", "4", "8", "default", "1"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--order-log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    records = []
    if args.order_log.exists():
        records = json.loads(args.order_log.read_text())
    completed = {(x["workload"], x["candidate"], x["session"]) for x in records if x["status"] == "completed"}
    for session, order in ORDERS.items():
        for workload in ("S1", "S2", "S3"):
            for candidate in order:
                key = (workload, candidate, session)
                out = args.raw_dir / f"{workload}-{candidate}-{session}.json"
                if key in completed and out.exists():
                    continue
                started = time.time()
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("run_vllm_max_num_seqs_session.py")),
                    "--plan", str(args.plans / f"{workload}-{candidate}.json"),
                    "--workload-manifest", str(args.workload_manifest),
                    "--workload", workload,
                    "--session", str(session),
                    "--port", str(args.port),
                    "--out", str(out),
                    "--log", str(args.log_dir / f"{workload}-{candidate}-{session}.log"),
                    "--startup-timeout", "240",
                ]
                status = "completed"
                error = None
                try:
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as exc:
                    status = "runner_failed"
                    error = str(exc)
                records.append({"workload": workload, "candidate": candidate, "session": session, "started_unix": started, "elapsed_seconds": time.time() - started, "status": status, "error": error})
                args.order_log.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
                if status != "completed":
                    raise SystemExit(error)


if __name__ == "__main__":
    main()
