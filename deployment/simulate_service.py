"""Local HTTP service exposing the existing RuntimeScheduler/PagedKVLifecycle
simulation as POST /simulate.

All responses are result_type="simulated": this runs the same deterministic
discrete-event RuntimeScheduler used by scripts/generate_llm_runtime_artifacts.py,
not a live GPU/CUDA kernel. Stdlib only, no new dependency.
"""

import argparse
import json
import random
import subprocess
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.llm_runtime_decision import (  # noqa: E402
    CostModel,
    MemoryPlanner,
    Request,
    RuntimeScheduler,
    summarize_policy,
)

DEFAULT_POLICY = "inflight_paged_kv_continuous_batching"
TOTAL_BLOCKS = 512
BLOCK_SIZE_TOKENS = 16
KV_MB_PER_BLOCK = 3.125


def git_commit() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return output.decode("utf-8").strip()


def run_simulation(
    prompt_tokens: int,
    max_output_tokens: int,
    request_id: str,
    policy: str,
    seed: int = 0,
) -> dict:
    request = Request(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=max_output_tokens,
        arrival_ms=0.0,
    )
    scheduler = RuntimeScheduler(
        policy=policy,
        cost_model=CostModel(),
        memory=MemoryPlanner(
            total_blocks=TOTAL_BLOCKS,
            block_size_tokens=BLOCK_SIZE_TOKENS,
            kv_mb_per_block=KV_MB_PER_BLOCK,
        ),
        rng=random.Random(seed),
    )
    result = scheduler.run([request])
    summary = summarize_policy(result)
    return {
        "request_id": request_id,
        "result_type": "simulated",
        "policy": policy,
        "git_commit": git_commit(),
        "ttft_ms": summary["ttft_p95_ms"],
        "tpot_ms": summary["tpot_p95_ms"],
        "e2e_latency_ms": summary["p95_latency_ms"],
        "kv_page_lifecycle": result.kv_page_lifecycle or {},
    }


class SimulateHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/simulate":
            self._send_json({"error": "not_found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, status=400)
            return

        if not isinstance(payload, dict):
            self._send_json({"error": "payload must be a JSON object"}, status=400)
            return

        try:
            prompt_tokens = int(payload.get("prompt_tokens", 512))
            max_output_tokens = int(payload.get("max_output_tokens", 64))
        except (TypeError, ValueError):
            self._send_json(
                {"error": "prompt_tokens and max_output_tokens must be integers"},
                status=400,
            )
            return
        if prompt_tokens <= 0 or max_output_tokens <= 0:
            self._send_json(
                {"error": "prompt_tokens and max_output_tokens must be positive"},
                status=400,
            )
            return

        request_id = payload.get("request_id") or f"sim-{uuid.uuid4().hex[:8]}"
        policy = payload.get("policy") or DEFAULT_POLICY

        try:
            result = run_simulation(prompt_tokens, max_output_tokens, request_id, policy)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)
            return

        self._send_json(result)

    def log_message(self, fmt, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local simulated runtime service (RuntimeScheduler/PagedKVLifecycle)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8901)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SimulateHandler)
    print(f"simulate service: http://{args.host}:{args.port}/simulate")
    print("All responses are result_type=simulated (RuntimeScheduler/PagedKVLifecycle), not measured GPU inference.")
    server.serve_forever()


if __name__ == "__main__":
    main()
