"""Local HTTP service exposing the existing RuntimeScheduler/PagedKVLifecycle
simulation as POST /simulate (single request) and POST /simulate_batch
(multiple requests sharing one RuntimeScheduler.run() call).

All responses are result_type="simulated": this runs the same deterministic
discrete-event RuntimeScheduler used by scripts/generate_llm_runtime_artifacts.py,
not a live GPU/CUDA kernel. Stdlib only, no new dependency.

/simulate_batch surfaces real cross-request pressure dynamics (rejections,
OOM events, pressure-driven prefetch skips) because RuntimeScheduler.run()
shares one PagedKVLifecycle across every request in the batch. It does NOT
make usefulness_score land between 0 and 1, decay usefulness_score_ema,
activate adaptive_guard_active, or raise adaptive_prefetch_skips: those all
require a request to be released while still holding an unconsumed
speculative prefetch, and RuntimeScheduler.run() always runs every request
to a terminal state (and releases its pages) before returning, by
construction. Confirmed empirically across 160+ workload/seed combinations
before adding this endpoint.
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


def _build_scheduler(policy: str, seed: int = 0) -> RuntimeScheduler:
    return RuntimeScheduler(
        policy=policy,
        cost_model=CostModel(),
        memory=MemoryPlanner(
            total_blocks=TOTAL_BLOCKS,
            block_size_tokens=BLOCK_SIZE_TOKENS,
            kv_mb_per_block=KV_MB_PER_BLOCK,
        ),
        rng=random.Random(seed),
    )


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
    result = _build_scheduler(policy, seed).run([request])
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


def parse_batch_requests(raw_requests) -> list:
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("requests must be a non-empty list")

    parsed = []
    for item in raw_requests:
        if not isinstance(item, dict):
            raise ValueError("each entry in requests must be a JSON object")
        try:
            prompt_tokens = int(item.get("prompt_tokens", 512))
            max_output_tokens = int(item.get("max_output_tokens", 64))
            arrival_ms = float(item.get("arrival_ms", 0.0))
        except (TypeError, ValueError):
            raise ValueError(
                "prompt_tokens, max_output_tokens, and arrival_ms must be numeric"
            )
        if prompt_tokens <= 0 or max_output_tokens <= 0:
            raise ValueError("prompt_tokens and max_output_tokens must be positive")
        request_id = item.get("request_id") or f"sim-{uuid.uuid4().hex[:8]}"
        parsed.append(
            Request(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=max_output_tokens,
                arrival_ms=arrival_ms,
            )
        )
    return parsed


def run_batch_simulation(requests: list, policy: str, seed: int = 0) -> dict:
    result = _build_scheduler(policy, seed).run(requests)
    summary = summarize_policy(result)
    return {
        "result_type": "simulated",
        "mode": "batch",
        "policy": policy,
        "git_commit": git_commit(),
        "request_count": len(requests),
        "ttft_ms": summary["ttft_p95_ms"],
        "tpot_ms": summary["tpot_p95_ms"],
        "e2e_latency_ms": summary["p95_latency_ms"],
        "rejected_requests": summary["rejected_requests"],
        "oom_events": summary["reject_oom_count"],
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
        if self.path not in ("/simulate", "/simulate_batch"):
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

        if self.path == "/simulate":
            self._handle_simulate(payload)
        else:
            self._handle_simulate_batch(payload)

    def _handle_simulate(self, payload: dict) -> None:
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

    def _handle_simulate_batch(self, payload: dict) -> None:
        policy = payload.get("policy") or DEFAULT_POLICY

        try:
            requests = parse_batch_requests(payload.get("requests"))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        try:
            result = run_batch_simulation(requests, policy)
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
    print(f"simulate batch service: http://{args.host}:{args.port}/simulate_batch")
    print("All responses are result_type=simulated (RuntimeScheduler/PagedKVLifecycle), not measured GPU inference.")
    server.serve_forever()


if __name__ == "__main__":
    main()
