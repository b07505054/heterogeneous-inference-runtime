"""Local HTTP service exposing the existing RuntimeScheduler/PagedKVLifecycle
simulation as POST /simulate (single request), POST /simulate_batch
(multiple requests sharing one RuntimeScheduler.run() call), and a stateful
POST /session family (caller-driven, multi-call KV lifecycle inspection).

All responses are result_type="simulated": this runs the same deterministic
discrete-event PagedKVLifecycle/RuntimeScheduler primitives used by
scripts/generate_llm_runtime_artifacts.py, not a live GPU/CUDA kernel.
Stdlib only, no new dependency.

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

The /session endpoints exist to make those mid-flight fields observable:
POST /session creates a session backed by a single long-lived
PagedKVLifecycle; POST /session/{id}/request admits one request (prefill
only); POST /session/{id}/step advances one decode tick across every
resident request using a plain round-robin queue (no priority, no batch
cap, no pressure-based admission delay); POST /session/{id}/cancel lets the
caller explicitly release a still-resident request's pages, which is the
only way prefetch waste / EMA decay / adaptive-guard activation /
adaptive_prefetch_skips become reachable. The server never evicts a
request on its own to make room or to manufacture metrics -- admission
rejects outright (insufficient_free_kv_pages) when there isn't room, the
same way a real admission-time rejection would, and every other state
change is a direct, unmodified call into PagedKVLifecycle in response to
an explicit caller action.
"""

import argparse
import json
import random
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.llm_runtime_decision import (  # noqa: E402
    CostModel,
    MemoryPlanner,
    PagedKVLifecycle,
    Request,
    RuntimeRequestState,
    RuntimeScheduler,
    summarize_policy,
)

DEFAULT_POLICY = "inflight_paged_kv_continuous_batching"
TOTAL_BLOCKS = 512
BLOCK_SIZE_TOKENS = 16
KV_MB_PER_BLOCK = 3.125

# Mirrors _run_inflight_paged_kv's prefetch_disable_threshold; fixed, not
# exposed through the session HTTP API.
SESSION_PREFETCH_DISABLE_THRESHOLD = 0.78


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


@dataclass
class SimulatedSession:
    """Caller-driven, multi-call wrapper around one PagedKVLifecycle.

    No eviction logic lives here: admission rejects outright when KV is
    full, and the only way a resident request loses its pages early is an
    explicit cancel_request() call from the caller.
    """

    kv: PagedKVLifecycle
    cost_model: CostModel = field(default_factory=CostModel)
    states: dict = field(default_factory=dict)
    decode_order: list = field(default_factory=list)
    finished_ids: set = field(default_factory=set)
    rejected_ids: set = field(default_factory=set)
    cancelled_ids: set = field(default_factory=set)
    ticks_elapsed: int = 0
    requests_submitted: int = 0

    def summary(self) -> dict:
        return {
            "result_type": "simulated",
            "note": "deterministic PagedKVLifecycle session simulation, not measured GPU inference",
            "resident_request_ids": list(self.decode_order),
            "finished_request_ids": sorted(self.finished_ids),
            "rejected_request_ids": sorted(self.rejected_ids),
            "cancelled_request_ids": sorted(self.cancelled_ids),
            "requests_submitted": self.requests_submitted,
            "ticks_elapsed": self.ticks_elapsed,
            "kv_page_lifecycle": self.kv.summary(),
        }


_SESSIONS: dict[str, SimulatedSession] = {}
_SESSIONS_LOCK = threading.Lock()


def create_session(
    total_pages: int = TOTAL_BLOCKS,
    page_size_tokens: int = BLOCK_SIZE_TOKENS,
    kv_mb_per_page: float = KV_MB_PER_BLOCK,
) -> str:
    kv = PagedKVLifecycle(
        total_pages=total_pages,
        page_size_tokens=page_size_tokens,
        kv_mb_per_page=kv_mb_per_page,
    )
    session_id = f"sess-{uuid.uuid4().hex[:12]}"
    with _SESSIONS_LOCK:
        _SESSIONS[session_id] = SimulatedSession(kv=kv)
    return session_id


def get_session(session_id: str) -> SimulatedSession | None:
    with _SESSIONS_LOCK:
        return _SESSIONS.get(session_id)


def delete_session(session_id: str) -> bool:
    with _SESSIONS_LOCK:
        return _SESSIONS.pop(session_id, None) is not None


def admit_request(session: SimulatedSession, request: Request) -> dict:
    """Prefill-only admission. Rejects outright; never evicts another request."""
    kv = session.kv
    session.requests_submitted += 1

    total_needed = kv.pages_needed_for_tokens(request.prompt_tokens + request.output_tokens)
    if total_needed > kv.total_pages:
        session.rejected_ids.add(request.request_id)
        return {"admitted": False, "reason": "request_exceeds_total_kv_pages"}

    prefill_needed = kv.pages_needed_for_tokens(request.prompt_tokens)
    if prefill_needed > len(kv.free_pages):
        session.rejected_ids.add(request.request_id)
        return {"admitted": False, "reason": "insufficient_free_kv_pages"}

    allocated = kv.allocate_range(
        request_id=request.request_id,
        token_begin=0,
        token_end=request.prompt_tokens,
        step=session.ticks_elapsed,
        initial_state="resident",
    )
    state = RuntimeRequestState(
        request=request,
        state="decode",
        prompt_tokens_prefilled=request.prompt_tokens,
        kv_pages=list(allocated),
    )
    session.states[request.request_id] = state
    session.decode_order.append(request.request_id)
    return {"admitted": True, "reason": "fits_session_kv_budget", "allocated_pages": allocated}


def cancel_request(session: SimulatedSession, request_id: str) -> dict:
    """Direct, caller-requested kv.release_request() call. No victim selection."""
    if request_id not in session.decode_order:
        if request_id in (session.finished_ids | session.rejected_ids | session.cancelled_ids):
            return {"error": "request_already_terminal"}
        return {"error": "unknown_request_id"}

    session.decode_order.remove(request_id)
    freed = session.kv.release_request(request_id)
    state = session.states.get(request_id)
    if state is not None:
        state.state = "cancelled"
        state.kv_pages = []
    session.cancelled_ids.add(request_id)
    return {"cancelled": True, "freed_pages": freed}


def step_session(session: SimulatedSession) -> dict:
    """Advance one decode tick using a plain round-robin policy: every
    currently resident request is visited once, in queue order, then the
    survivors are rotated to the back of the queue for the next tick.
    """
    kv = session.kv
    snapshot = list(session.decode_order)
    events: list[dict] = []
    survivors: list[str] = []

    for request_id in snapshot:
        state = session.states[request_id]
        token_index = state.prompt_tokens_total + state.output_tokens_generated

        if token_index % kv.page_size_tokens == 0 and not kv.has_page_for_token(request_id, token_index):
            token_end = min(
                state.prompt_tokens_total + state.request.output_tokens,
                token_index + kv.page_size_tokens,
            )
            try:
                pages = kv.allocate_range(
                    request_id=request_id,
                    token_begin=token_index,
                    token_end=token_end,
                    step=session.ticks_elapsed,
                )
                state.kv_pages.extend(pages)
            except MemoryError:
                kv.release_request(request_id)
                state.kv_pages = []
                state.state = "rejected"
                session.rejected_ids.add(request_id)
                events.append(
                    {
                        "request_id": request_id,
                        "event": "request_rejected",
                        "reason": "oom_during_decode_kv_growth",
                    }
                )
                continue

        access = kv.access_current_page(request_id, session.ticks_elapsed)
        if access["page_id"] is None:
            kv.release_request(request_id)
            state.kv_pages = []
            state.state = "rejected"
            session.rejected_ids.add(request_id)
            events.append(
                {
                    "request_id": request_id,
                    "event": "request_rejected",
                    "reason": "decode_without_resident_page",
                }
            )
            continue

        state.output_tokens_generated += 1
        if state.output_tokens_generated >= state.request.output_tokens:
            freed = kv.release_request(request_id)
            state.kv_pages = []
            state.state = "finished"
            session.finished_ids.add(request_id)
            events.append(
                {
                    "request_id": request_id,
                    "event": "request_finished",
                    "freed_pages": freed,
                }
            )
            continue

        prefetch = kv.prefetch_next_decode_page(
            request_id=request_id,
            step=session.ticks_elapsed,
            pressure_disable_threshold=SESSION_PREFETCH_DISABLE_THRESHOLD,
            token_index=token_index,
            request_token_budget=state.prompt_tokens_total + state.request.output_tokens,
        )
        prefetched_pages = prefetch.get("prefetched_pages") or []
        if prefetched_pages:
            state.kv_pages.extend(prefetched_pages)
        events.append({"request_id": request_id, **prefetch})
        survivors.append(request_id)

    session.decode_order = survivors
    session.ticks_elapsed += 1
    return {"events": events}


class SimulateHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def _parse_session_path(path: str) -> tuple | None:
    """Returns (session_id, action) for '/session/{id}/{action}', or
    (session_id, None) for '/session/{id}', or None if path doesn't match.
    """
    parts = path.strip("/").split("/")
    if len(parts) == 2 and parts[0] == "session" and parts[1]:
        return parts[1], None
    if len(parts) == 3 and parts[0] == "session" and parts[1]:
        return parts[1], parts[2]
    return None


class SimulateHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_payload(self) -> tuple:
        """Returns (payload, error) - error is an (status, body) tuple to
        send, or None if the payload was read successfully."""
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            return None, (400, {"error": "invalid_json"})
        if not isinstance(payload, dict):
            return None, (400, {"error": "payload must be a JSON object"})
        return payload, None

    def do_POST(self) -> None:
        if self.path in ("/simulate", "/simulate_batch"):
            payload, error = self._read_json_payload()
            if error:
                self._send_json(error[1], status=error[0])
                return
            if self.path == "/simulate":
                self._handle_simulate(payload)
            else:
                self._handle_simulate_batch(payload)
            return

        if self.path == "/session":
            payload, error = self._read_json_payload()
            if error:
                self._send_json(error[1], status=error[0])
                return
            self._handle_session_create(payload)
            return

        parsed = _parse_session_path(self.path)
        if parsed and parsed[1] in ("request", "step", "cancel"):
            session_id, action = parsed
            payload, error = self._read_json_payload()
            if error:
                self._send_json(error[1], status=error[0])
                return
            self._handle_session_action(session_id, action, payload)
            return

        self._send_json({"error": "not_found"}, status=404)

    def do_GET(self) -> None:
        parsed = _parse_session_path(self.path)
        if parsed and parsed[1] == "summary":
            session_id, _ = parsed
            session = get_session(session_id)
            if session is None:
                self._send_json({"error": "unknown_session_id"}, status=404)
                return
            self._send_json(session.summary())
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_DELETE(self) -> None:
        parsed = _parse_session_path(self.path)
        if parsed and parsed[1] is None:
            session_id, _ = parsed
            if not delete_session(session_id):
                self._send_json({"error": "unknown_session_id"}, status=404)
                return
            self._send_json({"deleted": True, "session_id": session_id, "result_type": "simulated"})
            return
        self._send_json({"error": "not_found"}, status=404)

    def _handle_session_create(self, payload: dict) -> None:
        try:
            total_pages = int(payload.get("total_pages", TOTAL_BLOCKS))
            page_size_tokens = int(payload.get("page_size_tokens", BLOCK_SIZE_TOKENS))
            kv_mb_per_page = float(payload.get("kv_mb_per_page", KV_MB_PER_BLOCK))
        except (TypeError, ValueError):
            self._send_json(
                {"error": "total_pages and page_size_tokens must be integers, kv_mb_per_page numeric"},
                status=400,
            )
            return
        if total_pages <= 0 or page_size_tokens <= 0 or kv_mb_per_page <= 0:
            self._send_json(
                {"error": "total_pages, page_size_tokens, and kv_mb_per_page must be positive"},
                status=400,
            )
            return

        session_id = create_session(total_pages, page_size_tokens, kv_mb_per_page)
        self._send_json(
            {
                "session_id": session_id,
                "result_type": "simulated",
                "note": "deterministic PagedKVLifecycle session simulation, not measured GPU inference",
            }
        )

    def _handle_session_action(self, session_id: str, action: str, payload: dict) -> None:
        session = get_session(session_id)
        if session is None:
            self._send_json({"error": "unknown_session_id"}, status=404)
            return

        if action == "request":
            self._handle_session_request(session, payload)
        elif action == "step":
            self._handle_session_step(session)
        else:
            self._handle_session_cancel(session, payload)

    def _handle_session_request(self, session: SimulatedSession, payload: dict) -> None:
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

        request_id = payload.get("request_id") or f"sess-req-{uuid.uuid4().hex[:8]}"
        if request_id in session.states:
            self._send_json({"error": "request_id already used in this session"}, status=400)
            return

        request = Request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            output_tokens=max_output_tokens,
            arrival_ms=0.0,
        )
        outcome = admit_request(session, request)
        self._send_json(
            {
                "request_id": request_id,
                "result_type": "simulated",
                "note": "deterministic PagedKVLifecycle session simulation, not measured GPU inference",
                **outcome,
                "kv_page_lifecycle": session.kv.summary(),
            }
        )

    def _handle_session_step(self, session: SimulatedSession) -> None:
        outcome = step_session(session)
        self._send_json(
            {
                "result_type": "simulated",
                "note": "deterministic PagedKVLifecycle session simulation, not measured GPU inference",
                **outcome,
                "kv_page_lifecycle": session.kv.summary(),
            }
        )

    def _handle_session_cancel(self, session: SimulatedSession, payload: dict) -> None:
        request_id = payload.get("request_id")
        if not request_id:
            self._send_json({"error": "request_id is required"}, status=400)
            return
        outcome = cancel_request(session, request_id)
        if outcome.get("error") == "unknown_request_id":
            self._send_json({"error": outcome["error"]}, status=404)
            return
        if outcome.get("error") == "request_already_terminal":
            self._send_json({"error": outcome["error"]}, status=409)
            return
        self._send_json(
            {
                "request_id": request_id,
                "result_type": "simulated",
                "note": "deterministic PagedKVLifecycle session simulation, not measured GPU inference",
                **outcome,
                "kv_page_lifecycle": session.kv.summary(),
            }
        )

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
    print(f"simulate session service: http://{args.host}:{args.port}/session")
    print("All responses are result_type=simulated (RuntimeScheduler/PagedKVLifecycle), not measured GPU inference.")
    server.serve_forever()


if __name__ == "__main__":
    main()
