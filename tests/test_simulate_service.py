import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.simulate_service import (  # noqa: E402
    DEFAULT_POLICY,
    SimulatedSession,
    SimulateHandler,
    admit_request,
    cancel_request,
    step_session,
)
from deployment.llm_runtime_decision import PagedKVLifecycle, Request  # noqa: E402


class _ServiceFixture:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SimulateHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path="/simulate"):
        return f"http://127.0.0.1:{self.port}{path}"

    def _request(self, path, method, payload=None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.url(path),
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def post(self, payload, path="/simulate"):
        return self._request(path, "POST", payload if payload is not None else {})

    def get(self, path):
        return self._request(path, "GET")

    def delete(self, path):
        return self._request(path, "DELETE")

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_simulate_returns_simulated_result_type_and_kv_fields():
    fixture = _ServiceFixture()
    try:
        status, data = fixture.post({"prompt_tokens": 256, "max_output_tokens": 32})
        assert status == 200
        assert data["result_type"] == "simulated"
        assert data["policy"] == DEFAULT_POLICY
        assert "git_commit" in data
        assert "ttft_ms" in data
        assert "tpot_ms" in data
        assert "e2e_latency_ms" in data

        lifecycle = data["kv_page_lifecycle"]
        for key in (
            "allocated_pages",
            "freed_pages",
            "prefetch_attempts",
            "prefetch_hits",
            "prefetch_misses",
            "prefetch_hit_rate",
            "usefulness_score",
            "usefulness_score_ema",
            "adaptive_guard_active",
            "adaptive_prefetch_skips",
            "kv_internal_fragmentation_ratio",
            "contiguous_free_run_ratio",
        ):
            assert key in lifecycle

        # explicit type/value checks for the honest KV fields this
        # service exists to surface (not just key presence)
        assert isinstance(lifecycle["adaptive_guard_active"], bool)
        assert isinstance(lifecycle["adaptive_prefetch_skips"], int)
        assert lifecycle["adaptive_prefetch_skips"] >= 0
        assert isinstance(lifecycle["kv_internal_fragmentation_ratio"], (int, float))
        assert 0.0 <= lifecycle["kv_internal_fragmentation_ratio"] <= 1.0
        assert isinstance(lifecycle["contiguous_free_run_ratio"], (int, float))
        assert 0.0 <= lifecycle["contiguous_free_run_ratio"] <= 1.0
    finally:
        fixture.close()


def test_simulate_echoes_request_id():
    fixture = _ServiceFixture()
    try:
        status, data = fixture.post(
            {"prompt_tokens": 128, "max_output_tokens": 16, "request_id": "fixed-id"}
        )
        assert status == 200
        assert data["request_id"] == "fixed-id"
    finally:
        fixture.close()


def test_simulate_prompt_tokens_changes_metrics():
    fixture = _ServiceFixture()
    try:
        _, short = fixture.post({"prompt_tokens": 64, "max_output_tokens": 16})
        _, long = fixture.post({"prompt_tokens": 4096, "max_output_tokens": 16})
        assert short["ttft_ms"] != long["ttft_ms"]
    finally:
        fixture.close()


def test_simulate_rejects_non_positive_prompt_tokens():
    fixture = _ServiceFixture()
    try:
        status, data = fixture.post({"prompt_tokens": 0, "max_output_tokens": 16})
        assert status == 400
        assert "error" in data
    finally:
        fixture.close()


def test_simulate_rejects_invalid_json():
    fixture = _ServiceFixture()
    try:
        req = urllib.request.Request(
            fixture.url(),
            data=b"{not json",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            raised = False
            status = None
            data = {}
        except urllib.error.HTTPError as exc:
            raised = True
            status = exc.code
            data = json.loads(exc.read().decode("utf-8"))
        assert raised
        assert status == 400
        assert data["error"] == "invalid_json"
    finally:
        fixture.close()


def test_simulate_rejects_unknown_path():
    fixture = _ServiceFixture()
    try:
        status, _ = fixture.post(
            {"prompt_tokens": 64, "max_output_tokens": 16}, path="/not-simulate"
        )
        assert status == 404
    finally:
        fixture.close()


def test_simulate_batch_returns_simulated_batch_result():
    fixture = _ServiceFixture()
    try:
        status, data = fixture.post(
            {
                "requests": [
                    {"prompt_tokens": 256, "max_output_tokens": 32, "arrival_ms": 0.0},
                ],
                "policy": DEFAULT_POLICY,
            },
            path="/simulate_batch",
        )
        assert status == 200
        assert data["result_type"] == "simulated"
        assert data["mode"] == "batch"
        assert data["policy"] == DEFAULT_POLICY
        assert "git_commit" in data
        assert "ttft_ms" in data
        assert "tpot_ms" in data
        assert "e2e_latency_ms" in data
        assert "rejected_requests" in data
        assert "oom_events" in data
        assert "kv_page_lifecycle" in data
    finally:
        fixture.close()


def test_simulate_batch_accepts_multiple_requests_and_matches_request_count():
    fixture = _ServiceFixture()
    try:
        requests = [
            {
                "request_id": f"r{i}",
                "prompt_tokens": 128,
                "max_output_tokens": 16,
                "arrival_ms": float(i),
            }
            for i in range(5)
        ]
        status, data = fixture.post({"requests": requests}, path="/simulate_batch")
        assert status == 200
        assert data["request_count"] == len(requests)
    finally:
        fixture.close()


def test_simulate_batch_load_changes_pressure_related_fields():
    fixture = _ServiceFixture()
    try:
        light_requests = [
            {"request_id": f"light-{i}", "prompt_tokens": 128, "max_output_tokens": 16, "arrival_ms": float(i)}
            for i in range(3)
        ]
        heavy_requests = [
            {
                "request_id": f"heavy-{i}",
                "prompt_tokens": 4096,
                "max_output_tokens": 256,
                "arrival_ms": float(i) * 0.05,
            }
            for i in range(60)
        ]
        _, light = fixture.post(
            {"requests": light_requests, "policy": DEFAULT_POLICY},
            path="/simulate_batch",
        )
        _, heavy = fixture.post(
            {"requests": heavy_requests, "policy": DEFAULT_POLICY},
            path="/simulate_batch",
        )
        assert (
            heavy["rejected_requests"] != light["rejected_requests"]
            or heavy["oom_events"] != light["oom_events"]
            or heavy["kv_page_lifecycle"]["pressure_prefetch_skips"]
            != light["kv_page_lifecycle"]["pressure_prefetch_skips"]
        )
    finally:
        fixture.close()


def test_simulate_batch_rejects_empty_requests_list():
    fixture = _ServiceFixture()
    try:
        status, data = fixture.post({"requests": []}, path="/simulate_batch")
        assert status == 400
        assert "error" in data
    finally:
        fixture.close()


def test_simulate_batch_rejects_missing_requests_key():
    fixture = _ServiceFixture()
    try:
        status, data = fixture.post({}, path="/simulate_batch")
        assert status == 400
        assert "error" in data
    finally:
        fixture.close()


def test_simulate_batch_rejects_non_positive_tokens_in_entry():
    fixture = _ServiceFixture()
    try:
        status, data = fixture.post(
            {
                "requests": [
                    {"prompt_tokens": 128, "max_output_tokens": 16},
                    {"prompt_tokens": 0, "max_output_tokens": 16},
                ]
            },
            path="/simulate_batch",
        )
        assert status == 400
        assert "error" in data
    finally:
        fixture.close()


def test_session_create_and_delete():
    fixture = _ServiceFixture()
    try:
        status, created = fixture.post({}, path="/session")
        assert status == 200
        assert created["result_type"] == "simulated"
        session_id = created["session_id"]

        status, deleted = fixture.delete(f"/session/{session_id}")
        assert status == 200
        assert deleted["deleted"] is True

        status, data = fixture.delete(f"/session/{session_id}")
        assert status == 404
        assert "error" in data
    finally:
        fixture.close()


def test_session_request_admits_and_summary_reflects_it():
    fixture = _ServiceFixture()
    try:
        _, created = fixture.post({}, path="/session")
        session_id = created["session_id"]

        status, admitted = fixture.post(
            {"prompt_tokens": 256, "max_output_tokens": 32, "request_id": "r1"},
            path=f"/session/{session_id}/request",
        )
        assert status == 200
        assert admitted["admitted"] is True
        assert admitted["result_type"] == "simulated"

        status, summary = fixture.get(f"/session/{session_id}/summary")
        assert status == 200
        assert "r1" in summary["resident_request_ids"]
        assert summary["kv_page_lifecycle"]["allocated_pages"] > 0
        assert summary["requests_submitted"] == 1
    finally:
        fixture.close()


def test_session_request_rejects_when_insufficient_pages():
    fixture = _ServiceFixture()
    try:
        _, created = fixture.post({"total_pages": 2, "page_size_tokens": 16}, path="/session")
        session_id = created["session_id"]

        status, admitted = fixture.post(
            {"prompt_tokens": 1024, "max_output_tokens": 1024, "request_id": "too-big"},
            path=f"/session/{session_id}/request",
        )
        assert status == 200
        assert admitted["admitted"] is False
        assert admitted["reason"] in ("request_exceeds_total_kv_pages", "insufficient_free_kv_pages")

        status, summary = fixture.get(f"/session/{session_id}/summary")
        assert status == 200
        assert summary["resident_request_ids"] == []
        assert "too-big" in summary["rejected_request_ids"]
    finally:
        fixture.close()


def test_session_step_advances_and_finishes_request():
    fixture = _ServiceFixture()
    try:
        _, created = fixture.post({"total_pages": 20, "page_size_tokens": 16}, path="/session")
        session_id = created["session_id"]

        fixture.post(
            {"prompt_tokens": 16, "max_output_tokens": 3, "request_id": "short"},
            path=f"/session/{session_id}/request",
        )

        finished = False
        for _ in range(10):
            status, step_result = fixture.post({}, path=f"/session/{session_id}/step")
            assert status == 200
            status, summary = fixture.get(f"/session/{session_id}/summary")
            if "short" in summary["finished_request_ids"]:
                finished = True
                break
        assert finished
        assert "short" not in summary["resident_request_ids"]
    finally:
        fixture.close()


def test_session_cancel_releases_pages_and_is_idempotent_failure():
    fixture = _ServiceFixture()
    try:
        _, created = fixture.post({"total_pages": 20, "page_size_tokens": 16}, path="/session")
        session_id = created["session_id"]

        fixture.post(
            {"prompt_tokens": 16, "max_output_tokens": 32, "request_id": "r1"},
            path=f"/session/{session_id}/request",
        )
        fixture.post({}, path=f"/session/{session_id}/step")

        status, cancelled = fixture.post(
            {"request_id": "r1"}, path=f"/session/{session_id}/cancel"
        )
        assert status == 200
        assert cancelled["cancelled"] is True

        status, summary = fixture.get(f"/session/{session_id}/summary")
        assert "r1" in summary["cancelled_request_ids"]
        assert "r1" not in summary["resident_request_ids"]

        status, data = fixture.post({"request_id": "r1"}, path=f"/session/{session_id}/cancel")
        assert status == 409

        status, data = fixture.post(
            {"request_id": "never-existed"}, path=f"/session/{session_id}/cancel"
        )
        assert status == 404
    finally:
        fixture.close()


def test_session_cancel_while_prefetched_creates_waste():
    fixture = _ServiceFixture()
    try:
        _, created = fixture.post({"total_pages": 20, "page_size_tokens": 4}, path="/session")
        session_id = created["session_id"]

        status, admitted = fixture.post(
            {"prompt_tokens": 4, "max_output_tokens": 40, "request_id": "long-lived"},
            path=f"/session/{session_id}/request",
        )
        assert status == 200 and admitted["admitted"] is True

        saw_prefetch = False
        for _ in range(20):
            status, step_result = fixture.post({}, path=f"/session/{session_id}/step")
            assert status == 200
            if any(
                event.get("event") == "page_prefetch" and event.get("request_id") == "long-lived"
                for event in step_result["events"]
            ):
                saw_prefetch = True
                break
        assert saw_prefetch

        status, cancelled = fixture.post(
            {"request_id": "long-lived"}, path=f"/session/{session_id}/cancel"
        )
        assert status == 200
        assert cancelled["cancelled"] is True

        status, summary = fixture.get(f"/session/{session_id}/summary")
        assert summary["kv_page_lifecycle"]["prefetch_waste"] >= 1
    finally:
        fixture.close()


def test_session_contiguous_free_run_ratio_drops_below_one():
    fixture = _ServiceFixture()
    try:
        _, created = fixture.post({"total_pages": 16, "page_size_tokens": 4}, path="/session")
        session_id = created["session_id"]

        for request_id in ("a", "b", "c"):
            fixture.post(
                {"prompt_tokens": 4, "max_output_tokens": 24, "request_id": request_id},
                path=f"/session/{session_id}/request",
            )
        for _ in range(4):
            fixture.post({}, path=f"/session/{session_id}/step")
        fixture.post({"request_id": "b"}, path=f"/session/{session_id}/cancel")
        for _ in range(2):
            fixture.post({}, path=f"/session/{session_id}/step")

        status, summary = fixture.get(f"/session/{session_id}/summary")
        assert status == 200
        assert summary["kv_page_lifecycle"]["contiguous_free_run_ratio"] < 1.0
    finally:
        fixture.close()


def test_session_unknown_id_returns_404():
    fixture = _ServiceFixture()
    try:
        status, _ = fixture.post(
            {"prompt_tokens": 16, "max_output_tokens": 16}, path="/session/does-not-exist/request"
        )
        assert status == 404
        status, _ = fixture.post({}, path="/session/does-not-exist/step")
        assert status == 404
        status, _ = fixture.post(
            {"request_id": "x"}, path="/session/does-not-exist/cancel"
        )
        assert status == 404
        status, _ = fixture.get("/session/does-not-exist/summary")
        assert status == 404
    finally:
        fixture.close()


def test_session_adaptive_guard_activates_and_skips_increment_unit_level():
    """Unit-level: drives the simulate_service._step/_cancel-equivalent
    helpers (step_session/cancel_request) directly against a PagedKVLifecycle
    constructed with a small usefulness_min_samples, bypassing the HTTP
    layer purely for test speed/determinism. Production sessions only reach
    this state via real cancel-while-prefetched cycles against the fixed
    production threshold (PagedKVLifecycle's default usefulness_min_samples),
    since /session does not expose this knob over HTTP.
    """
    kv = PagedKVLifecycle(total_pages=40, page_size_tokens=4, kv_mb_per_page=3.125, usefulness_min_samples=2)
    session = SimulatedSession(kv=kv)

    def run_waste_cycle(request_id: str) -> None:
        request = Request(request_id=request_id, prompt_tokens=4, output_tokens=40, arrival_ms=0.0)
        outcome = admit_request(session, request)
        assert outcome["admitted"] is True

        saw_prefetch = False
        for _ in range(20):
            result = step_session(session)
            if any(
                event.get("event") == "page_prefetch" and event.get("request_id") == request_id
                for event in result["events"]
            ):
                saw_prefetch = True
                break
        assert saw_prefetch

        cancel_outcome = cancel_request(session, request_id)
        assert cancel_outcome["cancelled"] is True

    run_waste_cycle("req-a")
    run_waste_cycle("req-b")

    # _update_adaptive_guard_state() only runs inside prefetch_next_decode_page,
    # so the flag is only re-evaluated on the next decode tick, not at cancel
    # time itself. req-c's first step is what reaches resolved_samples >= 2
    # with usefulness_ema <= usefulness_disable_threshold and flips it.
    request = Request(request_id="req-c", prompt_tokens=4, output_tokens=40, arrival_ms=0.0)
    outcome = admit_request(session, request)
    assert outcome["admitted"] is True
    step_session(session)
    assert kv.adaptive_guard_active is True

    saw_skip = False
    for _ in range(20):
        result = step_session(session)
        if any(
            event.get("event") == "page_prefetch_skipped"
            and event.get("reason") == "usefulness_below_adaptive_guard_threshold"
            for event in result["events"]
        ):
            saw_skip = True
            break
    assert saw_skip
    assert kv.adaptive_prefetch_skips > 0
