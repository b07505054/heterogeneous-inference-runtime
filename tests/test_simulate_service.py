import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deployment.simulate_service import DEFAULT_POLICY, SimulateHandler  # noqa: E402


class _ServiceFixture:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), SimulateHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path="/simulate"):
        return f"http://127.0.0.1:{self.port}{path}"

    def post(self, payload, path="/simulate"):
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        req = urllib.request.Request(
            self.url(path),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

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
