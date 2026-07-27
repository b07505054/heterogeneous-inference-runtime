import json
import threading
from unittest.mock import patch, MagicMock

from perf_model.token_timeline import TokenTimeline, inter_token_stats, stream_completion_with_timeline


def test_inter_token_stats_uniform_gaps():
    t = TokenTimeline(
        request_id="r1", ok=True, submit_time=0.0, first_token_time=0.1,
        token_arrival_times=[0.1, 0.11, 0.12, 0.13, 0.14], completion_time=0.15, output_tokens=5,
    )
    stats = inter_token_stats(t)
    assert abs(stats["ttft_ms"] - 100.0) < 1e-6
    assert abs(stats["median_itl_ms"] - 10.0) < 1e-6
    assert stats["stalls_above_2x_median"] == 0
    assert stats["max_stall_ms"] is not None


def test_inter_token_stats_detects_large_stall_not_just_mean_shift():
    # 9 fast tokens (10ms apart) + one huge 300ms stall -- mean would look
    # "moderately slow" but the stall detector must isolate the single spike.
    times = [0.0]
    for _ in range(9):
        times.append(times[-1] + 0.010)
    times.append(times[-1] + 0.300)
    t = TokenTimeline(request_id="r2", ok=True, submit_time=-0.05, first_token_time=0.0,
                       token_arrival_times=times, completion_time=times[-1] + 0.01, output_tokens=len(times))
    stats = inter_token_stats(t)
    assert stats["stalls_above_5x_median"] == 1
    assert stats["max_stall_ms"] > 250


def test_inter_token_stats_handles_failed_request():
    t = TokenTimeline(request_id="r3", ok=False, submit_time=0.0, first_token_time=None,
                       token_arrival_times=[], completion_time=0.01, output_tokens=0, error="boom")
    stats = inter_token_stats(t)
    assert stats["ttft_ms"] is None
    assert stats["mean_tpot_ms"] is None


def test_inter_token_stats_handles_single_token_output():
    t = TokenTimeline(request_id="r4", ok=True, submit_time=0.0, first_token_time=0.05,
                       token_arrival_times=[0.05], completion_time=0.06, output_tokens=1)
    stats = inter_token_stats(t)
    assert abs(stats["ttft_ms"] - 50.0) < 1e-6
    assert stats["mean_tpot_ms"] is None  # cannot compute a gap from a single arrival


class _FakeSSEResponse:
    """Minimal context-manager + line-iterator double for urllib's streamed response."""
    def __init__(self, n_tokens):
        lines = [f'data: {json.dumps({"choices": [{"text": "x"}]})}\n'.encode() for _ in range(n_tokens)]
        lines.append(b"data: [DONE]\n")
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._lines)


# --- 3. dynamically-triggered admission release (E2E-4 generalization of on_first_token) ---
def test_on_release_fires_exactly_when_release_after_tokens_reached():
    fired_at = []
    with patch("urllib.request.urlopen", return_value=_FakeSSEResponse(10)):
        stream_completion_with_timeline(
            "http://x", "m", "r1", [1, 2, 3], max_tokens=10,
            release_after_tokens=4, on_release=lambda: fired_at.append(True),
        )
    assert fired_at == [True]  # fired exactly once


def test_on_release_default_matches_on_first_token_trigger_point():
    order = []
    with patch("urllib.request.urlopen", return_value=_FakeSSEResponse(5)):
        stream_completion_with_timeline(
            "http://x", "m", "r1", [1, 2, 3], max_tokens=5,
            on_first_token=lambda: order.append("first_token"),
            on_release=lambda: order.append("release"),
        )
    # both fire, and since release_after_tokens defaults to 1, they fire on the same (first) token
    assert order == ["first_token", "release"] or order == ["release", "first_token"]


def test_on_release_not_called_if_fewer_tokens_than_threshold_are_emitted():
    fired = []
    with patch("urllib.request.urlopen", return_value=_FakeSSEResponse(3)):
        stream_completion_with_timeline(
            "http://x", "m", "r1", [1, 2, 3], max_tokens=3,
            release_after_tokens=8, on_release=lambda: fired.append(True),
        )
    assert fired == []
