"""Per-token streaming timeline capture for causal interference analysis.

This is a narrow addition alongside (not a replacement for) benchmark.backends.
openai_compatible.OpenAICompatibleBackend, which only records aggregate ttft/
tpot/e2e. Causally testing "does new prefill work stall an active decode
sequence" requires every individual token arrival timestamp, not just the
mean -- a single average TPOT cannot distinguish uniform slowdown from a few
large stalls (see module docstring intent in the E2E-3 slice report).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class TokenTimeline:
    request_id: str
    ok: bool
    submit_time: float
    first_token_time: float | None
    token_arrival_times: list[float] = field(default_factory=list)
    completion_time: float | None = None
    output_tokens: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id, "ok": self.ok, "submit_time": self.submit_time,
            "first_token_time": self.first_token_time, "token_arrival_times": self.token_arrival_times,
            "completion_time": self.completion_time, "output_tokens": self.output_tokens, "error": self.error,
        }


def stream_completion_with_timeline(
    base_url: str, model: str, request_id: str, prompt_token_ids: list[int], *,
    max_tokens: int, temperature: float = 0.0, seed: int = 1234, ignore_eos: bool = True,
    timeout_s: float = 120.0, on_first_token=None, release_after_tokens: int = 1, on_release=None,
) -> TokenTimeline:
    """on_first_token: optional zero-arg callback invoked the instant the first
    SSE token event is observed (E2E-3 behavior, unchanged, default trigger point).

    release_after_tokens / on_release: E2E-4 generalization -- on_release fires
    once `release_after_tokens` tokens have arrived (default 1, i.e. identical
    to on_first_token's trigger point when unspecified). This lets the E2E-4
    orchestrator establish a real pre-admission baseline decode interval
    (several tokens) before admitting new requests, instead of admitting on
    the very first token as E2E-3 did."""
    payload = {
        "model": model, "prompt": prompt_token_ids, "max_tokens": max_tokens, "temperature": temperature,
        "seed": seed, "ignore_eos": ignore_eos, "stream": True,
    }
    url = base_url.rstrip("/") + "/v1/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})

    submit_time = time.perf_counter()
    first_token_time = None
    arrivals: list[float] = []
    output_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices", []):
                    text = choice.get("text")
                    if text:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                            if on_first_token is not None:
                                on_first_token()
                        arrivals.append(now)
                        output_tokens += 1
                        if on_release is not None and len(arrivals) == release_after_tokens:
                            on_release()
        completion_time = time.perf_counter()
        return TokenTimeline(request_id, True, submit_time, first_token_time, arrivals, completion_time, output_tokens, None)
    except (urllib.error.URLError, TimeoutError) as exc:
        return TokenTimeline(request_id, False, submit_time, first_token_time, arrivals,
                              time.perf_counter(), output_tokens, str(exc))


def inter_token_stats(timeline: TokenTimeline) -> dict:
    """TTFT, mean TPOT, and the inter-token-gap distribution needed to
    distinguish uniform slowdown from a few large stalls."""
    if not timeline.ok or timeline.first_token_time is None or len(timeline.token_arrival_times) < 2:
        return {
            "ttft_ms": (
                (timeline.first_token_time - timeline.submit_time) * 1000.0
                if timeline.first_token_time else None
            ),
            "mean_tpot_ms": None, "median_itl_ms": None, "p95_itl_ms": None, "max_stall_ms": None,
            "stalls_above_2x_median": 0, "stalls_above_5x_median": 0, "gaps_ms": [],
        }
    gaps_ms = [
        (b - a) * 1000.0 for a, b in zip(timeline.token_arrival_times, timeline.token_arrival_times[1:])
    ]
    ordered = sorted(gaps_ms)
    median = ordered[len(ordered) // 2]
    p95_idx = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    ttft_ms = (timeline.first_token_time - timeline.submit_time) * 1000.0
    mean_tpot_ms = sum(gaps_ms) / len(gaps_ms)
    return {
        "ttft_ms": ttft_ms, "mean_tpot_ms": mean_tpot_ms, "median_itl_ms": median, "p95_itl_ms": ordered[p95_idx],
        "max_stall_ms": max(gaps_ms),
        "stalls_above_2x_median": sum(1 for g in gaps_ms if median > 0 and g > 2 * median),
        "stalls_above_5x_median": sum(1 for g in gaps_ms if median > 0 and g > 5 * median),
        "gaps_ms": gaps_ms,
    }
