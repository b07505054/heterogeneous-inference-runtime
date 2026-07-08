from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


Urlopen = Callable[[urllib.request.Request, float], object]


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    model: str
    concurrency: int = 1
    timeout_s: float = 60.0
    endpoint: str = "/v1/chat/completions"
    stream: bool = True


class OpenAICompatibleBackend:
    def __init__(self, config: OpenAICompatibleConfig, urlopen: Urlopen | None = None):
        self.config = config
        self.urlopen = urlopen or urllib.request.urlopen

    def fetch_model_metadata(self) -> dict:
        url = _join_openai_url(self.config.base_url, "/v1/models")
        request = urllib.request.Request(url, method="GET")
        try:
            with self.urlopen(request, timeout=self.config.timeout_s) as response:
                body = response.read().decode("utf-8")
            return {"ok": True, "response": json.loads(body)}
        except Exception as exc:  # best-effort metadata
            return {"ok": False, "error": str(exc)}

    def execute(self, trace_row: dict, warmup: bool = False) -> dict:
        payload = {
            "model": self.config.model,
            "stream": self.config.stream,
            **{key: value for key, value in trace_row.items() if key != "metadata"},
        }
        if "messages" not in payload and "prompt" in payload and "chat/completions" in self.config.endpoint:
            payload["messages"] = [{"role": "user", "content": str(payload.pop("prompt"))}]

        url = _join_openai_url(self.config.base_url, self.config.endpoint)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        first_token_at = None
        output_tokens = 0
        try:
            with self.urlopen(request, timeout=self.config.timeout_s) as response:
                if self.config.stream:
                    for token in _iter_sse_tokens(response):
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_tokens += token
                else:
                    response_body = response.read().decode("utf-8")
                    output_tokens = _count_non_stream_tokens(json.loads(response_body))
                    first_token_at = time.perf_counter()
            end = time.perf_counter()
        except urllib.error.HTTPError as exc:
            end = time.perf_counter()
            return _error_result(exc, start, end, warmup)
        except Exception as exc:
            end = time.perf_counter()
            return _error_result(exc, start, end, warmup)

        e2e_ms = (end - start) * 1000.0
        ttft_ms = (first_token_at - start) * 1000.0 if first_token_at is not None else None
        tpot_ms = None
        if first_token_at is not None and output_tokens > 1:
            tpot_ms = ((end - first_token_at) * 1000.0) / (output_tokens - 1)
        return {
            "ok": True,
            "warmup": warmup,
            "ttft_ms": round(ttft_ms, 6) if ttft_ms is not None else None,
            "tpot_ms": round(tpot_ms, 6) if tpot_ms is not None else None,
            "e2e_latency_ms": round(e2e_ms, 6),
            "output_tokens": output_tokens,
            "metadata": trace_row.get("metadata", {}),
        }



def _join_openai_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    path = "/" + endpoint.lstrip("/")
    if base.endswith("/v1") and path.startswith("/v1/"):
        path = path[len("/v1") :]
    return base + path

def _error_result(exc: Exception, start: float, end: float, warmup: bool) -> dict:
    return {
        "ok": False,
        "warmup": warmup,
        "e2e_latency_ms": round((end - start) * 1000.0, 6),
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def _iter_sse_tokens(response) -> int:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        yield _count_stream_event_tokens(event)


def _count_stream_event_tokens(event: dict) -> int:
    count = 0
    for choice in event.get("choices", []):
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text:
            count += 1
        completion_text = choice.get("text")
        if completion_text:
            count += 1
    return count


def _count_non_stream_tokens(event: dict) -> int:
    usage = event.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int):
        return completion_tokens
    count = 0
    for choice in event.get("choices", []):
        message = choice.get("message") or {}
        text = message.get("content") or choice.get("text") or ""
        count += len(str(text).split())
    return count
