"""D5: real streaming-latency benchmark harness for TP=1 vs TP=2 serving.

Measures TTFT (time-to-first-token), TPOT (mean inter-token time after the
first), end-to-end latency, and per-request/aggregate throughput against a
real, already-running vLLM server, using the real streaming
`/v1/completions` endpoint (`stream=True`) -- never a non-streaming
request with a computed approximation, since only a real streamed response
gives a genuine first-chunk-arrival timestamp.

This module only measures. It does not decide anything about TP1 vs TP2 --
that judgment belongs entirely to the cost model built from this data.
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from deployment.vllm_adapter.tp_workload_matrix import WorkloadSpec, build_prompt_of_token_length

DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 1234
WARMUP_REQUESTS_PER_WORKLOAD = 2
MEASURED_REPETITIONS_PER_WORKLOAD = 5


@dataclass
class StreamedRequestResult:
    request_index: int
    ok: bool
    error: str | None
    request_start_ts: float
    first_token_ts: float | None
    last_token_ts: float | None
    output_token_count: int
    prompt_token_count: int | None
    request_id: str | None = None
    request_kind: str | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_ts is None:
            return None
        return self.first_token_ts - self.request_start_ts

    @property
    def e2e_latency_s(self) -> float | None:
        if self.last_token_ts is None:
            return None
        return self.last_token_ts - self.request_start_ts

    @property
    def tpot_s(self) -> float | None:
        """Mean per-token decode time, excluding the prefill/first token."""
        if self.first_token_ts is None or self.last_token_ts is None:
            return None
        if self.output_token_count <= 1:
            return None
        return (self.last_token_ts - self.first_token_ts) / (self.output_token_count - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_index": self.request_index, "ok": self.ok, "error": self.error,
            "request_id": self.request_id, "request_kind": self.request_kind,
            "ttft_s": self.ttft_s, "tpot_s": self.tpot_s, "e2e_latency_s": self.e2e_latency_s,
            "output_token_count": self.output_token_count, "prompt_token_count": self.prompt_token_count,
        }


def _send_one_streaming_request(
    base_url: str, model: str, prompt: str, max_tokens: int, request_index: int, *, timeout_s: float,
    request_id: str | None = None, request_kind: str | None = None,
) -> StreamedRequestResult:
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": DEFAULT_TEMPERATURE, "seed": DEFAULT_SEED, "stream": True,
    }
    t_start = time.perf_counter()
    first_token_ts: float | None = None
    last_token_ts: float | None = None
    output_token_count = 0
    prompt_token_count: int | None = None
    try:
        headers = {"X-Request-Id": request_id} if request_id else None
        with requests.post(f"{base_url}/v1/completions", json=payload, headers=headers, stream=True, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                return StreamedRequestResult(
                    request_index=request_index, ok=False, error=f"http_{resp.status_code}: {resp.text[:500]}",
                    request_start_ts=t_start, first_token_ts=None, last_token_ts=None,
                    output_token_count=0, prompt_token_count=None, request_id=request_id, request_kind=request_kind,
                )
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                now = time.perf_counter()
                chunk = json.loads(data_str)
                choices = chunk.get("choices") or []
                text_piece = choices[0].get("text", "") if choices else ""
                usage = chunk.get("usage")
                if usage and usage.get("prompt_tokens") is not None:
                    prompt_token_count = usage["prompt_tokens"]
                if text_piece != "":
                    if first_token_ts is None:
                        first_token_ts = now
                    last_token_ts = now
                    output_token_count += 1
        return StreamedRequestResult(
            request_index=request_index, ok=first_token_ts is not None, error=None if first_token_ts is not None else "no_tokens_streamed",
            request_start_ts=t_start, first_token_ts=first_token_ts, last_token_ts=last_token_ts,
            output_token_count=output_token_count, prompt_token_count=prompt_token_count,
            request_id=request_id, request_kind=request_kind,
        )
    except requests.RequestException as exc:
        return StreamedRequestResult(
            request_index=request_index, ok=False, error=str(exc), request_start_ts=t_start,
            first_token_ts=None, last_token_ts=None, output_token_count=0, prompt_token_count=None,
            request_id=request_id, request_kind=request_kind,
        )


@dataclass
class WorkloadBenchmarkResult:
    workload: WorkloadSpec
    tp_degree: int
    warmup_count: int
    measured_results: list[StreamedRequestResult] = field(default_factory=list)
    wall_clock_batch_s: float = 0.0

    def _valid(self) -> list[StreamedRequestResult]:
        return [r for r in self.measured_results if r.ok]

    def summary(self) -> dict[str, Any]:
        valid = self._valid()
        n_ok = len(valid)
        n_total = len(self.measured_results)
        ttfts = [r.ttft_s for r in valid if r.ttft_s is not None]
        tpots = [r.tpot_s for r in valid if r.tpot_s is not None]
        e2es = [r.e2e_latency_s for r in valid if r.e2e_latency_s is not None]
        total_output_tokens = sum(r.output_token_count for r in valid)
        mean_tpot_s = (sum(tpots) / len(tpots)) if tpots else None
        tpot_std_s = statistics.stdev(tpots) if len(tpots) >= 2 else None
        return {
            "workload_id": self.workload.workload_id, "tp_degree": self.tp_degree,
            "requests_ok": n_ok, "requests_total": n_total,
            "mean_ttft_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "p50_ttft_s": _percentile(ttfts, 0.50) if len(ttfts) >= 5 else None,
            "p95_ttft_s": _percentile(ttfts, 0.95) if len(ttfts) >= 5 else None,
            "mean_tpot_s": mean_tpot_s,
            "p50_tpot_s": _percentile(tpots, 0.50) if len(tpots) >= 5 else None,
            "p95_tpot_s": _percentile(tpots, 0.95) if len(tpots) >= 5 else None,
            "tpot_cv": (tpot_std_s / mean_tpot_s) if tpot_std_s is not None and mean_tpot_s else None,
            "tpot_sample_count": len(tpots),
            "mean_e2e_latency_s": (sum(e2es) / len(e2es)) if e2es else None,
            "p95_e2e_latency_s": _percentile(e2es, 0.95) if len(e2es) >= 5 else None,
            "aggregate_throughput_tokens_per_s": (
                total_output_tokens / self.wall_clock_batch_s if self.wall_clock_batch_s > 0 else None
            ),
            "wall_clock_batch_s": self.wall_clock_batch_s,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.summary()
        d["per_request"] = [r.to_dict() for r in self.measured_results]
        return d


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def run_workload_benchmark(
    base_url: str, model: str, tokenizer, workload: WorkloadSpec, tp_degree: int, *,
    warmup_requests: int = WARMUP_REQUESTS_PER_WORKLOAD,
    measured_repetitions: int = MEASURED_REPETITIONS_PER_WORKLOAD,
    timeout_s: float = 120.0,
) -> WorkloadBenchmarkResult:
    """Runs warmup requests (discarded) then measured repetitions (kept) for
    one workload cell against an already-running, already-ready server.
    Concurrency is realized by firing `workload.concurrency` requests
    simultaneously via a real thread pool + real HTTP connections -- never
    simulated by sleeping or by a single sequential request scaled by a
    concurrency multiplier."""
    prompt = build_prompt_of_token_length(tokenizer, workload.input_length, seed_offset=workload.concurrency)

    tag_prefix = f"d8-tp{tp_degree}-{workload.workload_id}"
    for i in range(warmup_requests):
        _send_one_streaming_request(
            base_url, model, prompt, workload.output_length, -1 - i, timeout_s=timeout_s,
            request_id=f"{tag_prefix}-warmup-{i}", request_kind="warmup")

    result = WorkloadBenchmarkResult(workload=workload, tp_degree=tp_degree, warmup_count=warmup_requests)
    batch_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workload.concurrency) as pool:
        for rep in range(measured_repetitions):
            futures = [
                pool.submit(
                    _send_one_streaming_request, base_url, model, prompt, workload.output_length,
                    rep * workload.concurrency + slot, timeout_s=timeout_s,
                    request_id=f"{tag_prefix}-measured-{rep}-{slot}", request_kind="measured",
                )
                for slot in range(workload.concurrency)
            ]
            for fut in concurrent.futures.as_completed(futures):
                result.measured_results.append(fut.result())
    result.wall_clock_batch_s = time.perf_counter() - batch_start
    result.measured_results.sort(key=lambda r: r.request_index)
    return result
