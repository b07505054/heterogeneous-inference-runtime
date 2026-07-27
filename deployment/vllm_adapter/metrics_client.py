"""Prometheus text-exposition parser for vLLM's real GET /metrics endpoint.

Unlike the existing diagnostic-script parser (scripts/run_vllm_max_num_seqs_
diagnostic.py:_parse_prometheus), which only sums scalar gauge/counter lines
and would silently corrupt histogram data if pointed at it, this parser
reconstructs real histograms (bucket, sum, count) and keeps the raw text.
Field names verified live against vLLM 0.24.0 (tests/fixtures/metrics_sample.txt);
metric names have moved across vLLM major versions historically, so this is
PUBLIC_VERSION_SENSITIVE, not PUBLIC_STABLE.
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

_LINE_RE = re.compile(r"^([^\s{]+)(\{[^}]*\})?\s+([^\s]+)$")
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")

HISTOGRAM_METRICS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
)

GAUGE_OR_COUNTER_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
)


class MetricsUnavailable(RuntimeError):
    pass


def fetch_metrics_text(port: int, *, host: str = "127.0.0.1", timeout: float = 5.0) -> str:
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise MetricsUnavailable(f"/metrics returned HTTP {response.status}")
            return response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise MetricsUnavailable(f"/metrics unreachable: {exc}") from exc


def _parse_labels(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    return dict(_LABEL_RE.findall(raw))


@dataclass
class HistogramSample:
    label_key: tuple[tuple[str, str], ...]
    buckets: dict[str, float]  # le string -> cumulative count
    sum: float | None
    count: float | None

    def mean(self) -> float | None:
        if not self.sum or not self.count:
            return None
        return self.sum / self.count if self.count else None


def parse_prometheus_text(text: str) -> dict[str, Any]:
    """Returns {"histograms": {base_name: [HistogramSample, ...]},
                "gauges": {full_name: [{"labels":..., "value":...}, ...]},
                "raw_text": text}. Bucket data is preserved, not discarded."""
    histograms: dict[str, dict[tuple, HistogramSample]] = {}
    gauges: dict[str, list[dict[str, Any]]] = {}

    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = _LINE_RE.match(line.strip())
        if not match:
            continue
        name, label_blob, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = _parse_labels(label_blob[1:-1] if label_blob else None)

        base = None
        for suffix in _HISTOGRAM_SUFFIXES:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        if base is not None:
            if name.endswith("_created"):
                continue
            label_key = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
            bucket = histograms.setdefault(base, {}).setdefault(
                label_key, HistogramSample(label_key=label_key, buckets={}, sum=None, count=None)
            )
            if name.endswith("_bucket"):
                bucket.buckets[labels.get("le", "?")] = value
            elif name.endswith("_sum"):
                bucket.sum = value
            elif name.endswith("_count"):
                bucket.count = value
        elif not name.endswith("_created"):
            gauges.setdefault(name, []).append({"labels": labels, "value": value})

    return {
        "histograms": {base: list(samples.values()) for base, samples in histograms.items()},
        "gauges": gauges,
        "raw_text": text,
    }


def histogram_mean_ms(parsed: dict[str, Any], metric_name: str) -> float | None:
    """Aggregate mean across all label groups (engines), in milliseconds."""
    samples = parsed.get("histograms", {}).get(metric_name)
    if not samples:
        return None
    total_sum = sum(s.sum or 0.0 for s in samples)
    total_count = sum(s.count or 0.0 for s in samples)
    if not total_count:
        return None
    return (total_sum / total_count) * 1000.0


def histogram_count(parsed: dict[str, Any], metric_name: str) -> float | None:
    samples = parsed.get("histograms", {}).get(metric_name)
    if not samples:
        return None
    return sum(s.count or 0.0 for s in samples)


def gauge_value(parsed: dict[str, Any], metric_name: str) -> float | None:
    """Sums across label groups (e.g. multiple engines); fine for single-engine v1."""
    rows = parsed.get("gauges", {}).get(metric_name)
    if not rows:
        return None
    return sum(row["value"] for row in rows)


def snapshot_summary(parsed: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"histograms": {}, "gauges": {}}
    for metric in HISTOGRAM_METRICS:
        samples = parsed.get("histograms", {}).get(metric)
        summary["histograms"][metric] = {
            "mean_ms": histogram_mean_ms(parsed, metric),
            "count": histogram_count(parsed, metric),
            "buckets": [s.buckets for s in samples] if samples else None,
        }
    for metric in GAUGE_OR_COUNTER_METRICS:
        summary["gauges"][metric] = gauge_value(parsed, metric)
    return summary
