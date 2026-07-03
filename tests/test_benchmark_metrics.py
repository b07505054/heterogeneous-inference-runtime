from benchmark.metrics import latency_summary_ms, openai_latency_metrics, percentile, token_throughput


def test_percentile_and_latency_summary():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == 50
    summary = latency_summary_ms(values)
    assert summary["count"] == 5
    assert summary["p50"] == 30
    assert summary["p95"] == 50


def test_empty_latency_summary():
    summary = latency_summary_ms([])
    assert summary["count"] == 0
    assert summary["p50"] is None


def test_openai_latency_metrics_counts_success_and_errors():
    metrics = openai_latency_metrics(
        [
            {"ok": True, "ttft_ms": 10, "tpot_ms": 2, "e2e_latency_ms": 20, "output_tokens": 5},
            {"ok": False, "e2e_latency_ms": 3},
        ]
    )
    assert metrics["success_count"] == 1
    assert metrics["error_count"] == 1
    assert metrics["tokens_per_second"] == token_throughput(5, 20)

