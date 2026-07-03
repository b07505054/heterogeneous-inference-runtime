from benchmark.runner import BenchmarkRunner


def test_runner_warmup_run_finalize_and_export():
    calls = []
    exported = []

    def measure(item, warmup=False):
        calls.append((item, warmup))
        return {"item": item, "warmup": warmup}

    runner = BenchmarkRunner(
        measure_fn=measure,
        finalize_fn=lambda rows: {"count": len(rows), "rows": rows},
        export_fn=exported.append,
    )
    runner.warmup([1, 2])
    runner.run([3, 4])
    result = runner.finalize()
    runner.export(result)

    assert calls == [(1, True), (2, True), (3, False), (4, False)]
    assert result["count"] == 2
    assert exported == [result]


def test_runner_timing_wraps_measurement():
    runner = BenchmarkRunner(
        measure_fn=lambda _item, warmup=False: "ok",
        time_measurements=True,
    )
    row = runner.measure("x")
    assert row["value"] == "ok"
    assert row["elapsed_ms"] >= 0
