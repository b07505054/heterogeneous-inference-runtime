import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_kv_page_microbenchmark import (  # noqa: E402
    KVPagePool,
    TRUTH_BOUNDARY,
    dtype_from_name,
    gather_paged_kv,
    materialize_request_pages,
    run_benchmark,
)


def small_args(**overrides):
    values = {
        "output": "/tmp/kv_page_microbenchmark_report.json",
        "device": "cpu",
        "dtype": "float32",
        "request_count": 4,
        "iterations": 1,
        "page_size_tokens": 8,
        "page_count": 128,
        "num_layers": 1,
        "num_kv_heads": 2,
        "head_dim": 4,
        "fail_on_unavailable": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_report_schema_and_truth_boundary():
    report = run_benchmark(small_args())

    assert report["artifact_type"] == "kv_page_microbenchmark_report"
    assert report["status"] == "completed"
    for key in [
        "contiguous_baseline",
        "paged_kv",
        "movement_cost",
        "prefix_reuse_proxy",
        "comparison",
        "truth_boundary",
    ]:
        assert key in report
    assert "not vLLM PagedAttention" in report["truth_boundary"]
    assert "not TensorRT-LLM paged attention" in report["truth_boundary"]
    assert "not wired into live Qwen attention" in report["truth_boundary"]


def test_page_pool_checkout_release_has_no_leak():
    import torch

    pool = KVPagePool(
        page_count=8,
        page_size_tokens=4,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    pool.init(torch)
    pages = pool.checkout("req", 3)

    assert pool.used_pages() == 3
    assert pool.leaked_pages() == 3

    pool.release(pages)

    assert pool.used_pages() == 0
    assert pool.leaked_pages() == 0


def test_paged_gather_matches_contiguous_shape():
    import torch

    pool = KVPagePool(
        page_count=16,
        page_size_tokens=4,
        num_kv_heads=2,
        head_dim=4,
        dtype=dtype_from_name(torch, "float32"),
        device="cpu",
    )
    pool.init(torch)
    pages = pool.checkout("req", 3)
    materialize_request_pages(torch, pool, pages, total_tokens=10)

    gathered = gather_paged_kv(torch, pool, pages, total_tokens=10)
    contiguous = torch.empty((10, 2, 2, 4), dtype=torch.float32)

    assert gathered.shape == contiguous.shape

    pool.release(pages)
    assert pool.leaked_pages() == 0


def test_unavailable_device_report_does_not_crash():
    report = run_benchmark(small_args(device="cuda"))

    if report["status"] == "unavailable":
        assert report["artifact_type"] == "kv_page_microbenchmark_report"
        assert report["truth_boundary"] == TRUTH_BOUNDARY
    else:
        assert report["device"] == "cuda"
