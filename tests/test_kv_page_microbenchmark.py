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
        "churn_cycles": 16,
        "churn_max_pages_per_owner": 2,
        "churn_seed": 1234,
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


def test_allocator_churn_section_present_and_no_leak():
    report = run_benchmark(small_args(churn_cycles=64, churn_max_pages_per_owner=4, churn_seed=7))

    churn = report["allocator_churn"]
    assert churn["leaked_pages_after_churn"] == 0
    assert churn["cycles"] == 64
    for key in ["checkout_cost", "release_cost", "contiguous_free_run_ratio", "page_reuse"]:
        assert key in churn


def test_contiguous_free_run_ratio_bounds():
    report = run_benchmark(small_args(churn_cycles=64, churn_max_pages_per_owner=4, churn_seed=7))

    ratio = report["allocator_churn"]["contiguous_free_run_ratio"]
    for value in (ratio["min"], ratio["mean"], ratio["final"]):
        assert 0.0 <= value <= 1.0
    assert "free-list fragmentation" in ratio["description"]
    assert "Not GPU/CPU allocator" in ratio["description"]


def test_page_reuse_counters_are_consistent():
    report = run_benchmark(small_args(churn_cycles=64, churn_max_pages_per_owner=4, churn_seed=7))

    page_reuse = report["allocator_churn"]["page_reuse"]
    for key in ["unique_pages_seen", "pages_reused", "page_reuse_events"]:
        assert key in page_reuse
        assert isinstance(page_reuse[key], int)
    assert page_reuse["unique_pages_seen"] >= page_reuse["pages_reused"] >= 0
    assert page_reuse["page_reuse_events"] >= page_reuse["pages_reused"] >= 0


def test_provenance_fields_present():
    report = run_benchmark(small_args())

    provenance = report["provenance"]
    assert isinstance(provenance["git_commit"], str)
    assert isinstance(provenance["args"], dict)
    assert "timestamp_utc" in provenance


def test_scheduler_artifact_numbers_are_unaffected():
    sys.path.insert(0, str(ROOT))
    from deployment.llm_runtime_decision import MemoryPlanner  # noqa: E402

    planner = MemoryPlanner(total_blocks=16, block_size_tokens=8, kv_mb_per_block=1.0)
    assert planner.total_blocks == 16
    assert len(planner.free_blocks) == 16
