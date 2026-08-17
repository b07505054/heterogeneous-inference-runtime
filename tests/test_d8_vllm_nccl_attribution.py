import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/run_d8_vllm_nccl_attribution.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_d8_vllm_nccl_attribution", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_overlap_ratio_uses_wall_union_interval_intersections():
    m = _load_module()
    nccl = [m.Interval(100, 300), m.Interval(200, 500)]
    compute = [m.Interval(0, 150), m.Interval(250, 450)]
    assert m.interval_union_duration_us(nccl) == pytest.approx(0.4)
    assert m.interval_intersection_duration_us(nccl, compute) == pytest.approx(0.25)
    assert m.overlap_ratio(nccl, compute) == pytest.approx(0.625)


def test_wall_union_prevents_cross_rank_double_counting():
    m = _load_module()
    rank0 = m.Interval(1000, 2000)
    rank1 = m.Interval(1100, 2100)
    assert rank0.duration_us + rank1.duration_us == pytest.approx(2.0)
    assert m.interval_union_duration_us([rank0, rank1]) == pytest.approx(1.1)


def test_request_id_field_parser_and_measured_selection():
    m = _load_module()
    assert m.request_ids_from_fields({"req_ids": "warmup|measured"}) == ["warmup", "measured"]
    summary = {"workload_results": [{"per_request": [
        {"request_kind": "warmup", "request_id": "warmup-0"},
        {"request_kind": "measured", "request_id": "measured-0"},
    ]}]}
    assert m.measured_request_ids(summary) == {"measured-0"}


def test_prediction_error_summary_reports_required_metrics():
    m = _load_module()
    rows = [
        {"tensor_bytes": 1024, "predicted_nccl_us": 10.0, "observed_raw_nccl_us": 12.0},
        {"tensor_bytes": 2 * 1024 * 1024, "predicted_nccl_us": 100.0, "observed_raw_nccl_us": 80.0},
    ]
    summary = m.summarize_prediction_errors(rows)
    assert summary["mae_us"] == pytest.approx(11.0)
    assert summary["max_abs_error_us"] == pytest.approx(20.0)
    assert summary["mape"] == pytest.approx(((2.0 / 12.0) + (20.0 / 80.0)) / 2)
    assert summary["error_by_message_size_bucket"]["<16KiB"]["count"] == 1
    assert summary["error_by_message_size_bucket"][">=1MiB"]["count"] == 1
    assert rows[0]["absolute_error_us"] == pytest.approx(2.0)


def test_current_artifacts_are_fail_closed_when_vllm_missing():
    manifest_path = REPO_ROOT / "results/runtime_paths/distributed_d8_vllm_nccl_attribution/workload_manifest.json"
    if not manifest_path.exists():
        pytest.skip("D8 artifacts have not been generated in this checkout")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dependency_inventory"]["modules"].get("vllm"):
        pytest.skip("vLLM is available; fail-closed missing-vLLM artifact is not expected")
    assert manifest["status"] == "blocked_missing_vllm_or_torch"
    collectives = json.loads(
        (REPO_ROOT / "results/runtime_paths/distributed_d8_vllm_nccl_attribution"
         "/per_decode_step_collectives.json").read_text()
    )
    assert collectives["status"] == "blocked"
    assert collectives["rows"] == []
    assert collectives["no_synthesized_measurements"] is True


def test_extract_kernel_rows_resolves_nsys_string_ids(tmp_path):
    m = _load_module()
    db = tmp_path / "trace.sqlite"
    with sqlite3.connect(db) as con:
        con.execute("create table StringIds(id integer primary key, value text not null)")
        con.execute("insert into StringIds(id, value) values (1, 'ncclDevKernel_AllReduce_Sum_f16_RING_LL')")
        con.execute("insert into StringIds(id, value) values (2, 'triton_compute_kernel')")
        con.execute(
            "create table CUPTI_ACTIVITY_KIND_KERNEL("
            "start integer not null, end integer not null, shortName integer not null)"
        )
        con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (1000, 2500, 1)")
        con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (3000, 4500, 2)")
    rows = m.extract_kernel_rows(db)
    nccl = [row for row in rows if row["is_nccl"]]
    assert len(nccl) == 1
    assert nccl[0]["name"] == "ncclDevKernel_AllReduce_Sum_f16_RING_LL"
    assert nccl[0]["collective_kind"] == "all_reduce"
    assert nccl[0]["duration_us"] == pytest.approx(1.5)



def test_vllm_tp_profile_helper_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_TP_PROFILE_NVTX", raising=False)
    import vllm.tp_profile as tp_profile
    assert tp_profile.enabled() is False


def test_vllm_tp_profile_collective_label_and_bytes(monkeypatch):
    monkeypatch.setenv("VLLM_TP_PROFILE_NVTX", "1")
    import torch
    import vllm.tp_profile as tp_profile
    t = torch.empty((2, 896), dtype=torch.float16)
    assert tp_profile.tensor_bytes(t) == 2 * 896 * 2
    labels = []
    monkeypatch.setattr(tp_profile, "_range", lambda label: labels.append(label) or tp_profile.nullcontext())
    with tp_profile.nccl_range("all_reduce", t, 2):
        pass
    assert labels == [
        "vllm.nccl collective=all_reduce numel=1792 element_size=2 bytes=3584 dtype=torch.float16 world_size=2"
    ]


def test_vllm_tp_profile_step_label(monkeypatch):
    monkeypatch.setenv("VLLM_TP_PROFILE_NVTX", "1")
    import vllm.tp_profile as tp_profile
    labels = []
    monkeypatch.setattr(tp_profile, "_range", lambda label: labels.append(label) or tp_profile.nullcontext())
    with tp_profile.step_range("decode", 42, 1, 1, 0, 0, 1, 1, req_ids=["req 1"]):
        pass
    assert labels == [
        "vllm.step phase=decode step=42 tokens=1 requests=1 ctx_tokens=0 ctx_requests=0 decode_tokens=1 decode_requests=1 req_ids=req_1"
    ]


def test_vllm_pynccl_patch_has_collectives_and_no_explicit_sync_for_instrumentation():
    src = Path("/workspace/d8-vllm-env/lib/python3.12/site-packages/vllm/distributed/device_communicators/pynccl.py").read_text()
    assert 'tp_profile.nccl_range("all_reduce"' in src
    assert 'tp_profile.nccl_range("all_gather"' in src
    assert 'tp_profile.nccl_range("reduce_scatter"' in src
    assert 'tp_profile.nccl_range("broadcast"' in src
    assert "cudaDeviceSynchronize" not in src
