import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "parse_nccl_tests_results.py"
spec = importlib.util.spec_from_file_location("parse_nccl_tests_results", SCRIPT_PATH)
parser = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = parser
spec.loader.exec_module(parser)


def _write_result(path: Path, collective_name: str, rows: list[tuple[int, float, float]]) -> None:
    lines = [
        "# nccl-tests version 2.19.7 nccl-headers=22803 nccl-library=22803",
        f"# Collective test starting: {collective_name}_perf",
        "# nThread 1 nGpus 2 minBytes 1024 maxBytes 131072 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0 unalign: 0",
        "#",
        "# Using devices",
        "#  Rank  0 Group  0 Pid   1 on host device  0 [0000:04:00] NVIDIA GeForce RTX 4090",
        "#  Rank  1 Group  0 Pid   1 on host device  1 [0000:05:00] NVIDIA GeForce RTX 4090",
        "#",
        "#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong ",
        "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)         ",
    ]
    for size, oop_time, ip_time in rows:
        count = size // 4
        lines.append(
            f"{size:12d} {count:13d}     float     sum      -1"
            f" {oop_time:8.2f}    1.00    1.00       0"
            f" {ip_time:8.2f}    1.00    1.00       0"
        )
    lines.extend(
        [
            "# Out of bounds values : 0 OK",
            "# Avg bus bandwidth    : 1.0 ",
            "#",
            f"# Collective test concluded: {collective_name}_perf",
            "#",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _fixture_dir(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sizes = [1024 * (2**i) for i in range(8)]
    for collective, filename in parser.COLLECTIVE_FILES.items():
        rows = [
            (size, 10.0 + i * 3.0 + len(collective) * 0.1, 11.0 + i * 2.5 + len(collective) * 0.1)
            for i, size in enumerate(sizes)
        ]
        _write_result(tmp_path / filename, collective, rows)
    return tmp_path


def test_parse_file_preserves_raw_measurements(tmp_path):
    input_dir = _fixture_dir(tmp_path)
    metadata, rows = parser.parse_nccl_result_file(input_dir / "all_reduce.txt", "all_reduce")

    assert metadata["nccl_tests_version"] == "2.19.7"
    assert metadata["nccl_library_version"] == "22803"
    assert metadata["n_gpus"] == 2
    assert len(rows) == 8
    first = rows[0].to_dict()
    assert first["collective"] == "all_reduce"
    assert first["bytes"] == 1024
    assert first["datatype"] == "float"
    assert first["out_of_place"]["time_us"] == pytest.approx(11.0)
    assert first["in_place"]["time_us"] == pytest.approx(12.0)
    assert "raw_line" in first and "1024" in first["raw_line"]


def test_parse_all_builds_profile_and_fit_report(tmp_path):
    input_dir = _fixture_dir(tmp_path)
    profile, fit_report = parser.parse_all(input_dir)

    assert profile["profile_id"] == parser.PROFILE_ID
    assert profile["machine_calibration_boundary"]["gpu_model"] == "NVIDIA GeForce RTX 4090"
    assert profile["machine_calibration_boundary"]["topology_class"] == "PHB"
    assert profile["machine_calibration_boundary"]["cuda_p2p_available"] is False
    assert profile["machine_calibration_boundary"]["nccl_intra_node_transport"] == "SHM/direct/direct"
    assert set(profile["collectives"]) == {"all_reduce", "all_gather", "reduce_scatter", "broadcast"}
    for collective in profile["collectives"].values():
        assert len(collective["measurements"]) == 8
        for mode in ("out_of_place", "in_place"):
            report = collective["prediction_representations"][mode]
            assert len(report["raw_measured_lookup_points"]) == 8
            assert report["log_size_piecewise_interpolation"]["evaluation"]["mae_us"] >= 0
            assert report["alpha_beta_baseline"]["evaluation"]["max_relative_error"] >= 0

    assert fit_report["validity_note"].startswith("The alpha-beta representation")
    assert "source_artifact_hashes" in fit_report


def test_malformed_missing_concluded_marker_fails_closed(tmp_path):
    input_dir = _fixture_dir(tmp_path)
    bad = input_dir / "all_reduce.txt"
    bad.write_text(bad.read_text().replace("# Collective test concluded: all_reduce_perf\n", ""))

    with pytest.raises(parser.NcclResultsParseError, match="concluded"):
        parser.parse_nccl_result_file(bad, "all_reduce")


def test_cli_writes_expected_artifacts(tmp_path):
    input_dir = _fixture_dir(tmp_path / "input")
    out_dir = tmp_path / "out"
    profile, fit_report = parser.parse_all(input_dir)
    parser.write_json(out_dir / "communication_cost_profile.json", profile)
    parser.write_json(out_dir / "fit_report.json", fit_report)

    loaded = json.loads((out_dir / "communication_cost_profile.json").read_text())
    fit = json.loads((out_dir / "fit_report.json").read_text())
    assert loaded["schema_version"] == parser.SCHEMA_VERSION
    assert fit["collectives"]["all_reduce"]["out_of_place"]["alpha_beta_baseline"]["evaluation"]["mae_us"] >= 0
