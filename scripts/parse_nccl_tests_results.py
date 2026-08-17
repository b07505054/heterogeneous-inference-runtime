#!/usr/bin/env python3
"""Parse nccl-tests results into a communication calibration profile.

This is an offline Phase 1 artifact generator. It preserves every nccl-tests
measurement row, then emits two prediction summaries per collective and memory
placement mode:

* raw measured lookup points
* log-size piecewise interpolation
* alpha-beta baseline, latency_us = alpha_us + beta_us_per_byte * bytes

The alpha-beta fit is intentionally reported as a baseline only. It is evaluated
against held-out sizes, but the output does not imply a single linear model is
valid across the full 1 KiB to 1 GiB range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


COLLECTIVE_FILES = {
    "all_reduce": "all_reduce.txt",
    "all_gather": "all_gather.txt",
    "reduce_scatter": "reduce_scatter.txt",
    "broadcast": "broadcast.txt",
}

PROFILE_ID = "2x_rtx4090_phb_single_numa_p2p_unavailable_nccl_shm_direct"
SCHEMA_VERSION = "nccl.communication_calibration.v1"

ROW_RE = re.compile(
    r"^\s*(?P<size>\d+)\s+"
    r"(?P<count>\d+)\s+"
    r"(?P<datatype>\S+)\s+"
    r"(?P<redop>\S+)\s+"
    r"(?P<root>-?\d+)\s+"
    r"(?P<oop_time>\d+(?:\.\d+)?)\s+"
    r"(?P<oop_algbw>\d+(?:\.\d+)?)\s+"
    r"(?P<oop_busbw>\d+(?:\.\d+)?)\s+"
    r"(?P<oop_wrong>\d+)\s+"
    r"(?P<ip_time>\d+(?:\.\d+)?)\s+"
    r"(?P<ip_algbw>\d+(?:\.\d+)?)\s+"
    r"(?P<ip_busbw>\d+(?:\.\d+)?)\s+"
    r"(?P<ip_wrong>\d+)\s*$"
)

VERSION_RE = re.compile(
    r"^# nccl-tests version (?P<tests>\S+) nccl-headers=(?P<headers>\S+) nccl-library=(?P<library>\S+)"
)
START_RE = re.compile(r"^# Collective test starting: (?P<name>\S+)")
NGPUS_RE = re.compile(r"\bnGpus (?P<ngpus>\d+)\b")
GPU_RE = re.compile(r"device\s+(?P<device>\d+)\s+\[[^\]]+\]\s+(?P<name>.+)$")


class NcclResultsParseError(ValueError):
    """Raised for malformed or incomplete nccl-tests input."""


@dataclass(frozen=True)
class Measurement:
    collective: str
    bytes: int
    count: int
    datatype: str
    redop: str
    root: int
    out_of_place_time_us: float
    in_place_time_us: float
    out_of_place_algbw_gbps: float
    out_of_place_busbw_gbps: float
    in_place_algbw_gbps: float
    in_place_busbw_gbps: float
    raw_line: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "collective": self.collective,
            "bytes": self.bytes,
            "count": self.count,
            "datatype": self.datatype,
            "redop": self.redop,
            "root": self.root,
            "out_of_place": {
                "time_us": self.out_of_place_time_us,
                "algbw_gbps": self.out_of_place_algbw_gbps,
                "busbw_gbps": self.out_of_place_busbw_gbps,
            },
            "in_place": {
                "time_us": self.in_place_time_us,
                "algbw_gbps": self.in_place_algbw_gbps,
                "busbw_gbps": self.in_place_busbw_gbps,
            },
            "raw_line": self.raw_line,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_nccl_result_file(path: Path, collective: str) -> tuple[dict[str, Any], list[Measurement]]:
    if not path.exists():
        raise NcclResultsParseError(f"missing required nccl-tests result: {path}")
    text = path.read_text()
    if not text.strip():
        raise NcclResultsParseError(f"empty nccl-tests result: {path}")

    metadata: dict[str, Any] = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "nccl_tests_version": None,
        "nccl_headers_version": None,
        "nccl_library_version": None,
        "collective_test_name": None,
        "n_gpus": None,
        "gpu_names": [],
        "out_of_bounds_ok": False,
        "concluded": False,
    }
    measurements: list[Measurement] = []

    for line in text.splitlines():
        if m := VERSION_RE.match(line):
            metadata["nccl_tests_version"] = m.group("tests")
            metadata["nccl_headers_version"] = m.group("headers")
            metadata["nccl_library_version"] = m.group("library")
            continue
        if m := START_RE.match(line):
            metadata["collective_test_name"] = m.group("name")
            continue
        if "# nThread" in line:
            if m := NGPUS_RE.search(line):
                metadata["n_gpus"] = int(m.group("ngpus"))
            continue
        if "#  Rank" in line and "device" in line:
            if m := GPU_RE.search(line):
                metadata["gpu_names"].append(m.group("name").strip())
            continue
        if line.startswith("# Out of bounds values"):
            metadata["out_of_bounds_ok"] = " OK" in line
            continue
        if line.startswith("# Collective test concluded:"):
            metadata["concluded"] = True
            continue

        m = ROW_RE.match(line)
        if not m:
            continue
        if int(m.group("oop_wrong")) != 0 or int(m.group("ip_wrong")) != 0:
            raise NcclResultsParseError(f"{path}: nonzero #wrong in row: {line}")
        measurements.append(
            Measurement(
                collective=collective,
                bytes=int(m.group("size")),
                count=int(m.group("count")),
                datatype=m.group("datatype"),
                redop=m.group("redop"),
                root=int(m.group("root")),
                out_of_place_time_us=float(m.group("oop_time")),
                in_place_time_us=float(m.group("ip_time")),
                out_of_place_algbw_gbps=float(m.group("oop_algbw")),
                out_of_place_busbw_gbps=float(m.group("oop_busbw")),
                in_place_algbw_gbps=float(m.group("ip_algbw")),
                in_place_busbw_gbps=float(m.group("ip_busbw")),
                raw_line=line,
            )
        )

    required = (
        "nccl_tests_version",
        "nccl_headers_version",
        "nccl_library_version",
        "collective_test_name",
        "n_gpus",
    )
    missing = [key for key in required if metadata[key] in (None, "")]
    if missing:
        raise NcclResultsParseError(f"{path}: missing required metadata fields: {missing}")
    if not metadata["out_of_bounds_ok"]:
        raise NcclResultsParseError(f"{path}: out-of-bounds validation did not report OK")
    if not metadata["concluded"]:
        raise NcclResultsParseError(f"{path}: missing collective concluded marker")
    if len(measurements) < 4:
        raise NcclResultsParseError(f"{path}: expected at least 4 measurement rows, found {len(measurements)}")
    sizes = [m.bytes for m in measurements]
    if sizes != sorted(sizes) or len(set(sizes)) != len(sizes):
        raise NcclResultsParseError(f"{path}: measurement sizes must be unique and sorted")
    return metadata, measurements


def select_held_out_indices(n: int) -> list[int]:
    if n < 7:
        raise NcclResultsParseError(f"need at least 7 points for train/held-out evaluation, found {n}")
    candidates = {2, n // 2, n - 3}
    return sorted(i for i in candidates if 0 <= i < n)


def split_train_holdout(points: list[dict[str, float]]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    held = set(select_held_out_indices(len(points)))
    train = [p for i, p in enumerate(points) if i not in held]
    holdout = [p for i, p in enumerate(points) if i in held]
    if len(train) < 2 or not holdout:
        raise NcclResultsParseError("invalid train/held-out split")
    return train, holdout


def log_size_piecewise_predict(train: list[dict[str, float]], bytes_value: int) -> float:
    ordered = sorted(train, key=lambda p: p["bytes"])
    if bytes_value <= ordered[0]["bytes"]:
        return ordered[0]["time_us"]
    if bytes_value >= ordered[-1]["bytes"]:
        return ordered[-1]["time_us"]
    x = math.log2(bytes_value)
    for left, right in zip(ordered, ordered[1:]):
        if left["bytes"] <= bytes_value <= right["bytes"]:
            x0 = math.log2(left["bytes"])
            x1 = math.log2(right["bytes"])
            frac = (x - x0) / (x1 - x0)
            return left["time_us"] + frac * (right["time_us"] - left["time_us"])
    raise AssertionError("unreachable interpolation state")


def fit_alpha_beta(train: list[dict[str, float]]) -> tuple[float, float]:
    xs = [p["bytes"] for p in train]
    ys = [p["time_us"] for p in train]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        raise NcclResultsParseError("cannot fit alpha-beta model with zero byte variance")
    beta = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    alpha = y_mean - beta * x_mean
    return alpha, beta


def evaluate_predictions(actuals: list[dict[str, float]], predictions: list[float]) -> dict[str, float]:
    if len(actuals) != len(predictions) or not actuals:
        raise NcclResultsParseError("prediction evaluation received mismatched or empty inputs")
    abs_errors = [abs(pred - row["time_us"]) for row, pred in zip(actuals, predictions)]
    rel_errors = [err / row["time_us"] for err, row in zip(abs_errors, actuals)]
    return {
        "mae_us": mean(abs_errors),
        "mape": mean(rel_errors),
        "max_absolute_error_us": max(abs_errors),
        "max_relative_error": max(rel_errors),
    }


def build_prediction_report(measurements: list[Measurement], mode: str) -> dict[str, Any]:
    if mode == "out_of_place":
        points = [
            {"bytes": m.bytes, "time_us": m.out_of_place_time_us}
            for m in measurements
        ]
    elif mode == "in_place":
        points = [
            {"bytes": m.bytes, "time_us": m.in_place_time_us}
            for m in measurements
        ]
    else:
        raise ValueError(f"unknown mode: {mode}")

    train, holdout = split_train_holdout(points)
    interp_predictions = [log_size_piecewise_predict(train, int(row["bytes"])) for row in holdout]
    alpha, beta = fit_alpha_beta(train)
    alpha_beta_predictions = [alpha + beta * row["bytes"] for row in holdout]
    return {
        "raw_measured_lookup_points": points,
        "held_out_bytes": [int(row["bytes"]) for row in holdout],
        "training_bytes": [int(row["bytes"]) for row in train],
        "log_size_piecewise_interpolation": {
            "domain": "log2(bytes)",
            "held_out_predictions": [
                {"bytes": int(row["bytes"]), "actual_time_us": row["time_us"], "predicted_time_us": pred}
                for row, pred in zip(holdout, interp_predictions)
            ],
            "evaluation": evaluate_predictions(holdout, interp_predictions),
        },
        "alpha_beta_baseline": {
            "formula": "latency_us = alpha_us + beta_us_per_byte * bytes",
            "alpha_us": alpha,
            "beta_us_per_byte": beta,
            "validity_note": (
                "Baseline fit only; not assumed valid across the full 1 KiB to 1 GiB range."
            ),
            "held_out_predictions": [
                {"bytes": int(row["bytes"]), "actual_time_us": row["time_us"], "predicted_time_us": pred}
                for row, pred in zip(holdout, alpha_beta_predictions)
            ],
            "evaluation": evaluate_predictions(holdout, alpha_beta_predictions),
        },
    }


def profile_identity(nccl_version: str, nccl_tests_version: str, source_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "profile_id": PROFILE_ID,
        "schema_version": SCHEMA_VERSION,
        "machine_calibration_boundary": {
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "gpu_count": 2,
            "topology_class": "PHB",
            "numa_count": 1,
            "cuda_p2p_available": False,
            "nccl_intra_node_transport": "SHM/direct/direct",
            "nccl_version": nccl_version,
            "nccl_tests_version": nccl_tests_version,
        },
        "source_artifact_hashes": source_hashes,
    }


def parse_all(input_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    per_collective: dict[str, Any] = {}
    fit_collectives: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    nccl_versions: set[str] = set()
    nccl_tests_versions: set[str] = set()

    for collective, filename in COLLECTIVE_FILES.items():
        metadata, measurements = parse_nccl_result_file(input_dir / filename, collective)
        source_hashes[filename] = metadata["source_sha256"]
        nccl_versions.add(str(metadata["nccl_library_version"]))
        nccl_tests_versions.add(str(metadata["nccl_tests_version"]))

        prediction_models = {
            "out_of_place": build_prediction_report(measurements, "out_of_place"),
            "in_place": build_prediction_report(measurements, "in_place"),
        }
        per_collective[collective] = {
            "source_metadata": metadata,
            "measurements": [m.to_dict() for m in measurements],
            "prediction_representations": prediction_models,
        }
        fit_collectives[collective] = {
            mode: {
                "held_out_bytes": report["held_out_bytes"],
                "training_bytes": report["training_bytes"],
                "log_size_piecewise_interpolation": report["log_size_piecewise_interpolation"]["evaluation"],
                "alpha_beta_baseline": {
                    "alpha_us": report["alpha_beta_baseline"]["alpha_us"],
                    "beta_us_per_byte": report["alpha_beta_baseline"]["beta_us_per_byte"],
                    "evaluation": report["alpha_beta_baseline"]["evaluation"],
                },
            }
            for mode, report in prediction_models.items()
        }

    if len(nccl_versions) != 1:
        raise NcclResultsParseError(f"inconsistent NCCL library versions: {sorted(nccl_versions)}")
    if len(nccl_tests_versions) != 1:
        raise NcclResultsParseError(f"inconsistent nccl-tests versions: {sorted(nccl_tests_versions)}")

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    identity = profile_identity(
        nccl_version=next(iter(nccl_versions)),
        nccl_tests_version=next(iter(nccl_tests_versions)),
        source_hashes=source_hashes,
    )
    profile = {
        **identity,
        "generated_at": generated_at,
        "truth_boundary": (
            "Measured nccl-tests calibration for this exact 2x RTX 4090 PHB host "
            "with CUDA P2P unavailable and NCCL SHM/direct/direct transport. "
            "Not a vLLM decode latency model and not portable to other topology."
        ),
        "collectives": per_collective,
    }
    fit_report = {
        **identity,
        "generated_at": generated_at,
        "evaluation_method": {
            "held_out_selection": (
                "Deterministic internal holdout by measurement index: third, middle, and third-from-last sizes."
            ),
            "log_size_piecewise_interpolation": "Train on remaining points; interpolate linearly in log2(bytes).",
            "alpha_beta_baseline": (
                "Least-squares latency_us = alpha_us + beta_us_per_byte * bytes on training points only."
            ),
        },
        "validity_note": (
            "The alpha-beta representation is a baseline and is not assumed valid across 1 KiB to 1 GiB."
        ),
        "collectives": fit_collectives,
    }
    return profile, fit_report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/workspace/nccl-results"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/runtime_paths/nccl_calibration"),
    )
    args = parser.parse_args()
    profile, fit_report = parse_all(args.input_dir)
    write_json(args.out_dir / "communication_cost_profile.json", profile)
    write_json(args.out_dir / "fit_report.json", fit_report)
    print(args.out_dir / "communication_cost_profile.json")
    print(args.out_dir / "fit_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
