from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from benchmark.provenance import (
    command_metadata,
    git_commit,
    git_dirty,
    hardware_metadata,
    software_versions,
    timestamp_utc,
)


REQUIRED_MEASURED_KEYS = {
    "artifact_type",
    "evidence_type",
    "benchmark_target",
    "hardware",
    "software_versions",
    "command",
    "git_commit",
    "metrics",
    "notes",
}


def measured_envelope(
    *,
    artifact_type: str,
    benchmark_target: dict,
    metrics: dict,
    notes: list | None = None,
    command: Sequence[str] | None = None,
    software: dict | None = None,
    hardware: dict | None = None,
    status: str = "ok",
    extra: dict | None = None,
) -> dict:
    payload = {
        "artifact_type": artifact_type,
        "evidence_type": "measured",
        "status": status,
        "benchmark_target": benchmark_target,
        "hardware": hardware or hardware_metadata(),
        "software_versions": software_versions(software),
        "command": command_metadata(command),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp_utc": timestamp_utc(),
        "metrics": metrics,
        "notes": notes or [],
    }
    if extra:
        payload.update(extra)
    missing = REQUIRED_MEASURED_KEYS - payload.keys()
    if missing:
        raise ValueError(f"measured envelope missing required keys: {sorted(missing)}")
    return payload


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

