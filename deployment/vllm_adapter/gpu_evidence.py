"""D4B Part H/I: real GPU and NCCL evidence collection.

Everything here is derived from live `nvidia-smi` queries and real server
log content on this host at call time -- never inferred solely from
`--tensor-parallel-size 2` having been passed on a command line.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any


def _run_csv_noheader(cmd: list[str], fields: list[str]) -> list[dict[str, str]]:
    """Run an `nvidia-smi --format=csv,noheader` query and zip each row
    against the EXACT field list that was requested (in order) -- noheader
    output has no header row to parse, so the field names must be supplied
    by the caller, not inferred from the first data row (that earlier bug
    silently dropped GPU 0's row and mislabeled GPU 1's values)."""
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    lines = [l for l in out.splitlines() if l.strip()]
    rows = []
    for line in lines:
        values = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(fields, values)))
    return rows


_GPU_QUERY_FIELDS = [
    "index", "name", "uuid", "pci.bus_id", "memory.total", "memory.used", "memory.free",
    "utilization.gpu", "compute_cap", "driver_version",
]
_COMPUTE_APPS_FIELDS = ["pid", "process_name", "used_memory", "gpu_uuid"]


def query_gpu_inventory() -> list[dict[str, Any]]:
    return _run_csv_noheader(
        ["nvidia-smi", f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}", "--format=csv,noheader,nounits"],
        _GPU_QUERY_FIELDS,
    )


def query_compute_apps() -> list[dict[str, Any]]:
    try:
        return _run_csv_noheader(
            ["nvidia-smi", f"--query-compute-apps={','.join(_COMPUTE_APPS_FIELDS)}", "--format=csv,noheader,nounits"],
            _COMPUTE_APPS_FIELDS,
        )
    except subprocess.CalledProcessError:
        return []


@dataclass(frozen=True)
class GpuMemoryCleanupResult:
    baseline_used_mb: dict[str, float]
    final_used_mb: dict[str, float]
    drain_latency_s: float
    within_tolerance: bool
    tolerance_mb: float
    polls: int
    timed_out: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_used_mb": self.baseline_used_mb, "final_used_mb": self.final_used_mb,
            "drain_latency_s": self.drain_latency_s, "within_tolerance": self.within_tolerance,
            "tolerance_mb": self.tolerance_mb, "polls": self.polls, "timed_out": self.timed_out,
        }


def wait_for_gpu_memory_baseline(
    baseline_used_mb: dict[str, float], *, tolerance_mb: float = 64.0, timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
) -> GpuMemoryCleanupResult:
    """Bounded poll for real GPU memory to actually drain back to (near) its
    pre-launch baseline -- nvidia-smi can lag a few seconds behind actual
    process exit while the CUDA driver reclaims device memory, so a single
    immediate post-stop snapshot is not sufficient evidence of cleanup."""
    t0 = time.time()
    polls = 0
    final: dict[str, float] = {}
    timed_out = True
    while time.time() - t0 < timeout_s:
        polls += 1
        rows = query_gpu_inventory()
        final = {r["index"]: float(r["memory.used"]) for r in rows}
        if all(
            final.get(idx, 0.0) <= baseline_used_mb.get(idx, 0.0) + tolerance_mb
            for idx in baseline_used_mb
        ):
            timed_out = False
            break
        time.sleep(poll_interval_s)
    return GpuMemoryCleanupResult(
        baseline_used_mb=baseline_used_mb, final_used_mb=final, drain_latency_s=time.time() - t0,
        within_tolerance=not timed_out, tolerance_mb=tolerance_mb, polls=polls, timed_out=timed_out,
    )


def build_gpu_snapshot(label: str) -> dict[str, Any]:
    return {
        "label": label, "captured_at": time.time(),
        "gpu_inventory": query_gpu_inventory(), "compute_apps": query_compute_apps(),
        "method": "nvidia-smi --query-gpu / --query-compute-apps (real live query, not inferred)",
    }


@dataclass(frozen=True)
class ProcessGpuMapping:
    tracked_pids: tuple[int, ...]
    gpu_uuids_used_by_tracked_pids: dict[str, list[int]]
    distinct_gpu_uuids_used: tuple[str, ...]
    two_distinct_gpus_used: bool
    duplicate_assignment: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracked_pids": list(self.tracked_pids),
            "gpu_uuids_used_by_tracked_pids": self.gpu_uuids_used_by_tracked_pids,
            "distinct_gpu_uuids_used": list(self.distinct_gpu_uuids_used),
            "two_distinct_gpus_used": self.two_distinct_gpus_used,
            "duplicate_assignment": self.duplicate_assignment,
        }


def compute_process_gpu_mapping(tracked_pids: list[int]) -> ProcessGpuMapping:
    apps = query_compute_apps()
    tracked_set = set(tracked_pids)
    mapping: dict[str, list[int]] = {}
    for row in apps:
        try:
            pid = int(row.get("pid", "-1"))
        except ValueError:
            continue
        if pid in tracked_set:
            uuid = row.get("gpu_uuid", "")
            mapping.setdefault(uuid, []).append(pid)
    distinct = tuple(sorted(mapping.keys()))
    duplicate = any(len(set(pids)) != len(pids) for pids in mapping.values())
    return ProcessGpuMapping(
        tracked_pids=tuple(tracked_pids), gpu_uuids_used_by_tracked_pids=mapping,
        distinct_gpu_uuids_used=distinct, two_distinct_gpus_used=len(distinct) >= 2,
        duplicate_assignment=duplicate,
    )


# Real log-line patterns emitted by vLLM 0.24.0 / torch.distributed / NCCL
# init -- matched against the actual server log content, never assumed.
_NCCL_VERSION_PATTERNS = (
    re.compile(r"NCCL version ([\d.]+)", re.IGNORECASE),
    re.compile(r"nccl[_ ]version[=:\s]+([\d.]+)", re.IGNORECASE),
)
_RANK_INIT_PATTERNS = (
    re.compile(r"rank[=\s]*(\d+).{0,80}?world_size[=\s]*(\d+)", re.IGNORECASE),
    re.compile(r"init.{0,20}process group.{0,80}?rank[=\s]*(\d+)", re.IGNORECASE),
)
_BACKEND_PATTERNS = (
    re.compile(r"\bnccl\b", re.IGNORECASE),
    re.compile(r"\bgloo\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class NCCLEvidence:
    nccl_version_strings_found: tuple[str, ...]
    rank_mentions_found: tuple[str, ...]
    backend_mentions: dict[str, int]
    world_size_mentions: tuple[str, ...]
    raw_matching_lines: tuple[str, ...]
    evidence_found: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "nccl_version_strings_found": list(self.nccl_version_strings_found),
            "rank_mentions_found": list(self.rank_mentions_found),
            "backend_mentions": self.backend_mentions,
            "world_size_mentions": list(self.world_size_mentions),
            "raw_matching_lines": list(self.raw_matching_lines),
            "evidence_found": self.evidence_found,
        }


def extract_nccl_evidence(log_text: str) -> NCCLEvidence:
    versions: list[str] = []
    ranks: list[str] = []
    world_sizes: list[str] = []
    backend_counts = {"nccl": 0, "gloo": 0}
    matching_lines: list[str] = []

    for line in log_text.splitlines():
        matched = False
        for pat in _NCCL_VERSION_PATTERNS:
            m = pat.search(line)
            if m:
                versions.append(m.group(1))
                matched = True
        for pat in _RANK_INIT_PATTERNS:
            m = pat.search(line)
            if m:
                ranks.append(m.group(0))
                matched = True
        if re.search(r"\bnccl\b", line, re.IGNORECASE):
            backend_counts["nccl"] += 1
            matched = True
        if re.search(r"\bgloo\b", line, re.IGNORECASE):
            backend_counts["gloo"] += 1
        if re.search(r"world_size", line, re.IGNORECASE):
            world_sizes.append(line.strip()[:200])
            matched = True
        if matched and len(matching_lines) < 200:
            matching_lines.append(line.strip()[:300])

    return NCCLEvidence(
        nccl_version_strings_found=tuple(dict.fromkeys(versions)),
        rank_mentions_found=tuple(ranks[:50]),
        backend_mentions=backend_counts,
        world_size_mentions=tuple(world_sizes[:50]),
        raw_matching_lines=tuple(matching_lines),
        evidence_found=bool(versions or ranks or backend_counts["nccl"] > 0),
    )
