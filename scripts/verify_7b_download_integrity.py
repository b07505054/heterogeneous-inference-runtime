"""D5: verify the downloaded Qwen2.5-7B-Instruct checkpoint against the real
Hugging Face Hub blob-size metadata (queried live from the Hub API, not
guessed), plus real disk-space and GPU-memory checks, before any 7B
benchmark is run. Fails loudly (non-zero exit) on any mismatch or
insufficient headroom rather than silently continuing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

HF_HOME = os.environ.get("HF_HOME", "/workspace/.hf_home")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MIN_FREE_DISK_GB = 3.0
MIN_FREE_GPU_MB_PER_DEVICE = 100.0  # both GPUs must be genuinely idle before benchmarking


def _real_hub_blob_sizes() -> dict[str, int]:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{MODEL_ID}?blobs=true")
    data = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return {s["rfilename"]: s["size"] for s in data.get("siblings", []) if s.get("size")}


def _local_snapshot_dir() -> Path:
    slug = "models--" + MODEL_ID.replace("/", "--")
    model_dir = Path(HF_HOME) / "hub" / slug
    snapshots = model_dir / "snapshots"
    candidates = sorted(snapshots.iterdir()) if snapshots.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"no snapshot directory found under {snapshots}")
    return candidates[0]


def main() -> None:
    print(f"== verifying {MODEL_ID} download integrity ==")
    expected = _real_hub_blob_sizes()
    snapshot_dir = _local_snapshot_dir()
    mismatches = []
    checked = 0
    for rfilename, expected_size in expected.items():
        local_path = snapshot_dir / rfilename
        if not local_path.exists():
            mismatches.append({"file": rfilename, "issue": "missing", "expected_size": expected_size})
            continue
        actual_size = local_path.resolve().stat().st_size
        checked += 1
        if actual_size != expected_size:
            mismatches.append({"file": rfilename, "issue": "size_mismatch",
                                "expected_size": expected_size, "actual_size": actual_size})
    result = {"model_id": MODEL_ID, "files_checked": checked, "files_expected": len(expected), "mismatches": mismatches}
    print(json.dumps(result, indent=2))
    if mismatches:
        print("INTEGRITY_CHECK_FAILED")
        sys.exit(1)

    df = subprocess.run(["df", "-B1", "/"], capture_output=True, text=True, check=True).stdout.splitlines()[1].split()
    free_gb = int(df[3]) / (1024 ** 3)
    print(f"disk free: {free_gb:.2f} GB")
    if free_gb < MIN_FREE_DISK_GB:
        print(f"INTEGRITY_CHECK_FAILED: only {free_gb:.2f} GB free, need >= {MIN_FREE_DISK_GB} GB")
        sys.exit(1)

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    for line in smi:
        idx, used_mb = [x.strip() for x in line.split(",")]
        print(f"GPU {idx}: {used_mb} MiB used")
        if float(used_mb) > MIN_FREE_GPU_MB_PER_DEVICE:
            print(f"INTEGRITY_CHECK_FAILED: GPU {idx} has {used_mb} MiB used, expected near-idle before benchmarking")
            sys.exit(1)

    print("INTEGRITY_CHECK_PASSED")


if __name__ == "__main__":
    main()
