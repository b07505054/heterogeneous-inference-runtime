#!/usr/bin/env python3
"""Validate the pinned ExecuTorch source tree used by E3."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EXPECTED_EXECUTORCH_COMMIT = "e2f18eb23c45bd22ca332b0b8b49a81de304b472"
EXPECTED_EXECUTORCH_TAG = "v1.3.1"
EXPECTED_XNNPACK_COMMIT = "1adaa7c709d4839d29e1f219cb962b01c9e6a905"
XNNPACK_SUBMODULE = "backends/xnnpack/third-party/XNNPACK"


def run_git(source: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def validate(source: Path) -> dict:
    failures: list[str] = []
    head = run_git(source, "rev-parse", "HEAD")
    if head != EXPECTED_EXECUTORCH_COMMIT:
        failures.append("executorch_head_mismatch")
    tags = run_git(source, "tag", "--points-at", "HEAD").splitlines()
    if EXPECTED_EXECUTORCH_TAG not in tags:
        failures.append("executorch_tag_missing")
    dirty = run_git(source, "status", "--short")
    if dirty:
        failures.append("source_tree_dirty")
    submodules = run_git(source, "submodule", "status", "--recursive").splitlines()
    bad_markers = [line for line in submodules if line[:1] in {"+", "-", "U"}]
    if bad_markers:
        failures.append("submodule_status_mismatch")
    xnnpack = run_git(source / XNNPACK_SUBMODULE, "rev-parse", "HEAD")
    if xnnpack != EXPECTED_XNNPACK_COMMIT:
        failures.append("xnnpack_head_mismatch")
    return {
        "schema": "e3_executorch_source_validation",
        "source": str(source),
        "expected_tag": EXPECTED_EXECUTORCH_TAG,
        "executorch_commit": head,
        "expected_executorch_commit": EXPECTED_EXECUTORCH_COMMIT,
        "xnnpack_commit": xnnpack,
        "expected_xnnpack_commit": EXPECTED_XNNPACK_COMMIT,
        "submodule_count": len(submodules),
        "bad_submodule_markers": bad_markers,
        "dirty_status": dirty.splitlines(),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    result = validate(Path(args.source).resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
