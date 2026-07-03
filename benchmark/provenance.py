from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
from typing import Sequence


def _run_command(args: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def git_commit() -> str | None:
    return _run_command(["git", "rev-parse", "--short", "HEAD"])


def git_dirty() -> bool:
    return bool(_run_command(["git", "status", "--short"]))


def timestamp_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hardware_metadata() -> dict:
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def software_versions(extra: dict | None = None) -> dict:
    versions = {
        "python": platform.python_version(),
        "executable": sys.executable,
    }
    if extra:
        versions.update(extra)
    return versions


def command_metadata(argv: Sequence[str] | None = None) -> list[str]:
    return list(argv if argv is not None else sys.argv)

