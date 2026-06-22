#!/usr/bin/env bash
# Canonical validation entrypoint for this repository.
#
# This repo uses a project-local .venv only. This script never falls back
# to a system/global Python, and it never installs, upgrades, or
# uninstalls packages. If .venv is missing or a required dependency is
# missing inside it, it reports the problem and stops.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON=".venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo ".venv not found at $PYTHON" >&2
    echo "Create it manually, e.g.:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements.txt -r requirements-dev.txt" >&2
    echo "This script does not create or install into .venv automatically." >&2
    exit 1
fi

echo "Using interpreter: $PYTHON ($($PYTHON --version 2>&1))"

REQUIRED_MISSING=()
for module in pytest numpy; do
    if ! "$PYTHON" -c "import ${module}" >/dev/null 2>&1; then
        REQUIRED_MISSING+=("$module")
    fi
done

if [ "${#REQUIRED_MISSING[@]}" -ne 0 ]; then
    echo "Missing required dependencies in .venv: ${REQUIRED_MISSING[*]}" >&2
    echo "Install them manually, e.g.:" >&2
    echo "  .venv/bin/pip install -r requirements-dev.txt" >&2
    echo "This script does not install packages automatically." >&2
    exit 1
fi

if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "torch not found in .venv - tests that depend on it will report" >&2
    echo "status: \"unavailable\" rather than fail. Install with:" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
fi

exec env PYTHONPATH="$PWD" "$PYTHON" -m pytest -vv agentic_eval/tests tests
