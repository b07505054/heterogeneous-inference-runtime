#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -x .venv/bin/python ]]; then
  echo "error: .venv/bin/python not found; create the venv first" >&2
  exit 1
fi

exec env PYTHONPATH="$PWD" .venv/bin/python -m pytest -vv agentic_eval/tests tests
