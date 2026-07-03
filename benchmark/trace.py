from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_jsonl_trace(path: str | Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {line_no}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {line_no} must be an object")
        rows.append(row)
    return rows


def normalize_openai_trace(rows: Iterable[dict]) -> list[dict]:
    normalized = []
    for index, row in enumerate(rows):
        if "messages" in row:
            request = {"messages": row["messages"]}
        elif "prompt" in row:
            request = {"prompt": row["prompt"]}
        else:
            raise ValueError(f"trace row {index} must include messages or prompt")
        if "max_tokens" in row:
            request["max_tokens"] = int(row["max_tokens"])
        if "temperature" in row:
            request["temperature"] = float(row["temperature"])
        if "metadata" in row:
            request["metadata"] = row["metadata"]
        normalized.append(request)
    return normalized

