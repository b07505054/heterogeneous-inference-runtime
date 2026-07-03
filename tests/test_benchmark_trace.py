from pathlib import Path

import pytest

from benchmark.trace import load_jsonl_trace, normalize_openai_trace


def test_load_jsonl_trace_skips_blank_and_comment_lines(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    path.write_text('\n# comment\n{"prompt": "hi", "max_tokens": 2}\n', encoding="utf-8")
    rows = load_jsonl_trace(path)
    assert rows == [{"prompt": "hi", "max_tokens": 2}]
    assert normalize_openai_trace(rows) == [{"prompt": "hi", "max_tokens": 2}]


def test_load_jsonl_trace_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSONL"):
        load_jsonl_trace(path)


def test_normalize_openai_trace_requires_prompt_or_messages():
    with pytest.raises(ValueError, match="messages or prompt"):
        normalize_openai_trace([{"max_tokens": 1}])

