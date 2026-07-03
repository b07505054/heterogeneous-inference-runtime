import json

import pytest

from scripts.generate_llm_request_trace import build_parser, generate_rows, write_jsonl


def test_generate_trace_is_deterministic_with_same_seed():
    args = {
        "num_requests": 8,
        "prompt_set": "small",
        "max_tokens": 64,
        "arrival_pattern": "uniform",
        "seed": 7,
    }
    assert generate_rows(**args) == generate_rows(**args)


def test_generated_rows_have_valid_jsonl_schema(tmp_path):
    rows = generate_rows(
        num_requests=3,
        prompt_set="coding",
        max_tokens=32,
        arrival_pattern="uniform",
        seed=0,
    )
    path = tmp_path / "trace.jsonl"
    write_jsonl(path, rows)
    loaded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(loaded) == 3
    for index, row in enumerate(loaded):
        assert row["request_id"] == f"req_{index:04d}"
        assert isinstance(row["prompt"], str)
        assert row["max_tokens"] == 32
        assert isinstance(row["arrival_ms"], int)


def test_arrival_ms_is_monotonic_for_uniform_and_burst():
    for pattern in ("uniform", "burst"):
        rows = generate_rows(
            num_requests=16,
            prompt_set="mixed",
            max_tokens=64,
            arrival_pattern=pattern,
            seed=0,
        )
        arrivals = [row["arrival_ms"] for row in rows]
        assert arrivals == sorted(arrivals)


def test_prompt_set_selection_works():
    small = generate_rows(
        num_requests=8,
        prompt_set="small",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=1,
    )
    coding = generate_rows(
        num_requests=8,
        prompt_set="coding",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=1,
    )
    assert {row["prompt"] for row in small} != {row["prompt"] for row in coding}


def test_invalid_cli_args_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--prompt-set", "invalid"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--arrival-pattern", "invalid"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--num-requests", "0"])
