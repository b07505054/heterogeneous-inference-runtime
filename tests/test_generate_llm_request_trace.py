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


# ---------------------------------------------------------------------------
# --common-prefix
# ---------------------------------------------------------------------------

def test_common_prefix_prepended_to_every_prompt():
    prefix = "SYSTEM: You are a helpful assistant.\n\n"
    rows = generate_rows(
        num_requests=8,
        prompt_set="small",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=0,
        common_prefix=prefix,
    )
    for row in rows:
        assert row["prompt"].startswith(prefix), f"prefix missing: {row['prompt'][:40]!r}"


def test_common_prefix_empty_by_default_leaves_prompts_unchanged():
    rows_no_prefix = generate_rows(
        num_requests=8,
        prompt_set="small",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=5,
    )
    rows_empty_prefix = generate_rows(
        num_requests=8,
        prompt_set="small",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=5,
        common_prefix="",
    )
    assert rows_no_prefix == rows_empty_prefix


def test_common_prefix_does_not_change_arrival_or_max_tokens():
    rows = generate_rows(
        num_requests=4,
        prompt_set="mixed",
        max_tokens=128,
        arrival_pattern="burst",
        seed=3,
        common_prefix="PREFIX ",
    )
    for i, row in enumerate(rows):
        assert row["max_tokens"] == 128
        assert isinstance(row["arrival_ms"], int)


def test_common_prefix_is_deterministic():
    kwargs = dict(num_requests=6, prompt_set="coding", max_tokens=32, arrival_pattern="uniform", seed=9, common_prefix="TEST ")
    assert generate_rows(**kwargs) == generate_rows(**kwargs)


# ---------------------------------------------------------------------------
# --unique-prompts
# ---------------------------------------------------------------------------

def test_unique_prompts_limits_prompt_diversity():
    rows = generate_rows(
        num_requests=32,
        prompt_set="mixed",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=0,
        unique_prompts=2,
    )
    unique = {row["prompt"] for row in rows}
    assert len(unique) <= 2


def test_unique_prompts_none_uses_full_set():
    rows = generate_rows(
        num_requests=32,
        prompt_set="mixed",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=0,
    )
    unique = {row["prompt"] for row in rows}
    assert len(unique) > 2


def test_unique_prompts_larger_than_set_uses_full_set():
    rows = generate_rows(
        num_requests=16,
        prompt_set="small",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=0,
        unique_prompts=100,
    )
    # small set has 4 prompts; capping at 100 should still use at most 4
    unique = {row["prompt"] for row in rows}
    assert len(unique) <= 4


def test_unique_prompts_one_produces_single_prompt_trace():
    rows = generate_rows(
        num_requests=8,
        prompt_set="coding",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=0,
        unique_prompts=1,
    )
    unique = {row["prompt"] for row in rows}
    assert len(unique) == 1


def test_shared_prefix_workload_is_cache_friendly():
    prefix = "You are an assistant. Answer the following:\n\n"
    rows = generate_rows(
        num_requests=32,
        prompt_set="mixed",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=0,
        common_prefix=prefix,
        unique_prompts=4,
    )
    assert len(rows) == 32
    for row in rows:
        assert row["prompt"].startswith(prefix)
    unique = {row["prompt"] for row in rows}
    assert len(unique) <= 4


def test_no_shared_prefix_workload_uses_all_mixed_prompts():
    rows = generate_rows(
        num_requests=32,
        prompt_set="mixed",
        max_tokens=64,
        arrival_pattern="uniform",
        seed=0,
    )
    unique = {row["prompt"] for row in rows}
    assert len(unique) >= 4


def test_unique_prompts_cli_flag_accepted():
    parser = build_parser()
    args = parser.parse_args(["--unique-prompts", "3"])
    assert args.unique_prompts == 3


def test_unique_prompts_zero_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--unique-prompts", "0"])
