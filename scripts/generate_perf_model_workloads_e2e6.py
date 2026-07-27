#!/usr/bin/env python3
"""E2E-6 workload: pools of up to 8 IDENTICAL requests per prompt length, used
for simultaneous (non-staggered) steady-state batch-size experiments. No
"anchor" vs "admission" distinction this slice -- all N requests are peers.
"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "the compiler schedules kernels across heterogeneous accelerators. "
)
PROMPT_LENGTHS = (32, 64, 128, 512)
OUTPUT_TOKENS = 96
MAX_BATCH = 8
SEED = 20260729


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exact_prompt_token_ids(tokenizer, target_tokens: int) -> list[int]:
    text = FILLER
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += FILLER
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    assert len(ids) == target_tokens
    return ids


def build_request(token_ids, output_tokens, request_id) -> dict:
    metadata = {"request_id": request_id, "prompt_token_count": len(token_ids),
                "requested_output_token_count": output_tokens, "seed": SEED}
    row = {"prompt": token_ids, "max_tokens": output_tokens, "temperature": 0.0, "seed": SEED,
           "ignore_eos": True, "metadata": metadata}
    metadata["sha256"] = digest(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    pools = {}
    for length in PROMPT_LENGTHS:
        ids = exact_prompt_token_ids(tokenizer, length)
        pools[str(length)] = [build_request(ids, OUTPUT_TOKENS, f"L{length}-R{i}") for i in range(MAX_BATCH)]

    manifest = {
        "schema_version": "perf_model.e2e6.workloads.v1", "model": args.model, "random_seed": SEED,
        "prompt_lengths": list(PROMPT_LENGTHS), "output_tokens": OUTPUT_TOKENS, "max_batch": MAX_BATCH,
        "pools": pools,
    }
    manifest["sha256"] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} prompt_lengths={PROMPT_LENGTHS}")


if __name__ == "__main__":
    main()
