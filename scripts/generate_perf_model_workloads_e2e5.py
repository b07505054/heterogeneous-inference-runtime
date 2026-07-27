#!/usr/bin/env python3
"""E2E-5 workload: up to 2 identical anchor requests (128 prompt / 96 output,
decode-heavy, long enough for baseline + post-admission + recovery) and a
pool of up to 4 identical admission requests (128 prompt / 32 output).
Prompt length is deliberately NOT varied this slice (E2E-4 already showed it
doesn't matter) -- the only swept factors are counts, not sizes.
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
PROMPT_TOKENS = 128
ANCHOR_OUTPUT_TOKENS = 96
ADMISSION_OUTPUT_TOKENS = 32
MAX_ANCHORS = 2
MAX_ADMISSIONS = 4
SEED = 20260728


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
    prompt_ids = exact_prompt_token_ids(tokenizer, PROMPT_TOKENS)

    anchors = [build_request(prompt_ids, ANCHOR_OUTPUT_TOKENS, f"ANCHOR-{i}") for i in range(MAX_ANCHORS)]
    admissions = [build_request(prompt_ids, ADMISSION_OUTPUT_TOKENS, f"ADMIT-{i}") for i in range(MAX_ADMISSIONS)]

    manifest = {
        "schema_version": "perf_model.e2e5.workloads.v1", "model": args.model, "random_seed": SEED,
        "prompt_tokens": PROMPT_TOKENS, "anchor_output_tokens": ANCHOR_OUTPUT_TOKENS,
        "admission_output_tokens": ADMISSION_OUTPUT_TOKENS, "anchors": anchors, "admissions": admissions,
    }
    manifest["sha256"] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} anchors={len(anchors)} admissions={len(admissions)}")


if __name__ == "__main__":
    main()
