#!/usr/bin/env python3
"""E2E-4 workload: one fixed anchor (decode-heavy, long enough to observe a
pre-admission baseline, a post-admission stall, and recovery) plus, for each
swept admitted-prompt-length, up to 2 distinct admission requests (Factor B:
multiplicity). The anchor is held constant across the whole prompt-length
sweep so Factor A varies only the admitted requests' prefill size, isolating
its effect on the anchor's decode timeline.
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
ANCHOR_PROMPT_TOKENS = 128
ANCHOR_OUTPUT_TOKENS = 96
ADMISSION_OUTPUT_TOKENS = 32
PROMPT_LENGTHS = (32, 64, 128, 256, 512)
MAX_MULTIPLICITY = 2
SEED = 20260727


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exact_prompt_token_ids(tokenizer, target_tokens: int) -> list[int]:
    text = FILLER
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += FILLER
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    assert len(ids) == target_tokens
    return ids


def build_request(token_ids: list[int], output_tokens: int, request_id: str) -> dict:
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

    anchor = build_request(exact_prompt_token_ids(tokenizer, ANCHOR_PROMPT_TOKENS), ANCHOR_OUTPUT_TOKENS, "ANCHOR")

    admission_pools = {}
    for length in PROMPT_LENGTHS:
        admission_pools[str(length)] = [
            build_request(exact_prompt_token_ids(tokenizer, length), ADMISSION_OUTPUT_TOKENS, f"ADMIT-{length}-{i}")
            for i in range(MAX_MULTIPLICITY)
        ]

    manifest = {
        "schema_version": "perf_model.e2e4.workloads.v1", "model": args.model, "random_seed": SEED,
        "anchor_prompt_tokens": ANCHOR_PROMPT_TOKENS, "anchor_output_tokens": ANCHOR_OUTPUT_TOKENS,
        "admission_output_tokens": ADMISSION_OUTPUT_TOKENS, "prompt_lengths": list(PROMPT_LENGTHS),
        "max_multiplicity": MAX_MULTIPLICITY, "anchor": anchor, "admission_pools": admission_pools,
    }
    manifest["sha256"] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} prompt_lengths={PROMPT_LENGTHS}")


if __name__ == "__main__":
    main()
