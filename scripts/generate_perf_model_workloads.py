#!/usr/bin/env python3
"""Generate the small A/B/C/D perf-model workload manifest for Slice E2E-2.

Prompts are submitted as exact token-ID arrays (not text) so prompt token
counts are exact by construction, not approximated from text length.
`ignore_eos` is set so every request produces exactly `output_tokens`
tokens, removing early-stop variance as a TPOT/E2E measurement confound.
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

# workload_id -> (prompt_tokens_target, output_tokens, concurrency, request_count)
SPECS = {
    "A": (32, 64, 1, 12),   # decode dominated
    "B": (512, 16, 1, 12),  # prefill dominated
    "C": (128, 32, 4, 16),  # moderate concurrency
    "D": (128, 32, 8, 16),  # higher concurrency (memory-safety checked at run time)
}
WARMUP_REQUESTS = 2
SEED = 20260726


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exact_prompt_token_ids(tokenizer, target_tokens: int) -> list[int]:
    text = FILLER
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += FILLER
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    assert len(ids) == target_tokens
    return ids


def build_workload(tokenizer, workload_id: str) -> dict:
    prompt_tokens_target, output_tokens, concurrency, request_count = SPECS[workload_id]
    requests = []
    for i in range(request_count):
        token_ids = exact_prompt_token_ids(tokenizer, prompt_tokens_target)
        metadata = {
            "request_id": f"{workload_id}-R{i:03d}",
            "prompt_token_count": len(token_ids),
            "requested_output_token_count": output_tokens,
            "arrival_offset_ms": 0,
            "priority": 0,
            "seed": SEED,
        }
        row = {
            "prompt": token_ids,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "seed": SEED,
            "ignore_eos": True,
            "metadata": metadata,
        }
        metadata["sha256"] = digest(row)
        requests.append(row)
    payload = {
        "workload_id": workload_id,
        "description": {
            "A": "decode_dominated", "B": "prefill_dominated",
            "C": "moderate_concurrency", "D": "higher_safe_concurrency",
        }[workload_id],
        "prompt_tokens_target": prompt_tokens_target,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "warmup_requests": WARMUP_REQUESTS,
        "measured_requests": request_count - WARMUP_REQUESTS,
        "arrival_schedule": "closed_loop_immediate_submission",
        "requests": requests,
    }
    payload["sha256"] = digest(requests)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", nargs="+", default=list(SPECS.keys()))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    workloads = [build_workload(tokenizer, wid) for wid in args.workloads]
    manifest = {
        "schema_version": "perf_model.workloads.v1", "model": args.model, "random_seed": SEED,
        "workloads": workloads,
    }
    manifest["sha256"] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} workloads={[w['workload_id'] for w in workloads]}")


if __name__ == "__main__":
    main()
