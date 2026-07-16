#!/usr/bin/env python3
"""Generate the deterministic, single-dimension vLLM workload manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


PROMPTS = {
    "short": [
        "State one practical use of a compiler in one sentence.",
        "Name the largest planet and give one fact about it.",
        "Explain caching in one concise sentence.",
        "Give two safe habits for handling passwords.",
    ],
    "medium": [
        "A service receives requests from several users at once. Explain in three concise sentences how batching can improve throughput and also increase queueing latency.",
        "Compare a contiguous array with a paged representation. Give two advantages and two disadvantages, using plain language.",
        "Describe how to validate a performance benchmark so that setup, warmup, and measured work are not accidentally mixed.",
        "Summarize why deterministic request traces are useful when comparing server configurations across repeated sessions.",
    ],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request(tokenizer, workload, index, kind, output_tokens, seed):
    prompt = PROMPTS[kind][index % len(PROMPTS[kind])]
    metadata = {
        "request_id": f"{workload}-R{index:03d}",
        "prompt_token_count": len(tokenizer.encode(prompt, add_special_tokens=False)),
        "requested_output_token_count": output_tokens,
        "arrival_offset_ms": 0,
        "priority": 0,
        "seed": seed,
    }
    row = {"prompt": prompt, "max_tokens": output_tokens, "temperature": 0.0, "seed": seed, "metadata": metadata}
    metadata["sha256"] = digest(row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seed = 20260715
    specs = (("S1", 1, "short", 16), ("S2", 4, "medium", 24), ("S3", 8, "medium", 32))
    workloads = []
    for workload_id, concurrency, kind, output_tokens in specs:
        rows = [request(tokenizer, workload_id, i, kind, output_tokens, seed) for i in range(55)]
        workloads.append({
            "workload_id": workload_id,
            "concurrency": concurrency,
            "warmup_requests": 5,
            "measured_requests": 50,
            "arrival_schedule": "closed_loop_immediate_submission",
            "requests": rows,
            "sha256": digest(rows),
        })
    manifest = {"schema_version": "vllm.max_num_seqs.workloads.v1", "model": args.model, "random_seed": seed, "workloads": workloads}
    manifest["sha256"] = digest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
