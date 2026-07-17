#!/usr/bin/env python3
"""Merge measured CUDA/Triton RMSNorm sweeps into one exact-candidate artifact."""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    cuda, triton = json.loads(args.cuda.read_text()), json.loads(args.triton.read_text())
    if cuda.get("profile_status") != "measured" or triton.get("profile_status") != "measured":
        raise SystemExit("both input profiles must be measured")
    rows = []
    for source_path, payload in ((args.cuda, cuda), (args.triton, triton)):
        for original in payload["exact_candidates"]:
            row = dict(original)
            row["artifact_hash"] = digest(source_path)
            row["source_artifact"] = str(source_path)
            row["target"] = {key: payload["environment"].get(key) for key in ("gpu_name", "gpu_uuid", "compute_capability")}
            row["measurement_kind"] = "measured"
            rows.append(row)
    # One independently timed PyTorch fallback row per shape; CUDA profile is
    # used consistently so fallback measurements are not mixed across sessions.
    seen = set()
    for source in cuda["exact_candidates"]:
        key = (source["tokens"], source["hidden"])
        if key in seen: continue
        seen.add(key)
        rows.append({"candidate_id": "torch_rmsnorm_fp32_v1", "operator": "rmsnorm", "semantics": "weighted_rmsnorm",
            "backend": "torch", "kernel_family": "pytorch_rmsnorm_fallback", "kernel_entry_point": "torch_rmsnorm",
            "dtype": "fp32", "tokens": key[0], "hidden": key[1], "epsilon": source["epsilon"],
            "block_size": None, "num_warps": None, "num_stages": None, "source_hash": None,
            "artifact_hash": digest(args.cuda), "source_artifact": str(args.cuda),
            "target": {key: cuda["environment"].get(key) for key in ("gpu_name", "gpu_uuid", "compute_capability")},
            "mean_ms": source["fallback_latency_ms"], "p50_ms": source["fallback_p50_ms"], "p95_ms": source["fallback_p95_ms"],
            "min_ms": source["fallback_min_ms"], "max_ms": source["fallback_max_ms"],
            "effective_bandwidth_gbps": source["fallback_effective_bandwidth_gbps"], "correct": True,
            "selection_ready": True, "failure_reason": None, "measurement_kind": "measured"})
    for row in rows:
        row.setdefault("mean_ms", row.get("custom_latency_ms")); row.setdefault("p50_ms", row.get("custom_p50_ms")); row.setdefault("p95_ms", row.get("custom_p95_ms"))
        row.setdefault("min_ms", row.get("custom_min_ms")); row.setdefault("max_ms", row.get("custom_max_ms")); row.setdefault("effective_bandwidth_gbps", row.get("custom_effective_bandwidth_gbps"))
    winners = []
    for tokens, hidden in sorted(seen):
        legal = [r for r in rows if r["tokens"] == tokens and r["hidden"] == hidden and r["correct"] and r["selection_ready"] and r["p50_ms"] is not None]
        legal.sort(key=lambda r: (r["p50_ms"], r["candidate_id"]))
        winners.append({"tokens": tokens, "hidden": hidden, "winner": legal[0]["candidate_id"], "backend": legal[0]["backend"], "p50_ms": legal[0]["p50_ms"], "p95_ms": legal[0]["p95_ms"]})
    payload = {"artifact_type": "rmsnorm_exact_candidate_benchmark", "format": "rmsnorm.exact_candidates.v1",
        "semantics": "weighted_rmsnorm", "dtype": "fp32", "epsilon": 1e-6,
        "environment": cuda["environment"], "input_artifacts": [{"path": str(args.cuda), "sha256": digest(args.cuda)}, {"path": str(args.triton), "sha256": digest(args.triton)}],
        "methodology": {"timing": "CUDA events", "cuda_warmup": cuda["benchmark_config"]["warmup"], "cuda_runs": cuda["benchmark_config"]["runs"], "triton_warmup": triton["benchmark_config"]["warmup"], "triton_runs": triton["benchmark_config"]["runs"], "correctness": {"reference": "independent weighted PyTorch expression", "rtol": 1e-4, "atol": 1e-4}},
        "exact_candidates": rows, "winner_by_shape": winners, "winner_distribution": dict(Counter(w["backend"] for w in winners))}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["tokens","hidden","candidate_id","backend","block_size","num_warps","num_stages","mean_ms","p50_ms","p95_ms","effective_bandwidth_gbps","correct","selection_ready","failure_reason"]
    with args.csv_output.open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows({k:r.get(k) for k in fields} for r in rows)
    lines=["# Exact RMSNorm GPU Candidate Benchmark","", "Measured operator-level weighted FP32 RMSNorm; not full-model integration.","", "| Tokens | Hidden | Winner | Backend | p50 ms | p95 ms |","|---:|---:|---|---|---:|---:|"]
    lines += [f"| {w['tokens']} | {w['hidden']} | `{w['winner']}` | {w['backend']} | {w['p50_ms']} | {w['p95_ms']} |" for w in winners]
    args.report_output.write_text("\n".join(lines)+"\n")

if __name__ == "__main__": main()
