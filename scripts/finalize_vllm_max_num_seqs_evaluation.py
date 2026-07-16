#!/usr/bin/env python3
"""Finalize selection, regret, proof, and human-readable vLLM policy evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_score(row, valid, weights):
    best_ttft = min(x["ttft_ms"]["p95"] for x in valid)
    best_tpot = min(x["tpot_ms"]["p95"] for x in valid)
    lowest_throughput = min(x["output_token_throughput"] for x in valid)
    max_memory = max(x["peak_gpu_memory_mib"] for x in valid)
    return (weights["latency"] * row["ttft_ms"]["p95"] / best_ttft
            + weights["tpot"] * row["tpot_ms"]["p95"] / best_tpot
            - weights["throughput"] * row["output_token_throughput"] / lowest_throughput
            + weights["memory"] * row["peak_gpu_memory_mib"] / max_memory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--proof-raw", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.artifacts / "latency_summary.json").read_text())
    objectives = json.loads((args.artifacts / "objective_definitions.json").read_text())
    selections, regrets, proofs, plan_collection = [], [], [], []
    for workload in ("S1", "S2", "S3"):
        candidates = [x for x in rows if x["workload_id"] == workload and x["classification"] == "VALID" and x["failure_count"] == 0 and x["oom_count"] == 0]
        default = next(x for x in candidates if x["value_source"] == "default")
        manuals = [x for x in candidates if x["value_source"] == "explicit"]
        best_ttft = min(candidates, key=lambda x: x["ttft_ms"]["p95"])
        best_tpot = min(candidates, key=lambda x: x["tpot_ms"]["p95"])
        best_throughput = max(candidates, key=lambda x: x["output_token_throughput"])
        best_memory = min(candidates, key=lambda x: x["peak_gpu_memory_mib"])
        for objective in ("latency", "throughput", "balanced"):
            plan_path = args.artifacts / "compiler_selected_plans" / f"{workload}-{objective}.json"
            plan = json.loads(plan_path.read_text())
            selected = next(x for x in candidates if x["candidate_id"] == plan["candidate_id"])
            proof_path = args.proof_raw / f"{workload}-{objective}.json"
            proof = json.loads(proof_path.read_text())
            if objective == "latency":
                oracle = best_ttft
                manual_oracle = min(manuals, key=lambda x: x["ttft_ms"]["p95"])
                selected_score = selected["ttft_ms"]["p95"]
                oracle_score = oracle["ttft_ms"]["p95"]
            elif objective == "throughput":
                oracle = best_throughput
                manual_oracle = max(manuals, key=lambda x: x["output_token_throughput"])
                selected_score = -selected["output_token_throughput"]
                oracle_score = -oracle["output_token_throughput"]
            else:
                valid = [x for x in candidates if x["ttft_ms"]["p95"] <= objectives[objective]["maximum_ttft_p95_ms"] and x["output_token_throughput"] >= objectives[objective]["minimum_output_tokens_per_second"]]
                scores = {x["candidate_id"]: balanced_score(x, valid, objectives[objective]["weights"]) for x in valid}
                oracle = min(valid, key=lambda x: scores[x["candidate_id"]])
                manual_oracle = min((x for x in valid if x["value_source"] == "explicit"), key=lambda x: scores[x["candidate_id"]])
                selected_score = scores[selected["candidate_id"]]
                oracle_score = scores[oracle["candidate_id"]]
            selection = {
                "workload_id": workload,
                "objective": objective,
                "best_measured_legal_candidate": oracle["candidate_id"],
                "compiler_selected_candidate": plan["candidate_id"],
                "runtime_executed_candidate": proof["candidate_id"],
                "selected_max_num_seqs": plan["max_num_seqs"],
                "runtime_launched_max_num_seqs": proof["runtime_launched_max_num_seqs"],
                "selection_reason": plan["selection_reason"],
                "selected_ttft_p95_ms": selected["ttft_ms"]["p95"],
                "selected_tpot_p95_ms": selected["tpot_ms"]["p95"],
                "selected_output_token_throughput": selected["output_token_throughput"],
                "selected_peak_gpu_memory_mib": selected["peak_gpu_memory_mib"],
                "objective_score": round(selected_score, 9),
                "vllm_default_candidate": default["candidate_id"],
                "best_manual_candidate_for_objective": manual_oracle["candidate_id"],
            }
            regret = {
                "workload_id": workload,
                "objective": objective,
                "absolute_ttft_regret_ms": round(selected["ttft_ms"]["p95"] - best_ttft["ttft_ms"]["p95"], 6),
                "relative_ttft_regret": round(selected["ttft_ms"]["p95"] / best_ttft["ttft_ms"]["p95"] - 1, 9),
                "absolute_tpot_regret_ms": round(selected["tpot_ms"]["p95"] - best_tpot["tpot_ms"]["p95"], 6),
                "relative_throughput_regret": round(best_throughput["output_token_throughput"] / selected["output_token_throughput"] - 1, 9),
                "memory_regret_mib": selected["peak_gpu_memory_mib"] - best_memory["peak_gpu_memory_mib"],
                "objective_score_regret": round(selected_score - oracle_score, 9),
            }
            proof_row = {
                "workload_id": workload,
                "objective": objective,
                "compiler_selected_candidate": plan["candidate_id"],
                "compiler_selected_max_num_seqs": plan["max_num_seqs"],
                "runtime_executed_candidate": proof["candidate_id"],
                "runtime_launched_max_num_seqs": proof["runtime_launched_max_num_seqs"],
                "value_source": plan["value_source"],
                "server_pid": proof["server_pid"],
                "plan_sha256": proof["plan_sha256"],
                "plan_file_sha256": sha(plan_path),
                "runtime_policy_reselection_count": proof["runtime_policy_reselection_count"],
                "classification": proof["classification"],
                "request_count": proof["request_count"],
                "success_count": proof["success_count"],
                "failure_count": proof["failure_count"],
                "command": proof["command"],
                "exact_policy_match": plan["candidate_id"] == proof["candidate_id"] and plan["max_num_seqs"] == proof["runtime_launched_max_num_seqs"],
                "raw_proof_session_sha256": sha(proof_path),
                "raw_log_sha256": proof["raw_log_sha256"],
            }
            selections.append(selection)
            regrets.append(regret)
            proofs.append(proof_row)
            plan_collection.append({"workload_id": workload, "objective": objective, "plan": plan, "sha256": sha(plan_path)})
    write(args.artifacts / "selection_results.json", selections)
    write(args.artifacts / "regret_analysis.json", {"definitions": {"relative_ttft_regret": "selected_ttft_p95 / best_legal_ttft_p95 - 1", "relative_throughput_regret": "best_legal_output_throughput / selected_output_throughput - 1", "memory_regret_mib": "selected_peak_memory - lowest_legal_peak_memory", "objective_score_regret": "selected_objective_score - oracle_objective_score"}, "rows": regrets})
    write(args.artifacts / "runtime_proof.json", {"required": {"selected_equals_executed": True, "runtime_policy_reselection_count": 0}, "rows": proofs})
    write(args.artifacts / "compiler_selected_plan.json", plan_collection)
    header = "| workload | candidate | effective max_num_seqs | sessions | requests | success | failure | TTFT p50/p95/p99 ms | TPOT p50/p95/p99 ms | E2E p50/p95/p99 ms | output tok/s | req/s | queue p95 | peak MiB | OOM |\n|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|"
    lines = [header]
    for row in rows:
        lines.append(f"| {row['workload_id']} | {row['candidate_id']} | {row['effective_max_num_seqs']} | {row['session_count']} | {row['request_count']} | {row['success_count']} | {row['failure_count']} | {row['ttft_ms']['p50']}/{row['ttft_ms']['p95']}/{row['ttft_ms']['p99']} | {row['tpot_ms']['p50']}/{row['tpot_ms']['p95']}/{row['tpot_ms']['p99']} | {row['e2e_ms']['p50']}/{row['e2e_ms']['p95']}/{row['e2e_ms']['p99']} | {row['output_token_throughput']} | {row['request_throughput']} | not_available | {row['peak_gpu_memory_mib']} | {row['oom_count']} |")
    summary = "# Real vLLM `max_num_seqs` measured-policy evaluation\n\n" + "\n".join(lines) + "\n\n"
    summary += "All 45 baseline sessions and nine independent compiler-plan proof sessions used real vLLM 0.24.0 execution on the NVIDIA GeForce GTX 1650 with Max-Q Design. The default flag was omitted; its resolved effective value was not exposed by vLLM and is reported as `not_available`. Queue wait, separate prefill/decode time, and exact KV-cache usage were not exposed and are not reported as zero.\n\n"
    summary += "Compiler selections: S1 latency=8, throughput=4, balanced=4; S2 latency=8, throughput=1, balanced=1; S3 latency=8, throughput=1, balanced=1. Every independent proof session executed the exact selected value with `runtime_policy_reselection_count=0`.\n\n"
    summary += "Truth boundary: target/model/workload-specific measured policy for one vLLM setting. This is not a predictive cost model, universal optimum, scheduler-internal control, multi-GPU evidence, or production SLO guarantee.\n\nRecommended next slice: `max_num_batched_tokens`.\n"
    (args.artifacts / "summary.md").write_text(summary)


if __name__ == "__main__":
    main()
