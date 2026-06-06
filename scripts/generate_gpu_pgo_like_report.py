#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path, fallback=None):
    path = Path(path)
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dtype_bucket(dtype):
    return {"float32": "fp32", "f32": "fp32", "float16": "fp16", "f16": "fp16"}.get(dtype, dtype)


def shape_bucket(row):
    shape = row.get("representative_shape") or row.get("shape") or {}
    return f"{shape.get('tokens', '?')}x{shape.get('hidden', '?')}:{dtype_bucket(shape.get('dtype', 'fp32'))}"


def candidate_latency(row):
    for key in ("custom_p95_ms", "custom_latency_ms", "custom_p50_ms"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


def fallback_latency(row):
    for key in ("fallback_p95_ms", "fallback_latency_ms", "fallback_p50_ms"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return value
    return None


def fallback_candidate(row, source):
    latency = fallback_latency(row)
    return {
        "kernel": row.get("fallback_kernel", "torch_rmsnorm"),
        "backend": "PyTorch",
        "source": source,
        "correct": True,
        "selection_ready": latency is not None,
        "p95_ms": latency,
        "p50_ms": row.get("fallback_p50_ms"),
        "mean_ms": row.get("fallback_latency_ms"),
        "effective_bandwidth_gbps": row.get("fallback_effective_bandwidth_gbps"),
        "reason": None if latency is not None else "fallback latency missing",
    }


def custom_candidate(row, source):
    kernel = row.get("custom_kernel")
    if not kernel:
        return None
    backend = "CUDA" if "cuda" in kernel else "Triton" if "triton" in kernel else "GPU"
    latency = candidate_latency(row)
    return {
        "kernel": kernel,
        "backend": backend,
        "source": source,
        "correct": row.get("correct"),
        "selection_ready": bool(row.get("selection_ready")) and latency is not None,
        "p95_ms": latency,
        "p50_ms": row.get("custom_p50_ms"),
        "mean_ms": row.get("custom_latency_ms"),
        "effective_bandwidth_gbps": row.get("custom_effective_bandwidth_gbps"),
        "speedup_vs_fallback": row.get("speedup"),
        "reason": None if latency is not None else "candidate latency missing",
    }


def collect_candidates(profile, source):
    if not profile:
        return {}
    if profile.get("profile_status") != "measured":
        return {
            "unavailable": [
                {
                    "source": source,
                    "profile_status": profile.get("profile_status"),
                    "reason": profile.get("reason", "profile not measured"),
                }
            ]
        }

    by_shape = {}
    for row in profile.get("sweep", []):
        if row.get("fusion_candidate") != "rmsnorm":
            continue
        bucket = shape_bucket(row)
        entry = by_shape.setdefault(bucket, [])
        if not any(candidate["kernel"] == row.get("fallback_kernel", "torch_rmsnorm") for candidate in entry):
            entry.append(fallback_candidate(row, source))
        candidate = custom_candidate(row, source)
        if candidate:
            entry.append(candidate)
    return by_shape


def merge_candidates(*candidate_maps):
    merged = {}
    unavailable = []
    for candidate_map in candidate_maps:
        if not candidate_map:
            continue
        unavailable.extend(candidate_map.get("unavailable", []))
        for bucket, candidates in candidate_map.items():
            if bucket == "unavailable":
                continue
            merged.setdefault(bucket, []).extend(candidates)
    return merged, unavailable


def select_candidate(candidates):
    legal = [
        candidate for candidate in candidates
        if candidate.get("selection_ready") and candidate.get("correct") is not False
    ]
    if not legal:
        fallback = next((candidate for candidate in candidates if candidate.get("kernel") == "torch_rmsnorm"), None)
        return fallback or candidates[0], "fallback_no_legal_profile_candidate"
    winner = min(legal, key=lambda candidate: candidate.get("p95_ms", float("inf")))
    if winner.get("kernel") == "torch_rmsnorm":
        return winner, "gpu_pgo_like_fallback_lowest_p95_latency"
    return winner, "gpu_pgo_like_lowest_p95_latency"


def build_shape_decisions(candidates_by_shape):
    decisions = []
    for bucket, candidates in sorted(candidates_by_shape.items()):
        winner, reason = select_candidate(candidates)
        fallback = next((candidate for candidate in candidates if candidate.get("kernel") == "torch_rmsnorm"), None)
        baseline_ms = fallback.get("p95_ms") if fallback else None
        selected_ms = winner.get("p95_ms")
        delta_ms = (
            round(baseline_ms - selected_ms, 6)
            if isinstance(baseline_ms, (int, float)) and isinstance(selected_ms, (int, float))
            else None
        )
        decisions.append({
            "fusion_candidate": "rmsnorm",
            "shape_bucket": bucket,
            "selected_kernel": winner.get("kernel"),
            "selected_backend": winner.get("backend"),
            "selection_reason": reason,
            "baseline_kernel": "torch_rmsnorm",
            "baseline_p95_ms": baseline_ms,
            "selected_p95_ms": selected_ms,
            "p95_latency_delta_ms": delta_ms,
            "candidate_kernels": candidates,
        })
    return decisions


def representative_decision(decisions):
    for decision in decisions:
        if decision["shape_bucket"] == "16x4096:fp32":
            return decision
    selectable = [d for d in decisions if isinstance(d.get("p95_latency_delta_ms"), (int, float))]
    return max(selectable, key=lambda d: d["p95_latency_delta_ms"], default=decisions[0] if decisions else None)


def serving_impact(decision, runtime_profile, prefill_decode):
    if not decision:
        return {}
    delta_ms = decision.get("p95_latency_delta_ms") or 0.0
    tpot = prefill_decode.get("p95_decode_latency_ms")
    throughput = runtime_profile.get("tokens_per_second")
    optimized_tpot = max(tpot - delta_ms, 0.0) if isinstance(tpot, (int, float)) else None
    throughput_gain = (
        round((tpot / max(optimized_tpot, 1e-9) - 1.0) * throughput, 3)
        if isinstance(tpot, (int, float)) and isinstance(throughput, (int, float)) and optimized_tpot
        else None
    )
    return {
        "method": "single-op p95 latency delta projected onto decode TPOT proxy",
        "shape_bucket": decision["shape_bucket"],
        "selected_kernel": decision["selected_kernel"],
        "baseline_tpot_p95_ms": tpot,
        "projected_tpot_p95_ms": round(optimized_tpot, 6) if optimized_tpot is not None else None,
        "tpot_delta_ms": round(delta_ms, 6),
        "baseline_tokens_per_second": throughput,
        "projected_tokens_per_second_gain": throughput_gain,
        "note": "This is a serving-level projection from kernel p95 evidence; full serving validation should replay the selected kernel in the decode path.",
    }


def write_markdown(path, payload):
    lines = [
        "# GPU PGO-like RMSNorm Feedback Report",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Technology Gate",
        "",
        f"- Input: `{payload['technology_gate']['input']}`",
        f"- Decision: `{payload['technology_gate']['decision']}`",
        f"- Metric: `{payload['technology_gate']['metric']}`",
        "",
        "## Candidate Selection",
        "",
        "| Shape | Selected kernel | Backend | Selected p95 ms | Baseline p95 ms | Delta ms | Reason |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for decision in payload.get("shape_decisions", []):
        lines.append(
            "| {shape} | {kernel} | {backend} | {selected} | {baseline} | {delta} | {reason} |".format(
                shape=decision["shape_bucket"],
                kernel=decision["selected_kernel"],
                backend=decision["selected_backend"],
                selected=decision["selected_p95_ms"],
                baseline=decision["baseline_p95_ms"],
                delta=decision["p95_latency_delta_ms"],
                reason=decision["selection_reason"],
            )
        )
    impact = payload.get("serving_impact", {})
    lines.extend([
        "",
        "## Serving Impact Projection",
        "",
        f"- Baseline TPOT p95: `{impact.get('baseline_tpot_p95_ms')}` ms/token",
        f"- Projected TPOT p95: `{impact.get('projected_tpot_p95_ms')}` ms/token",
        f"- TPOT delta: `{impact.get('tpot_delta_ms')}` ms/token",
        f"- Baseline tokens/sec: `{impact.get('baseline_tokens_per_second')}`",
        f"- Projected tokens/sec gain: `{impact.get('projected_tokens_per_second_gain')}`",
        "",
        "## Remaining Work",
        "",
    ])
    for item in payload.get("remaining_work", []):
        lines.append(f"- {item}")
    write_text(path, "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-profile", default=ROOT / "results/cuda_transformer/rmsnorm_benchmark.json", type=Path)
    parser.add_argument("--triton-profile", default=ROOT / "results/cuda_transformer/rmsnorm_triton_benchmark.json", type=Path)
    parser.add_argument("--runtime-profile", default=ROOT / "results/llm_runtime_artifacts/runtime_profile.json", type=Path)
    parser.add_argument("--prefill-decode", default=ROOT / "results/llm_runtime_artifacts/prefill_decode_benchmark.json", type=Path)
    parser.add_argument("--output", default=ROOT / "results/cuda_transformer/gpu_pgo_like_rmsnorm_report.json", type=Path)
    parser.add_argument("--report-output", default=ROOT / "results/cuda_transformer/gpu_pgo_like_rmsnorm_report.md", type=Path)
    args = parser.parse_args()

    cuda_profile = load_json(args.cuda_profile, {})
    triton_profile = load_json(args.triton_profile, {})
    runtime_profile = load_json(args.runtime_profile, {})
    prefill_decode = load_json(args.prefill_decode, {})

    candidates_by_shape, unavailable = merge_candidates(
        collect_candidates(cuda_profile, str(args.cuda_profile)),
        collect_candidates(triton_profile, str(args.triton_profile)),
    )
    decisions = build_shape_decisions(candidates_by_shape)
    representative = representative_decision(decisions)

    payload = {
        "artifact_type": "gpu_pgo_like_kernel_selection_report",
        "format": "runtime.gpu_pgo_like.v1",
        "status": "passed" if decisions else "skipped",
        "technology_gate": {
            "technology": "gpu_pgo_like_kernel_selection",
            "input": "compiler-emitted HIR RMSNorm op plus runtime shape/workload distribution",
            "decision": "profile-guided kernel selection among CUDA/Triton/PyTorch candidates by shape bucket",
            "metric": "kernel p95 latency, effective bandwidth, TPOT projection, throughput projection",
            "passes_gate": bool(decisions),
        },
        "input_sources": {
            "cuda_profile": str(args.cuda_profile),
            "triton_profile": str(args.triton_profile),
            "runtime_profile": str(args.runtime_profile),
            "prefill_decode": str(args.prefill_decode),
        },
        "profile_status": {
            "cuda": cuda_profile.get("profile_status", "missing"),
            "triton": triton_profile.get("profile_status", "missing"),
            "unavailable_candidates": unavailable,
        },
        "shape_decisions": decisions,
        "representative_decision": representative,
        "serving_impact": serving_impact(representative, runtime_profile, prefill_decode),
        "remaining_work": [
            "Replay selected RMSNorm kernel inside a full decode loop instead of projecting from per-kernel p95 latency.",
            "Add fp16/bf16 candidate rows and vectorized CUDA candidate when kernels are implemented.",
            "Attach real Nsight Compute occupancy/DRAM/stall metrics when ncu capture is available.",
        ],
    }
    write_json(args.output, payload)
    write_markdown(args.report_output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
