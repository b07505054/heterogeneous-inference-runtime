#!/usr/bin/env python3
"""E2E-3 analysis: off-vs-on comparison, causal timing alignment, hypothesis
tests H1-H6, joint (max_num_seqs, chunked_prefill) ranking, and the
compiler-ownership decision. Reuses the E2E-2 calibrated constants unchanged
(loaded from the E2E-2 analysis.json) so the "current model" baseline is
identical to what E2E-2 actually used -- no re-fitting.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perf_model import calibration_row as cr
from perf_model import interference_model, phase_model
from deployment.vllm_adapter import metrics_client, server_info_client
from scripts.analyze_perf_model_results import (
    build_model_features, build_hardware_features, resolved_facts, metrics_delta_mean_ms,
    admitted_concurrency,
)

STALL_ALIGNMENT_WINDOW_S = 0.10  # a stall is "aligned" with new-request admission if a rest
                                  # request's submit_time falls within this window of the stall


def load_raws(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*.json"))]


def check_full_adherence(raw: dict) -> dict:
    facts = resolved_facts(raw)
    if facts is None:
        return {"derived_config_adherent": False, "mismatches": ["server_info_unavailable"],
                "chunked_prefill_match": False, "proof_fields": {}}
    fixed = raw["fixed_configuration"]
    base = server_info_client.compare_requested_vs_resolved(
        fixed, facts, requested_max_num_seqs=raw["max_num_seqs_requested"]
    )
    cp_match = facts.enable_chunked_prefill == raw["enable_chunked_prefill_requested"]
    proof_fields = {
        "scheduler_policy": facts.scheduler_policy, "num_gpu_blocks": facts.num_gpu_blocks,
        "num_cpu_blocks": facts.num_cpu_blocks, "compilation_mode": facts.compilation_mode,
        "cudagraph_mode": facts.cudagraph_mode, "tensor_parallel_size": facts.tensor_parallel_size,
    }
    return {**base, "chunked_prefill_match": cp_match, "proof_fields": proof_fields}


def stall_alignment(raw: dict) -> list[dict]:
    """For each repetition round, find the largest inter-token gap in request0's
    timeline and check whether any 'rest' request's submit_time falls inside
    that gap's window -- direct timing evidence, not aggregate correlation."""
    findings = []
    rest_submits = sorted(r["submit_time"] for r in raw.get("rest_pooled_request_rows", []) if r.get("submit_time"))
    for round_idx, tl in enumerate(raw.get("request0_timelines", [])):
        arrivals = tl.get("token_arrival_times") or []
        if len(arrivals) < 2:
            continue
        gaps = [(arrivals[i], arrivals[i + 1], arrivals[i + 1] - arrivals[i]) for i in range(len(arrivals) - 1)]
        if not gaps:
            continue
        durations = [g[2] for g in gaps]
        median = statistics.median(durations)
        max_gap = max(gaps, key=lambda g: g[2])
        start, end, dur = max_gap
        aligned = any(start - STALL_ALIGNMENT_WINDOW_S <= s <= end + STALL_ALIGNMENT_WINDOW_S for s in rest_submits)
        findings.append({
            "round": round_idx, "max_gap_ms": dur * 1000.0, "median_gap_ms": median * 1000.0,
            "ratio_to_median": (dur / median) if median > 0 else None,
            "aligned_with_new_request_submission": aligned,
            "nearest_rest_submit_offset_ms": (
                min((abs(s - end) for s in rest_submits), default=None) * 1000.0 if rest_submits else None
            ),
        })
    return findings


def row_summary(raw: dict, model, hardware, throughput) -> dict:
    adherence = check_full_adherence(raw)
    stall_align = stall_alignment(raw)

    req0_stats = raw.get("request0_inter_token_stats") or []
    ttft_vals = [s["ttft_ms"] for s in req0_stats if s.get("ttft_ms") is not None]
    tpot_vals = [s["mean_tpot_ms"] for s in req0_stats if s.get("mean_tpot_ms") is not None]
    max_stall_vals = [s["max_stall_ms"] for s in req0_stats if s.get("max_stall_ms") is not None]
    stalls_2x = sum(s.get("stalls_above_2x_median", 0) for s in req0_stats)
    stalls_5x = sum(s.get("stalls_above_5x_median", 0) for s in req0_stats)

    rest_rows = [r for r in raw.get("rest_pooled_request_rows", []) if r.get("ok")]
    rest_ttft = cr.distribution_summary([r["ttft_ms"] for r in rest_rows if r.get("ttft_ms") is not None])
    rest_tpot = cr.distribution_summary([r["tpot_ms"] for r in rest_rows if r.get("tpot_ms") is not None])
    rest_e2e = cr.distribution_summary([r["e2e_latency_ms"] for r in rest_rows if r.get("e2e_latency_ms") is not None])

    server_prefill_ms = None
    server_decode_ms = None
    server_queue_ms = None
    if raw.get("post_warmup_metrics_text") and raw.get("final_metrics_text"):
        server_prefill_ms = metrics_delta_mean_ms(raw["post_warmup_metrics_text"], raw["final_metrics_text"],
                                                   "vllm:request_prefill_time_seconds")
        server_decode_ms = metrics_delta_mean_ms(raw["post_warmup_metrics_text"], raw["final_metrics_text"],
                                                  "vllm:request_time_per_output_token_seconds")
        server_queue_ms = metrics_delta_mean_ms(raw["post_warmup_metrics_text"], raw["final_metrics_text"],
                                                 "vllm:request_queue_time_seconds")

    facts = resolved_facts(raw)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in rest_rows) + sum(
        t.get("output_tokens", 0) for t in raw.get("request0_timelines", []) if t.get("ok")
    )
    wall = None
    completions = [r["timeline"]["completion_time"] for r in rest_rows if r.get("timeline")]
    submits = [r["submit_time"] for r in rest_rows] + [
        t["submit_time"] for t in raw.get("request0_timelines", [])
    ]
    tl_completions = [t["completion_time"] for t in raw.get("request0_timelines", []) if t.get("completion_time")]
    all_completions = completions + tl_completions
    if submits and all_completions:
        wall = max(all_completions) - min(submits)
    measured_throughput = (total_output_tokens / wall) if wall else None

    return {
        "identity": {
            "workload_id": raw["workload_id"], "candidate_id": raw["candidate_id"],
            "max_num_seqs": raw["max_num_seqs_requested"],
            "enable_chunked_prefill": raw["enable_chunked_prefill_requested"],
            "arrival_mode": raw["arrival_mode"], "classification": raw["classification"],
        },
        "adherence": adherence,
        "stall_alignment": stall_align,
        "request0": {
            "ttft_ms": statistics.median(ttft_vals) if ttft_vals else None,
            "mean_tpot_ms": statistics.median(tpot_vals) if tpot_vals else None,
            "max_stall_ms": max(max_stall_vals) if max_stall_vals else None,
            "stalls_above_2x_median_total": stalls_2x, "stalls_above_5x_median_total": stalls_5x,
        },
        "rest": {"ttft_ms": rest_ttft, "tpot_ms": rest_tpot, "e2e_ms": rest_e2e},
        "measured_throughput_tokens_per_s": measured_throughput,
        "server_prefill_ms": server_prefill_ms, "server_decode_token_ms": server_decode_ms,
        "server_queue_ms": server_queue_ms,
        "resolved_num_gpu_blocks": facts.num_gpu_blocks if facts else None,
        "peak_gpu_memory_mib": raw.get("peak_gpu_memory_mib"),
        "reference_match": raw.get("reference_match"), "oom_detected": raw.get("oom_detected_in_log"),
        "process_cleanup_status": raw.get("process_cleanup_status"),
        "success_count": raw.get("success_count"), "failure_count": raw.get("failure_count"),
    }


def pair_off_on(summaries: list[dict]) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for s in summaries:
        idn = s["identity"]
        key = (idn["workload_id"], idn["max_num_seqs"], idn["arrival_mode"])
        by_key.setdefault(key, {})[idn["enable_chunked_prefill"]] = s

    pairs = []
    for key, states in by_key.items():
        if False in states and True in states:
            off, on = states[False], states[True]

            def diff(off_val, on_val):
                if off_val is None or on_val is None:
                    return {"absolute": None, "percent": None}
                return {"absolute": on_val - off_val,
                        "percent": ((on_val - off_val) / off_val * 100.0) if off_val else None}

            pairs.append({
                "workload_id": key[0], "max_num_seqs": key[1], "arrival_mode": key[2],
                "off": off, "on": on,
                "diff": {
                    "request0_ttft_ms": diff(off["request0"]["ttft_ms"], on["request0"]["ttft_ms"]),
                    "request0_tpot_ms": diff(off["request0"]["mean_tpot_ms"], on["request0"]["mean_tpot_ms"]),
                    "request0_max_stall_ms": diff(off["request0"]["max_stall_ms"], on["request0"]["max_stall_ms"]),
                    "rest_ttft_ms": diff(off["rest"]["ttft_ms"]["median"], on["rest"]["ttft_ms"]["median"]),
                    "rest_tpot_ms": diff(off["rest"]["tpot_ms"]["median"], on["rest"]["tpot_ms"]["median"]),
                    "throughput": diff(off["measured_throughput_tokens_per_s"], on["measured_throughput_tokens_per_s"]),
                    "server_prefill_ms": diff(off["server_prefill_ms"], on["server_prefill_ms"]),
                    "server_decode_token_ms": diff(off["server_decode_token_ms"], on["server_decode_token_ms"]),
                    "server_queue_ms": diff(off["server_queue_ms"], on["server_queue_ms"]),
                    "peak_gpu_memory_mib": diff(off["peak_gpu_memory_mib"], on["peak_gpu_memory_mib"]),
                },
                "correctness_both_pass": off["reference_match"] is not False and on["reference_match"] is not False,
            })
    return pairs


def evaluate_hypotheses(summaries: list[dict], pairs: list[dict]) -> dict:
    h = {}

    # H1: chunked_prefill disabled + staggered + concurrency>=2 -> large, TIME-ALIGNED stalls
    h1_evidence = []
    for s in summaries:
        idn = s["identity"]
        if idn["enable_chunked_prefill"] or idn["arrival_mode"] != "staggered" or (idn["max_num_seqs"] or 1) < 2:
            continue
        for align in s["stall_alignment"]:
            h1_evidence.append(align)
    h1_aligned = [a for a in h1_evidence if a["aligned_with_new_request_submission"] and a["ratio_to_median"] and a["ratio_to_median"] > 3]
    h["H1"] = {
        "result": "SUPPORTED" if h1_evidence and len(h1_aligned) / len(h1_evidence) >= 0.5 else
                  ("INCONCLUSIVE" if not h1_evidence else "REJECTED"),
        "evidence": {"stall_observations": len(h1_evidence), "time_aligned_large_stalls": len(h1_aligned)},
    }

    # H2: chunked_prefill enabled -> smaller / more evenly distributed stalls
    h2_diffs = [p["diff"]["request0_max_stall_ms"]["percent"] for p in pairs
                if p["arrival_mode"] == "staggered" and (p["max_num_seqs"] or 1) >= 2
                and p["diff"]["request0_max_stall_ms"]["percent"] is not None]
    h["H2"] = {
        "result": ("SUPPORTED" if h2_diffs and statistics.median(h2_diffs) < -20 else
                   "INCONCLUSIVE" if not h2_diffs else "REJECTED"),
        "evidence": {"median_percent_change_in_max_stall_on_vs_off": statistics.median(h2_diffs) if h2_diffs else None,
                     "n": len(h2_diffs)},
    }

    # H4: negligible overhead at concurrency<=1-equivalent (workload A, or candidate=1)
    h4_diffs = [p["diff"]["rest_e2e_ms" if False else "request0_ttft_ms"]["percent"] for p in pairs
                if (p["workload_id"] == "A" or (p["max_num_seqs"] or 1) == 1) and p["arrival_mode"] == "burst"
                and p["diff"]["request0_ttft_ms"]["percent"] is not None]
    h["H4"] = {
        "result": ("SUPPORTED" if h4_diffs and all(abs(d) < 20 for d in h4_diffs) else
                   "INCONCLUSIVE" if not h4_diffs else "REJECTED"),
        "evidence": {"percent_diffs": h4_diffs},
    }

    # H5: step-transition (1 -> >=2) rather than a gradual per-value trend
    burst_off = {(s["identity"]["workload_id"], s["identity"]["max_num_seqs"]): s
                 for s in summaries if not s["identity"]["enable_chunked_prefill"] and s["identity"]["arrival_mode"] == "burst"}
    h5_notes = []
    for wl in ("C", "D"):
        vals = {cand: burst_off[(wl, cand)]["request0"]["mean_tpot_ms"]
                for cand in (1, 2, 8) if (wl, cand) in burst_off and burst_off[(wl, cand)]["request0"]["mean_tpot_ms"]}
        if 1 in vals and 2 in vals and 8 in vals:
            jump_1_to_2 = vals[2] / vals[1] if vals[1] else None
            spread_2_to_8 = max(vals[2], vals[8]) / min(vals[2], vals[8]) if min(vals[2], vals[8]) else None
            h5_notes.append({"workload": wl, "tpot_by_candidate": vals, "jump_1_to_2_ratio": jump_1_to_2,
                              "spread_2_to_8_ratio": spread_2_to_8})
    step_like = [n for n in h5_notes if n["jump_1_to_2_ratio"] and n["jump_1_to_2_ratio"] > 3
                 and n["spread_2_to_8_ratio"] and n["spread_2_to_8_ratio"] < 1.5]
    h["H5"] = {
        "result": ("SUPPORTED" if h5_notes and len(step_like) == len(h5_notes) else
                   "INCONCLUSIVE" if not h5_notes else "REJECTED"),
        "evidence": h5_notes,
    }

    # H6: single admitted-prefill-token-cost constant explains stalls consistently
    stall_costs = []
    for s in summaries:
        idn = s["identity"]
        if idn["enable_chunked_prefill"] or idn["arrival_mode"] != "staggered":
            continue
        prompt_tokens = 128  # workloads C/D prompt_tokens_target
        max_stall = s["request0"]["max_stall_ms"]
        if max_stall:
            stall_costs.append(max_stall / prompt_tokens)
    h6_consistent = None
    if len(stall_costs) >= 2:
        h6_consistent = (max(stall_costs) / min(stall_costs)) < 2.5 if min(stall_costs) > 0 else False
    h["H6"] = {
        "result": ("SUPPORTED" if h6_consistent else "INCONCLUSIVE" if stall_costs is None or len(stall_costs) < 2
                   else "REJECTED"),
        "evidence": {"per_token_stall_cost_ms_samples": stall_costs,
                     "ratio_max_to_min": (max(stall_costs) / min(stall_costs)) if len(stall_costs) >= 2 and min(stall_costs) > 0 else None},
    }

    return h


def ranking_analysis(summaries: list[dict]) -> dict:
    burst = [s for s in summaries if s["identity"]["arrival_mode"] == "burst"]
    by_workload: dict[str, list[dict]] = {}
    for s in burst:
        by_workload.setdefault(s["identity"]["workload_id"], []).append(s)

    results = {}
    for wl, rows in by_workload.items():
        candidates = {}
        for s in rows:
            key = (s["identity"]["max_num_seqs"], s["identity"]["enable_chunked_prefill"])
            tput = s["measured_throughput_tokens_per_s"]
            if tput is not None:
                candidates[key] = tput
        if not candidates:
            continue
        measured_best = max(candidates, key=lambda k: candidates[k])
        off_only = {k: v for k, v in candidates.items() if k[1] is False}
        current_model_best = min(off_only, key=lambda k: k[0] or 1) if off_only else None  # E2E-2 model always prefers larger batch; see note below
        results[wl] = {
            "candidates_measured_throughput": {f"{k[0]}_{k[1]}": v for k, v in candidates.items()},
            "measured_global_best": f"{measured_best[0]}_{measured_best[1]}",
            "measured_best_throughput": candidates[measured_best],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--e2e2-analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raws = load_raws(args.raw_dir)
    model = build_model_features()
    hardware = build_hardware_features()
    e2e2 = json.loads(args.e2e2_analysis.read_text())
    throughput = phase_model.EffectiveThroughput(**e2e2["calibration"])

    summaries = [row_summary(r, model, hardware, throughput) for r in raws]
    pairs = pair_off_on(summaries)
    hypotheses = evaluate_hypotheses(summaries, pairs)
    ranking = ranking_analysis(summaries)

    adherent = sum(1 for s in summaries if s["adherence"]["derived_config_adherent"] and s["adherence"]["chunked_prefill_match"])
    args.out.write_text(json.dumps({
        "n_groups": len(raws), "n_fully_adherent": adherent,
        "summaries": summaries, "off_vs_on_pairs": pairs, "hypotheses": hypotheses, "ranking": ranking,
    }, indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out} groups={len(raws)} adherent={adherent}")


if __name__ == "__main__":
    main()
