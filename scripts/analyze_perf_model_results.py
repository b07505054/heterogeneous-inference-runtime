#!/usr/bin/env python3
"""Phase 2: offline analysis. Loads raw experiment results (Phase 1 output),
derives the two calibrated throughput constants from designated single-
request calibration sources, computes predictions for every row (including
the calibration sources themselves, since only 2 scalars are fit from 2
numbers -- everything else is an out-of-sample test), and produces:
  - one normalized calibration row per (workload, candidate)
  - an aggregate report: memory/latency/throughput error, ranking usefulness,
    OOM correctness, adherence status, baseline comparison.

No GPU access required -- pure analysis over already-captured JSON/text.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perf_model import calibration_row as cr
from perf_model import compute_model, memory_model, phase_model
from perf_model.schema import HardwareFeature, HardwareFeatures, ModelFeatures, RuntimeConfiguration, WorkloadFeatures
from deployment.vllm_adapter import metrics_client, server_info_client

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Qwen2.5-0.5B-Instruct model features, derived from the real config.json
# cached at ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct ---
HIDDEN = 896
INTERMEDIATE = 4864
LAYERS = 24
HEADS = 14
KV_HEADS = 2
HEAD_DIM = 64
VOCAB = 151936


def _real_parameter_count() -> int:
    embedding = VOCAB * HIDDEN
    per_layer = (
        (HIDDEN * HIDDEN + HIDDEN)             # q_proj + bias
        + (HIDDEN * KV_HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM)  # k_proj + bias
        + (HIDDEN * KV_HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM)  # v_proj + bias
        + (HIDDEN * HIDDEN)                    # o_proj, no bias
        + HIDDEN + HIDDEN                      # input/post-attn RMSNorm
        + 3 * (HIDDEN * INTERMEDIATE)           # gate+up+down MLP
    )
    return embedding + per_layer * LAYERS + HIDDEN  # + final norm


def build_model_features() -> ModelFeatures:
    param_count = _real_parameter_count()
    weight_bytes = param_count * 2  # fp16, matches memory_model.bytes_per_element
    return ModelFeatures(
        model_id="Qwen/Qwen2.5-0.5B-Instruct", architecture="qwen2", parameter_count=param_count,
        layer_count=LAYERS, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
        attention_head_count=HEADS, kv_head_count=KV_HEADS, head_dimension=HEAD_DIM,
        vocabulary_size=VOCAB, dtype="float16", quantization="none", maximum_model_length=2048,
        estimated_weight_bytes=weight_bytes, estimated_weight_bytes_source="analytical_flop_bandwidth",
        tie_word_embeddings=True,
    )


def build_hardware_features() -> HardwareFeatures:
    # GPU name/memory/compute-capability: device-reported (nvidia-smi), captured live during Phase 1.
    # Bandwidth/FLOPs: public vendor-quoted figures for GTX 1650 Max-Q (TU117) -- explicitly
    # VENDOR_SPEC, uncertain for the laptop Max-Q clock profile, and NEVER used as the
    # denominator in any prediction (predictions use the calibrated constants below only).
    return HardwareFeatures(
        gpu_name="NVIDIA GeForce GTX 1650 with Max-Q Design",
        gpu_memory_bytes=HardwareFeature(4096 * 1024 * 1024, "device_reported"),
        gpu_count=1, cuda_version="13.0",
        compute_capability=HardwareFeature("7.5", "device_reported"),
        memory_bandwidth_bytes_per_s=HardwareFeature(128e9, "vendor_spec"),
        compute_throughput_flops=HardwareFeature(2.5e12, "vendor_spec"),
    )


def load_raw_results(raw_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(raw_dir.glob("*.json"))]


def metrics_delta_mean_ms(post_warmup_text: str, final_text: str, metric_name: str) -> float | None:
    """Isolates the measured window from warmup by differencing cumulative
    histogram sum/count between the two /metrics snapshots."""
    before = metrics_client.parse_prometheus_text(post_warmup_text)
    after = metrics_client.parse_prometheus_text(final_text)

    def totals(parsed):
        samples = parsed.get("histograms", {}).get(metric_name) or []
        return sum(s.sum or 0.0 for s in samples), sum(s.count or 0.0 for s in samples)

    sum_before, count_before = totals(before)
    sum_after, count_after = totals(after)
    d_sum, d_count = sum_after - sum_before, count_after - count_before
    if d_count <= 0:
        return None
    return (d_sum / d_count) * 1000.0


def resolved_facts(raw: dict) -> server_info_client.ResolvedRuntimeFacts | None:
    if not raw.get("server_info_raw"):
        return None
    return server_info_client.parse_server_info(
        raw["server_info_raw"], attention_backend_from_log=raw.get("attention_backend_from_log")
    )


def admitted_concurrency(workload_concurrency: int, resolved_max_num_seqs: int | None) -> int:
    if resolved_max_num_seqs is None:
        return workload_concurrency
    return min(workload_concurrency, resolved_max_num_seqs)


def derive_calibration(model: ModelFeatures, raw_by_key: dict[tuple[str, int], dict]) -> phase_model.EffectiveThroughput:
    """A workload (single-request, concurrency=1) supplies the compute-bound
    prefill measurement; A workload supplies the memory-bound decode
    measurement. Any candidate works since max_num_seqs does not affect
    single-in-flight-request behavior; candidate=1 is used for cleanliness.
    """
    b1 = raw_by_key.get(("B", 1))
    a1 = raw_by_key.get(("A", 1))
    if not b1 or not a1 or b1["classification"] != "VALID" or a1["classification"] != "VALID":
        return phase_model.UNCALIBRATED

    prefill_ms = metrics_delta_mean_ms(b1["post_warmup_metrics_text"], b1["final_metrics_text"],
                                       "vllm:request_prefill_time_seconds")
    decode_ms = metrics_delta_mean_ms(a1["post_warmup_metrics_text"], a1["final_metrics_text"],
                                       "vllm:request_time_per_output_token_seconds")
    if not prefill_ms or not decode_ms:
        return phase_model.UNCALIBRATED

    b_wl = b1["workload_definition"]
    prefill_flops = compute_model.prefill_op_counts(model, b_wl["prompt_tokens_target"]).total_flops

    a_wl = a1["workload_definition"]
    avg_kv_context = a_wl["prompt_tokens_target"] + (a_wl["output_tokens"] - 1) / 2.0
    decode_bytes = compute_model.decode_step_memory_traffic_bytes(
        model, weight_bytes=model.estimated_weight_bytes,
        kv_bytes_per_token=memory_model.kv_bytes_per_token(model, kv_cache_dtype_bytes=2),
        kv_context_tokens=avg_kv_context,
    )
    return phase_model.calibrate(
        prefill_flops=prefill_flops, measured_prefill_ms=prefill_ms,
        decode_memory_bytes_batch1=decode_bytes, measured_decode_token_ms=decode_ms,
        calibrated_from={"prefill_source": "B-1", "decode_source": "A-1",
                          "measured_prefill_ms": prefill_ms, "measured_decode_token_ms": decode_ms},
    )


def analyze_row(
    raw: dict, model: ModelFeatures, hardware: HardwareFeatures, throughput: phase_model.EffectiveThroughput,
    runtime_overhead_bytes: float,
) -> dict:
    workload_id = raw["workload_id"]
    candidate_id = raw["candidate_id"]
    wl = raw["workload_definition"]
    fixed = raw["fixed_configuration"]
    facts = resolved_facts(raw)

    requested_rc = RuntimeConfiguration(
        max_num_seqs=raw["max_num_seqs_requested"], max_num_batched_tokens=fixed["max_num_batched_tokens"],
        max_model_len=fixed["max_model_len"], gpu_memory_utilization=fixed["gpu_memory_utilization"],
        tensor_parallel_size=fixed["tensor_parallel_size"], dtype=fixed["dtype"], quantization=fixed["quantization"],
        recorded_not_owned={"block_size": fixed["block_size"], "enable_prefix_caching": fixed["enable_prefix_caching"],
                             "enable_chunked_prefill": fixed["enable_chunked_prefill"]},
    )

    adherence = {"derived_config_adherent": False, "mismatches": ["server_info_unavailable"]}
    if facts is not None:
        adherence = server_info_client.compare_requested_vs_resolved(
            fixed, facts, requested_max_num_seqs=raw["max_num_seqs_requested"]
        )

    concurrency = wl["concurrency"]
    resolved_max_num_seqs = facts.max_num_seqs if facts else raw["max_num_seqs_requested"]
    batch = admitted_concurrency(concurrency, resolved_max_num_seqs)
    prompt_tokens = wl["prompt_tokens_target"]
    output_tokens = wl["output_tokens"]
    block_size = facts.block_size if facts else fixed["block_size"]
    kv_dtype_bytes = 2  # cache_dtype "auto" resolves to the model dtype (fp16) on this GPU/version

    weight_bytes = model.estimated_weight_bytes
    kv_bpt = memory_model.kv_bytes_per_token(model, kv_cache_dtype_bytes=kv_dtype_bytes)

    # --- memory ---
    kv_estimate = memory_model.kv_peak_bytes(
        model, kv_cache_dtype_bytes=kv_dtype_bytes,
        per_sequence_token_counts=[prompt_tokens + output_tokens] * batch, block_size=block_size,
    )
    total_predicted_bytes = memory_model.total_predicted_memory_bytes(
        weight_bytes=weight_bytes, kv_peak_estimate=kv_estimate,
        runtime_overhead_bytes=int(runtime_overhead_bytes), safety_margin_bytes=0,
    )
    predicted_oom = memory_model.predict_oom(total_predicted_bytes, hardware)

    observed_total_bytes = raw["peak_gpu_memory_mib"] * 1024 * 1024 if raw["peak_gpu_memory_mib"] > 0 else None
    resolved_kv_pool_bytes = (
        facts.num_gpu_blocks * facts.block_size * kv_bpt if facts and facts.num_gpu_blocks and facts.block_size
        else None
    )

    # --- latency (concurrent-serving prediction, matched to how the measurement was actually run) ---
    avg_kv_context_decode = prompt_tokens + (output_tokens - 1) / 2.0
    prefill_est, prefill_breakdown = phase_model.predict_prefill_ms(model, prompt_tokens, weight_bytes, throughput)
    decode_est, decode_breakdown = phase_model.predict_decode_token_ms(
        model, kv_context_tokens=avg_kv_context_decode, weight_bytes=weight_bytes, kv_bytes_per_token=kv_bpt,
        batch_size=batch, throughput=throughput,
    )
    avg_service_ms = (prefill_est.value or 0.0) + output_tokens * (decode_est.value or 0.0)
    # The client is a closed-loop ThreadPoolExecutor(max_workers=concurrency): at most
    # `concurrency` requests are ever simultaneously in flight, regardless of how many
    # measured requests exist in total. Queue position must therefore be modeled over
    # one representative wave of `concurrency` requests, not the whole measured pool --
    # using measured_count here was a bug that fabricated queueing even at concurrency=1
    # (where the real server-side queue is provably ~0, since the next request is never
    # submitted until the previous one completes).
    queue_estimates = [
        phase_model.predict_queue_ms_positional(i, batch, avg_service_ms) for i in range(concurrency)
    ]
    avg_queue_ms = statistics.mean(queue_estimates) if queue_estimates else 0.0
    ttft_est = phase_model.predict_ttft_ms(
        queue_ms=avg_queue_ms, prefill_estimate=prefill_est, first_decode_step_estimate=decode_est,
        request_overhead_ms=0.0,
    )
    tpot_est = phase_model.predict_tpot_ms(decode_est)
    e2e_est = phase_model.predict_e2e_ms(ttft_est, output_tokens, tpot_est)
    throughput_est = phase_model.predict_output_tokens_per_second_concurrent(batch, decode_est)

    # --- measurements ---
    good_rows = [r for r in raw["pooled_request_rows"] if r.get("ok")]
    ttft_dist = cr.distribution_summary([r["ttft_ms"] for r in good_rows if r.get("ttft_ms") is not None])
    tpot_dist = cr.distribution_summary([r["tpot_ms"] for r in good_rows if r.get("tpot_ms") is not None])
    e2e_dist = cr.distribution_summary([r["e2e_latency_ms"] for r in good_rows if r.get("e2e_latency_ms") is not None])
    total_output_tokens = sum(r.get("output_tokens", 0) for r in good_rows)
    server_prefill_mean = metrics_delta_mean_ms(raw["post_warmup_metrics_text"], raw["final_metrics_text"],
                                                 "vllm:request_prefill_time_seconds") if raw.get("post_warmup_metrics_text") and raw.get("final_metrics_text") else None
    server_decode_mean = metrics_delta_mean_ms(raw["post_warmup_metrics_text"], raw["final_metrics_text"],
                                                "vllm:request_time_per_output_token_seconds") if raw.get("post_warmup_metrics_text") and raw.get("final_metrics_text") else None
    server_queue_mean = metrics_delta_mean_ms(raw["post_warmup_metrics_text"], raw["final_metrics_text"],
                                               "vllm:request_queue_time_seconds") if raw.get("post_warmup_metrics_text") and raw.get("final_metrics_text") else None

    # approximate measured concurrent throughput: total measured output tokens / total measured wall time.
    # wall time isn't tracked directly in the pooled rows' e2e sum (overlapping requests), so use the max
    # completion timestamp minus min submit timestamp across the pooled rows as the wall-clock window.
    if good_rows:
        submits = [r["submit_time"] for r in good_rows]
        completions = [r["timeline"]["completion_time"] for r in good_rows if r.get("timeline")]
        wall = (max(completions) - min(submits)) if completions else None
    else:
        wall = None
    measured_throughput = (total_output_tokens / wall) if wall else None

    predictions = {
        "predicted_weight_memory_bytes": weight_bytes,
        "predicted_kv_memory_bytes": kv_estimate.block_rounded_bytes,
        "predicted_kv_memory_bytes_theoretical": kv_estimate.theoretical_bytes,
        "predicted_total_memory_bytes": total_predicted_bytes,
        "predicted_oom": predicted_oom.to_dict(),
        "predicted_prefill_ms": prefill_est.to_dict(),
        "predicted_decode_token_ms": decode_est.to_dict(),
        "predicted_ttft_ms": ttft_est.to_dict(),
        "predicted_tpot_ms": tpot_est.to_dict(),
        "predicted_e2e_ms": e2e_est.to_dict(),
        "predicted_output_tokens_per_second": throughput_est.to_dict(),
        "batch_size_used_for_prediction": batch,
        "avg_predicted_queue_ms": avg_queue_ms,
        "component_breakdown": {"prefill": prefill_breakdown, "decode_step": decode_breakdown},
    }

    measurements = {
        "ttft_ms": ttft_dist, "tpot_ms": tpot_dist, "e2e_ms": e2e_dist,
        "output_token_throughput": measured_throughput,
        "server_prefill_time_ms_mean": server_prefill_mean,
        "server_decode_token_ms_mean": server_decode_mean,
        "server_queue_time_ms_mean": server_queue_mean,
        "resolved_num_gpu_blocks": facts.num_gpu_blocks if facts else None,
        "resolved_num_cpu_blocks": facts.num_cpu_blocks if facts else None,
        "resolved_kv_pool_bytes": resolved_kv_pool_bytes,
        "observed_peak_gpu_memory_bytes": observed_total_bytes,
        "correctness_reference_match": raw.get("reference_match"),
        "oom_detected": raw.get("oom_detected_in_log"),
        "process_cleanup_status": raw.get("process_cleanup_status"),
        "success_count": raw.get("success_count"), "failure_count": raw.get("failure_count"),
    }

    errors = {}
    for metric_name, pred_est, measured_dist in (
        ("predicted_ttft_ms", ttft_est, ttft_dist), ("predicted_tpot_ms", tpot_est, tpot_dist),
        ("predicted_e2e_ms", e2e_est, e2e_dist),
    ):
        err = cr.compute_error(pred_est.value, measured_dist["median"])
        findings = cr.attribute_error(
            metric_name=metric_name, error=err, measured_distribution=measured_dist,
            adherence_mismatches=adherence["mismatches"], concurrency=concurrency,
            resolved_max_num_seqs=resolved_max_num_seqs, warmup_count=wl["warmup_requests"],
        )
        errors[metric_name] = {**err, "attribution": findings}

    throughput_err = cr.compute_error(throughput_est.value, measured_throughput)
    errors["predicted_output_tokens_per_second"] = {
        **throughput_err,
        "attribution": cr.attribute_error(
            metric_name="predicted_output_tokens_per_second", error=throughput_err, measured_distribution=None,
            adherence_mismatches=adherence["mismatches"], concurrency=concurrency,
            resolved_max_num_seqs=resolved_max_num_seqs, warmup_count=wl["warmup_requests"],
        ),
    }

    oom_correct = (predicted_oom.value is False) and not raw.get("oom_detected_in_log")
    errors["predicted_oom"] = {"predicted": predicted_oom.value, "actual": raw.get("oom_detected_in_log"),
                                "correct": oom_correct}

    identity = {
        "experiment_id": f"{workload_id}-{candidate_id}", "model_id": model.model_id,
        "model_features_hash": model.features_hash(), "hardware_id": hardware.gpu_name,
        "hardware_features_hash": hardware.features_hash(), "workload_id": workload_id,
        "candidate_id": candidate_id,
    }
    configuration = {
        "requested": requested_rc.to_dict(),
        "resolved": facts.to_dict() if facts else None,
        "adherence": adherence,
    }
    return cr.build_calibration_row(
        identity=identity, configuration=configuration, predictions=predictions,
        measurements=measurements, errors=errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raws = load_raw_results(args.raw_dir)
    raw_by_key = {(r["workload_id"], r["max_num_seqs_requested"]): r for r in raws}

    model = build_model_features()
    hardware = build_hardware_features()
    throughput = derive_calibration(model, raw_by_key)

    # runtime_overhead calibrated once from any successful row's resolved num_gpu_blocks
    # (num_gpu_blocks is workload/candidate independent -- same fixed config each time).
    runtime_overhead_bytes = 0.0
    for raw in raws:
        facts = resolved_facts(raw)
        if facts and facts.num_gpu_blocks and facts.block_size:
            kv_bpt = memory_model.kv_bytes_per_token(model, kv_cache_dtype_bytes=2)
            observed_non_kv = (
                facts.gpu_memory_utilization * hardware.gpu_memory_bytes.value
                - facts.num_gpu_blocks * facts.block_size * kv_bpt
            )
            runtime_overhead_bytes = observed_non_kv - model.estimated_weight_bytes
            break

    rows = [analyze_row(raw, model, hardware, throughput, runtime_overhead_bytes) for raw in raws]
    args.out.write_text(json.dumps({
        "model_features": model.to_dict(), "hardware_features": hardware.to_dict(),
        "calibration": throughput.to_dict(), "runtime_overhead_bytes": runtime_overhead_bytes,
        "rows": rows,
    }, indent=2, sort_keys=True, default=str))
    print(f"wrote {args.out} rows={len(rows)} calibration_source={throughput.source}")


if __name__ == "__main__":
    main()
