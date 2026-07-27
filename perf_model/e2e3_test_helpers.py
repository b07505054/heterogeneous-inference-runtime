"""Tiny synthetic-data builders shared by E2E-3 tests, kept out of the test
files themselves so the fixtures stay readable."""
from __future__ import annotations


def synthetic_raw(
    *, workload_id: str, max_num_seqs: int, chunked_prefill: bool, arrival_mode: str,
    req0_gaps_s: list[float], rest_submit_times: list[float], req0_submit: float = 0.0,
    server_info: dict | None = None, classification: str = "VALID",
) -> dict:
    arrivals = [req0_submit + 0.05]
    for g in req0_gaps_s:
        arrivals.append(arrivals[-1] + g)
    timeline = {
        "request_id": "R000", "ok": True, "submit_time": req0_submit, "first_token_time": arrivals[0],
        "token_arrival_times": arrivals, "completion_time": arrivals[-1] + 0.01,
        "output_tokens": len(arrivals), "error": None,
    }
    from perf_model.token_timeline import TokenTimeline, inter_token_stats
    stats = inter_token_stats(TokenTimeline(**timeline))

    rest_rows = [
        {"ok": True, "ttft_ms": 100.0 + i, "tpot_ms": 12.0, "e2e_latency_ms": 400.0 + i, "output_tokens": 32,
         "submit_time": t, "timeline": {"connection_start_time": t, "completion_time": t + 0.4}}
        for i, t in enumerate(rest_submit_times)
    ]

    return {
        "raw_result_schema_version": "perf_model.e2e3.raw_result.v1",
        "workload_id": workload_id, "candidate_id": f"vllm_max_num_seqs_{max_num_seqs}",
        "max_num_seqs_requested": max_num_seqs, "enable_chunked_prefill_requested": chunked_prefill,
        "arrival_mode": arrival_mode, "repetitions": 1, "classification": classification,
        "command": [], "fixed_configuration": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float16", "max_model_len": 2048,
            "max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.75, "block_size": 16,
            "enable_prefix_caching": False, "enable_chunked_prefill": chunked_prefill,
            "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
        },
        "startup_seconds": 40.0, "server_pid": 1,
        "server_info_raw": server_info, "server_info_error": None if server_info else "unavailable",
        "attention_backend_from_log": "TRITON_ATTN",
        "post_warmup_metrics_text": None, "final_metrics_text": None,
        "reference_completion": {"http_status": 200, "text": "Paris"}, "reference_baseline": {"text": "Paris"},
        "reference_match": True, "oom_detected_in_log": False,
        "idle_gpu_memory_mib": 4, "peak_gpu_memory_mib": 3000, "after_shutdown_gpu_memory_mib": 4,
        "process_cleanup_status": "graceful_sigterm",
        "request_count": len(rest_rows) + 1, "success_count": len(rest_rows) + 1, "failure_count": 0,
        "request0_timelines": [timeline], "request0_inter_token_stats": [stats],
        "rest_pooled_request_rows": rest_rows,
        "workload_definition": {"prompt_tokens_target": 128, "output_tokens": 32, "concurrency": 4},
    }


def synthetic_server_info(*, enable_chunked_prefill: bool, max_num_seqs: int) -> dict:
    return {"vllm_config": {
        "model_config": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "served_model_name": "qwen2.5-0.5b",
                          "dtype": "torch.float16", "quantization": None, "max_model_len": 2048},
        "cache_config": {"block_size": 16, "cache_dtype": "auto", "gpu_memory_utilization": 0.75,
                          "num_gpu_blocks": 8900, "num_cpu_blocks": None, "kv_cache_size_tokens": 142400,
                          "kv_cache_memory_bytes": None, "enable_prefix_caching": False},
        "parallel_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1,
                             "distributed_executor_backend": "uni"},
        "scheduler_config": {"max_num_seqs": max_num_seqs, "max_num_batched_tokens": 2048,
                              "enable_chunked_prefill": enable_chunked_prefill, "policy": "fcfs"},
        "compilation_config": {"mode": 3, "cudagraph_mode": [2, 1]},
        "attention_config": {"backend": None},
    }}
