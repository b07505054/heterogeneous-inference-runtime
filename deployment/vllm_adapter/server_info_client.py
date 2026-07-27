"""Client for vLLM's real dev-mode GET /server_info?config_format=json endpoint.

This is the strongest available runtime-truth source for this slice: it is
the server's own resolved VllmConfig (post-profiling for cache_config
fields), not an echo of the CLI we sent. Requires the server to have been
launched with VLLM_SERVER_DEV_MODE=1 -- no vLLM source modification, no
fork. Field paths below were verified against a live capture on vLLM 0.24.0
(see tests/fixtures/server_info_sample.json) and may shift across vLLM
versions -- this module is PUBLIC_VERSION_SENSITIVE, not PUBLIC_STABLE.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ServerInfoUnavailable(RuntimeError):
    """Raised when /server_info cannot be fetched or parsed. Callers MUST
    treat this as fail-closed for derived-config adherence -- never fall
    back to CLI-only proof silently."""


def fetch_server_info(port: int, *, host: str = "127.0.0.1", timeout: float = 5.0) -> dict[str, Any]:
    url = f"http://{host}:{port}/server_info?config_format=json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise ServerInfoUnavailable(f"/server_info returned HTTP {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ServerInfoUnavailable(f"/server_info unreachable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ServerInfoUnavailable(f"/server_info returned invalid JSON: {exc}") from exc


@dataclass(frozen=True)
class ResolvedRuntimeFacts:
    model: str | None
    served_model_name: str | None
    dtype: str | None
    quantization: str | None
    max_model_len: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    gpu_memory_utilization: float | None
    block_size: int | None
    kv_cache_dtype: str | None
    num_gpu_blocks: int | None
    num_cpu_blocks: int | None
    kv_cache_size_tokens: int | None
    kv_cache_memory_bytes: int | None
    enable_prefix_caching: bool | None
    enable_chunked_prefill: bool | None
    tensor_parallel_size: int | None
    pipeline_parallel_size: int | None
    data_parallel_size: int | None
    distributed_executor_backend: str | None
    scheduler_policy: str | None
    compilation_mode: Any
    cudagraph_mode: Any
    attention_backend: str | None  # None if not resolvable from this endpoint alone
    attention_backend_source: str  # "server_info_field" | "log_scrape" | "unavailable"
    raw_vllm_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        return d


def parse_server_info(raw: dict[str, Any], *, attention_backend_from_log: str | None = None) -> ResolvedRuntimeFacts:
    vc = raw.get("vllm_config")
    if not isinstance(vc, dict) or not vc:
        raise ServerInfoUnavailable("vllm_config missing, empty, or not an object in /server_info response")

    model_config = vc.get("model_config") or {}
    cache_config = vc.get("cache_config") or {}
    parallel_config = vc.get("parallel_config") or {}
    scheduler_config = vc.get("scheduler_config") or {}
    compilation_config = vc.get("compilation_config") or {}
    attention_config = vc.get("attention_config") or {}

    field_backend = attention_config.get("backend")
    if field_backend:
        attention_backend, attention_source = field_backend, "server_info_field"
    elif attention_backend_from_log:
        attention_backend, attention_source = attention_backend_from_log, "log_scrape"
    else:
        attention_backend, attention_source = None, "unavailable"

    return ResolvedRuntimeFacts(
        model=model_config.get("model"),
        served_model_name=model_config.get("served_model_name"),
        dtype=model_config.get("dtype"),
        quantization=model_config.get("quantization"),
        max_model_len=model_config.get("max_model_len"),
        max_num_seqs=scheduler_config.get("max_num_seqs"),
        max_num_batched_tokens=scheduler_config.get("max_num_batched_tokens"),
        gpu_memory_utilization=cache_config.get("gpu_memory_utilization"),
        block_size=cache_config.get("block_size"),
        kv_cache_dtype=cache_config.get("cache_dtype"),
        num_gpu_blocks=cache_config.get("num_gpu_blocks"),
        num_cpu_blocks=cache_config.get("num_cpu_blocks"),
        kv_cache_size_tokens=cache_config.get("kv_cache_size_tokens"),
        kv_cache_memory_bytes=cache_config.get("kv_cache_memory_bytes"),
        enable_prefix_caching=cache_config.get("enable_prefix_caching"),
        enable_chunked_prefill=scheduler_config.get("enable_chunked_prefill"),
        tensor_parallel_size=parallel_config.get("tensor_parallel_size"),
        pipeline_parallel_size=parallel_config.get("pipeline_parallel_size"),
        data_parallel_size=parallel_config.get("data_parallel_size"),
        distributed_executor_backend=parallel_config.get("distributed_executor_backend"),
        scheduler_policy=scheduler_config.get("policy"),
        compilation_mode=compilation_config.get("mode"),
        cudagraph_mode=compilation_config.get("cudagraph_mode"),
        attention_backend=attention_backend,
        attention_backend_source=attention_source,
        raw_vllm_config=vc,
    )


REQUESTED_VS_RESOLVED_FIELDS = (
    # (requested_key_in_fixed_configuration, resolved_attr_on_ResolvedRuntimeFacts)
    ("model", "model"),
    ("dtype", "dtype"),
    ("max_model_len", "max_model_len"),
    ("max_num_batched_tokens", "max_num_batched_tokens"),
    ("gpu_memory_utilization", "gpu_memory_utilization"),
    ("block_size", "block_size"),
    ("enable_prefix_caching", "enable_prefix_caching"),
    ("enable_chunked_prefill", "enable_chunked_prefill"),
    ("tensor_parallel_size", "tensor_parallel_size"),
    ("pipeline_parallel_size", "pipeline_parallel_size"),
)


def _normalize(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("torch."):
        return value[len("torch."):]
    return value


def compare_requested_vs_resolved(
    requested: dict[str, Any], resolved: ResolvedRuntimeFacts, *, requested_max_num_seqs: int | None,
) -> dict[str, Any]:
    mismatches: list[str] = []
    comparisons: dict[str, Any] = {}
    for requested_key, resolved_attr in REQUESTED_VS_RESOLVED_FIELDS:
        req_val = _normalize(requested.get(requested_key))
        res_val = _normalize(getattr(resolved, resolved_attr))
        match = req_val == res_val
        comparisons[requested_key] = {"requested": req_val, "resolved": res_val, "match": match}
        if not match:
            mismatches.append(requested_key)

    if requested_max_num_seqs is not None:
        match = requested_max_num_seqs == resolved.max_num_seqs
        comparisons["max_num_seqs"] = {
            "requested": requested_max_num_seqs, "resolved": resolved.max_num_seqs, "match": match,
        }
        if not match:
            mismatches.append("max_num_seqs")

    return {"comparisons": comparisons, "mismatches": mismatches, "derived_config_adherent": not mismatches}
