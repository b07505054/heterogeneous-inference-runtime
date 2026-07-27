import json
from pathlib import Path

import pytest

from deployment.vllm_adapter import server_info_client as sic

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "server_info_sample.json").read_text())


def test_parse_server_info_extracts_resolved_facts_from_real_capture():
    facts = sic.parse_server_info(FIXTURE, attention_backend_from_log="TRITON_ATTN")
    assert facts.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert facts.dtype == "torch.float16"
    assert facts.max_num_seqs == 4
    assert facts.max_num_batched_tokens == 2048
    assert facts.block_size == 16
    assert facts.num_gpu_blocks == 8888
    assert facts.kv_cache_size_tokens == 142208
    assert facts.tensor_parallel_size == 1
    assert facts.enable_prefix_caching is False
    assert facts.enable_chunked_prefill is False


def test_attention_backend_falls_back_to_log_scrape_when_field_is_null():
    facts = sic.parse_server_info(FIXTURE, attention_backend_from_log="TRITON_ATTN")
    assert facts.attention_backend == "TRITON_ATTN"
    assert facts.attention_backend_source == "log_scrape"


def test_attention_backend_unavailable_without_log_line():
    facts = sic.parse_server_info(FIXTURE, attention_backend_from_log=None)
    assert facts.attention_backend is None
    assert facts.attention_backend_source == "unavailable"


def test_parse_server_info_fails_closed_on_missing_vllm_config():
    with pytest.raises(sic.ServerInfoUnavailable):
        sic.parse_server_info({"not_vllm_config": {}})


def test_compare_requested_vs_resolved_all_match():
    facts = sic.parse_server_info(FIXTURE)
    requested = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float16", "max_model_len": 2048,
        "max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.75, "block_size": 16,
        "enable_prefix_caching": False, "enable_chunked_prefill": False,
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
    }
    result = sic.compare_requested_vs_resolved(requested, facts, requested_max_num_seqs=4)
    assert result["derived_config_adherent"] is True
    assert result["mismatches"] == []


def test_compare_requested_vs_resolved_detects_mismatch():
    facts = sic.parse_server_info(FIXTURE)
    requested = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct", "dtype": "float16", "max_model_len": 2048,
        "max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.75, "block_size": 16,
        "enable_prefix_caching": False, "enable_chunked_prefill": False,
        "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
    }
    result = sic.compare_requested_vs_resolved(requested, facts, requested_max_num_seqs=8)
    assert result["derived_config_adherent"] is False
    assert "max_num_seqs" in result["mismatches"]
