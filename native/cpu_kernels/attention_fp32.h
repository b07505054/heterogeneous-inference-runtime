#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
struct HirAttentionStatus { int32_t code; const char *message; };
const char *hir_attention_artifact_version();
const char *hir_contiguous_kv_artifact_version();
HirAttentionStatus hir_contiguous_kv_initialize(
    float *k_cache, size_t k_count, float *v_cache, size_t v_count,
    int64_t batch, int64_t heads, int64_t capacity, int64_t head_dim);
HirAttentionStatus hir_contiguous_kv_prefill_write(
    float *k_cache, size_t k_count, float *v_cache, size_t v_count,
    const float *k, size_t input_k_count, const float *v, size_t input_v_count,
    int64_t batch, int64_t heads, int64_t tokens, int64_t capacity, int64_t head_dim);
HirAttentionStatus hir_contiguous_kv_append(
    float *k_cache, size_t k_count, float *v_cache, size_t v_count,
    const float *k, size_t input_k_count, const float *v, size_t input_v_count,
    int64_t batch, int64_t heads, int64_t append_index, int64_t capacity, int64_t head_dim);
HirAttentionStatus hir_contiguous_kv_reset(
    float *k_cache, size_t k_count, float *v_cache, size_t v_count,
    int64_t batch, int64_t heads, int64_t capacity, int64_t head_dim);
HirAttentionStatus hir_cpu_attention_prefill_fp32(
    const float *q, size_t q_count, const float *k, size_t k_count,
    const float *v, size_t v_count, float *out, size_t out_count,
    float *workspace, size_t workspace_count, int64_t batch, int64_t heads,
    int64_t query_length, int64_t context_length, int64_t head_dim);
HirAttentionStatus hir_cpu_attention_decode_fp32(
    const float *q, size_t q_count, const float *k, size_t k_count,
    const float *v, size_t v_count, float *out, size_t out_count,
    float *workspace, size_t workspace_count, int64_t batch, int64_t heads,
    int64_t query_length, int64_t context_length, int64_t head_dim);
HirAttentionStatus hir_cpu_attention_decode_contiguous_kv_fp32(
    const float *q, size_t q_count, const float *k_cache, size_t k_count,
    const float *v_cache, size_t v_count, float *out, size_t out_count,
    float *workspace, size_t workspace_count, int64_t batch, int64_t heads,
    int64_t valid_tokens, int64_t capacity_tokens, int64_t head_dim);
}
