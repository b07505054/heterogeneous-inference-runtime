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
HirAttentionStatus hir_cpu_attention_decode_contiguous_kv_reordered_fp32(
    const float *q, size_t q_count, const float *k_cache, size_t k_count,
    const float *v_cache, size_t v_count, float *out, size_t out_count,
    float *workspace, size_t workspace_count, int64_t batch, int64_t heads,
    int64_t valid_tokens, int64_t capacity_tokens, int64_t head_dim);
// Historical benchmark alias; compiler-generated plans do not select it.
HirAttentionStatus hir_cpu_attention_decode_contiguous_kv_reordered_control_fp32(
    const float *q, size_t q_count, const float *k_cache, size_t k_count,
    const float *v_cache, size_t v_count, float *out, size_t out_count,
    float *workspace, size_t workspace_count, int64_t batch, int64_t heads,
    int64_t valid_tokens, int64_t capacity_tokens, int64_t head_dim);
const char *hir_paged_kv_artifact_version();
HirAttentionStatus hir_paged_kv_initialize(
    float *k_pages, size_t k_count, float *v_pages, size_t v_count,
    int32_t *block_table, size_t table_count, int64_t physical_pages,
    int64_t heads, int64_t page_tokens, int64_t head_dim, int32_t sentinel);
HirAttentionStatus hir_paged_kv_reset(
    float *k_pages, size_t k_count, float *v_pages, size_t v_count,
    int32_t *block_table, size_t table_count, int64_t physical_pages,
    int64_t heads, int64_t page_tokens, int64_t head_dim, int32_t sentinel);
HirAttentionStatus hir_paged_kv_prefill_write(
    float *k_pages, size_t k_count, float *v_pages, size_t v_count,
    const int32_t *block_table, size_t table_count,
    const float *k, size_t input_k_count, const float *v, size_t input_v_count,
    int64_t tokens, int64_t physical_pages, int64_t heads,
    int64_t page_tokens, int64_t head_dim, int32_t sentinel);
HirAttentionStatus hir_paged_kv_append(
    float *k_pages, size_t k_count, float *v_pages, size_t v_count,
    const int32_t *block_table, size_t table_count,
    const float *k, size_t input_k_count, const float *v, size_t input_v_count,
    int64_t logical_index, int64_t physical_pages, int64_t heads,
    int64_t page_tokens, int64_t head_dim, int32_t sentinel);
HirAttentionStatus hir_cpu_attention_decode_paged_kv_fp32(
    const float *q, size_t q_count, const float *k_pages, size_t k_count,
    const float *v_pages, size_t v_count, const int32_t *block_table,
    size_t table_count, float *out, size_t out_count, float *workspace,
    size_t workspace_count, int64_t valid_tokens, int64_t physical_pages,
    int64_t heads, int64_t page_tokens, int64_t head_dim, int32_t sentinel);
HirAttentionStatus hir_cpu_attention_decode_paged_kv_page_major_fp32(
    const float *q, size_t q_count, const float *k_pages, size_t k_count,
    const float *v_pages, size_t v_count, const int32_t *block_table,
    size_t table_count, int32_t *physical_page_cache, size_t cache_count,
    float *out, size_t out_count, float *workspace,
    size_t workspace_count, int64_t valid_tokens, int64_t physical_pages,
    int64_t heads, int64_t page_tokens, int64_t head_dim, int32_t sentinel);
}
