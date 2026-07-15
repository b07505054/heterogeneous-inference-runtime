#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
struct HirAttentionStatus { int32_t code; const char *message; };
const char *hir_attention_artifact_version();
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
}
