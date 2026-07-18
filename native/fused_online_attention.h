#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {

struct HirFusedAttentionParams {
  const float *q;
  const float *k;
  const float *v;
  float *output;
  int64_t batch, query_length, context_length;
  int64_t query_heads, kv_heads, head_dimension;
  int64_t q_strides[4], k_strides[4], v_strides[4], out_strides[4];
  float scale;
  int32_t causal;
  int64_t query_position_offset;
  int64_t query_tile, key_tile;
  int64_t worker_count;
  int64_t query_head_offset;
  int64_t total_query_heads;
};

struct HirFusedAttentionStats {
  uint64_t temporary_bytes;
  uint64_t largest_allocation_bytes;
  uint64_t allocations_per_invocation;
  uint64_t allocations_per_query_row;
  uint64_t score_tile_bytes;
  uint64_t output_accumulator_bytes;
  uint64_t worker_scratch_bytes;
};

struct HirFusedAttentionStatus {
  int32_t code;
  const char *message;
};

const char *hir_fused_attention_artifact_version();
int32_t hir_fused_attention_has_avx2();
HirFusedAttentionStatus hir_fused_online_attention_scalar(
    const HirFusedAttentionParams *, HirFusedAttentionStats *);
HirFusedAttentionStatus hir_fused_online_attention_avx2(
    const HirFusedAttentionParams *, HirFusedAttentionStats *);
}
