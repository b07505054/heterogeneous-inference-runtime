#include "fused_online_attention.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

namespace {
HirFusedAttentionStatus error(int32_t code, const char *message) {
  return {code, message};
}

bool positive_strides(const int64_t s[4]) {
  return s[0] > 0 && s[1] > 0 && s[2] > 0 && s[3] > 0;
}

HirFusedAttentionStatus validate(const HirFusedAttentionParams *p) {
  if (!p || !p->q || !p->k || !p->v || !p->output)
    return error(1, "null_pointer");
  if (p->batch <= 0 || p->query_length <= 0 || p->context_length <= 0 ||
      p->query_heads <= 0 || p->kv_heads <= 0 || p->head_dimension <= 0)
    return error(2, "invalid_dimension");
  if (p->query_heads % p->kv_heads != 0 ||
      p->total_query_heads % p->kv_heads != 0)
    return error(3, "invalid_gqa_mapping");
  if (p->query_head_offset < 0 ||
      p->query_head_offset + p->query_heads > p->total_query_heads)
    return error(4, "invalid_query_head_range");
  if (!positive_strides(p->q_strides) || !positive_strides(p->k_strides) ||
      !positive_strides(p->v_strides) || !positive_strides(p->out_strides))
    return error(5, "unsupported_stride");
  if (!std::isfinite(p->scale) || p->scale <= 0)
    return error(6, "invalid_scale");
  if (p->causal != 1) return error(7, "causal_required");
  if (p->query_tile <= 0 || p->key_tile <= 0 || p->worker_count <= 0)
    return error(8, "invalid_tile_or_worker_count");
  if (p->query_position_offset < 0 ||
      p->query_position_offset + p->query_length > p->context_length)
    return error(9, "invalid_causal_position_range");
  return {0, "ok"};
}

inline size_t offset(const int64_t s[4], int64_t b, int64_t h, int64_t t,
                     int64_t d) {
  return size_t(b * s[0] + h * s[1] + t * s[2] + d * s[3]);
}

float dot_scalar(const float *q, const float *k, int64_t d,
                 int64_t qs, int64_t ks) {
  float sum = 0.0f;
  for (int64_t i = 0; i < d; ++i) sum += q[i * qs] * k[i * ks];
  return sum;
}

__attribute__((target("avx2,fma")))
float dot_avx2(const float *q, const float *k, int64_t d,
               int64_t qs, int64_t ks) {
  if (qs != 1 || ks != 1) return dot_scalar(q, k, d, qs, ks);
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= d; i += 8)
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(k + i), acc);
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 x = _mm_add_ps(lo, hi);
  x = _mm_hadd_ps(x, x); x = _mm_hadd_ps(x, x);
  float sum = _mm_cvtss_f32(x);
  for (; i < d; ++i) sum += q[i] * k[i];
  return sum;
}

template <bool Simd>
HirFusedAttentionStatus run(const HirFusedAttentionParams *p,
                            HirFusedAttentionStats *stats) {
  auto status = validate(p);
  if (status.code) return status;
  const size_t score_bytes = size_t(p->key_tile) * sizeof(float);
  const size_t accum_bytes = size_t(p->head_dimension) * sizeof(float);
  // One bounded scratch allocation per complete native invocation. It is
  // reused for every row and never scales as query_length*context_length.
  std::vector<float> scratch(size_t(p->key_tile + p->head_dimension));
  float *scores = scratch.data();
  float *accumulator = scores + p->key_tile;
  if (stats) {
    *stats = {score_bytes + accum_bytes, std::max(score_bytes, accum_bytes),
              1, 0, score_bytes, accum_bytes, score_bytes + accum_bytes};
  }
  const int64_t group = p->total_query_heads / p->kv_heads;
  for (int64_t b = 0; b < p->batch; ++b)
    for (int64_t h = 0; h < p->query_heads; ++h) {
      const int64_t global_h = p->query_head_offset + h;
      const int64_t kv_h = global_h / group;
      for (int64_t qi = 0; qi < p->query_length; ++qi) {
        std::fill(accumulator, accumulator + p->head_dimension, 0.0f);
        float running_max = -std::numeric_limits<float>::infinity();
        float denominator = 0.0f;
        const float *qrow = p->q + offset(p->q_strides, b, h, qi, 0);
        const int64_t max_key = p->query_position_offset + qi;
        for (int64_t ks = 0; ks < p->context_length; ks += p->key_tile) {
          const int64_t ke = std::min(ks + p->key_tile, p->context_length);
          float tile_max = -std::numeric_limits<float>::infinity();
          for (int64_t key_i = ks; key_i < ke; ++key_i) {
            if (key_i > max_key) {
              scores[key_i - ks] = -std::numeric_limits<float>::infinity();
              continue;
            }
            const float *krow = p->k + offset(p->k_strides, b, kv_h, key_i, 0);
            float score = (Simd ? dot_avx2(qrow, krow, p->head_dimension,
                                           p->q_strides[3], p->k_strides[3])
                                : dot_scalar(qrow, krow, p->head_dimension,
                                             p->q_strides[3], p->k_strides[3]))
                          * p->scale;
            scores[key_i - ks] = score;
            tile_max = std::max(tile_max, score);
          }
          if (!std::isfinite(tile_max)) continue; // Entire tile is masked.
          const float new_max = std::max(running_max, tile_max);
          // Old l and o were normalized relative to running_max. Rescale both
          // by exp(running_max-new_max) before adding weights in the new frame.
          const float old_scale = std::isfinite(running_max)
                                      ? std::exp(running_max - new_max) : 0.0f;
          denominator *= old_scale;
          if constexpr (Simd) {
            int64_t d = 0; __m256 alpha = _mm256_set1_ps(old_scale);
            for (; d + 8 <= p->head_dimension; d += 8)
              _mm256_storeu_ps(accumulator + d,
                  _mm256_mul_ps(_mm256_loadu_ps(accumulator + d), alpha));
            for (; d < p->head_dimension; ++d) accumulator[d] *= old_scale;
          } else {
            for (int64_t d = 0; d < p->head_dimension; ++d)
              accumulator[d] *= old_scale;
          }
          for (int64_t key_i = ks; key_i < ke && key_i <= max_key; ++key_i) {
            const float weight = std::exp(scores[key_i - ks] - new_max);
            denominator += weight;
            const float *vrow = p->v + offset(p->v_strides, b, kv_h, key_i, 0);
            if constexpr (Simd) {
              int64_t d = 0; __m256 w = _mm256_set1_ps(weight);
              if (p->v_strides[3] == 1) {
                for (; d + 8 <= p->head_dimension; d += 8)
                  _mm256_storeu_ps(accumulator + d,
                    _mm256_fmadd_ps(_mm256_loadu_ps(vrow + d), w,
                                    _mm256_loadu_ps(accumulator + d)));
              }
              for (; d < p->head_dimension; ++d)
                accumulator[d] += weight * vrow[d * p->v_strides[3]];
            } else {
              for (int64_t d = 0; d < p->head_dimension; ++d)
                accumulator[d] += weight * vrow[d * p->v_strides[3]];
            }
          }
          running_max = new_max;
        }
        if (!(denominator > 0.0f) || !std::isfinite(denominator))
          return error(10, "invalid_softmax_denominator");
        float *out = p->output + offset(p->out_strides, b, h, qi, 0);
        if constexpr (Simd) {
          int64_t d = 0; __m256 inv = _mm256_set1_ps(1.0f / denominator);
          for (; d + 8 <= p->head_dimension; d += 8)
            _mm256_storeu_ps(out + d,
                _mm256_mul_ps(_mm256_loadu_ps(accumulator + d), inv));
          for (; d < p->head_dimension; ++d) out[d] = accumulator[d] / denominator;
        } else {
          for (int64_t d = 0; d < p->head_dimension; ++d)
            out[d * p->out_strides[3]] = accumulator[d] / denominator;
        }
      }
    }
  return {0, "ok"};
}
}

extern "C" const char *hir_fused_attention_artifact_version() {
  return "hir.fused_online_attention.v1";
}
extern "C" int32_t hir_fused_attention_has_avx2() {
#if defined(__x86_64__)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return 0;
#endif
}
extern "C" HirFusedAttentionStatus hir_fused_online_attention_scalar(
    const HirFusedAttentionParams *p, HirFusedAttentionStats *s) {
  return run<false>(p, s);
}
extern "C" __attribute__((target("avx2,fma")))
HirFusedAttentionStatus hir_fused_online_attention_avx2(
    const HirFusedAttentionParams *p, HirFusedAttentionStats *s) {
  if (!hir_fused_attention_has_avx2()) return error(11, "avx2_fma_unavailable");
  return run<true>(p, s);
}
