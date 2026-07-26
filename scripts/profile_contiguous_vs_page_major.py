#!/usr/bin/env python3
"""Differential native profiling for contiguous and page-major paged decode."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_paged_vs_contiguous_pi import stat


TOKENS = (1, 7, 8, 9, 16, 32, 64)
STAGES = (
    "validation_setup",
    "metadata_preparation",
    "qk_score_generation_plus_max",
    "softmax_exp_sum",
    "standalone_reciprocal",
    "v_accumulation",
    "output_store_read",
    "logical_page_count_setup",
    "block_table_validation_cache",
    "k_page_base_generation",
    "v_page_base_generation",
    "page_loop_tail_handling",
    "contiguous_capacity_base_setup",
    "contiguous_stride_setup",
)
DIAGNOSTIC_STAGES = {
    "k_page_base_generation",
    "v_page_base_generation",
    "page_loop_tail_handling",
}
EXCLUSIVE_STAGES = tuple(stage for stage in STAGES if stage not in DIAGNOSTIC_STAGES)


CPP_SOURCE = r"""
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

struct HirAttentionStatus { int32_t code; const char* message; };
extern "C" HirAttentionStatus hir_cpu_attention_decode_contiguous_kv_fp32(
    const float*, size_t, const float*, size_t, const float*, size_t,
    float*, size_t, float*, size_t, int64_t, int64_t, int64_t, int64_t, int64_t);
extern "C" HirAttentionStatus hir_cpu_attention_decode_paged_kv_page_major_fp32(
    const float*, size_t, const float*, size_t, const float*, size_t,
    const int32_t*, size_t, int32_t*, size_t,
    float*, size_t, float*, size_t, int64_t, int64_t, int64_t, int64_t, int64_t, int32_t);

using Clock = std::chrono::steady_clock;
static constexpr int H = 2;
static constexpr int D = 8;
static constexpr int CAP = 64;
static constexpr int PT = 8;
static constexpr int PAGES = 8;
static constexpr int SENTINEL = -1;
static volatile float g_sink = 0.0f;

static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static float data(size_t i, int seed) {
  return float(((i * 1103515245u + seed * 12345u) & 0xffffu) / 32768.0 - 1.0);
}

static void emit_values(const char* name, const std::vector<double>& xs) {
  std::cout << "\"" << name << "\":[";
  for (size_t i = 0; i < xs.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << xs[i];
  }
  std::cout << "]";
}

static double timer_overhead_ms(int samples) {
  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    auto a = Clock::now();
    auto b = Clock::now();
    total += ms(a, b);
  }
  return total / samples;
}

struct CaseData {
  int valid;
  size_t logical_pages;
  std::vector<float> q, kc, vc, kp, vp, oc, op, wc, wp;
  std::vector<int32_t> bt, physical;
  explicit CaseData(int tokens)
      : valid(tokens),
        logical_pages((size_t(tokens) + PT - 1) / PT),
        q(H * D),
        kc(H * CAP * D),
        vc(H * CAP * D),
        kp(PAGES * H * PT * D),
        vp(PAGES * H * PT * D),
        oc(H * D),
        op(H * D),
        wc(CAP),
        wp(CAP),
        bt(logical_pages),
        physical(logical_pages, SENTINEL) {
    for (size_t i = 0; i < q.size(); ++i) q[i] = data(i, 100 + tokens);
    std::iota(bt.begin(), bt.end(), 0);
    for (int hi = 0; hi < H; ++hi) {
      for (int t = 0; t < tokens; ++t) {
        for (int di = 0; di < D; ++di) {
          float kval = data(size_t((hi * tokens + t) * D + di), 200 + tokens);
          float vval = data(size_t((hi * tokens + t) * D + di), 300 + tokens);
          kc[((hi * CAP + t) * D) + di] = kval;
          vc[((hi * CAP + t) * D) + di] = vval;
          int page = t / PT;
          int off = t % PT;
          kp[(((page * H + hi) * PT + off) * D) + di] = kval;
          vp[(((page * H + hi) * PT + off) * D) + di] = vval;
        }
      }
    }
  }
};

#define NOINLINE __attribute__((noinline))

static NOINLINE void stage_contig_validation(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    size_t cacheN = size_t(H) * CAP * D;
    size_t qn = size_t(H) * D;
    if (!c.q.data() || !c.kc.data() || !c.vc.data() || !c.oc.data() || !c.wc.data()) g_sink += 1.0f;
    if (c.valid <= 0 || c.valid > CAP || c.kc.size() < cacheN || c.vc.size() < cacheN || c.q.size() < qn) g_sink += 2.0f;
  }
}

static NOINLINE void stage_paged_validation(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    size_t poolN = size_t(PAGES) * H * PT * D;
    size_t qn = size_t(H) * D;
    if (!c.q.data() || !c.kp.data() || !c.vp.data() || !c.bt.data() || !c.physical.data() || !c.op.data() || !c.wp.data()) g_sink += 1.0f;
    if (c.valid <= 0 || PT <= 0 || c.kp.size() < poolN || c.vp.size() < poolN || c.q.size() < qn) g_sink += 2.0f;
  }
}

static NOINLINE void stage_contig_metadata(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    float scale = 1.0f / std::sqrt(float(D));
    size_t capacity_stride = size_t(CAP) * D;
    size_t head_stride = capacity_stride;
    g_sink += scale + float(head_stride) * 0.0f;
  }
}

static NOINLINE void stage_paged_metadata(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    float scale = 1.0f / std::sqrt(float(D));
    size_t page_stride = size_t(H) * PT * D;
    size_t head_stride = size_t(PT) * D;
    g_sink += scale + float(page_stride + head_stride) * 0.0f;
  }
}

static NOINLINE void stage_paged_logical_count(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    size_t logical_pages = (size_t(c.valid) + PT - 1) / PT;
    if (logical_pages > c.bt.size() || logical_pages > c.physical.size()) g_sink += 1.0f;
    g_sink += float(logical_pages) * 0.0f;
  }
}

static NOINLINE void stage_paged_block_cache(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    for (size_t block = 0; block < c.logical_pages; ++block) {
      int32_t page = c.bt[block];
      if (page == SENTINEL || page < 0 || page >= PAGES) g_sink += 1.0f;
      c.physical[block] = page;
    }
  }
}

static NOINLINE void stage_contig_qk_max(CaseData& c, int inner) {
  const float scale = 1.0f / std::sqrt(float(D));
  for (int n = 0; n < inner; ++n) {
    for (int hi = 0; hi < H; ++hi) {
      size_t qb = size_t(hi) * D;
      float mx = -std::numeric_limits<float>::infinity();
      for (int t = 0; t < c.valid; ++t) {
        size_t kb = ((size_t(hi) * CAP + t) * D);
        float s = 0.0f;
        for (int di = 0; di < D; ++di) s += c.q[qb + di] * c.kc[kb + di];
        c.wc[t] = s * scale;
        mx = std::max(mx, c.wc[t]);
      }
      g_sink += mx * 0.0f;
    }
  }
}

static NOINLINE void stage_paged_qk_max(CaseData& c, int inner) {
  const float scale = 1.0f / std::sqrt(float(D));
  const size_t page_stride = size_t(H) * PT * D;
  const size_t head_stride = size_t(PT) * D;
  for (int n = 0; n < inner; ++n) {
    for (int hi = 0; hi < H; ++hi) {
      size_t qb = size_t(hi) * D;
      float mx = -std::numeric_limits<float>::infinity();
      size_t logical = 0;
      for (size_t block = 0; block < c.logical_pages; ++block) {
        const float* kbase = c.kp.data() + size_t(c.physical[block]) * page_stride + size_t(hi) * head_stride;
        size_t in_page = std::min(size_t(PT), size_t(c.valid) - logical);
        for (size_t off = 0; off < in_page; ++off, ++logical) {
          const float* kt = kbase + off * D;
          float s = 0.0f;
          for (int di = 0; di < D; ++di) s += c.q[qb + di] * kt[di];
          c.wp[logical] = s * scale;
          mx = std::max(mx, c.wp[logical]);
        }
      }
      g_sink += mx * 0.0f;
    }
  }
}

static NOINLINE void stage_contig_exp_sum(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    for (int hi = 0; hi < H; ++hi) {
      float mx = -std::numeric_limits<float>::infinity();
      for (int t = 0; t < c.valid; ++t) mx = std::max(mx, c.wc[t]);
      float sum = 0.0f;
      for (int t = 0; t < c.valid; ++t) {
        c.wc[t] = std::exp(c.wc[t] - mx);
        sum += c.wc[t];
      }
      g_sink += sum * 0.0f;
    }
  }
}

static NOINLINE void stage_paged_exp_sum(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    for (int hi = 0; hi < H; ++hi) {
      float mx = -std::numeric_limits<float>::infinity();
      for (int t = 0; t < c.valid; ++t) mx = std::max(mx, c.wp[t]);
      float sum = 0.0f;
      for (int t = 0; t < c.valid; ++t) {
        c.wp[t] = std::exp(c.wp[t] - mx);
        sum += c.wp[t];
      }
      g_sink += sum * 0.0f;
    }
  }
}

static NOINLINE void stage_contig_v(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    for (int hi = 0; hi < H; ++hi) {
      size_t qb = size_t(hi) * D;
      float sum = 0.0f;
      for (int t = 0; t < c.valid; ++t) sum += c.wc[t];
      for (int di = 0; di < D; ++di) {
        float x = 0.0f;
        for (int t = 0; t < c.valid; ++t) {
          size_t vb = ((size_t(hi) * CAP + t) * D);
          x += (c.wc[t] / sum) * c.vc[vb + di];
        }
        c.oc[qb + di] = x;
      }
    }
  }
}

static NOINLINE void stage_paged_v(CaseData& c, int inner) {
  const size_t page_stride = size_t(H) * PT * D;
  const size_t head_stride = size_t(PT) * D;
  for (int n = 0; n < inner; ++n) {
    for (int hi = 0; hi < H; ++hi) {
      size_t qb = size_t(hi) * D;
      float sum = 0.0f;
      for (int t = 0; t < c.valid; ++t) sum += c.wp[t];
      std::fill(c.op.begin() + qb, c.op.begin() + qb + D, 0.0f);
      size_t logical = 0;
      for (size_t block = 0; block < c.logical_pages; ++block) {
        const float* vbase = c.vp.data() + size_t(c.physical[block]) * page_stride + size_t(hi) * head_stride;
        size_t in_page = std::min(size_t(PT), size_t(c.valid) - logical);
        for (size_t off = 0; off < in_page; ++off, ++logical) {
          const float weight = c.wp[logical] / sum;
          const float* vt = vbase + off * D;
          for (int di = 0; di < D; ++di) c.op[qb + di] += weight * vt[di];
        }
      }
    }
  }
}

static NOINLINE void stage_paged_k_base(CaseData& c, int inner) {
  const size_t page_stride = size_t(H) * PT * D;
  const size_t head_stride = size_t(PT) * D;
  for (int n = 0; n < inner; ++n)
    for (int hi = 0; hi < H; ++hi)
      for (size_t block = 0; block < c.logical_pages; ++block)
        g_sink += *(c.kp.data() + size_t(c.physical[block]) * page_stride + size_t(hi) * head_stride) * 0.0f;
}

static NOINLINE void stage_paged_v_base(CaseData& c, int inner) {
  const size_t page_stride = size_t(H) * PT * D;
  const size_t head_stride = size_t(PT) * D;
  for (int n = 0; n < inner; ++n)
    for (int hi = 0; hi < H; ++hi)
      for (size_t block = 0; block < c.logical_pages; ++block)
        g_sink += *(c.vp.data() + size_t(c.physical[block]) * page_stride + size_t(hi) * head_stride) * 0.0f;
}

static NOINLINE void stage_paged_page_loop(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    size_t count = 0;
    for (int hi = 0; hi < H; ++hi) {
      size_t logical = 0;
      for (size_t block = 0; block < c.logical_pages; ++block) {
        size_t in_page = std::min(size_t(PT), size_t(c.valid) - logical);
        for (size_t off = 0; off < in_page; ++off, ++logical) count += off + block + hi;
      }
    }
    g_sink += float(count) * 0.0f;
  }
}

static NOINLINE void stage_output_read(CaseData& c, bool paged, int inner) {
  for (int n = 0; n < inner; ++n) {
    const auto& o = paged ? c.op : c.oc;
    for (float x : o) g_sink += x * 0.0f;
  }
}

static NOINLINE void direct_contig(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    auto st = hir_cpu_attention_decode_contiguous_kv_fp32(
        c.q.data(), c.q.size(), c.kc.data(), c.kc.size(), c.vc.data(), c.vc.size(),
        c.oc.data(), c.oc.size(), c.wc.data(), c.wc.size(), 1, H, c.valid, CAP, D);
    if (st.code) g_sink += 1.0f;
  }
}

static NOINLINE void direct_paged(CaseData& c, int inner) {
  for (int n = 0; n < inner; ++n) {
    auto st = hir_cpu_attention_decode_paged_kv_page_major_fp32(
        c.q.data(), c.q.size(), c.kp.data(), c.kp.size(), c.vp.data(), c.vp.size(),
        c.bt.data(), c.bt.size(), c.physical.data(), c.physical.size(),
        c.op.data(), c.op.size(), c.wp.data(), c.wp.size(), c.valid, PAGES, H, PT, D, SENTINEL);
    if (st.code) g_sink += 1.0f;
  }
}

template <typename F>
static double timed(F fn, int inner) {
  auto a = Clock::now();
  fn();
  auto b = Clock::now();
  return ms(a, b) / inner;
}

static void run_case(int valid, int reps, int inner) {
  CaseData c(valid);
  stage_paged_block_cache(c, 1);
  std::vector<double> cv, cm, cq, ce, cr, cvv, co, ccaps, cstride, ctotal;
  std::vector<double> pv, pm, pq, pe, pr, pvv, po, plog, pbt, pkb, pvb, ploop, ptotal;
  std::vector<double> direct_c, direct_p;
  for (int r = 0; r < reps; ++r) {
    cv.push_back(timed([&] { stage_contig_validation(c, inner); }, inner));
    cm.push_back(timed([&] { stage_contig_metadata(c, inner); }, inner));
    ccaps.push_back(timed([&] { stage_contig_metadata(c, inner); }, inner));
    cstride.push_back(0.0);
    cq.push_back(timed([&] { stage_contig_qk_max(c, inner); }, inner));
    ce.push_back(timed([&] { stage_contig_exp_sum(c, inner); }, inner));
    cr.push_back(0.0);
    cvv.push_back(timed([&] { stage_contig_v(c, inner); }, inner));
    co.push_back(timed([&] { stage_output_read(c, false, inner); }, inner));
    direct_c.push_back(timed([&] { direct_contig(c, inner); }, inner));

    pv.push_back(timed([&] { stage_paged_validation(c, inner); }, inner));
    plog.push_back(timed([&] { stage_paged_logical_count(c, inner); }, inner));
    pbt.push_back(timed([&] { stage_paged_block_cache(c, inner); }, inner));
    pm.push_back(timed([&] { stage_paged_metadata(c, inner); }, inner));
    pkb.push_back(timed([&] { stage_paged_k_base(c, inner); }, inner));
    pvb.push_back(timed([&] { stage_paged_v_base(c, inner); }, inner));
    ploop.push_back(timed([&] { stage_paged_page_loop(c, inner); }, inner));
    pq.push_back(timed([&] { stage_paged_qk_max(c, inner); }, inner));
    pe.push_back(timed([&] { stage_paged_exp_sum(c, inner); }, inner));
    pr.push_back(0.0);
    pvv.push_back(timed([&] { stage_paged_v(c, inner); }, inner));
    po.push_back(timed([&] { stage_output_read(c, true, inner); }, inner));
    direct_p.push_back(timed([&] { direct_paged(c, inner); }, inner));

    ctotal.push_back(cv.back()+cm.back()+cq.back()+ce.back()+cr.back()+cvv.back()+co.back()+ccaps.back()+cstride.back());
    ptotal.push_back(pv.back()+pm.back()+pq.back()+pe.back()+pr.back()+pvv.back()+po.back()+plog.back()+pbt.back()+pkb.back()+pvb.back()+ploop.back());
  }
  std::cout << "{\"valid_tokens\":" << valid << ",\"contiguous\":{";
  emit_values("validation_setup_ms", cv); std::cout << ",";
  emit_values("metadata_preparation_ms", cm); std::cout << ",";
  emit_values("qk_score_generation_plus_max_ms", cq); std::cout << ",";
  emit_values("softmax_exp_sum_ms", ce); std::cout << ",";
  emit_values("standalone_reciprocal_ms", cr); std::cout << ",";
  emit_values("v_accumulation_ms", cvv); std::cout << ",";
  emit_values("output_store_read_ms", co); std::cout << ",";
  emit_values("contiguous_capacity_base_setup_ms", ccaps); std::cout << ",";
  emit_values("contiguous_stride_setup_ms", cstride); std::cout << ",";
  emit_values("modeled_total_ms", ctotal); std::cout << ",";
  emit_values("direct_exported_kernel_ms", direct_c); std::cout << "},\"page_major\":{";
  emit_values("validation_setup_ms", pv); std::cout << ",";
  emit_values("metadata_preparation_ms", pm); std::cout << ",";
  emit_values("qk_score_generation_plus_max_ms", pq); std::cout << ",";
  emit_values("softmax_exp_sum_ms", pe); std::cout << ",";
  emit_values("standalone_reciprocal_ms", pr); std::cout << ",";
  emit_values("v_accumulation_ms", pvv); std::cout << ",";
  emit_values("output_store_read_ms", po); std::cout << ",";
  emit_values("logical_page_count_setup_ms", plog); std::cout << ",";
  emit_values("block_table_validation_cache_ms", pbt); std::cout << ",";
  emit_values("k_page_base_generation_ms", pkb); std::cout << ",";
  emit_values("v_page_base_generation_ms", pvb); std::cout << ",";
  emit_values("page_loop_tail_handling_ms", ploop); std::cout << ",";
  emit_values("modeled_total_ms", ptotal); std::cout << ",";
  emit_values("direct_exported_kernel_ms", direct_p); std::cout << "}}";
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? std::stoi(argv[1]) : 80;
  int inner = argc > 2 ? std::stoi(argv[2]) : 500;
  std::cout << "{\"timer_overhead_ms\":" << timer_overhead_ms(1000)
            << ",\"repetitions\":" << reps
            << ",\"inner_iterations\":" << inner
            << ",\"token_counts\":[1,7,8,9,16,32,64],\"stage_rows\":[";
  bool first = true;
  for (int t : {1,7,8,9,16,32,64}) {
    if (!first) std::cout << ",";
    first = false;
    run_case(t, reps, inner);
  }
  std::cout << "]}\n";
}
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, Any]:
    return stat(values, warmups=0)


def median_of(section: dict[str, Any], stage: str) -> float:
    key = f"{stage}_ms"
    if key not in section:
        return 0.0
    return section[key]["summary"]["median_ms"]


def enrich(payload: dict[str, Any]) -> dict[str, Any]:
    for row in payload["stage_rows"]:
        for side in ("contiguous", "page_major"):
            raw = row[side]
            for key, values in list(raw.items()):
                if key.endswith("_ms"):
                    raw[key] = {"summary": summarize(values), "raw_ms": values}
        c_direct = median_of(row["contiguous"], "direct_exported_kernel")
        p_direct = median_of(row["page_major"], "direct_exported_kernel")
        direct_gap = p_direct - c_direct
        modeled_gap = median_of(row["page_major"], "modeled_total") - median_of(row["contiguous"], "modeled_total")
        diffs = []
        for stage in STAGES:
            c = median_of(row["contiguous"], stage)
            p = median_of(row["page_major"], stage)
            delta = p - c
            diffs.append({
                "stage": stage,
                "contiguous_ms": c,
                "page_major_ms": p,
                "delta_ms": delta,
                "ratio": (p / c if c > 0 else None),
                "percent_of_direct_total_gap": (delta / direct_gap * 100.0 if direct_gap else None),
                "percent_of_modeled_gap": (delta / modeled_gap * 100.0 if modeled_gap else None),
            })
        measured_delta = sum(item["delta_ms"] for item in diffs if item["stage"] in EXCLUSIVE_STAGES)
        row["direct_total_gap_ms"] = direct_gap
        row["modeled_total_gap_ms"] = modeled_gap
        row["stage_differential"] = diffs
        row["exclusive_stage_delta_sum_ms"] = measured_delta
        row["unclassified_differential_remainder_ms"] = direct_gap - measured_delta
        row["unclassified_differential_remainder_percent"] = (
            (direct_gap - measured_delta) / direct_gap * 100.0 if direct_gap else None
        )
        positive = [d for d in diffs if d["delta_ms"] > 0 and d["stage"] in EXCLUSIVE_STAGES]
        row["largest_positive_delta_stage"] = max(positive, key=lambda x: x["delta_ms"]) if positive else None
    return payload


def parse_objdump(path: Path, symbol: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    insns = []
    for line in text.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+\t?([^<\s]+)", line)
        if m:
            insns.append((int(m.group(1), 16), m.group(3), line))
    mnems = [x[1] for x in insns]
    size = (insns[-1][0] - insns[0][0] + 4) if insns else 0
    return {
        "symbol": symbol,
        "instruction_bytes_approx": size,
        "instruction_count": len(insns),
        "branch_count": sum(1 for m in mnems if m.startswith("b") or m in {"cbz", "cbnz", "tbz", "tbnz"}),
        "load_count": sum(1 for m in mnems if m.startswith("ldr") or m.startswith("ldp") or m.startswith("ldur")),
        "store_count": sum(1 for m in mnems if m.startswith("str") or m.startswith("stp") or m.startswith("stur")),
        "integer_multiply_count": sum(1 for m in mnems if m in {"mul", "madd", "msub", "smull", "umull"}),
        "division_count": sum(1 for m in mnems if m in {"udiv", "sdiv"} or m.startswith("fdiv")),
        "neon_or_vector_fp_count": sum(1 for _, _, line in insns if re.search(r"\bv[0-9]+\\.", line)),
        "scalar_fp_count": sum(1 for m in mnems if m.startswith("f") and m not in {"fmov"}),
        "fma_count": sum(1 for m in mnems if m in {"fmadd", "fmla", "fmls"}),
        "expf_calls": text.count("<expf@plt>"),
        "sqrtf_calls": text.count("<sqrtf@plt>"),
        "stack_pair_traffic": sum(1 for _, _, line in insns if re.search(r"\b(stp|ldp)\b.*sp", line)),
    }


def render_stage_report(payload: dict[str, Any], path: Path) -> None:
    lines = ["# Contiguous vs Page-Major Differential Stage Breakdown", ""]
    lines.append(f"- Repetitions: {payload['repetitions']}")
    lines.append(f"- Inner iterations: {payload['inner_iterations']}")
    lines.append(f"- Timer overhead: {payload['timer_overhead_ms']:.9f} ms")
    lines.append("")
    for row in payload["stage_rows"]:
        lines.append(f"## {row['valid_tokens']} Tokens")
        lines.append(f"- Direct exported-kernel gap: {row['direct_total_gap_ms']:.9f} ms")
        lines.append(f"- Modeled stage gap: {row['modeled_total_gap_ms']:.9f} ms")
        lines.append(f"- Unclassified differential remainder: {row['unclassified_differential_remainder_ms']:.9f} ms")
        lines.append("")
        lines.append("| Stage | Contiguous ms | Page-major ms | Delta ms | Ratio | % direct gap |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for item in row["stage_differential"]:
            label = item["stage"]
            if label in DIAGNOSTIC_STAGES:
                label += " (diagnostic proxy, not exclusive)"
            ratio = "" if item["ratio"] is None else f"{item['ratio']:.2f}"
            pct = "" if item["percent_of_direct_total_gap"] is None else f"{item['percent_of_direct_total_gap']:.2f}"
            lines.append(
                f"| {label} | {item['contiguous_ms']:.9f} | {item['page_major_ms']:.9f} | "
                f"{item['delta_ms']:.9f} | {ratio} | {pct} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_assembly_report(contig: dict[str, Any], paged: dict[str, Any], path: Path) -> None:
    lines = ["# Contiguous vs Page-Major Assembly Report", ""]
    lines.append("| Metric | Contiguous | Page-major |")
    lines.append("| --- | ---: | ---: |")
    for key in sorted(k for k in contig if k != "symbol"):
        lines.append(f"| {key} | {contig[key]} | {paged[key]} |")
    lines.append("")
    lines.append(f"- Contiguous contains NEON/vector FP operands: {'yes' if contig['neon_or_vector_fp_count'] else 'no'}")
    lines.append(f"- Page-major contains NEON/vector FP operands: {'yes' if paged['neon_or_vector_fp_count'] else 'no'}")
    lines.append(f"- expf call-site references: contiguous={contig['expf_calls']}, page-major={paged['expf_calls']}")
    lines.append(f"- Approximate function-size delta: {paged['instruction_bytes_approx'] - contig['instruction_bytes_approx']} bytes")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--inner", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    source = args.build_dir / "contiguous_vs_page_major_probe.cpp"
    binary = args.build_dir / "contiguous_vs_page_major_probe"
    shared = args.build_dir / "libattention_fp32.so"
    source.write_text(textwrap.dedent(CPP_SOURCE), encoding="utf-8")
    build_shared = ["g++", "-O3", "-std=c++17", "-fPIC", "-shared", str(ROOT / "native/cpu_kernels/attention_fp32.cpp"), "-o", str(shared)]
    build_probe = ["g++", "-O3", "-std=c++17", str(source), str(shared), "-Wl,-rpath," + str(args.build_dir), "-o", str(binary)]
    subprocess.run(build_shared, check=True)
    subprocess.run(build_probe, check=True)
    raw = subprocess.run([str(binary), str(args.repetitions), str(args.inner)], check=True, text=True, capture_output=True).stdout
    payload = enrich(json.loads(raw))
    payload["build_shared_command"] = build_shared
    payload["build_probe_command"] = build_probe
    payload["probe_source_sha256"] = sha256(source)
    payload["native_artifact_sha256"] = sha256(shared)

    json_path = args.output_dir / "contiguous_vs_page_major_stage_breakdown.json"
    report_path = args.output_dir / "contiguous_vs_page_major_stage_breakdown_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_stage_report(payload, report_path)

    full_objdump = subprocess.run(["objdump", "-d", "-C", str(shared)], check=True, text=True, capture_output=True).stdout
    symbols = {
        "contiguous": "hir_cpu_attention_decode_contiguous_kv_fp32",
        "page_major": "hir_cpu_attention_decode_paged_kv_page_major_fp32",
    }
    assembly_metrics = {}
    for key, symbol in symbols.items():
        m = re.search(rf"(^[0-9a-f]+ <{symbol}>:\n.*?)(?=^\S|\Z)", full_objdump, re.M | re.S)
        text = m.group(1) if m else ""
        out = args.output_dir / ("contiguous_decode_objdump.txt" if key == "contiguous" else "page_major_decode_objdump.txt")
        out.write_text(text, encoding="utf-8")
        assembly_metrics[key] = parse_objdump(out, symbol)
    asm_report = args.output_dir / "contiguous_vs_page_major_assembly_report.md"
    render_assembly_report(assembly_metrics["contiguous"], assembly_metrics["page_major"], asm_report)
    payload["assembly_metrics"] = assembly_metrics
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "json": str(json_path), "report": str(report_path), "assembly_report": str(asm_report)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
