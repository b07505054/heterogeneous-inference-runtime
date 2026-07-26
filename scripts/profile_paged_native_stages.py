#!/usr/bin/env python3
"""Native C++ stage decomposition for page-major paged-KV decode softmax."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_paged_vs_contiguous_pi import stage_breakdown


CPP_SOURCE = r"""
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static float data(size_t i, int seed) {
  return float(((i * 1103515245u + seed * 12345u) & 0xffffu) / 32768.0 - 1.0);
}

static double elapsed_ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
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
    total += elapsed_ms(a, b);
  }
  return total / samples;
}

struct Buffers {
  static constexpr int h = 2;
  static constexpr int d = 8;
  static constexpr int pt = 8;
  static constexpr int pages = 8;
  static constexpr size_t dim = d;
  static constexpr size_t page_stride = size_t(h) * pt * d;
  static constexpr size_t head_stride = size_t(pt) * d;
  int valid;
  size_t logical_pages;
  size_t full_pages;
  size_t tail_tokens;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> o;
  std::vector<float> scores;
  std::vector<int32_t> bt;
  std::vector<int32_t> physical;

  explicit Buffers(int tokens)
      : valid(tokens),
        logical_pages((size_t(tokens) + pt - 1) / pt),
        full_pages(size_t(tokens) / pt),
        tail_tokens(size_t(tokens) % pt),
        q(h * d),
        k(size_t(pages) * h * pt * d),
        v(size_t(pages) * h * pt * d),
        o(h * d),
        scores(tokens),
        bt(logical_pages),
        physical(logical_pages) {
    for (size_t i = 0; i < q.size(); ++i) q[i] = data(i, 100 + tokens);
    for (size_t i = 0; i < k.size(); ++i) {
      k[i] = data(i, 200 + tokens);
      v[i] = data(i, 300 + tokens);
    }
    std::iota(bt.begin(), bt.end(), 0);
  }
};

static void fill_scores(Buffers& b, int inner, float scale) {
  for (int call = 0; call < inner; ++call) {
    for (int hi = 0; hi < Buffers::h; ++hi) {
      const size_t qb = size_t(hi) * Buffers::dim;
      const size_t head_base = size_t(hi) * Buffers::head_stride;
      const float* qh = b.q.data() + qb;
      size_t logical = 0;
      for (size_t block = 0; block < b.full_pages; ++block) {
        const float* kt = b.k.data() + size_t(b.physical[block]) * Buffers::page_stride + head_base;
        for (size_t offset = 0; offset < size_t(Buffers::pt); ++offset, ++logical, kt += Buffers::dim) {
          float s = 0.0f;
          for (size_t di = 0; di < Buffers::dim; ++di) s += qh[di] * kt[di];
          b.scores[logical] = s * scale;
        }
      }
      if (b.tail_tokens) {
        const float* kt = b.k.data() + size_t(b.physical[b.full_pages]) * Buffers::page_stride + head_base;
        for (size_t offset = 0; offset < b.tail_tokens; ++offset, ++logical, kt += Buffers::dim) {
          float s = 0.0f;
          for (size_t di = 0; di < Buffers::dim; ++di) s += qh[di] * kt[di];
          b.scores[logical] = s * scale;
        }
      }
    }
  }
}

static float max_reduce(Buffers& b, int inner) {
  float mx = -std::numeric_limits<float>::infinity();
  for (int call = 0; call < inner; ++call) {
    for (int hi = 0; hi < Buffers::h; ++hi) {
      mx = -std::numeric_limits<float>::infinity();
      for (int t = 0; t < b.valid; ++t) mx = std::max(mx, b.scores[t]);
    }
  }
  return mx;
}

static float exp_sum(Buffers& b, int inner, float mx) {
  float sum = 0.0f;
  for (int call = 0; call < inner; ++call) {
    for (int hi = 0; hi < Buffers::h; ++hi) {
      sum = 0.0f;
      for (int t = 0; t < b.valid; ++t) {
        b.scores[t] = std::exp(b.scores[t] - mx);
        sum += b.scores[t];
      }
    }
  }
  return sum;
}

static float reciprocal(Buffers& b, int inner) {
  float inv = 0.0f;
  for (int call = 0; call < inner; ++call) {
    for (int hi = 0; hi < Buffers::h; ++hi) {
      float sum = 0.0f;
      for (int t = 0; t < b.valid; ++t) sum += b.scores[t];
      inv = 1.0f / sum;
    }
  }
  return inv;
}

static void v_accumulate_fused(Buffers& b, int inner, float inv_sum) {
  for (int call = 0; call < inner; ++call) {
    for (int hi = 0; hi < Buffers::h; ++hi) {
      const size_t qb = size_t(hi) * Buffers::dim;
      const size_t head_base = size_t(hi) * Buffers::head_stride;
      float* outp = b.o.data() + qb;
      std::fill(outp, outp + Buffers::dim, 0.0f);
      size_t logical = 0;
      for (size_t block = 0; block < b.full_pages; ++block) {
        const float* vt = b.v.data() + size_t(b.physical[block]) * Buffers::page_stride + head_base;
        for (size_t offset = 0; offset < size_t(Buffers::pt); ++offset, ++logical, vt += Buffers::dim) {
          const float weight = b.scores[logical] * inv_sum;
          for (size_t di = 0; di < Buffers::dim; ++di) outp[di] += weight * vt[di];
        }
      }
      if (b.tail_tokens) {
        const float* vt = b.v.data() + size_t(b.physical[b.full_pages]) * Buffers::page_stride + head_base;
        for (size_t offset = 0; offset < b.tail_tokens; ++offset, ++logical, vt += Buffers::dim) {
          const float weight = b.scores[logical] * inv_sum;
          for (size_t di = 0; di < Buffers::dim; ++di) outp[di] += weight * vt[di];
        }
      }
    }
  }
}

static void run_case(int valid, int reps, int inner) {
  Buffers b(valid);
  std::vector<double> setup, page, qk, smx, sexp, recip, norm, vv, total;
  volatile float sink = 0.0f;
  for (int r = 0; r < reps; ++r) {
    auto t0 = Clock::now();
    const float scale = 1.0f / std::sqrt(float(Buffers::d));
    auto t1 = Clock::now();
    for (size_t block = 0; block < b.logical_pages; ++block) b.physical[block] = b.bt[block];
    auto t2 = Clock::now();
    fill_scores(b, inner, scale);
    auto t3 = Clock::now();
    float mx = max_reduce(b, inner);
    auto t4 = Clock::now();
    float sum = exp_sum(b, inner, mx);
    auto t5 = Clock::now();
    float inv = reciprocal(b, inner);
    auto t6 = Clock::now();
    v_accumulate_fused(b, inner, inv);
    auto t7 = Clock::now();
    for (float x : b.o) sink += x + sum;

    setup.push_back(elapsed_ms(t0, t1));
    page.push_back(elapsed_ms(t1, t2));
    qk.push_back(elapsed_ms(t2, t3) / inner);
    smx.push_back(elapsed_ms(t3, t4) / inner);
    sexp.push_back(elapsed_ms(t4, t5) / inner);
    recip.push_back(elapsed_ms(t5, t6) / inner);
    norm.push_back(0.0);
    vv.push_back(elapsed_ms(t6, t7) / inner);
    total.push_back(setup.back() + page.back() + qk.back() + smx.back() + sexp.back() + recip.back() + norm.back() + vv.back());
  }
  std::cout << "{\"valid_tokens\":" << valid << ",";
  emit_values("setup_ms", setup); std::cout << ",";
  emit_values("logical_page_setup_ms", page); std::cout << ",";
  emit_values("qk_score_generation_ms", qk); std::cout << ",";
  emit_values("softmax_max_reduction_ms", smx); std::cout << ",";
  emit_values("softmax_exp_sum_ms", sexp); std::cout << ",";
  emit_values("softmax_reciprocal_ms", recip); std::cout << ",";
  emit_values("softmax_normalization_writeback_ms", norm); std::cout << ",";
  emit_values("v_accumulation_fused_normalization_ms", vv); std::cout << ",";
  emit_values("complete_kernel_modeled_ms", total);
  std::cout << "}";
  (void)sink;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::stoi(argv[1]) : 80;
  const int inner = argc > 2 ? std::stoi(argv[2]) : 500;
  std::cout << "{\"stage_rows\":[";
  bool first = true;
  for (int valid : {1, 9, 16, 32, 64}) {
    if (!first) std::cout << ",";
    first = false;
    run_case(valid, reps, inner);
  }
  std::cout << "],\"repetitions\":" << reps
            << ",\"inner_iterations\":" << inner
            << ",\"timer_overhead_ms\":" << timer_overhead_ms(1000)
            << "}\n";
}
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, float | int | bool | str]:
    ordered = sorted(values)
    mean = statistics.fmean(ordered)
    median = statistics.median(ordered)
    sd = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    mad = statistics.median(abs(x - median) for x in ordered)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    cv = sd / mean if mean else 0.0
    return {
        "median_ms": median,
        "mean_ms": mean,
        "minimum_ms": ordered[0],
        "p95_ms": p95,
        "mad_ms": mad,
        "stddev_ms": sd,
        "coefficient_of_variation": cv,
        "samples": len(ordered),
        "unstable": bool(cv > 0.25 or p95 > median * 2.0),
        "stability_rule": "stable iff coefficient_of_variation <= 0.25 and p95 <= 2.0 * median",
    }


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    summaries = {key[:-3]: summarize(value) for key, value in row.items() if key.endswith("_ms")}
    median_stages = {
        "setup": summaries["setup"]["median_ms"],
        "logical_page_setup": summaries["logical_page_setup"]["median_ms"],
        "qk_score_generation": summaries["qk_score_generation"]["median_ms"],
        "softmax_max_reduction": summaries["softmax_max_reduction"]["median_ms"],
        "softmax_exp_sum": summaries["softmax_exp_sum"]["median_ms"],
        "softmax_reciprocal": summaries["softmax_reciprocal"]["median_ms"],
        "softmax_normalization_writeback": summaries["softmax_normalization_writeback"]["median_ms"],
        "v_accumulation_fused_normalization": summaries["v_accumulation_fused_normalization"]["median_ms"],
    }
    total = summaries["complete_kernel_modeled"]["median_ms"]
    row["stage_summaries"] = summaries
    row["softmax_core_ms"] = (
        median_stages["softmax_max_reduction"]
        + median_stages["softmax_exp_sum"]
        + median_stages["softmax_reciprocal"]
        + median_stages["softmax_normalization_writeback"]
    )
    row["attention_math_ms"] = median_stages["qk_score_generation"] + median_stages["v_accumulation_fused_normalization"]
    row["breakdown"] = stage_breakdown(total, median_stages)
    return row


def render_report(payload: dict[str, Any], output: Path) -> None:
    lines = ["# Paged Softmax Stage Breakdown", ""]
    lines.append(f"- Repetitions: {payload['repetitions']}")
    lines.append(f"- Inner iterations: {payload['inner_iterations']}")
    lines.append(f"- Timer overhead: {payload['timer_overhead_ms']:.9f} ms")
    lines.append("")
    lines.append("| Tokens | Total ms | QK % | Max % | Exp/Sum % | Reciprocal % | Normalization writeback % | V+norm % | Softmax core ms | CV |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["stage_rows"]:
        b = row["breakdown"]["stages"]
        total = row["stage_summaries"]["complete_kernel_modeled"]["median_ms"]
        cv = row["stage_summaries"]["complete_kernel_modeled"]["coefficient_of_variation"]
        lines.append(
            f"| {row['valid_tokens']} | {total:.6f} | "
            f"{b['qk_score_generation']['percent_of_total']:.2f} | "
            f"{b['softmax_max_reduction']['percent_of_total']:.2f} | "
            f"{b['softmax_exp_sum']['percent_of_total']:.2f} | "
            f"{b['softmax_reciprocal']['percent_of_total']:.2f} | "
            f"{b['softmax_normalization_writeback']['percent_of_total']:.2f} | "
            f"{b['v_accumulation_fused_normalization']['percent_of_total']:.2f} | "
            f"{row['softmax_core_ms']:.6f} | {cv:.4f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=80)
    parser.add_argument("--inner", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    source = args.build_dir / "paged_softmax_stage_probe.cpp"
    binary = args.build_dir / "paged_softmax_stage_probe"
    source.write_text(textwrap.dedent(CPP_SOURCE), encoding="utf-8")
    build_cmd = ["g++", "-O3", "-std=c++17", str(source), "-o", str(binary)]
    subprocess.run(build_cmd, check=True)
    raw = subprocess.run(
        [str(binary), str(args.repetitions), str(args.inner)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    payload = json.loads(raw)
    for row in payload["stage_rows"]:
        enrich_row(row)
    payload["build_command"] = build_cmd
    payload["probe_source_sha256"] = sha256(source)
    json_path = args.output_dir / "paged_softmax_stage_breakdown.json"
    report_path = args.output_dir / "paged_softmax_stage_breakdown_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_report(payload, report_path)
    print(json.dumps({"status": "passed", "json": str(json_path), "report": str(report_path), "probe_sha256": payload["probe_source_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
