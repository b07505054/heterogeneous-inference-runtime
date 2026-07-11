#!/usr/bin/env bash
# RMSNorm CUDA block-size policy lab — full sweep driver (Phase 1).
#
# Runs correctness across the full tokens x hidden x block-size matrix,
# then the benchmark sweep with CSV/JSON/markdown outputs. Requires a CUDA
# host; both scripts SKIP cleanly (exit 0, correctness) or emit an
# "unavailable" artifact (benchmark) when CUDA is absent, and no
# performance claim is made without a measured run.
#
# Truth boundary: kernel-level correctness and microbenchmark evidence
# only — not an end-to-end Qwen or vLLM execution path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "error: $PYTHON not found; create the project venv first" >&2
  exit 1
fi

TOKENS="${TOKENS:-1,16,128}"
HIDDEN="${HIDDEN:-768,1024,4096,8192}"
BLOCK_SIZES="${BLOCK_SIZES:-64,128,256,512}"

echo "[sweep] correctness: tokens=${TOKENS} hidden=${HIDDEN} block_sizes=${BLOCK_SIZES}"
"$PYTHON" scripts/test_rmsnorm_cuda_correctness.py \
  --tokens "$TOKENS" \
  --hidden "$HIDDEN" \
  --block-sizes "$BLOCK_SIZES"

echo "[sweep] benchmark: tokens=${TOKENS} hidden=${HIDDEN} block_sizes=${BLOCK_SIZES}"
"$PYTHON" scripts/benchmark_rmsnorm_cuda.py \
  --tokens "$TOKENS" \
  --hidden "$HIDDEN" \
  --block-sizes "$BLOCK_SIZES" \
  --output results/cuda_transformer/rmsnorm_benchmark.json \
  --csv-output results/cuda_transformer/rmsnorm_block_size_sweep.csv \
  --report-output results/cuda_transformer/rmsnorm_benchmark_report.md

echo "[sweep] done"
