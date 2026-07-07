#!/usr/bin/env bash
# run_qwen_quant_benchmark.sh
#
# Phase C benchmark (minimal, AWQ only): 3-way comparison for Qwen 2.5-0.5B.
#
#   A = baseline          — original Qwen weights, manual vLLM config, quantization=none
#   B = compiler no-quant — original Qwen weights, compiler ExecutionPlan, quantization=none
#   C = compiler quant    — AWQ Qwen checkpoint,   compiler ExecutionPlan, quantization=awq
#
# Truth boundaries (read before interpreting any results this script produces):
#   - C vs B isolates quantized weights (both use the compiler ExecutionPlan;
#     only the weights + --quantization flag differ).
#   - C vs A combines quantized weights AND compiler execution plan policy
#     (KV layout, memory budget, serving topology) -- do not attribute a C-vs-A
#     delta to quantization alone.
#   - Do not claim a speedup from these results unless actually measured here.
#   - Do not claim accuracy parity -- this script runs no accuracy evaluation.
#   - Do not claim GTX 1650 has native INT4 Tensor Core support. C's compiler
#     plan (nvidia_gtx1650_maxq_awq_forced.json) is an explicit experimental
#     forced-quant override; see that profile's truth_boundary.
#
# Compiler artifacts:
#   B: execution_plan.json (schema 2.0.0) from compiler's no-quant GTX1650 profile
#      Default: ../ml-graph-compiler-runtime/artifacts/qwen/execution_plan.json
#   C: execution_plan.json (schema 2.0.0) from compiler's forced-AWQ GTX1650 profile
#      Default: ../ml-graph-compiler-runtime/artifacts/qwen_awq_plan/execution_plan.json
#
# Quantized model artifact (C only):
#   Default: ../ml-graph-compiler-runtime/artifacts/qwen_awq
#   Produced by: (cd ../ml-graph-compiler-runtime && .venv/bin/python tools/export_qwen_awq.py)
#   This script does NOT produce that artifact. If it is absent, or vLLM/AutoAWQ
#   are not installed in this environment, C is materialized (command written)
#   but not run -- see "Behavior when C cannot run" below. This is not a failure.
#
# Outputs (written to results/qwen_quant/):
#   traces/                    — generated trace files (one per workload)
#   baseline_*.json            — benchmark results for path A
#   compiler_noquant_*.json    — benchmark results for path B
#   compiler_awq_*.json        — benchmark results for path C (only if run)
#   *_command.txt              — materialized server command for each path (always written)
#   quant_comparison.md        — markdown comparison report with truth boundaries
#
# Behavior when C cannot run:
#   - vLLM not importable in .venv, or the quantized artifact directory is
#     missing/empty: this script still writes compiler_awq_command.txt (the
#     materialized --model/--quantization command) and a status note, then
#     continues with A and B if they can run, or exits 0 having materialized
#     all three commands. It does not raise or exit non-zero for this reason.
#
# Env overrides:
#   COMPILER_PLAN_NOQUANT   path to B's execution_plan.json
#   COMPILER_PLAN_AWQ       path to C's execution_plan.json
#   AWQ_MODEL_DIR           path to the AWQ quantized model artifact (C's --model)
#   BASELINE_MODEL          HuggingFace model ID for path A and B's original weights
#   OUTPUT_DIR              results directory (default: results/qwen_quant)
#   VLLM_HOST / VLLM_PORT   vLLM server host/port (default: 127.0.0.1 / 8000)
#   CONCURRENCY             benchmark concurrency (default: 1)
#   WARMUP                  warmup request count (default: 4)
#   VLLM_STARTUP_S          seconds to wait for vLLM startup (default: 60)
#   DRY_RUN                 set to 1 to print commands without running (default: 0)
#
# Usage:
#   bash scripts/run_qwen_quant_benchmark.sh
#   DRY_RUN=1 bash scripts/run_qwen_quant_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_PLAN_NOQUANT="${COMPILER_PLAN_NOQUANT:-$REPO_ROOT/../ml-graph-compiler-runtime/artifacts/qwen/execution_plan.json}"
COMPILER_PLAN_AWQ="${COMPILER_PLAN_AWQ:-$REPO_ROOT/../ml-graph-compiler-runtime/artifacts/qwen_awq_plan/execution_plan.json}"
AWQ_MODEL_DIR="${AWQ_MODEL_DIR:-$REPO_ROOT/../ml-graph-compiler-runtime/artifacts/qwen_awq}"
BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/qwen_quant}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
CONCURRENCY="${CONCURRENCY:-1}"
WARMUP="${WARMUP:-4}"
VLLM_STARTUP_S="${VLLM_STARTUP_S:-60}"
DRY_RUN="${DRY_RUN:-0}"

PYTHON="$REPO_ROOT/.venv/bin/python"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}"
TRACE_DIR="$OUTPUT_DIR/traces"

mkdir -p "$OUTPUT_DIR" "$TRACE_DIR"

log() { echo "[quant-benchmark] $*"; }

run_or_print() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[DRY_RUN] $*"
    else
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# Step 0: validate inputs
# ---------------------------------------------------------------------------

log "compiler plan (B, no-quant): $COMPILER_PLAN_NOQUANT"
log "compiler plan (C, awq):      $COMPILER_PLAN_AWQ"
log "AWQ model artifact (C):      $AWQ_MODEL_DIR"

if [[ ! -f "$COMPILER_PLAN_NOQUANT" ]]; then
    echo "error: B's execution_plan.json not found at: $COMPILER_PLAN_NOQUANT" >&2
    echo "  Generate it first:" >&2
    echo "    cd ../ml-graph-compiler-runtime && bash tools/run_qwen_compiler_pipeline.sh" >&2
    exit 1
fi
if [[ ! -f "$COMPILER_PLAN_AWQ" ]]; then
    echo "error: C's execution_plan.json not found at: $COMPILER_PLAN_AWQ" >&2
    echo "  Generate it first:" >&2
    echo "    cd ../ml-graph-compiler-runtime && bash tools/run_qwen_awq_compiler_pipeline.sh" >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "error: .venv not found at $PYTHON. Create it and install dependencies first." >&2
    exit 1
fi

AWQ_ARTIFACT_PRESENT=0
if [[ -f "$AWQ_MODEL_DIR/provenance.json" ]]; then
    AWQ_ARTIFACT_PRESENT=1
else
    log "note: AWQ quantized model artifact not found at $AWQ_MODEL_DIR (no provenance.json)."
    log "  C will be materialized (command written) but not run."
    log "  Produce it with: cd ../ml-graph-compiler-runtime && .venv/bin/python tools/export_qwen_awq.py"
fi

VLLM_AVAILABLE=1
if ! "$PYTHON" -c "import vllm" >/dev/null 2>&1; then
    VLLM_AVAILABLE=0
    log "note: vllm is not importable in $PYTHON."
    log "  All three paths will be materialized (commands written) but not run."
fi

# ---------------------------------------------------------------------------
# Step 1: materialize A/B/C vLLM commands from compiler plans
# ---------------------------------------------------------------------------

log "materializing runtime config for A / B / C..."

MATERIALIZED="$(COMPILER_PLAN_NOQUANT="$COMPILER_PLAN_NOQUANT" COMPILER_PLAN_AWQ="$COMPILER_PLAN_AWQ" BASELINE_MODEL="$BASELINE_MODEL" BASE_URL="$BASE_URL" $PYTHON - <<'PYEOF'
import sys, json, os
sys.path.insert(0, ".")
from dataclasses import replace

from deployment.execution_plan.loader import load_execution_plan
from deployment.execution_plan.stage_builder import build_execution_stages
from deployment.execution_plan.path_builder import build_execution_paths, build_baseline_vllm_path
from deployment.execution_plan.schema import ExecutionPathKind
from deployment.vllm_adapter.backend_adapter import VLLMBackendAdapter

baseline_model = os.environ.get("BASELINE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
adapter = VLLMBackendAdapter()


def first_vllm_path(plan_path):
    plan = load_execution_plan(plan_path)
    stages = build_execution_stages(plan)
    paths = build_execution_paths(plan, stages)
    vllm_paths = [p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM]
    if not vllm_paths:
        return plan, None
    return plan, vllm_paths[0]


out = {}

# A: baseline (manual vLLM config, no compiler plan involved).
baseline_path = build_baseline_vllm_path(
    model_id=baseline_model,
    output_artifact="results/qwen_quant/baseline_{workload}.json",
)
baseline_mat = adapter.materialize(baseline_path)
out["baseline_command"] = list(baseline_mat.command)
out["baseline_config"] = baseline_mat.config

# B: compiler no-quant. Override model/tokenizer: the compiler plan stores a
# short model_id ("qwen2.5-0.5b"), not a HuggingFace path. B uses the same
# original Qwen weights as A; only runtime policy differs (see
# ../ml-graph-compiler-runtime/CLAUDE.md and results/qwen_no_quant/).
plan_b, path_b = first_vllm_path(os.environ["COMPILER_PLAN_NOQUANT"])
if path_b is None:
    print(json.dumps({"error": "no COMPILER_GUIDED_VLLM path found in B's plan"}))
    sys.exit(1)
path_b = replace(path_b, runtime_config={**path_b.runtime_config, "model": baseline_model, "tokenizer": baseline_model})
compiler_b_mat = adapter.materialize(path_b)
out["compiler_noquant_command"] = list(compiler_b_mat.command)
out["compiler_noquant_config"] = compiler_b_mat.config
out["plan_id_b"] = plan_b.plan_id

# C: compiler quant (AWQ).
plan_c, path_c = first_vllm_path(os.environ["COMPILER_PLAN_AWQ"])
if path_c is None:
    print(json.dumps({"error": "no COMPILER_GUIDED_VLLM path found in C's plan"}))
    sys.exit(1)
compiler_c_mat = adapter.materialize(path_c)
out["compiler_awq_command"] = list(compiler_c_mat.command)
out["compiler_awq_config"] = compiler_c_mat.config
out["plan_id_c"] = plan_c.plan_id
out["hardware_profile_ref_c"] = plan_c.provenance.capability_bundle.hardware_profile_ref
quant = plan_c.global_decisions.quantization
out["quant_strategy_c"] = quant.get("strategy")
out["quant_algorithm_c"] = quant.get("algorithm")
out["quant_artifact_ref_c"] = quant.get("quantized_model_artifact_ref")
out["quant_truth_boundary_c"] = quant.get("truth_boundary")

print(json.dumps(out))
PYEOF
)"

if echo "$MATERIALIZED" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if 'error' not in d else 1)" 2>/dev/null; then
    :
else
    echo "error: materialization failed: $MATERIALIZED" >&2
    exit 1
fi

extract() {
    echo "$MATERIALIZED" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); print(d.get('$1', ''))"
}
extract_cmd() {
    echo "$MATERIALIZED" | $PYTHON -c "
import json, sys, shlex
d = json.load(sys.stdin)
print(shlex.join(str(x) for x in d['$1']))
"
}

BASELINE_CMD="$(extract_cmd baseline_command)"
COMPILER_NOQUANT_CMD="$(extract_cmd compiler_noquant_command)"
COMPILER_AWQ_CMD="$(extract_cmd compiler_awq_command)"

QUANT_STRATEGY_C="$(extract quant_strategy_c)"
QUANT_ALGORITHM_C="$(extract quant_algorithm_c)"
QUANT_ARTIFACT_REF_C="$(extract quant_artifact_ref_c)"
QUANT_TRUTH_BOUNDARY_C="$(extract quant_truth_boundary_c)"

echo "$BASELINE_CMD" > "$OUTPUT_DIR/baseline_command.txt"
echo "$COMPILER_NOQUANT_CMD" > "$OUTPUT_DIR/compiler_noquant_command.txt"
echo "$COMPILER_AWQ_CMD" > "$OUTPUT_DIR/compiler_awq_command.txt"

log ""
log "A (baseline) server command:"
log "  $BASELINE_CMD"
log ""
log "B (compiler no-quant) server command:"
log "  $COMPILER_NOQUANT_CMD"
log ""
log "C (compiler awq) server command:"
log "  $COMPILER_AWQ_CMD"
log ""
log "C quantization: strategy=$QUANT_STRATEGY_C algorithm=$QUANT_ALGORITHM_C"
log "C artifact_ref: $QUANT_ARTIFACT_REF_C"
log "C truth_boundary: $QUANT_TRUTH_BOUNDARY_C"
log ""

CAN_RUN=0
if [[ "$VLLM_AVAILABLE" == "1" ]]; then
    CAN_RUN=1
fi

if [[ "$CAN_RUN" == "0" && "$DRY_RUN" != "1" ]]; then
    log "vllm unavailable -- stopping after command materialization."
    log "Materialized commands are in $OUTPUT_DIR/*_command.txt."
    echo "materialized_only: vllm not importable in $PYTHON" > "$OUTPUT_DIR/status.txt"
    echo "materialized_only: vllm not importable in $PYTHON" > "$OUTPUT_DIR/compiler_awq_status.txt"
fi

if [[ "$CAN_RUN" == "1" || "$DRY_RUN" == "1" ]]; then

# ---------------------------------------------------------------------------
# Step 2: generate workload traces (shared across A/B/C)
# ---------------------------------------------------------------------------

SHARED_PREFIX="You are a helpful AI assistant. Please answer the following question carefully:\n\n"
TRACE_SHORT="$TRACE_DIR/short.jsonl"
TRACE_SHARED_PREFIX="$TRACE_DIR/shared_prefix.jsonl"
TRACE_NO_SHARED_PREFIX="$TRACE_DIR/no_shared_prefix.jsonl"

log "generating traces..."

$PYTHON scripts/generate_llm_request_trace.py \
    --num-requests 32 --prompt-set mixed --max-tokens 64 \
    --arrival-pattern uniform --seed 0 --output "$TRACE_SHORT"

$PYTHON scripts/generate_llm_request_trace.py \
    --num-requests 32 --prompt-set mixed --max-tokens 64 \
    --arrival-pattern uniform --seed 0 \
    --common-prefix "$SHARED_PREFIX" --unique-prompts 4 --output "$TRACE_SHARED_PREFIX"

$PYTHON scripts/generate_llm_request_trace.py \
    --num-requests 32 --prompt-set mixed --max-tokens 64 \
    --arrival-pattern uniform --seed 0 --output "$TRACE_NO_SHARED_PREFIX"

log "traces written to $TRACE_DIR"

# ---------------------------------------------------------------------------
# Step 3: run helpers
# ---------------------------------------------------------------------------

_vllm_pid=""

start_vllm() {
    local label="$1" server_cmd="$2"
    log "starting vLLM ($label): $server_cmd"
    if [[ "$DRY_RUN" == "1" ]]; then
        log "[DRY_RUN] would start: $server_cmd"
        _vllm_pid=""
        return
    fi
    eval "$server_cmd" &
    _vllm_pid=$!
    log "waiting ${VLLM_STARTUP_S}s for server startup (pid $_vllm_pid)..."
    local waited=0
    while (( waited < VLLM_STARTUP_S )); do
        if curl -s --max-time 2 "$BASE_URL/health" >/dev/null 2>&1; then
            log "server ready"
            return
        fi
        sleep 5
        (( waited += 5 ))
    done
    log "warning: server did not respond to /health after ${VLLM_STARTUP_S}s; proceeding anyway"
}

stop_vllm() {
    if [[ -n "$_vllm_pid" ]]; then
        log "stopping vLLM (pid $_vllm_pid)"
        kill "$_vllm_pid" 2>/dev/null || true
        wait "$_vllm_pid" 2>/dev/null || true
        _vllm_pid=""
    fi
}

run_benchmark_workload() {
    local label="$1" workload="$2" trace="$3" model_name="$4"
    local output="$OUTPUT_DIR/${label}_${workload}.json"
    log "running benchmark: $label / $workload -> $output"
    run_or_print "$PYTHON" scripts/benchmark_openai_compatible_server.py \
        --base-url "$BASE_URL" --model "$model_name" --trace "$trace" \
        --concurrency "$CONCURRENCY" --warmup "$WARMUP" \
        --claimed-server "vllm_quant_${label}" --output "$output"
}

run_path() {
    local label="$1" server_cmd="$2" model_name="$3"
    log ""
    log "=== $label vLLM path ==="
    start_vllm "$label" "$server_cmd"
    for workload in short shared_prefix no_shared_prefix; do
        case "$workload" in
            short)            trace="$TRACE_SHORT" ;;
            shared_prefix)    trace="$TRACE_SHARED_PREFIX" ;;
            no_shared_prefix) trace="$TRACE_NO_SHARED_PREFIX" ;;
        esac
        run_benchmark_workload "$label" "$workload" "$trace" "$model_name"
    done
    stop_vllm
}

BASELINE_MODEL_NAME="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
print(d['baseline_config'].get('served_model_name') or d['baseline_config']['model'])
")"
COMPILER_NOQUANT_MODEL_NAME="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
print(d['compiler_noquant_config'].get('served_model_name') or d['compiler_noquant_config']['model'])
")"
COMPILER_AWQ_MODEL_NAME="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
print(d['compiler_awq_config'].get('served_model_name') or d['compiler_awq_config']['model'])
")"

# ---------------------------------------------------------------------------
# Step 4: run A and B (both use original Qwen weights; always runnable if
# vllm is available, regardless of AWQ artifact state).
# ---------------------------------------------------------------------------

run_path "baseline" "$BASELINE_CMD" "$BASELINE_MODEL_NAME"
run_path "compiler_noquant" "$COMPILER_NOQUANT_CMD" "$COMPILER_NOQUANT_MODEL_NAME"

# ---------------------------------------------------------------------------
# Step 5: run C only if the quantized artifact is actually present locally.
# ---------------------------------------------------------------------------

if [[ "$AWQ_ARTIFACT_PRESENT" == "1" ]]; then
    run_path "compiler_awq" "$COMPILER_AWQ_CMD" "$COMPILER_AWQ_MODEL_NAME"
    echo "ran: AWQ artifact present at $AWQ_MODEL_DIR" > "$OUTPUT_DIR/compiler_awq_status.txt"
else
    log ""
    log "=== compiler awq vLLM path === SKIPPED (materialized only)"
    log "quantized_model_artifact_ref points at $AWQ_MODEL_DIR, which does not exist locally."
    log "Produce it with: cd ../ml-graph-compiler-runtime && .venv/bin/python tools/export_qwen_awq.py"
    echo "materialized_only: AWQ artifact not present at $AWQ_MODEL_DIR" > "$OUTPUT_DIR/compiler_awq_status.txt"
fi

fi  # end CAN_RUN || DRY_RUN block (Steps 2-5)

# ---------------------------------------------------------------------------
# Step 6: write comparison markdown (always written, even materialized-only)
# ---------------------------------------------------------------------------

COMPARISON_MD="$OUTPUT_DIR/quant_comparison.md"

cat > "$COMPARISON_MD" <<MDEOF
# Qwen A/B/C Quantization Benchmark Comparison

**Plan (B, no-quant):** \`$(extract plan_id_b)\`
**Plan (C, awq):**      \`$(extract plan_id_c)\`
**Hardware profile (C):** \`$(extract hardware_profile_ref_c)\`
**Date:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")

## Paths

| Path | Weights | Execution plan | Quantization |
|---|---|---|---|
| A: baseline | original HF Qwen | none (manual vLLM config) | none |
| B: compiler no-quant | original HF Qwen | compiler ExecutionPlan | none |
| C: compiler quant | AWQ Qwen checkpoint (\`$QUANT_ARTIFACT_REF_C\`) | compiler ExecutionPlan | awq (strategy=$QUANT_STRATEGY_C) |

## Truth Boundaries

- **C's quantization decision truth_boundary:** \`$QUANT_TRUTH_BOUNDARY_C\`
- **C vs B** isolates quantized weights: both use the compiler ExecutionPlan;
  only the weights and \`--quantization\` flag differ.
- **C vs A** combines quantized weights AND compiler execution plan policy
  (KV layout, memory budget, serving topology) -- a C-vs-A delta cannot be
  attributed to quantization alone.
- Do not claim a speedup unless a measured result below shows one beyond
  repeatability noise (see \`results/qwen_no_quant/repeatability_summary.md\`
  for what "noise" looked like for B vs A: ~0.5-1.1% across 3 trials).
- Do not claim accuracy parity -- no accuracy evaluation (perplexity, task
  benchmarks) was run for C.
- Do not claim GTX 1650 has native INT4 Tensor Core support. C's compiler
  plan is an explicit experimental forced-quant override
  (\`nvidia_gtx1650_maxq_awq_forced.json\`); per-op kernel planning for this
  target is unchanged and still reports no native int4 kernel path.

## Server Commands

**A (baseline):**
\`\`\`
$BASELINE_CMD
\`\`\`

**B (compiler no-quant):**
\`\`\`
$COMPILER_NOQUANT_CMD
\`\`\`

**C (compiler awq):**
\`\`\`
$COMPILER_AWQ_CMD
\`\`\`

## Results

| Workload | A (baseline) | B (compiler no-quant) | C (compiler awq) |
|---|---|---|---|
| short | \`baseline_short.json\` | \`compiler_noquant_short.json\` | \`compiler_awq_short.json\` |
| shared_prefix | \`baseline_shared_prefix.json\` | \`compiler_noquant_shared_prefix.json\` | \`compiler_awq_shared_prefix.json\` |
| no_shared_prefix | \`baseline_no_shared_prefix.json\` | \`compiler_noquant_no_shared_prefix.json\` | \`compiler_awq_no_shared_prefix.json\` |

C's result files are populated only when \`$AWQ_MODEL_DIR\` exists locally
(see \`compiler_awq_status.txt\`) -- otherwise C is materialized-only, and
these paths are pending measurement.

## How to Run

\`\`\`bash
# Dry-run (prints commands, no server started):
DRY_RUN=1 bash scripts/run_qwen_quant_benchmark.sh

# Produce the AWQ artifact first (CUDA-capable Linux host):
(cd ../ml-graph-compiler-runtime && .venv/bin/python tools/export_qwen_awq.py)

# Full benchmark on Linux GPU host:
bash scripts/run_qwen_quant_benchmark.sh
\`\`\`
MDEOF

log ""
log "comparison markdown: $COMPARISON_MD"
log ""
log "done. results in: $OUTPUT_DIR"
