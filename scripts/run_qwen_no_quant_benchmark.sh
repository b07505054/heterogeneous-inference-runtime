#!/usr/bin/env bash
# run_qwen_no_quant_benchmark.sh
#
# Benchmark: baseline vLLM vs compiler-guided no-quant vLLM for Qwen 2.5-0.5B.
#
# Truth:
#   Compiler-guided no-quant Qwen uses the original HuggingFace Qwen weights.
#   Differences between paths come from runtime policy decisions extracted from
#   the compiler execution plan, not from weight optimization or quantization.
#
# Compiler artifact:
#   execution_plan.json (schema 2.0.0, produced by compile-for-target)
#   Default: ../ml-graph-compiler-runtime/artifacts/qwen/execution_plan.json
#
# Workloads:
#   short          — 32 requests, mixed prompts, 64 tokens, uniform arrival
#   shared_prefix  — 32 requests, 4 unique prompts + shared prefix, exercises prefix cache
#   no_shared_prefix — 32 requests, all 6 mixed prompts, no shared prefix
#
# Outputs (written to results/qwen_no_quant/):
#   traces/            — generated trace files (one per workload)
#   baseline_*.json    — benchmark results for baseline vLLM
#   compiler_*.json    — benchmark results for compiler-guided vLLM
#   no_quant_comparison.md — markdown comparison report
#
# Env overrides:
#   COMPILER_PLAN      path to execution_plan.json
#   BASELINE_MODEL     HuggingFace model ID for baseline path
#   COMPILER_MODEL     HuggingFace model ID for compiler-guided path (default: same as BASELINE_MODEL)
#                      Both paths use the same Qwen weights; differences are runtime policy only.
#   OUTPUT_DIR         results directory (default: results/qwen_no_quant)
#   VLLM_HOST          vLLM server host (default: 127.0.0.1)
#   VLLM_PORT          vLLM server port (default: 8000)
#   CONCURRENCY        benchmark concurrency (default: 1)
#   WARMUP             warmup request count (default: 4)
#   VLLM_STARTUP_S     seconds to wait for vLLM startup (default: 60)
#   DRY_RUN            set to 1 to print commands without running (default: 0)
#
# Usage:
#   bash scripts/run_qwen_no_quant_benchmark.sh
#   DRY_RUN=1 bash scripts/run_qwen_no_quant_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPILER_PLAN="${COMPILER_PLAN:-$REPO_ROOT/../ml-graph-compiler-runtime/artifacts/qwen/execution_plan.json}"
BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
# Both paths use the same model weights. COMPILER_MODEL defaults to BASELINE_MODEL.
# The compiler plan stores a short model_id (e.g. "qwen2.5-0.5b"), not the HuggingFace path.
COMPILER_MODEL="${COMPILER_MODEL:-$BASELINE_MODEL}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/qwen_no_quant}"
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[benchmark] $*"; }

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

log "compiler plan: $COMPILER_PLAN"
if [[ ! -f "$COMPILER_PLAN" ]]; then
    echo "error: execution_plan.json not found at: $COMPILER_PLAN" >&2
    echo "  Generate it first:" >&2
    echo "    cd ../ml-graph-compiler-runtime && bash tools/run_qwen_compiler_pipeline.sh" >&2
    exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "error: .venv not found at $PYTHON. Create it and install dependencies first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1: materialize vLLM commands from compiler plan
# ---------------------------------------------------------------------------

log "materializing runtime config from compiler plan..."

MATERIALIZED="$(COMPILER_PLAN="$COMPILER_PLAN" BASELINE_MODEL="$BASELINE_MODEL" COMPILER_MODEL="$COMPILER_MODEL" BASE_URL="$BASE_URL" $PYTHON - <<'PYEOF'
import sys, json
sys.path.insert(0, ".")
import os
from dataclasses import replace

plan_path = os.environ["COMPILER_PLAN"]
baseline_model = os.environ.get("BASELINE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
compiler_model = os.environ.get("COMPILER_MODEL", baseline_model)

from deployment.execution_plan.loader import load_execution_plan
from deployment.execution_plan.stage_builder import build_execution_stages
from deployment.execution_plan.path_builder import build_execution_paths, build_baseline_vllm_path
from deployment.execution_plan.schema import ExecutionPath, ExecutionPathKind
from deployment.vllm_adapter.backend_adapter import VLLMBackendAdapter

plan = load_execution_plan(plan_path)
stages = build_execution_stages(plan)
paths = build_execution_paths(plan, stages)

vllm_paths = [p for p in paths if p.path_kind == ExecutionPathKind.COMPILER_GUIDED_VLLM]
if not vllm_paths:
    print(json.dumps({"error": "no COMPILER_GUIDED_VLLM path found in plan"}))
    sys.exit(1)

compiler_path = vllm_paths[0]

# Override model/tokenizer: compiler plan stores a short model_id ("qwen2.5-0.5b"),
# not a HuggingFace path. Both paths use the same weights; differences are runtime policy.
overridden_config = {**compiler_path.runtime_config, "model": compiler_model, "tokenizer": compiler_model}
compiler_path = replace(compiler_path, runtime_config=overridden_config)

adapter = VLLMBackendAdapter()
compiler_mat = adapter.materialize(compiler_path)

baseline_path = build_baseline_vllm_path(
    model_id=baseline_model,
    output_artifact="results/qwen_no_quant/baseline_{workload}.json",
)
baseline_mat = adapter.materialize(baseline_path)

print(json.dumps({
    "plan_id": plan.plan_id,
    "model_id": str(plan.model_identity.get("model_id", "")),
    "hardware_profile_ref": plan.provenance.capability_bundle.hardware_profile_ref,
    "compiler_server_command": list(compiler_mat.command),
    "baseline_server_command": list(baseline_mat.command),
    "compiler_config": compiler_mat.config,
    "baseline_config": baseline_mat.config,
    "truth_boundary": compiler_mat.truth_boundary,
}))
PYEOF
)"

# Check for materialization error
if echo "$MATERIALIZED" | $PYTHON -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if 'error' not in d else 1)" 2>/dev/null; then
    :
else
    echo "error: materialization failed: $MATERIALIZED" >&2
    exit 1
fi

PLAN_ID="$(echo "$MATERIALIZED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['plan_id'])")"
MODEL_ID="$(echo "$MATERIALIZED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['model_id'])")"
HW_REF="$(echo "$MATERIALIZED" | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['hardware_profile_ref'])")"

log "plan_id:  $PLAN_ID"
log "model_id: $MODEL_ID"
log "hardware: $HW_REF"

# Print compiler server command for dry-run visibility
COMPILER_SERVER_CMD="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
print(' '.join(str(x) for x in d['compiler_server_command']))
")"
BASELINE_SERVER_CMD="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
print(' '.join(str(x) for x in d['baseline_server_command']))
")"

log ""
log "baseline server command:"
log "  $BASELINE_SERVER_CMD"
log ""
log "compiler-guided server command:"
log "  $COMPILER_SERVER_CMD"
log ""

# ---------------------------------------------------------------------------
# Step 2: generate workload traces
# ---------------------------------------------------------------------------

SHARED_PREFIX="You are a helpful AI assistant. Please answer the following question carefully:\n\n"
TRACE_SHORT="$TRACE_DIR/short.jsonl"
TRACE_SHARED_PREFIX="$TRACE_DIR/shared_prefix.jsonl"
TRACE_NO_SHARED_PREFIX="$TRACE_DIR/no_shared_prefix.jsonl"

log "generating traces..."

$PYTHON scripts/generate_llm_request_trace.py \
    --num-requests 32 \
    --prompt-set mixed \
    --max-tokens 64 \
    --arrival-pattern uniform \
    --seed 0 \
    --output "$TRACE_SHORT"

$PYTHON scripts/generate_llm_request_trace.py \
    --num-requests 32 \
    --prompt-set mixed \
    --max-tokens 64 \
    --arrival-pattern uniform \
    --seed 0 \
    --common-prefix "$SHARED_PREFIX" \
    --unique-prompts 4 \
    --output "$TRACE_SHARED_PREFIX"

$PYTHON scripts/generate_llm_request_trace.py \
    --num-requests 32 \
    --prompt-set mixed \
    --max-tokens 64 \
    --arrival-pattern uniform \
    --seed 0 \
    --output "$TRACE_NO_SHARED_PREFIX"

log "traces written to $TRACE_DIR"

# ---------------------------------------------------------------------------
# Step 3: define benchmark runner function
# ---------------------------------------------------------------------------

_vllm_pid=""

start_vllm() {
    local label="$1"
    local server_cmd="$2"
    local model_name="$3"

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
    local label="$1"     # e.g. "baseline" or "compiler"
    local workload="$2"  # e.g. "short", "shared_prefix", "no_shared_prefix"
    local trace="$3"
    local model_name="$4"
    local output="$OUTPUT_DIR/${label}_${workload}.json"

    log "running benchmark: $label / $workload -> $output"
    run_or_print "$PYTHON" scripts/benchmark_openai_compatible_server.py \
        --base-url "$BASE_URL" \
        --model "$model_name" \
        --trace "$trace" \
        --concurrency "$CONCURRENCY" \
        --warmup "$WARMUP" \
        --claimed-server "vllm_no_quant_${label}" \
        --output "$output"
}

# ---------------------------------------------------------------------------
# Step 4: run baseline path for each workload
# ---------------------------------------------------------------------------

BASELINE_SERVER_CMD_EVAL="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys,shlex
d=json.load(sys.stdin)
print(shlex.join(str(x) for x in d['baseline_server_command']))
")"
BASELINE_MODEL_NAME="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
print(d['baseline_config'].get('served_model_name') or d['baseline_config']['model'])
")"

log ""
log "=== baseline vLLM path ==="
start_vllm "baseline" "$BASELINE_SERVER_CMD_EVAL" "$BASELINE_MODEL_NAME"

for workload in short shared_prefix no_shared_prefix; do
    case "$workload" in
        short)            trace="$TRACE_SHORT" ;;
        shared_prefix)    trace="$TRACE_SHARED_PREFIX" ;;
        no_shared_prefix) trace="$TRACE_NO_SHARED_PREFIX" ;;
    esac
    run_benchmark_workload "baseline" "$workload" "$trace" "$BASELINE_MODEL_NAME"
done

stop_vllm

# ---------------------------------------------------------------------------
# Step 5: run compiler-guided path for each workload
# ---------------------------------------------------------------------------

COMPILER_SERVER_CMD_EVAL="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys,shlex
d=json.load(sys.stdin)
print(shlex.join(str(x) for x in d['compiler_server_command']))
")"
COMPILER_MODEL_NAME="$(echo "$MATERIALIZED" | $PYTHON -c "
import json,sys
d=json.load(sys.stdin)
print(d['compiler_config'].get('served_model_name') or d['compiler_config']['model'])
")"

log ""
log "=== compiler-guided no-quant vLLM path ==="
start_vllm "compiler" "$COMPILER_SERVER_CMD_EVAL" "$COMPILER_MODEL_NAME"

for workload in short shared_prefix no_shared_prefix; do
    case "$workload" in
        short)            trace="$TRACE_SHORT" ;;
        shared_prefix)    trace="$TRACE_SHARED_PREFIX" ;;
        no_shared_prefix) trace="$TRACE_NO_SHARED_PREFIX" ;;
    esac
    run_benchmark_workload "compiler" "$workload" "$trace" "$COMPILER_MODEL_NAME"
done

stop_vllm

# ---------------------------------------------------------------------------
# Step 6: write comparison markdown
# ---------------------------------------------------------------------------

COMPARISON_MD="$OUTPUT_DIR/no_quant_comparison.md"

cat > "$COMPARISON_MD" <<MDEOF
# Qwen No-Quant Benchmark Comparison

**Plan:** \`$PLAN_ID\`
**Model:** \`$MODEL_ID\`
**Hardware profile:** \`$HW_REF\`
**Date:** $(date -u +"%Y-%m-%dT%H:%M:%SZ")

## Truth Boundary

Compiler-guided no-quant Qwen uses the **original HuggingFace Qwen weights**.
Differences between paths come from **runtime policy decisions** extracted from
the compiler execution plan (KV layout, memory budget, serving topology, prefix
reuse eligibility), not from weight optimization or quantization.

The compiler plan declares:
- Quantization: **none** (no \`global_decisions.quantization\` emitted for this profile)
- Per-op: \`fp16_fallback\` for accuracy-sensitive ops (stays in fp16)
- GPU memory utilization: **0.75** (from compiler \`memory_budget_fraction\`)
- KV layout: **paged**

Neither path modifies model weights. Measured differences are **declared profile evidence**,
not measured silicon performance.

## Workloads

| Workload | Description |
|---|---|
| \`short\` | 32 requests, mixed prompts, 64 max_tokens, uniform arrival, no shared prefix |
| \`shared_prefix\` | 32 requests, 4 unique prompts, shared system prefix, exercises prefix cache |
| \`no_shared_prefix\` | 32 requests, all 6 mixed prompts, no shared prefix, no cache benefit expected |

## Server Commands

**Baseline vLLM:**
\`\`\`
$BASELINE_SERVER_CMD
\`\`\`

**Compiler-guided vLLM:**
\`\`\`
$COMPILER_SERVER_CMD
\`\`\`

## Results

Results are written to \`results/qwen_no_quant/\` when the benchmark runs.
Each result file is a \`measured_envelope\` with \`evidence_type: "measured"\`.

| Workload | Baseline | Compiler-guided |
|---|---|---|
| short | \`baseline_short.json\` | \`compiler_short.json\` |
| shared_prefix | \`baseline_shared_prefix.json\` | \`compiler_shared_prefix.json\` |
| no_shared_prefix | \`baseline_no_shared_prefix.json\` | \`compiler_no_shared_prefix.json\` |

Results pending measurement. Files above are populated by running this script
on a Linux host with a GTX 1650 GPU and vLLM installed.

## How to Run

\`\`\`bash
# Dry-run (prints commands, no server started):
DRY_RUN=1 bash scripts/run_qwen_no_quant_benchmark.sh

# Full benchmark on Linux GPU host:
BASELINE_MODEL=Qwen/Qwen2.5-0.5B-Instruct \\
COMPILER_PLAN=../ml-graph-compiler-runtime/artifacts/qwen/execution_plan.json \\
bash scripts/run_qwen_no_quant_benchmark.sh
\`\`\`
MDEOF

log ""
log "comparison markdown: $COMPARISON_MD"
log ""
log "done. results in: $OUTPUT_DIR"
