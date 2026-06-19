#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/generate_llm_runtime_artifacts.py"
DEFAULT_OUTPUT = ROOT / "results/llm_runtime_artifacts/mode_comparison"
MODES = ("scheduler_focused", "paged_attention")
KEY_ARTIFACTS = (
    "runtime_mode_comparison.json",
    "scheduler_decision_report.json",
    "runtime_profile.json",
    "prefill_decode_benchmark.json",
    "page_prefetch_report.json",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_generator(mode, output_dir, passthrough_args):
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--runtime-mode",
        mode,
        "--output-dir",
        str(output_dir),
        *passthrough_args,
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def copy_key_artifacts(source_dir, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in KEY_ARTIFACTS:
        shutil.copy2(source_dir / name, target_dir / name)


def build_summary(output_dir):
    modes = {}
    for mode in MODES:
        comparison = load_json(output_dir / mode / "runtime_mode_comparison.json")
        modes[mode] = comparison

    scheduler = modes["scheduler_focused"]
    paged = modes["paged_attention"]
    return {
        "artifact_type": "llm_runtime_two_mode_summary",
        "source": "scripts/generate_llm_runtime_mode_comparison.py",
        "modes": modes,
        "interpretation": {
            "scheduler_focused": (
                "Performance-oriented scheduler artifact. Paged-attention read-cost "
                "accounting is disabled to isolate continuous batching-style scheduling, "
                "memory-pressure admission, and KV page prefetch impact."
            ),
            "paged_attention": (
                "More conservative artifact. It keeps the same scheduler policies but "
                "adds paged-attention read-cost modeling and paged-KV lifecycle evidence."
            ),
        },
        "headline_comparison": {
            "scheduler_focused_tokens_per_second": scheduler["selected_policy"]["tokens_per_second"],
            "scheduler_focused_p95_latency_ms": scheduler["selected_policy"]["p95_latency_ms"],
            "scheduler_focused_tpot_p95_ms": scheduler["selected_policy"]["tpot_p95_ms"],
            "paged_attention_tokens_per_second": paged["selected_policy"]["tokens_per_second"],
            "paged_attention_p95_latency_ms": paged["selected_policy"]["p95_latency_ms"],
            "paged_attention_tpot_p95_ms": paged["selected_policy"]["tpot_p95_ms"],
            "paged_attention_cost_p95_ms": paged["selected_policy"]["paged_attention_latency_p95_ms"],
        },
        "claim_boundary": (
            "Both modes are artifact-backed local runtime simulator results, not "
            "production vLLM, SGLang, or TensorRT-LLM forks."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "generator_args",
        nargs=argparse.REMAINDER,
        help="Optional arguments forwarded to generate_llm_runtime_artifacts.py after --.",
    )
    args = parser.parse_args()

    passthrough = args.generator_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hir_llm_modes_") as tmp:
        tmp_root = Path(tmp)
        for mode in MODES:
            generated = tmp_root / mode
            run_generator(mode, generated, passthrough)
            copy_key_artifacts(generated, args.output_dir / mode)

    summary = build_summary(args.output_dir)
    write_json(args.output_dir / "summary.json", summary)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
