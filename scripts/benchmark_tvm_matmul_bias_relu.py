import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tvm_experiments.matmul_bias_relu import (
    MatmulBiasReluShape,
    ScheduleConfig,
    benchmark_module,
    compile_module,
    create_scheduled_module,
    create_unscheduled_module,
    import_tvm,
    make_inputs,
    numpy_reference,
    run_module,
)


DEFAULT_OUTPUT = ROOT / "results" / "tvm_tensorir" / "matmul_bias_relu_benchmark.json"
DEFAULT_REPORT = ROOT / "results" / "tvm_tensorir" / "matmul_bias_relu_report.md"
DEFAULT_TIR_DIR = ROOT / "results" / "tvm_tensorir" / "tir"


def run_command(args):
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def git_commit_hash():
    return run_command(["git", "rev-parse", "--short", "HEAD"])


def git_dirty():
    output = run_command(["git", "status", "--short"])
    return bool(output)


def percentile(values, p):
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p / 100))
    return ordered[idx]


def summarize(values):
    return {
        "mean_ms": round(statistics.mean(values), 6),
        "p50_ms": round(percentile(values, 50), 6),
        "p95_ms": round(percentile(values, 95), 6),
        "min_ms": round(min(values), 6),
        "max_ms": round(max(values), 6),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_shape(value):
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be MxNxK, for example 128x128x128")
    return MatmulBiasReluShape(*(int(part) for part in parts))


def default_shapes():
    return [
        MatmulBiasReluShape(64, 64, 64),
        MatmulBiasReluShape(128, 128, 128),
        MatmulBiasReluShape(256, 256, 256),
    ]


def environment_metadata(tvm):
    return {
        "tvm_version": getattr(tvm, "__version__", None),
        "llvm_enabled": bool(tvm.runtime.enabled("llvm")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "commit_hash": git_commit_hash(),
        "git_dirty": git_dirty(),
    }


def write_tensorir_artifacts(tir_dir, shape, unscheduled_mod, scheduled_mod):
    tir_dir.mkdir(parents=True, exist_ok=True)
    base = shape.bucket.replace(":", "_").replace("x", "_")
    unscheduled_path = tir_dir / f"{base}_unscheduled.py"
    scheduled_path = tir_dir / f"{base}_scheduled.py"
    write_text(unscheduled_path, unscheduled_mod.script() + "\n")
    write_text(scheduled_path, scheduled_mod.script() + "\n")
    try:
        unscheduled_ref = str(unscheduled_path.relative_to(ROOT))
        scheduled_ref = str(scheduled_path.relative_to(ROOT))
    except ValueError:
        unscheduled_ref = str(unscheduled_path)
        scheduled_ref = str(scheduled_path)
    return {
        "unscheduled_tensorir": unscheduled_ref,
        "scheduled_tensorir": scheduled_ref,
    }


def write_markdown_report(path, payload):
    env = payload["environment"]
    config = payload["benchmark_config"]
    lines = [
        "# TVM TensorIR MatMul-Bias-ReLU Benchmark",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Environment",
        "",
        f"- TVM version: `{env.get('tvm_version')}`",
        f"- LLVM enabled: `{env.get('llvm_enabled')}`",
        f"- Target: `{payload.get('target')}`",
        f"- Platform: `{env.get('platform')}`",
        f"- Machine: `{env.get('machine')}`",
        f"- Commit: `{env.get('commit_hash')}`",
        f"- Git dirty: `{env.get('git_dirty')}`",
        f"- Warmup runs: `{config.get('warmup')}`",
        f"- Timed repeats: `{config.get('repeat')}`",
        f"- Number per repeat: `{config.get('number')}`",
        "",
        "## Schedule",
        "",
        f"- Tile M/N/K: `{config['schedule']['tile_m']} / {config['schedule']['tile_n']} / {config['schedule']['tile_k']}`",
        f"- Vectorized N lanes: `{config['schedule']['vectorize_n']}`",
        "- Transformations: `split`, `reorder`, `parallel`, `vectorize`, `reverse_compute_at`",
        "",
        "## Shape Sweep",
        "",
        "| Shape | Correct | Max abs diff | Unscheduled p50 ms | Scheduled p50 ms | Speedup | Scheduled TensorIR |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["shape_sweep"]:
        lines.append(
            "| {shape} | {correct} | {diff} | {base} | {sched} | {speedup} | `{tir}` |".format(
                shape=row["shape_bucket"],
                correct=row["correct"],
                diff=row["max_abs_diff"],
                base=row["unscheduled_latency"]["p50_ms"],
                sched=row["scheduled_latency"]["p50_ms"],
                speedup=row["speedup_p50"],
                tir=row["artifacts"]["scheduled_tensorir"],
            )
        )
    lines.extend(
        [
            "",
            "## Interview Notes",
            "",
            "- This is a real executable TensorIR path, not a static report.",
            "- The unscheduled and scheduled versions share the same fused MatMul-Bias-ReLU semantics.",
            "- The scheduled path exposes hardware-aware loop decisions that can be compared against MLIR/HIR lowering.",
            "- The result is useful as a TVM reference point, while the main compiler story remains MLIR/HIR.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def benchmark_shape(shape, args, schedule_config):
    unscheduled_mod = create_unscheduled_module(shape)
    scheduled_mod = create_scheduled_module(shape, schedule_config)
    artifacts = write_tensorir_artifacts(args.tir_dir, shape, unscheduled_mod, scheduled_mod)

    unscheduled_rt = compile_module(unscheduled_mod, args.target)
    scheduled_rt = compile_module(scheduled_mod, args.target)

    inputs = make_inputs(shape, seed=shape.m * 1_000_000 + shape.n * 1_000 + shape.k)
    expected = numpy_reference(*inputs)
    unscheduled_out = run_module(unscheduled_rt, shape, inputs)
    scheduled_out = run_module(scheduled_rt, shape, inputs)

    max_abs_diff = float(abs(scheduled_out - expected).max())
    correct = bool(
        abs(unscheduled_out - expected).max() <= args.atol
        and abs(scheduled_out - expected).max() <= args.atol
    )

    # Warmup before timing so LLVM codegen/runtime setup is not counted.
    for _ in range(args.warmup):
        run_module(unscheduled_rt, shape, inputs)
        run_module(scheduled_rt, shape, inputs)

    unscheduled_times = benchmark_module(unscheduled_rt, shape, inputs, args.number, args.repeat)
    scheduled_times = benchmark_module(scheduled_rt, shape, inputs, args.number, args.repeat)
    unscheduled_latency = summarize(unscheduled_times)
    scheduled_latency = summarize(scheduled_times)
    speedup = unscheduled_latency["p50_ms"] / max(scheduled_latency["p50_ms"], 1e-9)

    return {
        "shape": {"m": shape.m, "n": shape.n, "k": shape.k, "dtype": shape.dtype},
        "shape_bucket": shape.bucket,
        "correct": correct,
        "max_abs_diff": round(max_abs_diff, 8),
        "unscheduled_latency": unscheduled_latency,
        "scheduled_latency": scheduled_latency,
        "speedup_p50": round(speedup, 4),
        "artifacts": artifacts,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TVM TensorIR MatMul-Bias-ReLU scheduling.")
    parser.add_argument("--shape", action="append", type=parse_shape, help="Shape MxNxK. Repeat for a sweep.")
    parser.add_argument("--target", default="llvm", help="TVM compile target.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tir-dir", type=Path, default=DEFAULT_TIR_DIR)
    args = parser.parse_args()

    tvm = import_tvm()
    if not tvm.runtime.enabled("llvm"):
        raise RuntimeError("TVM was built without LLVM support; cannot run the TensorIR CPU benchmark.")

    schedule_config = ScheduleConfig()
    shapes = args.shape or default_shapes()
    rows = [benchmark_shape(shape, args, schedule_config) for shape in shapes]
    payload = {
        "artifact_type": "tvm_tensorir_schedule_benchmark",
        "source": "scripts/benchmark_tvm_matmul_bias_relu.py",
        "status": "ok" if all(row["correct"] for row in rows) else "failed_correctness",
        "target": args.target,
        "environment": environment_metadata(tvm),
        "benchmark_config": {
            "warmup": args.warmup,
            "number": args.number,
            "repeat": args.repeat,
            "atol": args.atol,
            "schedule": schedule_config.as_dict(),
        },
        "shape_sweep": rows,
    }
    write_json(args.output, payload)
    write_markdown_report(args.report_output, payload)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report_output}")
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
