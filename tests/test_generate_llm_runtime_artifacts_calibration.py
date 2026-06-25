import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_llm_runtime_artifacts.py"
ARTIFACT_PATH = ROOT / "results" / "llm_runtime_artifacts" / "gpu_decode_batch_scaling_gtx1650maxq.json"

CALIBRATED_ARTIFACT_NAMES = (
    "runtime_profile.json",
    "scheduler_decision_report.json",
    "prefill_decode_benchmark.json",
    "serving_framework_report.json",
    "runtime_mode_comparison.json",
)

EXPECTED_TRUTH_BOUNDARY = (
    "Decode/prefill estimates are calibrated from a synthetic transformer "
    "workload measured on GTX 1650 Max-Q; this is not production serving."
)


def run_generator(output_dir, *, gpu_batch_scaling_artifact=None, seed=42, requests=8):
    args = [
        sys.executable,
        str(SCRIPT),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--requests",
        str(requests),
    ]
    if gpu_batch_scaling_artifact is not None:
        args += ["--gpu-batch-scaling-artifact", str(gpu_batch_scaling_artifact)]
    result = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    return result


def load_artifacts(output_dir):
    return {
        name: json.loads((output_dir / name).read_text())
        for name in CALIBRATED_ARTIFACT_NAMES
    }


def test_no_flag_uses_formula_metadata_and_no_gpu_key(tmp_path):
    output_dir = tmp_path / "formula"
    run_generator(output_dir)

    for name, payload in load_artifacts(output_dir).items():
        assert payload["cost_model_source"] == "formula", name
        assert "gpu_batch_scaling_artifact" not in payload, name
        assert "calibration_truth_boundary" not in payload, name


def test_with_real_artifact_uses_calibrated_metadata(tmp_path):
    output_dir = tmp_path / "calibrated"
    run_generator(output_dir, gpu_batch_scaling_artifact=ARTIFACT_PATH)

    for name, payload in load_artifacts(output_dir).items():
        assert payload["cost_model_source"] == "gpu_batch_scaling_artifact", name
        assert payload["gpu_batch_scaling_artifact"] == str(ARTIFACT_PATH), name
        assert payload["calibration_truth_boundary"] == EXPECTED_TRUTH_BOUNDARY, name


def test_calibrated_metrics_differ_from_formula_baseline(tmp_path):
    formula_dir = tmp_path / "formula"
    calibrated_dir = tmp_path / "calibrated"
    run_generator(formula_dir)
    run_generator(calibrated_dir, gpu_batch_scaling_artifact=ARTIFACT_PATH)

    formula_benchmark = json.loads((formula_dir / "prefill_decode_benchmark.json").read_text())
    calibrated_benchmark = json.loads((calibrated_dir / "prefill_decode_benchmark.json").read_text())

    assert formula_benchmark["avg_decode_latency_ms"] != calibrated_benchmark["avg_decode_latency_ms"]
    assert formula_benchmark["tokens_per_second"] != calibrated_benchmark["tokens_per_second"]


def test_missing_artifact_path_falls_back_to_formula_without_crash(tmp_path):
    output_dir = tmp_path / "missing"
    missing_path = tmp_path / "does" / "not" / "exist.json"

    run_generator(output_dir, gpu_batch_scaling_artifact=missing_path)

    for name, payload in load_artifacts(output_dir).items():
        assert payload["cost_model_source"] == "formula", name
        assert "gpu_batch_scaling_artifact" not in payload, name
