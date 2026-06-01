import json
import subprocess
import sys
from pathlib import Path


MODELS = [
    ("FP32", "models/mobilenet_v2_fp32.onnx"),
    ("Optimized", "models/mobilenet_v2_optimized.onnx"),
    ("INT8", "models/mobilenet_v2_int8.onnx"),
]


def main():
    results = []

    for name, path in MODELS:
        cmd = [
            sys.executable,
            "benchmarks/measure_single_model_memory.py",
            "--model-name", name,
            "--model-path", path,
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        result = json.loads(proc.stdout)
        results.append(result)
        print(result)

    output_path = Path("results/memory_usage.json")
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()