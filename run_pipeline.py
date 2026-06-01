import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Edge AI Benchmark Tool")
    parser.add_argument("--model", default="mobilenet")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()

    print(f"Running benchmark: model={args.model}, threads={args.threads}, batch={args.batch}")

    subprocess.run([
        "python",
        "scripts/benchmark.py",
        "--backend", "onnx",
        "--model-path", "models/mobilenet_v2_optimized.onnx",
        "--threads", str(args.threads),
        "--batch-size", str(args.batch),
        "--iterations", "100"
    ])

if __name__ == "__main__":
    main()