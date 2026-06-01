import argparse
import subprocess
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="mv2")
    parser.add_argument("--output-dir", default="models/executorch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Exporting {args.model_name} to ExecuTorch .pte with XNNPACK...")

    cmd = [
        sys.executable,
        "-m",
        "executorch.examples.xnnpack.aot_compiler",
        "--model_name",
        args.model_name,
        "--delegate",
    ]

    subprocess.run(cmd, check=True)

    generated = Path(f"{args.model_name}_xnnpack_fp32.pte")
    if not generated.exists():
        raise FileNotFoundError(f"Expected {generated} was not created.")

    target = output_dir / generated.name
    generated.replace(target)

    print(f"[OK] Saved ExecuTorch model to: {target}")


if __name__ == "__main__":
    main()