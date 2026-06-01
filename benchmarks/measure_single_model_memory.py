import argparse
import json
import os
import psutil
import time
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    process = psutil.Process(os.getpid())

    mem_before = process.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()

    session = ort.InferenceSession(
        args.model_path,
        providers=["CPUExecutionProvider"]
    )

    load_time_ms = (time.perf_counter() - start) * 1000
    mem_after = process.memory_info().rss / (1024 * 1024)

    result = {
        "model": args.model_name,
        "model_path": args.model_path,
        "file_size_mb": round(os.path.getsize(args.model_path) / (1024 * 1024), 2),
        "load_time_ms": round(load_time_ms, 2),
        "rss_before_mb": round(mem_before, 2),
        "rss_after_mb": round(mem_after, 2),
        "rss_delta_mb": round(mem_after - mem_before, 2),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()