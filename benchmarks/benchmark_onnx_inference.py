import argparse
import time
import numpy as np
import onnxruntime as ort


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    sess_options = ort.SessionOptions()
    if args.profile:
        sess_options.enable_profiling = True

    session = ort.InferenceSession(
        args.model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    print("Providers:", session.get_providers())

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    dummy_input = np.random.rand(
        *[d if isinstance(d, int) else 1 for d in input_shape]
    ).astype(np.float32)

    latencies = []

    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    print("=== ONNX Runtime Inference ===")
    print(f"avg: {np.mean(latencies):.4f} ms")
    print(f"p95: {percentile(latencies, 95):.4f} ms")
    print(f"p99: {percentile(latencies, 99):.4f} ms")

    if args.profile:
        profile_file = session.end_profiling()
        print(f"Profile saved to: {profile_file}")


if __name__ == "__main__":
    main()