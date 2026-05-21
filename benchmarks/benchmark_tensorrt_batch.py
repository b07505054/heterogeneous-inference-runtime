import csv
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


ENGINE_PATH = Path("models/mobilenet_v2_fp16.engine")
OUTPUT_PATH = Path("results/tensorrt_batch_scaling.csv")

BATCH_SIZES = [1, 2, 4, 8, 16]
CHANNELS = 3
HEIGHT = 224
WIDTH = 224
NUM_CLASSES = 1000

WARMUP = 10
RUNS = 100

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def benchmark_batch(engine, batch_size):
    context = engine.create_execution_context()

    input_shape = (batch_size, CHANNELS, HEIGHT, WIDTH)
    output_shape = (batch_size, NUM_CLASSES)

    input_data = np.random.rand(*input_shape).astype(np.float32)
    output_data = np.empty(output_shape, dtype=np.float32)

    input_nbytes = input_data.nbytes
    output_nbytes = output_data.nbytes

    d_input = cuda.mem_alloc(input_nbytes)
    d_output = cuda.mem_alloc(output_nbytes)

    stream = cuda.Stream()

    tensor_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
    ]

    input_name = tensor_names[0]
    output_name = tensor_names[1]

    context.set_input_shape(input_name, input_shape)

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    for _ in range(WARMUP):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()

    latencies = []

    for _ in range(RUNS):
        start = time.perf_counter()

        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output_data, d_output, stream)
        stream.synchronize()

        end = time.perf_counter()

        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)

    avg = float(np.mean(latencies))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    throughput_qps = (batch_size * 1000.0) / avg

    return {
        "backend": "TensorRT",
        "precision": "FP16",
        "batch_size": batch_size,
        "avg_latency_ms": round(avg, 4),
        "p95_latency_ms": round(p95, 4),
        "p99_latency_ms": round(p99, 4),
        "throughput_qps": round(throughput_qps, 4),
    }


def main():
    with ENGINE_PATH.open("rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())

    results = []

    for batch_size in BATCH_SIZES:
        print(f"[INFO] Benchmarking batch={batch_size}")
        result = benchmark_batch(engine, batch_size)
        print(result)
        results.append(result)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "precision",
                "batch_size",
                "avg_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "throughput_qps",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()