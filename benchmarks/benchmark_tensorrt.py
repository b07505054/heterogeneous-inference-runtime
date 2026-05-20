import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pandas as pd

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ENGINE_PATH = "models/mobilenet_v2_fp16.engine"

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

input_shape = (1, 3, 224, 224)

input_data = np.random.rand(*input_shape).astype(np.float32)

input_bytes = input_data.nbytes

d_input = cuda.mem_alloc(input_bytes)

output_shape = (1, 1000)
output_data = np.empty(output_shape, dtype=np.float32)

d_output = cuda.mem_alloc(output_data.nbytes)

stream = cuda.Stream()

tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

input_name = tensor_names[0]
output_name = tensor_names[1]

context.set_input_shape(input_name, input_shape)

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

warmup = 10
runs = 100

for _ in range(warmup):
    cuda.memcpy_htod_async(d_input, input_data, stream)

    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(output_data, d_output, stream)

    stream.synchronize()

latencies = []

for _ in range(runs):
    start = time.perf_counter()

    cuda.memcpy_htod_async(d_input, input_data, stream)

    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(output_data, d_output, stream)

    stream.synchronize()

    end = time.perf_counter()

    latencies.append((end - start) * 1000)

avg_latency = np.mean(latencies)
p95_latency = np.percentile(latencies, 95)
p99_latency = np.percentile(latencies, 99)

throughput = 1000.0 / avg_latency

result = {
    "backend": "TensorRT",
    "precision": "FP16",
    "device": "CUDA",
    "avg_latency_ms": round(avg_latency, 4),
    "p95_latency_ms": round(p95_latency, 4),
    "p99_latency_ms": round(p99_latency, 4),
    "throughput_qps": round(throughput, 4),
}

print(result)

df = pd.DataFrame([result])

output_path = "results/tensorrt_benchmark.csv"

df.to_csv(output_path, index=False)

print(f"Saved to {output_path}")