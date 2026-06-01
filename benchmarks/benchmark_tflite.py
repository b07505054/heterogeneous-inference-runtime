import time
import numpy as np
import tensorflow as tf


MODEL_PATH = "models/mobilenet_v2.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]

dummy_input = np.random.rand(*input_shape).astype(np.float32)

RUNS = 100

latencies = []

for _ in range(10):
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

for _ in range(RUNS):
    start = time.perf_counter()

    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    latency_ms = (time.perf_counter() - start) * 1000
    latencies.append(latency_ms)

latencies = np.array(latencies)

print("=== TFLite Benchmark ===")
print(f"avg: {latencies.mean():.4f} ms")
print(f"p95: {np.percentile(latencies,95):.4f} ms")
print(f"p99: {np.percentile(latencies,99):.4f} ms")