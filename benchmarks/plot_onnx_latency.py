import matplotlib.pyplot as plt
import numpy as np

models = ["FP32", "Optimized", "INT8"]

avg = [5.2902, 4.8443, 113.8294]
p95 = [7.3215, 6.6451, 130.7832]
p99 = [7.7902, 7.7841, 138.1241]

x = np.arange(len(models))
width = 0.25

plt.figure()

plt.bar(x - width, avg, width, label="avg")
plt.bar(x, p95, width, label="p95")
plt.bar(x + width, p99, width, label="p99")

plt.xticks(x, models)
plt.xlabel("Model Variant")
plt.ylabel("Latency (ms)")
plt.title("ONNX Runtime Latency Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("results/onnx_latency_distribution.png")
plt.show()