import json
import matplotlib.pyplot as plt

with open("results/scaling_metrics.json") as f:
    data = json.load(f)

threads = [d["threads"] for d in data]
throughput = [d["throughput"] for d in data]
speedup = [d["speedup"] for d in data]

# Throughput
plt.figure()
plt.plot(threads, throughput, marker="o")
plt.xlabel("Threads")
plt.ylabel("Throughput (ops/sec)")
plt.title("Throughput Scaling (ExecuTorch C++)")
plt.grid(True)
plt.savefig("results/throughput_scaling.png")

# Speedup
plt.figure()
plt.plot(threads, speedup, marker="o")
plt.xlabel("Threads")
plt.ylabel("Speedup (x)")
plt.title("Speedup vs Threads")
plt.grid(True)
plt.savefig("results/speedup_scaling.png")

plt.show()