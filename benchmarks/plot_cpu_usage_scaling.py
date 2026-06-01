import json
import matplotlib.pyplot as plt

with open("results/executorch_cpu_usage.json") as f:
    data = json.load(f)

threads = [d["threads"] for d in data]
cpu = [d["cpu_usage"] for d in data]

plt.figure()
plt.plot(threads, cpu, marker="o")
plt.xlabel("Threads")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Utilization vs Threads")
plt.grid(True)

plt.savefig("results/cpu_usage_scaling.png")
plt.show()