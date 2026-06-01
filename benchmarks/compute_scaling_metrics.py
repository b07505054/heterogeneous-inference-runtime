import json

data = [
    {"threads": 1, "latency": 14.6},
    {"threads": 2, "latency": 8.6},
    {"threads": 4, "latency": 5.7},
    {"threads": 8, "latency": 5.8},
]

baseline = data[0]["latency"]

for d in data:
    d["throughput"] = 1000.0 / d["latency"]  # ops/sec
    d["speedup"] = baseline / d["latency"]

with open("results/scaling_metrics.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved results/scaling_metrics.json")