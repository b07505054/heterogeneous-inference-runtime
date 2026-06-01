import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = {
        "Runtime": ["PyTorch", "TorchScript", "ONNX Runtime"],
        "Latency": [26.1, 20.1, 4.5]
    }

    df = pd.DataFrame(data)

    plt.figure()
    plt.bar(df["Runtime"], df["Latency"])
    plt.ylabel("Avg Latency (ms)")
    plt.title("Runtime Comparison")

    plt.savefig("results/runtime_comparison.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()