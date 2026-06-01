import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = {
        "Batch": [1, 4, 8],
        "Latency": [7.7, 27.9, 50.7],
        "QPS": [129, 143, 158]
    }

    df = pd.DataFrame(data)

    plt.figure()
    plt.plot(df["Batch"], df["Latency"], marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("Batch vs Latency")

    plt.savefig("results/batch_latency.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()