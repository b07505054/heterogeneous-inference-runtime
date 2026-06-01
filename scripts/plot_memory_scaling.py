import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = {
        "Batch": [1, 4, 8],
        "Memory_MB": [5.85, 4.10, 4.91]
    }

    df = pd.DataFrame(data)

    plt.figure()
    plt.plot(df["Batch"], df["Memory_MB"], marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory vs Batch Size")

    plt.savefig("results/memory_scaling.png", dpi=200, bbox_inches="tight")

if __name__ == "__main__":
    main()
