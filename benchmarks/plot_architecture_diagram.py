from pathlib import Path

import matplotlib.pyplot as plt


def draw_box(ax, x, y, text, width=2.8, height=0.7):
    rect = plt.Rectangle((x, y), width, height, fill=False, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.5),
    )


def main():
    output_path = Path("results/edge_inference_architecture.png")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    draw_box(ax, 4.6, 6.0, "PyTorch Model")

    draw_box(ax, 0.8, 4.5, "ONNX Export")
    draw_box(ax, 4.6, 4.5, "ExecuTorch Export\n(.pte)")
    draw_box(ax, 8.4, 4.5, "TFLite Export")

    draw_box(ax, 0.8, 3.0, "ONNX Runtime\nCPU EP")
    draw_box(ax, 4.6, 3.0, "ExecuTorch C++\nXNNPACK")
    draw_box(ax, 8.4, 3.0, "TFLite\nXNNPACK CPU")

    draw_box(ax, 2.7, 1.5, "Benchmarking\nLatency / Throughput")
    draw_box(ax, 6.5, 1.5, "Profiling\nCPU / Memory / Operators")

    draw_box(ax, 4.6, 0.2, "Systems Analysis\nScaling / Quantization / Roofline-style Interpretation")

    draw_arrow(ax, 6.0, 6.0, 2.2, 5.2)
    draw_arrow(ax, 6.0, 6.0, 6.0, 5.2)
    draw_arrow(ax, 6.0, 6.0, 9.8, 5.2)

    draw_arrow(ax, 2.2, 4.5, 2.2, 3.7)
    draw_arrow(ax, 6.0, 4.5, 6.0, 3.7)
    draw_arrow(ax, 9.8, 4.5, 9.8, 3.7)

    draw_arrow(ax, 2.2, 3.0, 4.1, 2.2)
    draw_arrow(ax, 6.0, 3.0, 4.1, 2.2)
    draw_arrow(ax, 9.8, 3.0, 7.9, 2.2)

    draw_arrow(ax, 4.1, 1.5, 5.4, 0.9)
    draw_arrow(ax, 7.9, 1.5, 6.7, 0.9)

    ax.set_title("Edge AI Inference Evaluation Pipeline", fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()