import argparse
from collections import Counter
from pathlib import Path

import onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model = onnx.load(model_path)

    ops = Counter(node.op_type for node in model.graph.node)

    print(f"\nModel: {model_path}")
    print(f"IR version: {model.ir_version}")
    print(f"Opset imports: {[opset.version for opset in model.opset_import]}")
    print(f"Number of nodes: {len(model.graph.node)}")

    print("\nTop operators:")
    for op, count in ops.most_common():
        print(f"{op}: {count}")


if __name__ == "__main__":
    main()