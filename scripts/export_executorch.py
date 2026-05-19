import torch
from torchvision.models import mobilenet_v2

from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

MODEL_PATH = "models/mobilenet_v2_xnnpack.pte"


def main():
    model = mobilenet_v2(weights=None).eval()
    example_inputs = (torch.randn(1, 3, 224, 224),)

    exported_program = torch.export.export(
        model,
        example_inputs,
    )
    edge_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    executorch_program = edge_program.to_executorch()

    with open(MODEL_PATH, "wb") as f:
        executorch_program.write_to_file(f)

    print(f"Saved ExecuTorch program to {MODEL_PATH}")


if __name__ == "__main__":
    main()