#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Export torchvision MobileNetV2 to native CoreML .mlpackage.")
    parser.add_argument("--output", default="models/mobilenet_v2.mlpackage")
    args = parser.parse_args()

    try:
        import coremltools as ct
        import torch
        from torchvision import models
    except Exception as exc:
        print(f"CoreML export unavailable: {exc}", file=sys.stderr)
        raise SystemExit(2)

    torch.manual_seed(0)
    model = models.mobilenet_v2(weights=None).eval()
    sample = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, sample)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=sample.shape)],
        convert_to="mlprogram",
    )
    mlmodel.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
