#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys


DEFAULT_OUTPUTS = {
    ("fp16", "none"): "models/coreml/mobilenet_v2_fp16.mlpackage",
    ("fp16", "palettize"): "models/coreml/mobilenet_v2_fp16_palettized.mlpackage",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export torchvision MobileNetV2 to native CoreML .mlpackage.")
    parser.add_argument("--precision", choices=["fp16"], default="fp16")
    parser.add_argument("--compression", choices=["none", "palettize"], default="none")
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output or DEFAULT_OUTPUTS[(args.precision, args.compression)]

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
    compute_precision = _coreml_precision(ct, args.precision)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=sample.shape)],
        convert_to="mlprogram",
        compute_precision=compute_precision,
    )
    if args.compression == "palettize":
        mlmodel = palettize_model(ct, mlmodel)
    mlmodel.save(output)
    print(output)


def _coreml_precision(ct, precision: str):
    if precision == "fp16":
        value = getattr(ct.precision, "FLOAT16", None)
        if value is None:
            print(
                "CoreML FP16 export unavailable: coremltools.precision.FLOAT16 is not available.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return value
    raise ValueError(f"unsupported CoreML precision: {precision}")


def palettize_model(ct, mlmodel):
    optimize_coreml = getattr(getattr(ct, "optimize", None), "coreml", None)
    if optimize_coreml is None:
        print(
            "CoreML palettization unavailable: coremltools.optimize.coreml is not available.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    op_palettizer_config = getattr(optimize_coreml, "OpPalettizerConfig", None)
    optimization_config = getattr(optimize_coreml, "OptimizationConfig", None)
    palettize_weights = getattr(optimize_coreml, "palettize_weights", None)
    if op_palettizer_config is None or optimization_config is None or palettize_weights is None:
        print(
            "CoreML palettization unavailable: required coremltools palettization APIs are missing.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    config = optimization_config(global_config=op_palettizer_config(mode="kmeans"))
    return palettize_weights(mlmodel, config)


if __name__ == "__main__":
    main()
