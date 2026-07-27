"""E2E-10: MatMulBiasReLUDescriptor, the third operation-family payload,
registered into perf_model.operation_descriptor's payload-type registry
(register_payload_type) rather than requiring any change to
OperationDescriptor itself.

Field semantics and constraints are SOURCE_PROVEN from the real, existing,
committed fused MatMul-Bias-ReLU pipeline discovered in Phase 0:
  - native/cpu_kernels/portable_fused_matmul_bias_relu.cpp (FP32 tiled kernel)
  - native/cpu_kernels/portable_fused_matmul_bias_relu_int8.cpp (INT8 row-major)
  - native/cpu_kernels/portable_fused_matmul_bias_relu_int8_packed_b.cpp (INT8 packed-B)
  - deployment/execution_plan/int8_quantization.py (per-tensor symmetric
    quantization, zero_point=0 only, scale = max_abs/127)
  - deployment/execution_plan/portable_cpu_kernel_adapter.py (the exact
    quantization contract this descriptor's fields must satisfy to dispatch)

This descriptor does NOT invent quantization metadata for unquantized
(fp32) operations: input_scale/weight_scale/zero-points are only populated
(and only meaningful) when quantized=True.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perf_model.operation_descriptor import register_payload_type
from perf_model.operation_descriptor import OperationFamily

SUPPORTED_ACTIVATION = "relu"
SUPPORTED_ACCUMULATOR_DTYPES_QUANTIZED = ("int32",)
SUPPORTED_QUANTIZED_INPUT_DTYPES = ("int8",)
SUPPORTED_QUANTIZED_WEIGHT_DTYPES = ("int8",)
SUPPORTED_FP32_DTYPES = ("fp32",)
SUPPORTED_WEIGHT_LAYOUTS = ("row_major_kx_n", "packed_b_transposed_nxk")


class MatMulBiasReLUDescriptorError(ValueError):
    pass


@dataclass(frozen=True)
class MatMulBiasReLUDescriptor:
    M: int
    N: int
    K: int
    input_dtype: str
    weight_dtype: str
    accumulator_dtype: str
    output_dtype: str
    has_bias: bool
    activation: str
    input_layout: str
    weight_layout: str
    output_layout: str
    input_contiguous: bool
    weight_contiguous: bool
    output_contiguous: bool
    quantized: bool
    target_arch: str
    target_cpu: str
    thread_count: int
    input_scale: float | None = None
    weight_scale: float | None = None
    input_zero_point: int | None = None
    weight_zero_point: int | None = None
    per_tensor_or_per_channel: str | None = None
    packed_b_available: bool = False

    def __post_init__(self) -> None:
        if self.M <= 0 or self.N <= 0 or self.K <= 0:
            raise MatMulBiasReLUDescriptorError(f"M,N,K must all be positive, got M={self.M} N={self.N} K={self.K}")
        if self.activation != SUPPORTED_ACTIVATION:
            raise MatMulBiasReLUDescriptorError(
                f"activation '{self.activation}' is not supported for this subtype; "
                f"the only fused activation the existing kernels implement is '{SUPPORTED_ACTIVATION}'"
            )
        if not self.has_bias:
            raise MatMulBiasReLUDescriptorError(
                "has_bias=False is not representable: every existing kernel (FP32/INT8/packed-B) "
                "unconditionally fuses a bias add before the ReLU -- there is no bias-less kernel "
                "contract to dispatch to, so this descriptor refuses to invent one"
            )
        if self.weight_layout not in SUPPORTED_WEIGHT_LAYOUTS:
            raise MatMulBiasReLUDescriptorError(f"weight_layout '{self.weight_layout}' not in {SUPPORTED_WEIGHT_LAYOUTS}")

        if self.quantized:
            if self.input_dtype not in SUPPORTED_QUANTIZED_INPUT_DTYPES:
                raise MatMulBiasReLUDescriptorError(f"quantized input_dtype must be int8, got '{self.input_dtype}'")
            if self.weight_dtype not in SUPPORTED_QUANTIZED_WEIGHT_DTYPES:
                raise MatMulBiasReLUDescriptorError(f"quantized weight_dtype must be int8, got '{self.weight_dtype}'")
            if self.accumulator_dtype not in SUPPORTED_ACCUMULATOR_DTYPES_QUANTIZED:
                raise MatMulBiasReLUDescriptorError(f"quantized accumulator_dtype must be int32, got '{self.accumulator_dtype}'")
            if self.output_dtype != "fp32":
                raise MatMulBiasReLUDescriptorError(
                    f"quantized output_dtype must be fp32 (the kernel always dequantizes before bias+relu), "
                    f"got '{self.output_dtype}'"
                )
            if self.input_scale is None or self.weight_scale is None:
                raise MatMulBiasReLUDescriptorError(
                    "quantized=True requires input_scale and weight_scale -- this descriptor will not "
                    "invent default scale values"
                )
            if self.input_scale <= 0.0 or self.weight_scale <= 0.0:
                raise MatMulBiasReLUDescriptorError("input_scale and weight_scale must be positive")
            if self.input_zero_point is None or self.weight_zero_point is None:
                raise MatMulBiasReLUDescriptorError(
                    "quantized=True requires explicit input_zero_point/weight_zero_point -- this "
                    "descriptor will not invent a default zero point"
                )
            if self.input_zero_point != 0 or self.weight_zero_point != 0:
                raise MatMulBiasReLUDescriptorError(
                    f"nonzero zero points are not supported by any existing kernel (int8.cpp and "
                    f"int8_packed_b.cpp both hard-reject zero_point != 0): "
                    f"input_zero_point={self.input_zero_point}, weight_zero_point={self.weight_zero_point}"
                )
            if self.per_tensor_or_per_channel is not None and self.per_tensor_or_per_channel != "per_tensor":
                raise MatMulBiasReLUDescriptorError(
                    f"per-channel quantization is not supported by any existing kernel (both "
                    f"deployment/execution_plan/int8_quantization.py and the C++ kernels are strictly "
                    f"per-tensor symmetric); got per_tensor_or_per_channel="
                    f"'{self.per_tensor_or_per_channel}'"
                )
            if self.weight_layout == "packed_b_transposed_nxk" and not self.packed_b_available:
                raise MatMulBiasReLUDescriptorError(
                    "weight_layout='packed_b_transposed_nxk' requires packed_b_available=True "
                    "(the packed kernel refuses to repack at runtime -- packing is a compiler/offline "
                    "artifact, see int8_quantization.create_packed_weight_artifact)"
                )
        else:
            if self.input_dtype not in SUPPORTED_FP32_DTYPES or self.weight_dtype not in SUPPORTED_FP32_DTYPES:
                raise MatMulBiasReLUDescriptorError(
                    f"unquantized path only supports fp32 input/weight dtypes, got "
                    f"input_dtype='{self.input_dtype}' weight_dtype='{self.weight_dtype}'"
                )
            if self.accumulator_dtype != "fp32":
                raise MatMulBiasReLUDescriptorError(f"unquantized accumulator_dtype must be fp32, got '{self.accumulator_dtype}'")
            if self.output_dtype != "fp32":
                raise MatMulBiasReLUDescriptorError(f"unquantized output_dtype must be fp32, got '{self.output_dtype}'")
            for field_name, value in (("input_scale", self.input_scale), ("weight_scale", self.weight_scale),
                                       ("input_zero_point", self.input_zero_point), ("weight_zero_point", self.weight_zero_point)):
                if value is not None:
                    raise MatMulBiasReLUDescriptorError(
                        f"quantized=False but {field_name}={value!r} was supplied -- this descriptor "
                        f"refuses to carry quantization metadata for an unquantized operation "
                        f"(no invented/ignored fields)"
                    )
            if self.weight_layout == "packed_b_transposed_nxk":
                raise MatMulBiasReLUDescriptorError(
                    "packed_b_transposed_nxk weight_layout is only defined for the INT8 packed-B kernel; "
                    "no FP32 packed-weight kernel exists"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "M": self.M, "N": self.N, "K": self.K, "input_dtype": self.input_dtype, "weight_dtype": self.weight_dtype,
            "accumulator_dtype": self.accumulator_dtype, "output_dtype": self.output_dtype, "has_bias": self.has_bias,
            "activation": self.activation, "input_layout": self.input_layout, "weight_layout": self.weight_layout,
            "output_layout": self.output_layout, "input_contiguous": self.input_contiguous,
            "weight_contiguous": self.weight_contiguous, "output_contiguous": self.output_contiguous,
            "quantized": self.quantized, "target_arch": self.target_arch, "target_cpu": self.target_cpu,
            "thread_count": self.thread_count, "input_scale": self.input_scale, "weight_scale": self.weight_scale,
            "input_zero_point": self.input_zero_point, "weight_zero_point": self.weight_zero_point,
            "per_tensor_or_per_channel": self.per_tensor_or_per_channel, "packed_b_available": self.packed_b_available,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MatMulBiasReLUDescriptor":
        return MatMulBiasReLUDescriptor(**d)


register_payload_type(OperationFamily.MATMUL_BIAS_RELU, MatMulBiasReLUDescriptor)
