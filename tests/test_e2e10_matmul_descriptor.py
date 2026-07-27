import pytest

from perf_model.matmul_bias_relu_descriptor import MatMulBiasReLUDescriptor, MatMulBiasReLUDescriptorError
from perf_model.operation_descriptor import (
    DecisionKind, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype,
)


def _fp32_kwargs(**overrides):
    base = dict(M=8, N=8, K=8, input_dtype="fp32", weight_dtype="fp32", accumulator_dtype="fp32", output_dtype="fp32",
                has_bias=True, activation="relu", input_layout="row_major", weight_layout="row_major_kx_n",
                output_layout="row_major", input_contiguous=True, weight_contiguous=True, output_contiguous=True,
                quantized=False, target_arch="aarch64", target_cpu="cortex-a76", thread_count=1)
    base.update(overrides)
    return base


def _int8_kwargs(**overrides):
    base = dict(M=8, N=8, K=8, input_dtype="int8", weight_dtype="int8", accumulator_dtype="int32", output_dtype="fp32",
                has_bias=True, activation="relu", input_layout="row_major", weight_layout="row_major_kx_n",
                output_layout="row_major", input_contiguous=True, weight_contiguous=True, output_contiguous=True,
                quantized=True, target_arch="aarch64", target_cpu="cortex-a76", thread_count=1,
                input_scale=0.01, weight_scale=0.01, input_zero_point=0, weight_zero_point=0,
                per_tensor_or_per_channel="per_tensor")
    base.update(overrides)
    return base


def test_valid_fp32_descriptor_constructs():
    d = MatMulBiasReLUDescriptor(**_fp32_kwargs())
    assert d.quantized is False


def test_valid_symmetric_int8_descriptor_constructs():
    d = MatMulBiasReLUDescriptor(**_int8_kwargs(weight_layout="packed_b_transposed_nxk", packed_b_available=True))
    assert d.quantized is True


def test_invalid_nonzero_input_zero_point_rejected():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_int8_kwargs(input_zero_point=1))


def test_invalid_nonzero_weight_zero_point_rejected():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_int8_kwargs(weight_zero_point=3))


def test_missing_scales_rejected():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_int8_kwargs(input_scale=None))


def test_unsupported_per_channel_quantization_rejected():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_int8_kwargs(per_tensor_or_per_channel="per_channel"))


def test_invalid_activation_rejected():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_fp32_kwargs(activation="gelu"))


def test_invalid_shape_rejected():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_fp32_kwargs(M=0))
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_fp32_kwargs(K=-1))


def test_bias_false_rejected_no_invented_contract():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_fp32_kwargs(has_bias=False))


def test_packed_layout_without_packed_available_rejected():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_int8_kwargs(weight_layout="packed_b_transposed_nxk", packed_b_available=False))


def test_fp32_with_quantization_metadata_rejected_not_silently_ignored():
    with pytest.raises(MatMulBiasReLUDescriptorError):
        MatMulBiasReLUDescriptor(**_fp32_kwargs(input_scale=0.1))


def test_json_round_trip_fp32():
    env = OperationEnvelope(operation_family=OperationFamily.MATMUL_BIAS_RELU, operation_subtype=OperationSubtype.FUSED_MATMUL_BIAS_RELU,
                             dtype="fp32", device_type="cpu", target_arch="aarch64", phase="n/a", logical_shape=(8, 8, 8))
    payload = MatMulBiasReLUDescriptor(**_fp32_kwargs())
    op = OperationDescriptor(common=env, payload=payload)
    op2 = OperationDescriptor.from_json(op.to_json())
    assert op2 == op
    assert isinstance(op2.payload, MatMulBiasReLUDescriptor)


def test_json_round_trip_int8_packed():
    env = OperationEnvelope(operation_family=OperationFamily.MATMUL_BIAS_RELU, operation_subtype=OperationSubtype.FUSED_MATMUL_BIAS_RELU,
                             dtype="int8", device_type="cpu", target_arch="aarch64", phase="n/a", logical_shape=(384, 384, 384))
    payload = MatMulBiasReLUDescriptor(**_int8_kwargs(M=384, N=384, K=384, weight_layout="packed_b_transposed_nxk", packed_b_available=True))
    op = OperationDescriptor(common=env, payload=payload)
    op2 = OperationDescriptor.from_json(op.to_json())
    assert op2 == op


def test_decision_kind_kernel_implementation_exists():
    assert DecisionKind.KERNEL_IMPLEMENTATION.value == "kernel_implementation"


def test_operation_family_and_subtype_parse():
    assert OperationFamily("matmul_bias_relu") == OperationFamily.MATMUL_BIAS_RELU
    assert OperationSubtype("fused_matmul_bias_relu") == OperationSubtype.FUSED_MATMUL_BIAS_RELU
