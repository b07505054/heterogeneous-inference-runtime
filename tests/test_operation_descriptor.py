import pytest

from perf_model.operation_descriptor import (
    LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)


def _rmsnorm_op(tokens=16, hidden=4096):
    env = OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                             dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(tokens, hidden))
    payload = RMSNormDescriptor(token_count=tokens, hidden_size=hidden, epsilon=1e-6, has_weight=True,
                                 input_contiguous=True, output_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def _linear_op(m=4, n=151936, k=896):
    env = OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                             dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(m, n, k))
    payload = LinearDescriptor(M=m, N=n, K=k, has_bias=False, decode_or_prefill="decode", graph_captured=False,
                                eager_execution=True, tensor_parallel_size=1, weight_layout="row_major",
                                input_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def test_rmsnorm_descriptor_json_round_trip():
    op = _rmsnorm_op()
    op2 = OperationDescriptor.from_json(op.to_json())
    assert op2 == op
    assert isinstance(op2.payload, RMSNormDescriptor)


def test_linear_descriptor_json_round_trip():
    op = _linear_op()
    op2 = OperationDescriptor.from_json(op.to_json())
    assert op2 == op
    assert isinstance(op2.payload, LinearDescriptor)


def test_subtype_must_match_family():
    with pytest.raises(ValueError):
        OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.LM_HEAD,
                           dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode",
                           logical_shape=(1, 1))


def test_payload_must_match_family():
    env = OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                             dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(1, 1))
    wrong_payload = LinearDescriptor(M=1, N=1, K=1, has_bias=False, decode_or_prefill="decode",
                                      graph_captured=False, eager_execution=True, tensor_parallel_size=1,
                                      weight_layout="row_major", input_contiguous=True)
    with pytest.raises(ValueError):
        OperationDescriptor(common=env, payload=wrong_payload)


def test_from_dict_unknown_family_raises():
    d = {"common": {"operation_family": "not_a_family", "operation_subtype": "rms_norm_generic",
                     "dtype": "float32", "device_type": "cuda", "target_arch": "x", "phase": "decode",
                     "logical_shape": [1, 1], "static_attributes": {}},
         "payload": {}}
    with pytest.raises(ValueError):
        OperationDescriptor.from_dict(d)
