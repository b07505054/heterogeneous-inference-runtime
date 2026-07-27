"""E2E-10 Phase 17 architecture tests: proves (not just asserts) that adding
the third family did not require a family-specific branch in the generic
selector/registry orchestration, and that all three families are selectable
through one shared registry and one shared select_implementation() call."""
from __future__ import annotations

import inspect

from perf_model.cost_model_registry import CostModelRegistry
from perf_model.implementation_decision import ImplementationDecision, select_implementation
from perf_model.matmul_bias_relu_cost_model import MatMulBiasReLUCostModel
from perf_model.matmul_bias_relu_descriptor import MatMulBiasReLUDescriptor
import perf_model.matmul_bias_relu_legality  # noqa: F401
from perf_model.operation_descriptor import (
    LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)
from perf_model.rmsnorm_cost_model_adapter import RMSNormCostModel
from perf_model.tiny_m_linear_cost_model import TinyMLinearCostModel


def test_selector_source_contains_no_matmul_family_branch():
    source = inspect.getsource(select_implementation)
    assert "matmul" not in source.lower()  # no literal family branch text anywhere in the function body
    import perf_model.implementation_decision as m
    full_module_source = inspect.getsource(m)
    assert "matmul" not in full_module_source.lower()


def test_cost_model_registry_orchestration_contains_no_family_branch():
    import perf_model.cost_model_registry as m
    source = inspect.getsource(m)
    assert "matmul_bias_relu" not in source
    assert "MATMUL_BIAS_RELU" not in source


def test_all_three_families_selectable_through_one_registry_one_call():
    registry = CostModelRegistry()
    registry.register(OperationFamily.RMS_NORM, RMSNormCostModel())
    registry.register(OperationFamily.LINEAR, TinyMLinearCostModel())
    registry.register(OperationFamily.MATMUL_BIAS_RELU, MatMulBiasReLUCostModel())

    rms_op = OperationDescriptor(
        common=OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                                  dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode", logical_shape=(16, 4096)),
        payload=RMSNormDescriptor(token_count=16, hidden_size=4096, epsilon=1e-6, has_weight=True, input_contiguous=True, output_contiguous=True),
    )
    linear_op = OperationDescriptor(
        common=OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                                  dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode", logical_shape=(4, 151936, 896)),
        payload=LinearDescriptor(M=4, N=151936, K=896, has_bias=False, decode_or_prefill="decode", graph_captured=False,
                                  eager_execution=True, tensor_parallel_size=1, weight_layout="row_major", input_contiguous=True),
    )
    matmul_op = OperationDescriptor(
        common=OperationEnvelope(operation_family=OperationFamily.MATMUL_BIAS_RELU, operation_subtype=OperationSubtype.FUSED_MATMUL_BIAS_RELU,
                                  dtype="int8", device_type="cpu", target_arch="aarch64", phase="n/a", logical_shape=(384, 384, 384)),
        payload=MatMulBiasReLUDescriptor(M=384, N=384, K=384, input_dtype="int8", weight_dtype="int8", accumulator_dtype="int32",
                                          output_dtype="fp32", has_bias=True, activation="relu", input_layout="row_major",
                                          weight_layout="packed_b_transposed_nxk", output_layout="row_major",
                                          input_contiguous=True, weight_contiguous=True, output_contiguous=True, quantized=True,
                                          target_arch="aarch64", target_cpu="cortex-a76", thread_count=1,
                                          input_scale=0.01, weight_scale=0.01, input_zero_point=0, weight_zero_point=0,
                                          per_tensor_or_per_channel="per_tensor", packed_b_available=True),
    )

    decisions = [
        select_implementation(rms_op, registry, target={"dtype": "float32", "device_type": "cuda"}),
        select_implementation(linear_op, registry),
        select_implementation(matmul_op, registry, target={"device_type": "cpu"}),
    ]
    assert all(isinstance(d, ImplementationDecision) for d in decisions)
    assert {d.operation_family for d in decisions} == {"rms_norm", "linear", "matmul_bias_relu"}
    assert decisions[2].selected_candidate.candidate_id.startswith("matmul_bias_relu_")


def test_registered_generator_and_legality_checker_counts():
    from perf_model.implementation_candidate import _GENERATORS
    from perf_model.legality import _LEGALITY_CHECKERS
    assert set(_GENERATORS.keys()) == {OperationFamily.RMS_NORM, OperationFamily.LINEAR, OperationFamily.MATMUL_BIAS_RELU}
    assert set(_LEGALITY_CHECKERS.keys()) == {OperationFamily.RMS_NORM, OperationFamily.LINEAR, OperationFamily.MATMUL_BIAS_RELU}
