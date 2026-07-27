from perf_model.cost_model_registry import CostModelRegistry
from perf_model.execution_policy import ExecutionPolicy, attach_to_plan_dict, build_runtime_piecewise_policy, build_static_policy, extract_from_plan_dict
from perf_model.implementation_candidate import ImplementationKind
from perf_model.implementation_decision import select_implementation
from perf_model.operation_descriptor import (
    LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)
from perf_model.rmsnorm_cost_model_adapter import RMSNormCostModel
from perf_model.tiny_m_linear_cost_model import TinyMLinearCostModel


def _registry():
    r = CostModelRegistry()
    r.register(OperationFamily.RMS_NORM, RMSNormCostModel())
    r.register(OperationFamily.LINEAR, TinyMLinearCostModel())
    return r


def _rmsnorm_op(tokens=16, hidden=4096):
    env = OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                             dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(tokens, hidden))
    payload = RMSNormDescriptor(token_count=tokens, hidden_size=hidden, epsilon=1e-6, has_weight=True,
                                 input_contiguous=True, output_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def _linear_template(m=1):
    env = OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                             dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode",
                             logical_shape=(m, 151936, 896))
    payload = LinearDescriptor(M=m, N=151936, K=896, has_bias=False, decode_or_prefill="decode",
                                graph_captured=False, eager_execution=True, tensor_parallel_size=1,
                                weight_layout="row_major", input_contiguous=True)
    return OperationDescriptor(common=env, payload=payload)


def test_static_policy_round_trips_through_json():
    decision = select_implementation(_rmsnorm_op(), _registry(), target={"dtype": "float32", "device_type": "cuda"})
    policy = build_static_policy(_rmsnorm_op(), decision)
    policy2 = ExecutionPolicy.from_json(policy.to_json())
    assert policy2.resolve_candidate_id() == policy.resolve_candidate_id()
    assert policy2.selection_mode == "static"


def test_runtime_piecewise_policy_round_trips_and_resolves_by_m():
    policy = build_runtime_piecewise_policy(_linear_template(), _registry(), [1, 2, 3, 4, 6, 8])
    policy2 = ExecutionPolicy.from_json(policy.to_json())
    for m in (1, 2, 3, 4, 6, 8, 99):
        assert policy.resolve_candidate_id({"M": m}) == policy2.resolve_candidate_id({"M": m})
    assert policy2.resolve_candidate({"M": 4}).implementation_kind == ImplementationKind.LINEAR_ROW_WISE_GEMV
    assert policy2.resolve_candidate({"M": 1}).implementation_kind == ImplementationKind.LINEAR_DEFAULT_GEMM


def test_runtime_piecewise_unlisted_m_falls_back_to_default():
    policy = build_runtime_piecewise_policy(_linear_template(), _registry(), [1, 2, 3, 4, 6, 8])
    # M=100 was never in the derivation range and has no rule; must resolve via default, not raise
    cand = policy.resolve_candidate({"M": 100})
    assert cand.implementation_kind == ImplementationKind.LINEAR_DEFAULT_GEMM


def test_attach_and_extract_from_plan_dict_is_additive():
    decision = select_implementation(_rmsnorm_op(), _registry(), target={"dtype": "float32", "device_type": "cuda"})
    policy = build_static_policy(_rmsnorm_op(), decision)
    existing_plan = {"some_other_key": "untouched", "nested": {"a": 1}}
    new_plan = attach_to_plan_dict(existing_plan, {"rmsnorm": policy})
    assert new_plan["some_other_key"] == "untouched"
    assert new_plan["nested"] == {"a": 1}
    assert "operation_policies" in new_plan
    assert existing_plan is not new_plan  # original untouched (copy semantics)
    assert "operation_policies" not in existing_plan

    extracted = extract_from_plan_dict(new_plan)
    assert extracted["rmsnorm"].resolve_candidate_id() == policy.resolve_candidate_id()


def test_extract_from_plan_dict_with_no_operation_policies_key_is_empty():
    plan = {"unrelated": True}
    extracted = extract_from_plan_dict(plan)
    assert extracted == {}
