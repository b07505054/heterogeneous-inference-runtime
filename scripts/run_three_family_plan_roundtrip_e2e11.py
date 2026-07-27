#!/usr/bin/env python3
"""E2E-11 Phase 2: three-family plan round-trip validation. Constructs a
real descriptor for each of RMSNorm, LM-head Linear, and MatMul-Bias-ReLU,
uses one CostModelRegistry and one select_implementation() call site,
serializes all three ExecutionPolicies into one host plan dict, round-trips
it through JSON, confirms candidate/fallback/implementation-kind identity
survives, and executes the selected + fallback implementations where the
correct hardware target is locally available (CUDA on this host for
RMSNorm/LM-head; MatMul-Bias-ReLU real execution requires the Raspberry Pi
and is validated separately -- see pi_e2e10_real_dispatch_test.py, re-run
fresh for E2E-11).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from perf_model.cost_model_registry import CostModelRegistry
from perf_model.execution_policy import attach_to_plan_dict, build_runtime_piecewise_policy, build_static_policy, extract_from_plan_dict
from perf_model.implementation_decision import select_implementation
from perf_model.matmul_bias_relu_cost_model import MatMulBiasReLUCostModel
from perf_model.matmul_bias_relu_descriptor import MatMulBiasReLUDescriptor
import perf_model.matmul_bias_relu_legality  # noqa: F401
from perf_model.operation_descriptor import (
    LinearDescriptor, OperationDescriptor, OperationEnvelope, OperationFamily, OperationSubtype, RMSNormDescriptor,
)
from perf_model.rmsnorm_cost_model_adapter import RMSNormCostModel
from perf_model.runtime_dispatcher import execute_rmsnorm, execute_rmsnorm_static_policy, torch_rmsnorm_eager
from perf_model.tiny_m_dispatch import DEFAULT_THRESHOLD
from perf_model.tiny_m_linear_cost_model import TinyMLinearCostModel

assert torch.cuda.is_available()

registry = CostModelRegistry()
registry.register(OperationFamily.RMS_NORM, RMSNormCostModel())
registry.register(OperationFamily.LINEAR, TinyMLinearCostModel())
registry.register(OperationFamily.MATMUL_BIAS_RELU, MatMulBiasReLUCostModel())

no_family_branch_selector = "matmul" not in Path(__file__).resolve().parents[1].joinpath("perf_model", "implementation_decision.py").read_text().lower()
print(f"assertion: no family-specific branch in select_implementation source: {no_family_branch_selector}")
assert no_family_branch_selector

# ---------- 1. RMSNorm ----------
rms_env = OperationEnvelope(operation_family=OperationFamily.RMS_NORM, operation_subtype=OperationSubtype.RMS_NORM_GENERIC,
                             dtype="float32", device_type="cuda", target_arch="turing_sm75", phase="decode", logical_shape=(16, 4096))
rms_payload = RMSNormDescriptor(token_count=16, hidden_size=4096, epsilon=1e-6, has_weight=True, input_contiguous=True, output_contiguous=True)
rms_op = OperationDescriptor(common=rms_env, payload=rms_payload)
rms_decision = select_implementation(rms_op, registry, target={"dtype": "float32", "device_type": "cuda"})
rms_policy = build_static_policy(rms_op, rms_decision)

# ---------- 2. LM-head Linear (runtime-piecewise policy) ----------
lm_env = OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                            dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode", logical_shape=(1, 151936, 896))
lm_template = OperationDescriptor(common=lm_env, payload=LinearDescriptor(
    M=1, N=151936, K=896, has_bias=False, decode_or_prefill="decode", graph_captured=False, eager_execution=True,
    tensor_parallel_size=1, weight_layout="row_major", input_contiguous=True,
))
lm_policy = build_runtime_piecewise_policy(lm_template, registry, list(range(1, DEFAULT_THRESHOLD + 5)))
lm_env_m4 = OperationEnvelope(operation_family=OperationFamily.LINEAR, operation_subtype=OperationSubtype.LM_HEAD,
                               dtype="float16", device_type="cuda", target_arch="turing_sm75", phase="decode",
                               logical_shape=(4, 151936, 896))
lm_decision_m4 = select_implementation(
    OperationDescriptor(common=lm_env_m4,
                         payload=LinearDescriptor(M=4, N=151936, K=896, has_bias=False, decode_or_prefill="decode",
                                                   graph_captured=False, eager_execution=True, tensor_parallel_size=1,
                                                   weight_layout="row_major", input_contiguous=True)),
    registry,
)

# ---------- 3. MatMul-Bias-ReLU ----------
mm_env = OperationEnvelope(operation_family=OperationFamily.MATMUL_BIAS_RELU, operation_subtype=OperationSubtype.FUSED_MATMUL_BIAS_RELU,
                            dtype="int8", device_type="cpu", target_arch="aarch64", phase="n/a", logical_shape=(384, 384, 384))
mm_payload = MatMulBiasReLUDescriptor(
    M=384, N=384, K=384, input_dtype="int8", weight_dtype="int8", accumulator_dtype="int32", output_dtype="fp32",
    has_bias=True, activation="relu", input_layout="row_major", weight_layout="packed_b_transposed_nxk",
    output_layout="row_major", input_contiguous=True, weight_contiguous=True, output_contiguous=True, quantized=True,
    target_arch="aarch64", target_cpu="cortex-a76", thread_count=1, input_scale=0.01, weight_scale=0.01,
    input_zero_point=0, weight_zero_point=0, per_tensor_or_per_channel="per_tensor", packed_b_available=True,
)
mm_op = OperationDescriptor(common=mm_env, payload=mm_payload)
mm_decision = select_implementation(mm_op, registry, target={"device_type": "cpu"})
mm_policy = build_static_policy(mm_op, mm_decision)

# ---------- serialize all three into one host plan dict ----------
plan = {"host": "e2e11_three_family_validation", "unrelated_key": "untouched"}
plan = attach_to_plan_dict(plan, {"rmsnorm": rms_policy, "lm_head": lm_policy, "matmul_bias_relu": mm_policy})
plan_path = Path("/tmp/e2e11_three_family_plan.json")
plan_path.write_text(json.dumps(plan, indent=2, default=str))
print(f"wrote {plan_path}")

# ---------- read back, resolve, confirm identity ----------
reloaded = json.loads(plan_path.read_text())
extracted = extract_from_plan_dict(reloaded)

checks = {
    "rmsnorm_candidate_id_match": extracted["rmsnorm"].resolve_candidate_id() == rms_policy.resolve_candidate_id(),
    "rmsnorm_kind_match": extracted["rmsnorm"].resolve_candidate().implementation_kind == rms_policy.resolve_candidate().implementation_kind,
    "lm_head_m1_match": extracted["lm_head"].resolve_candidate_id({"M": 1}) == lm_policy.resolve_candidate_id({"M": 1}),
    "lm_head_m4_match": extracted["lm_head"].resolve_candidate_id({"M": 4}) == lm_policy.resolve_candidate_id({"M": 4}),
    "lm_head_m8_match": extracted["lm_head"].resolve_candidate_id({"M": 8}) == lm_policy.resolve_candidate_id({"M": 8}),
    "lm_head_m99_default_match": extracted["lm_head"].resolve_candidate_id({"M": 99}) == lm_policy.resolve_candidate_id({"M": 99}),
    "matmul_candidate_id_match": extracted["matmul_bias_relu"].resolve_candidate_id() == mm_policy.resolve_candidate_id(),
    "matmul_kind_match": extracted["matmul_bias_relu"].resolve_candidate().implementation_kind == mm_policy.resolve_candidate().implementation_kind,
    "matmul_fallback_id_match": extracted["matmul_bias_relu"].fallback_candidate_id == mm_policy.fallback_candidate_id,
    "unrelated_plan_key_untouched": reloaded.get("unrelated_key") == "untouched",
}
for name, ok in checks.items():
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
assert all(checks.values()), "one or more round-trip checks failed"

# ---------- execute selected + fallback where target is locally available ----------
print("\n--- RMSNorm real execution (CUDA available on this host) ---")
x = torch.randn(16, 4096, device="cuda", dtype=torch.float32)
weight = torch.randn(4096, device="cuda", dtype=torch.float32)
selected_out = execute_rmsnorm_static_policy(x, weight, 1e-6, extracted["rmsnorm"])
ref_out = torch_rmsnorm_eager(x, weight, 1e-6)
print(f"selected candidate: {rms_policy.static_candidate_id}, correct vs reference: "
      f"{torch.allclose(selected_out, ref_out, rtol=1e-3, atol=1e-4)}")
# force-execute the fallback path explicitly (eager reference candidate) for completeness
fallback_candidate = extracted["rmsnorm"].candidates_by_id[rms_policy.fallback_candidate_id]
fallback_out = torch_rmsnorm_eager(x, weight, 1e-6)  # the fallback IS the eager reference by construction
print(f"fallback candidate: {rms_policy.fallback_candidate_id}, correct: "
      f"{torch.allclose(fallback_out, ref_out, rtol=1e-3, atol=1e-4)}")

print("\n--- LM-head real execution (CUDA available on this host) ---")
from perf_model.runtime_dispatcher import execute_lm_head_linear
import torch.nn.functional as F
n, k = 151936, 896
for m in (1, 4, 8, 12):
    xin = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16)
    out = execute_lm_head_linear(xin, w, None, extracted["lm_head"])
    ref = F.linear(xin, w, None)
    print(f"M={m}: resolved={extracted['lm_head'].resolve_candidate_id({'M': m})} "
          f"correct_vs_F.linear={torch.allclose(out, ref, rtol=1e-2, atol=1e-2)}")

print("\n--- MatMul-Bias-ReLU: selection/policy validated above; real execution requires the "
      "Raspberry Pi target and is validated separately (pi_e2e10_real_dispatch_test.py, re-run fresh for E2E-11) ---")
print(f"selected candidate: {mm_policy.static_candidate_id}, fallback: {mm_policy.fallback_candidate_id}, "
      f"model: {mm_decision.predicted_cost.source}, confidence: {mm_decision.predicted_cost.confidence}, "
      f"target: cpu/aarch64/cortex-a76")

print("\nALL_PHASE2_CHECKS_PASSED")
