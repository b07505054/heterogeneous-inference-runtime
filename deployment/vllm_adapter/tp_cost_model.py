"""D5/D6: TP1-vs-TP2 throughput cost model -- offline calibration / oracle
reference utility.

D6 status (reclassified -- read this before using this module): this is
NOT part of the production TP-degree decision path. That decision is now
made by the real C++/MLIR compiler
(ml-graph-compiler-runtime/mlir_passes/lib/serving/
DistributedStrategyPlanningPass.cpp's distributed_profitability_contract_v1
selector), which reimplements this exact regression numerically (verified
bit-for-bit identical -- see
tools/generate_distributed_profitability_profile.py --check-against) and
consumes calibration coefficients from a versioned target-profile JSON,
never this Python module, at compile time.

This module's remaining, real roles are:
  1. Offline calibration fitting (`fit_linear_regression`,
     `TPCostModel.fit`) -- the actual source of the calibration
     coefficients embedded in
     ml-graph-compiler-runtime/configs/target_profiles/
     nvidia_rtx4090_d6_distributed_profitability.json, via
     tools/generate_distributed_profitability_profile.py.
  2. An independent oracle/reference implementation for cross-checking the
     compiler's C++ predictions against a separately-maintained Python
     implementation of the same math (see
     tests/test_distributed_d6_compiler_owned_tp_selection.py's
     production-path-independence tests).
  3. A regression-reference utility for anyone extending the calibration
     (e.g. adding a third model size) who wants to explore the fit in
     Python before regenerating the compiler-consumed profile.

No production script (scripts/run_d6_*.py, deployment/vllm_adapter/
distributed_materializer.py, or any real server-launch path) imports or
calls TPCostModel.decide() to choose a runtime TP degree. If you are
writing new code that launches a real vLLM server, the TP degree must
come from an ExecutionPlan's `distributed` block (produced by the real
compiler), never from calling this module directly.

Historical D5 note (mechanism unchanged from when this was the production
selector): predicts, from information available strictly *before*
execution (model identity, workload shape, GPU memory budget, and
constants regressed from already-collected calibration measurements),
which tensor-parallel degree a given workload should use. Never consults
held-out measurements. Never consults the actual outcome of the workload
being decided.

Two decision layers, in priority order:
  1. Feasibility (hard, analytical): if TP1 cannot legally hold the
     workload's weights + worst-case KV cache within the real GPU memory
     budget, TP2 is forced -- this is a closed-form memory calculation, not
     a regression, and is exactly the "TP2 capacity region" the spec asks
     us to search for.
  2. Performance (soft, regressed): if both TP degrees are feasible, a
     linear model -- fit ONLY on calibration-split throughput
     measurements -- predicts each degree's throughput for the workload's
     features; the compiler picks whichever is predicted higher.

Real, measured model-identity features (hidden_size, num_attention_heads,
num_kv_heads, num_layers) come directly from each model's real HF config
(deployment/execution_plan model_identity blocks / D3A-validated mapping).
weight_footprint_mb is the real measured/queried checkpoint size (see
distributed_materializer._estimate_model_footprint_mb and, for the 7B
entry, the real Hugging Face Hub blob-size total queried directly from
the Hub API on 2026-07-19 -- not a parameter-count guess).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

BYTES_PER_PARAM_FP16 = 2
KV_CACHE_HEADROOM_MB = 512.0  # matches distributed_materializer.KV_CACHE_HEADROOM_MB

# Real model-identity features. hidden_size/num_attention_heads/num_kv_heads/
# num_layers are the exact values used to build each model's real D2
# ExecutionPlan (results/runtime_paths/distributed_d2_qwen_pipeline/
# real_qwen*_execution_plan.json). weight_footprint_mb for the 0.5B model is
# the real value distributed_materializer._estimate_model_footprint_mb()
# measured against the actual cached checkpoint (1454.32 MB, matches every
# D5 0.5B sweep's own preflight artifact). weight_footprint_mb for the 7B
# model is the real total safetensors blob size returned by a live query
# against https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct?blobs=true
# (15,242,807,270 bytes = 14,539.06 MB) plus the same fixed headroom margin.
MODEL_IDENTITY_FEATURES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "model_plan_id": "qwen2.5-0.5b", "hidden_size": 896, "num_attention_heads": 14,
        "num_kv_heads": 2, "num_layers": 24, "weight_footprint_mb": 1454.3235168457031,
        "max_position_embeddings": 32768,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model_plan_id": "qwen2.5-7b", "hidden_size": 3584, "num_attention_heads": 28,
        "num_kv_heads": 4, "num_layers": 28,
        "weight_footprint_mb": 15242807270 / (1024 * 1024) + KV_CACHE_HEADROOM_MB,
        "max_position_embeddings": 32768,
    },
}

FEATURE_NAMES = (
    "per_gpu_weight_mb", "kv_cache_kb_per_token_per_gpu", "gpu_count",
    "input_length", "output_length", "concurrency",
)

COMMUNICATION_CONTRACT_VERSION = "nccl_communication_calibration.v1"
COMMUNICATION_COLLECTIVE_KIND = "all_reduce"
COMMUNICATION_MODE = "out_of_place"
D9_POLICY_ID = "d9_break_even_tp_selector_v1"
D9_DECISION_MARGIN_US = 250.0
D9_RUNTIME_RESIDUAL_US = 0.0
D9_OVERLAP_ASSUMPTION = "zero"
D9_COMPUTE_REFERENCE_WEIGHT_MB = MODEL_IDENTITY_FEATURES["Qwen/Qwen2.5-0.5B-Instruct"]["weight_footprint_mb"]
D9_COMPUTE_SAVINGS_US_PER_WEIGHT_MB_ABOVE_REFERENCE = 0.50
COMMUNICATION_PREDICTOR_SELECTION_METRIC = ("mape", "mae_us")
EXPECTED_COMMUNICATION_TOPOLOGY = {
    "topology_class": "PHB",
    "p2p_available": False,
    "nccl_transport": "SHM/direct/direct",
}


def head_dim(model_features: dict[str, Any]) -> float:
    return model_features["hidden_size"] / model_features["num_attention_heads"]


def kv_cache_bytes_per_token_per_gpu(model_features: dict[str, Any], tp_degree: int) -> float:
    """2 (K & V) * num_layers * (num_kv_heads / tp_degree) * head_dim * fp16 bytes.

    vLLM's tensor-parallel attention shards KV heads across ranks, so each
    GPU only stores 1/tp_degree of the total KV cache for a given sequence
    -- this is the real mechanism (not an assumption) behind why splitting
    a model across GPUs can relieve per-GPU memory pressure even when
    weights are also (independently) sharded.
    """
    kv_heads_per_gpu = model_features["num_kv_heads"] / tp_degree
    return 2 * model_features["num_layers"] * kv_heads_per_gpu * head_dim(model_features) * BYTES_PER_PARAM_FP16


def per_gpu_weight_mb(model_features: dict[str, Any], tp_degree: int) -> float:
    """Real weight footprint divided across ranks. An approximation of
    vLLM's actual per-rank shard size (embedding/lm-head are not perfectly
    even-split), declared as such -- not claimed to be exact bytes."""
    return model_features["weight_footprint_mb"] / tp_degree


def is_feasible(
    model_features: dict[str, Any], tp_degree: int, *, gpu_total_mb: float, gpu_memory_utilization: float,
    max_model_len: int, max_num_seqs: int,
) -> tuple[bool, dict[str, float]]:
    """Closed-form worst-case memory check: can this TP degree legally hold
    its weight shard plus a full max_num_seqs x max_model_len KV cache
    within the real GPU memory budget? This is the hard capacity boundary
    -- computed, never regressed, and never treated as a performance
    preference."""
    budget_mb = gpu_total_mb * gpu_memory_utilization
    weight_mb = per_gpu_weight_mb(model_features, tp_degree)
    worst_case_kv_mb = (max_num_seqs * max_model_len * kv_cache_bytes_per_token_per_gpu(model_features, tp_degree)) / (1024 * 1024)
    required_mb = weight_mb + worst_case_kv_mb
    return required_mb <= budget_mb, {
        "budget_mb": budget_mb, "weight_mb": weight_mb, "worst_case_kv_mb": worst_case_kv_mb, "required_mb": required_mb,
    }


def build_feature_vector(
    model_features: dict[str, Any], tp_degree: int, *, input_length: int, output_length: int, concurrency: int,
) -> list[float]:
    return [
        per_gpu_weight_mb(model_features, tp_degree),
        kv_cache_bytes_per_token_per_gpu(model_features, tp_degree) / 1024.0,
        float(tp_degree),
        float(input_length), float(output_length), float(concurrency),
    ]


class CommunicationCalibrationError(ValueError):
    """Raised when a communication calibration profile is missing or mismatched."""


@dataclass(frozen=True)
class CommunicationPoint:
    bytes: int
    time_us: float


@dataclass(frozen=True)
class AlphaBetaBaseline:
    alpha_us: float
    beta_us_per_byte: float
    evaluation: dict[str, float]

    def predict(self, bytes_value: int) -> float:
        return self.alpha_us + self.beta_us_per_byte * bytes_value


@dataclass(frozen=True)
class CommunicationPredictor:
    profile_id: str
    collective_kind: str
    predictor_kind: str
    topology_class: str
    p2p_available: bool
    nccl_transport: str
    nccl_version: str
    nccl_tests_version: str
    source_artifact_hashes: dict[str, str]
    points: tuple[CommunicationPoint, ...]
    alpha_beta_baseline: AlphaBetaBaseline
    selected_evaluation: dict[str, float]

    def _check_bytes_in_range(self, bytes_value: int) -> None:
        if bytes_value < 0:
            raise CommunicationCalibrationError("communication bytes must be non-negative")
        if bytes_value == 0:
            return
        if not self.points:
            raise CommunicationCalibrationError("communication predictor has no measured points")
        lo, hi = self.points[0].bytes, self.points[-1].bytes
        if bytes_value < lo or bytes_value > hi:
            raise CommunicationCalibrationError(
                f"communication bytes {bytes_value} outside calibrated range [{lo}, {hi}]"
            )

    def predict_time_us(self, bytes_value: int) -> float:
        self._check_bytes_in_range(bytes_value)
        if bytes_value == 0:
            return 0.0
        if self.predictor_kind == "alpha_beta":
            return self.alpha_beta_baseline.predict(bytes_value)
        if self.predictor_kind != "log_size_piecewise_interpolation":
            raise CommunicationCalibrationError(f"unknown communication predictor kind: {self.predictor_kind}")
        for p in self.points:
            if p.bytes == bytes_value:
                return p.time_us
        for left, right in zip(self.points, self.points[1:]):
            if left.bytes <= bytes_value <= right.bytes:
                x = math.log2(bytes_value)
                x0 = math.log2(left.bytes)
                x1 = math.log2(right.bytes)
                frac = (x - x0) / (x1 - x0)
                return left.time_us + frac * (right.time_us - left.time_us)
        raise CommunicationCalibrationError("failed to interpolate communication bytes")


def _select_predictor_kind(fit_collective: dict[str, Any], mode: str = COMMUNICATION_MODE) -> str:
    report = fit_collective[mode]
    piece = report["log_size_piecewise_interpolation"]
    alpha = report["alpha_beta_baseline"]["evaluation"]
    piece_key = (piece["mape"], piece["mae_us"])
    alpha_key = (alpha["mape"], alpha["mae_us"])
    return "log_size_piecewise_interpolation" if piece_key <= alpha_key else "alpha_beta"


def load_communication_predictor(
    communication_cost_profile: dict[str, Any],
    fit_report: dict[str, Any],
    *,
    collective_kind: str = COMMUNICATION_COLLECTIVE_KIND,
    mode: str = COMMUNICATION_MODE,
    expected_topology: dict[str, Any] | None = EXPECTED_COMMUNICATION_TOPOLOGY,
) -> CommunicationPredictor:
    profile_id = communication_cost_profile.get("profile_id")
    if not profile_id or fit_report.get("profile_id") != profile_id:
        raise CommunicationCalibrationError("communication profile id mismatch")

    boundary = communication_cost_profile.get("machine_calibration_boundary") or {}
    topology = {
        "topology_class": boundary.get("topology_class"),
        "p2p_available": boundary.get("cuda_p2p_available"),
        "nccl_transport": boundary.get("nccl_intra_node_transport"),
    }
    if expected_topology:
        for key, expected in expected_topology.items():
            if topology.get(key) != expected:
                raise CommunicationCalibrationError(
                    f"communication topology mismatch for {key}: got {topology.get(key)!r}, expected {expected!r}"
                )

    collective = (communication_cost_profile.get("collectives") or {}).get(collective_kind)
    fit_collective = (fit_report.get("collectives") or {}).get(collective_kind)
    if not collective or not fit_collective:
        raise CommunicationCalibrationError(f"missing communication collective calibration: {collective_kind}")
    predictor_kind = _select_predictor_kind(fit_collective, mode)
    measurements = collective.get("measurements") or []
    points = []
    for row in measurements:
        try:
            points.append(CommunicationPoint(bytes=int(row["bytes"]), time_us=float(row[mode]["time_us"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise CommunicationCalibrationError(f"malformed communication measurement row: {row!r}") from exc
    points = sorted(points, key=lambda p: p.bytes)
    if len(points) < 2 or len({p.bytes for p in points}) != len(points):
        raise CommunicationCalibrationError("communication predictor requires unique sorted measured points")

    alpha_block = fit_collective[mode]["alpha_beta_baseline"]
    return CommunicationPredictor(
        profile_id=profile_id,
        collective_kind=collective_kind,
        predictor_kind=predictor_kind,
        topology_class=str(topology["topology_class"]),
        p2p_available=bool(topology["p2p_available"]),
        nccl_transport=str(topology["nccl_transport"]),
        nccl_version=str(boundary.get("nccl_version")),
        nccl_tests_version=str(boundary.get("nccl_tests_version")),
        source_artifact_hashes=dict(communication_cost_profile.get("source_artifact_hashes") or {}),
        points=tuple(points),
        alpha_beta_baseline=AlphaBetaBaseline(
            alpha_us=float(alpha_block["alpha_us"]),
            beta_us_per_byte=float(alpha_block["beta_us_per_byte"]),
            evaluation=dict(alpha_block["evaluation"]),
        ),
        selected_evaluation=dict(
            fit_collective[mode]["log_size_piecewise_interpolation"]
            if predictor_kind == "log_size_piecewise_interpolation"
            else fit_collective[mode]["alpha_beta_baseline"]["evaluation"]
        ),
    )


def estimated_communication_bytes(model_features: dict[str, Any], tp_degree: int) -> int:
    if tp_degree <= 1:
        return 0
    hidden = int(model_features["hidden_size"])
    return int(tp_degree * hidden * BYTES_PER_PARAM_FP16 * 2)


def estimate_collective_call_count(model_features: dict[str, Any], *, concurrency: int, tp_degree: int = 2) -> int:
    if tp_degree <= 1:
        return 0
    # D9 intentionally uses structural/workload features, not model names.
    # One TP all-reduce class per layer per active decode lane is a conservative
    # call-count-aware upper estimate for selector use; Phase 4D showed count,
    # not total bytes alone, carries decision signal.
    return int(max(1, model_features.get("num_layers", 1)) * max(1, concurrency))


def estimate_collective_demand(model_features: dict[str, Any], *, concurrency: int, tp_degree: int = 2) -> dict[str, Any]:
    bytes_per_call = estimated_communication_bytes(model_features, tp_degree)
    call_count = estimate_collective_call_count(model_features, concurrency=concurrency, tp_degree=tp_degree)
    return {
        "collective_kind": COMMUNICATION_COLLECTIVE_KIND if call_count else "none",
        "bytes_per_collective_call": bytes_per_call,
        "estimated_collective_call_count": call_count,
        "estimated_total_communication_bytes": bytes_per_call * call_count,
    }


def estimate_communication_penalty_us(
    model_features: dict[str, Any], *, concurrency: int, tp_degree: int,
    communication_calibration: CommunicationPredictor,
) -> tuple[float, dict[str, Any]]:
    demand = estimate_collective_demand(model_features, concurrency=concurrency, tp_degree=tp_degree)
    if demand["estimated_collective_call_count"] == 0:
        return 0.0, demand
    per_call = communication_calibration.predict_time_us(int(demand["bytes_per_collective_call"]))
    demand["predicted_nccl_latency_us_per_call"] = per_call
    return per_call * int(demand["estimated_collective_call_count"]), demand


def throughput_to_latency_us(predicted_tokens_per_s: float) -> float:
    if predicted_tokens_per_s <= 0:
        return float("inf")
    return 1_000_000.0 / predicted_tokens_per_s


def estimate_structural_compute_savings_adjustment_us(model_features: dict[str, Any]) -> float:
    """Phase 4D break-even calibration using structural model scale.

    Larger weight footprint is the proxy for the model-compute increase that
    moved the measured boundary from TP1-favorable 0.5B cells to
    TP2-favorable 7B cells. This uses model facts, not model names.
    """
    weight_mb = float(model_features.get("weight_footprint_mb", 0.0))
    excess_mb = max(0.0, weight_mb - D9_COMPUTE_REFERENCE_WEIGHT_MB)
    return excess_mb * D9_COMPUTE_SAVINGS_US_PER_WEIGHT_MB_ABOVE_REFERENCE


def estimate_compute_savings_us(
    tp1_compute_latency_us: float, tp2_compute_latency_us: float,
    model_features: dict[str, Any],
) -> tuple[float, dict[str, float | str]]:
    base = tp1_compute_latency_us - tp2_compute_latency_us
    base_status = "finite"
    if not math.isfinite(base):
        base = 0.0
        base_status = "non_finite_regression_latency_delta_ignored"
    structural = estimate_structural_compute_savings_adjustment_us(model_features)
    return base + structural, {
        "regression_compute_savings_us": base,
        "regression_compute_savings_status": base_status,
        "structural_compute_savings_adjustment_us": structural,
        "compute_reference_weight_mb": D9_COMPUTE_REFERENCE_WEIGHT_MB,
        "compute_savings_us_per_weight_mb_above_reference": D9_COMPUTE_SAVINGS_US_PER_WEIGHT_MB_ABOVE_REFERENCE,
    }


def adjust_throughput_for_communication(
    predicted_tokens_per_s: float,
    *,
    estimated_nccl_comm_time_us: float,
    output_length: int,
    concurrency: int,
) -> float:
    if predicted_tokens_per_s <= 0:
        return predicted_tokens_per_s
    token_count = max(1, int(output_length) * int(concurrency))
    base_time_s = token_count / predicted_tokens_per_s
    communication_time_s = (estimated_nccl_comm_time_us / 1_000_000.0) * token_count
    return token_count / (base_time_s + communication_time_s)


@dataclass
class FittedRegression:
    tp_degree: int
    coefficients: list[float]  # [intercept, *FEATURE_NAMES]
    n_samples: int
    r_squared: float

    def predict(self, feature_vector: list[float]) -> float:
        x = np.array([1.0] + list(feature_vector))
        return float(np.dot(x, self.coefficients))

    def to_dict(self) -> dict[str, Any]:
        return {"tp_degree": self.tp_degree, "coefficients": self.coefficients,
                "feature_names": ["intercept", *FEATURE_NAMES], "n_samples": self.n_samples,
                "r_squared": self.r_squared}


def fit_linear_regression(X: list[list[float]], y: list[float]) -> FittedRegression:
    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)
    design = np.hstack([np.ones((X_arr.shape[0], 1)), X_arr])
    coeffs, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    preds = design @ coeffs
    ss_res = float(np.sum((y_arr - preds) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return FittedRegression(tp_degree=-1, coefficients=coeffs.tolist(), n_samples=len(y), r_squared=r_squared)


@dataclass
class TPCostModel:
    """Frozen after fit(): must not be refit or adjusted after seeing
    held-out results."""

    throughput_models: dict[int, FittedRegression] = field(default_factory=dict)
    frozen: bool = False

    def fit(self, calibration_rows: list[dict[str, Any]]) -> None:
        if self.frozen:
            raise RuntimeError("cost model is frozen; cannot refit")
        for tp_degree in (1, 2):
            rows = [r for r in calibration_rows if r["tp_degree"] == tp_degree]
            X = [r["feature_vector"] for r in rows]
            y = [r["aggregate_throughput_tokens_per_s"] for r in rows]
            reg = fit_linear_regression(X, y)
            reg.tp_degree = tp_degree
            self.throughput_models[tp_degree] = reg
        self.frozen = True

    def predict_throughput(self, feature_vector: list[float], tp_degree: int) -> float:
        return self.throughput_models[tp_degree].predict(feature_vector)

    def decide(
        self, *, model_features: dict[str, Any], input_length: int, output_length: int, concurrency: int,
        gpu_total_mb: float, gpu_memory_utilization: float, max_model_len: int, max_num_seqs: int,
        communication_calibration: CommunicationPredictor | None = None,
    ) -> dict[str, Any]:
        tp1_feasible, tp1_mem = is_feasible(model_features, 1, gpu_total_mb=gpu_total_mb,
                                             gpu_memory_utilization=gpu_memory_utilization,
                                             max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        tp2_feasible, tp2_mem = is_feasible(model_features, 2, gpu_total_mb=gpu_total_mb,
                                             gpu_memory_utilization=gpu_memory_utilization,
                                             max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        if not tp1_feasible and tp2_feasible:
            return {"decision": "tp2", "reason": "capacity_forced", "tp1_memory": tp1_mem, "tp2_memory": tp2_mem}
        if tp1_feasible and not tp2_feasible:
            return {"decision": "tp1", "reason": "capacity_forced", "tp1_memory": tp1_mem, "tp2_memory": tp2_mem}
        if not tp1_feasible and not tp2_feasible:
            return {"decision": "infeasible", "reason": "neither_tp_degree_fits", "tp1_memory": tp1_mem, "tp2_memory": tp2_mem}

        fv1 = build_feature_vector(model_features, 1, input_length=input_length, output_length=output_length, concurrency=concurrency)
        fv2 = build_feature_vector(model_features, 2, input_length=input_length, output_length=output_length, concurrency=concurrency)
        pred1_before = self.predict_throughput(fv1, 1)
        pred2_before = self.predict_throughput(fv2, 2)
        legacy = communication_calibration is None
        pred1_compute_latency_us = throughput_to_latency_us(pred1_before)
        pred2_compute_latency_us = throughput_to_latency_us(pred2_before)
        estimated_compute_savings_us, compute_savings_evidence = estimate_compute_savings_us(
            pred1_compute_latency_us, pred2_compute_latency_us, model_features)
        if legacy:
            comm1_us = 0.0
            comm2_us = 0.0
            demand1 = estimate_collective_demand(model_features, concurrency=concurrency, tp_degree=1)
            demand2 = estimate_collective_demand(model_features, concurrency=concurrency, tp_degree=2)
            comm_reason = "legacy_no_communication_calibration"
            comm_profile = None
            comm_predictor = None
            topology_class = None
            p2p_available = None
            nccl_transport = None
        else:
            comm1_us, demand1 = estimate_communication_penalty_us(
                model_features, concurrency=concurrency, tp_degree=1,
                communication_calibration=communication_calibration)
            comm2_us, demand2 = estimate_communication_penalty_us(
                model_features, concurrency=concurrency, tp_degree=2,
                communication_calibration=communication_calibration)
            comm_reason = "d9_collective_instance_calibrated_break_even"
            comm_profile = communication_calibration.profile_id
            comm_predictor = communication_calibration.predictor_kind
            topology_class = communication_calibration.topology_class
            p2p_available = communication_calibration.p2p_available
            nccl_transport = communication_calibration.nccl_transport

        estimated_runtime_residual_us = D9_RUNTIME_RESIDUAL_US
        estimated_net_tp2_benefit_us = (
            estimated_compute_savings_us - comm2_us - estimated_runtime_residual_us
        )
        pre_decision = "tp2" if estimated_compute_savings_us > D9_DECISION_MARGIN_US else "tp1"
        decision = "tp2" if estimated_net_tp2_benefit_us > D9_DECISION_MARGIN_US else "tp1"
        pred1_after = pred1_before
        pred2_after = 1_000_000.0 / (pred2_compute_latency_us + comm2_us + estimated_runtime_residual_us) if math.isfinite(pred2_compute_latency_us) else 0.0
        return {
            "decision": decision,
            "reason": "performance_regression" if legacy else "d9_break_even_net_benefit",
            "policy_id": D9_POLICY_ID if not legacy else "legacy_throughput_selector",
            "legacy_behavior": legacy,
            "legacy_reason": comm_reason if legacy else None,
            "pre_communication_decision": pre_decision,
            "communication_changed_decision": pre_decision != decision,
            "predicted_tp1_throughput": pred1_after,
            "predicted_tp2_throughput": pred2_after,
            "predicted_tp1_throughput_before_communication": pred1_before,
            "predicted_tp2_throughput_before_communication": pred2_before,
            "tp1_memory": tp1_mem, "tp2_memory": tp2_mem,
            "break_even": {
                "estimated_compute_savings_us": estimated_compute_savings_us,
                **compute_savings_evidence,
                "estimated_communication_penalty_us": comm2_us,
                "estimated_runtime_residual_us": estimated_runtime_residual_us,
                "estimated_net_tp2_benefit_us": estimated_net_tp2_benefit_us,
                "decision_margin_us": D9_DECISION_MARGIN_US,
                "overlap_assumption": D9_OVERLAP_ASSUMPTION,
            },
            "communication": {
                "estimated_communication_bytes_tp1": demand1["estimated_total_communication_bytes"],
                "estimated_communication_bytes_tp2": demand2["estimated_total_communication_bytes"],
                "estimated_nccl_comm_time_us_tp1": comm1_us,
                "estimated_nccl_comm_time_us_tp2": comm2_us,
                "estimated_collective_call_count": demand2["estimated_collective_call_count"],
                "collective_kind": demand2["collective_kind"],
                "bytes_per_collective_call": demand2["bytes_per_collective_call"],
                "communication_collective_kind": demand2["collective_kind"],
                "communication_profile_id": comm_profile,
                "communication_predictor_kind": comm_predictor,
                "topology_class": topology_class,
                "p2p_available": p2p_available,
                "nccl_transport": nccl_transport,
                "overlap_assumption": D9_OVERLAP_ASSUMPTION,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {"frozen": self.frozen, "throughput_models": {k: v.to_dict() for k, v in self.throughput_models.items()}}
